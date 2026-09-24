#!/usr/bin/env python3
"""Read and write Cities: Skylines II .Geometry files.

Layout (little-endian). Prefix: u16 1, u16 mesh count, u16 0, u32 3. Then one 88-byte
descriptor per mesh (offsets below are for a single mesh, i.e. descriptor at 10), then per
mesh its submesh table + bounds, then every stream (mesh by mesh). One mesh per material.
  10  7 bytes: vertex-attribute format, one nibble per Unity VertexAttribute
      slot (low nibble first), 0xF = attribute absent. Unity VertexAttributeFormat:
      0 Float32, 1 Float16, 2 UNorm8, 3 SNorm8, 4 UNorm16, 5 SNorm16, 6 UInt8,
      7 SInt8, 8 UInt16, 9 SInt16, 10 UInt32, 11 SInt32
  17  u8 0
  18  u32 dimensions: 2 bits per slot, value = dimension - 1
  22  14 x u32 compressed size of each attribute stream (0 = absent)
  78  u32 compressed size of the index stream
  82  u32 0
  86  u32 vertex count, u32 index count
  94  u32 submesh count (1), then per submesh: index start, index count, first vertex, vertex count
  114 6 x f32 bounds: center xyz, extents xyz
Payload: every mesh's index stream, then slot by slot each mesh's stream for that slot.
Every stream is meshoptimizer-encoded (index codec v1 / vertex codec v0) and then
zstd-compressed as its own frame. Indices are 32-bit.

Slots: 0 Position 1 Normal 2 Tangent 3 Color 4-11 TexCoord0-7 12 BlendWeight 13 BlendIndices
Normals: 2 x SNorm16 octahedral. Tangents: 32 bits = octahedral, x in bits 16-30 and
y in bits 0-14 (15-bit unsigned, v/16383.5-1), bit 31 set when bitangent sign is -1.
"""
import struct
import numpy as np
import meshoptimizer as mo
import zstandard

SLOTS = ['position', 'normal', 'tangent', 'color'] + [f'uv{i}' for i in range(8)] + ['blendweight', 'blendindices']
FORMATS = {0: np.float32, 1: np.float16, 2: np.uint8, 3: np.int8, 4: np.uint16, 5: np.int16,
           6: np.uint8, 7: np.int8, 8: np.uint16, 9: np.int16, 10: np.uint32, 11: np.int32}
HEADER = 138

# ---------------------------------------------------------------- packing helpers
def oct_encode(v):
    v = v / np.maximum(np.abs(v).sum(1, keepdims=True), 1e-20)
    x, y = v[:, 0].copy(), v[:, 1].copy(); neg = v[:, 2] < 0
    ox = np.where(x >= 0, 1.0, -1.0) * (1 - np.abs(y)); oy = np.where(y >= 0, 1.0, -1.0) * (1 - np.abs(x))
    x[neg], y[neg] = ox[neg], oy[neg]
    return x, y

def oct_decode(x, y):
    z = 1 - np.abs(x) - np.abs(y); t = np.clip(-z, 0, None)
    x = x + np.where(x >= 0, -t, t); y = y + np.where(y >= 0, -t, t)
    v = np.stack([x, y, z], 1)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-20)

def pack_normals(n):
    x, y = oct_encode(np.asarray(n, np.float64))
    return np.stack([np.round(x * 32767), np.round(y * 32767)], 1).astype(np.int16)

def unpack_normals(raw):
    return oct_decode(raw[:, 0] / 32767.0, raw[:, 1] / 32767.0)

def pack_tangents(t4):
    """t4: (n,4) tangent xyz + w (bitangent sign, +1/-1) -> uint32."""
    t4 = np.asarray(t4, np.float64)
    x, y = oct_encode(t4[:, :3])
    qx = np.clip(np.round((x + 1) * 16383.5), 0, 32767).astype(np.uint32)
    qy = np.clip(np.round((y + 1) * 16383.5), 0, 32767).astype(np.uint32)
    return (qx << 16) | qy | (np.where(t4[:, 3] < 0, 1, 0).astype(np.uint32) << 31)

def unpack_tangents(u):
    u = u.astype(np.uint32)
    x = ((u >> 16) & 0x7fff) / 16383.5 - 1; y = (u & 0x7fff) / 16383.5 - 1
    return np.concatenate([oct_decode(x, y), np.where(u >> 31, -1.0, 1.0)[:, None]], 1)

# ---------------------------------------------------------------- read
def _frames(buf, pos):
    out = []
    while pos < len(buf):
        d = zstandard.ZstdDecompressor().decompressobj()
        data = d.decompress(buf[pos:]); used = len(buf) - pos - len(d.unused_data)
        out.append(data); pos += used
    return out

def read_all(path):
    """Returns a list of meshes (one per material slot)."""
    b = open(path, 'rb').read()
    one, count, zero, three = struct.unpack_from('<HHHI', b, 0)
    pos, descs = 10, []
    for _ in range(count):
        fmt = [(b[pos + i // 2] >> (4 * (i % 2))) & 0xF for i in range(14)]
        dims = struct.unpack_from('<I', b, pos + 8)[0]
        nv, ni, nsub = struct.unpack_from('<3I', b, pos + 76)
        descs.append((fmt, dims, nv, ni, nsub)); pos += 88
    meshes = []
    for fmt, dims, nv, ni, nsub in descs:
        subs = [struct.unpack_from('<4I', b, pos + 16 * i) for i in range(nsub)]; pos += 16 * nsub
        bounds = struct.unpack_from('<6f', b, pos); pos += 24
        meshes.append(dict(vertex_count=nv, index_count=ni, submeshes=subs, center=bounds[:3], extents=bounds[3:],
                           _fmt=fmt, _dims=dims, attrs={}))
    # stream order: every mesh's index stream, then slot by slot, each mesh's stream for that slot
    frames = iter(_frames(b, pos))
    for m in meshes:
        m['indices'] = mo.decode_index_buffer(m['index_count'], 4, next(frames)).astype(np.uint32)
    for slot in range(14):
        for m in meshes:
            f = m['_fmt'][slot]
            if f == 0xF:
                continue
            dim = ((m['_dims'] >> (2 * slot)) & 3) + 1
            dt = np.dtype(FORMATS[f])
            raw = mo.decode_vertex_buffer(m['vertex_count'], dim * dt.itemsize, next(frames)).tobytes()
            m['attrs'][SLOTS[slot]] = (f, np.frombuffer(raw, dt).reshape(m['vertex_count'], dim).copy())
    for m in meshes:
        del m['_fmt'], m['_dims']
    return meshes

def read(path):
    """Single-mesh convenience wrapper."""
    meshes = read_all(path)
    if len(meshes) != 1:
        raise ValueError(f'{len(meshes)} meshes; use read_all')
    m = meshes[0]; m['submesh'] = m['submeshes'][0]
    return m

# ---------------------------------------------------------------- write
def write_all(path, meshes, level=19):
    """meshes: list of dicts with 'indices' and 'attrs' {slot_name: (format_id, array[n, dim])}."""
    zc = zstandard.ZstdCompressor(level=level)
    mo.encode_index_version(1); mo.encode_vertex_version(0)
    descs, tails, idx_blobs, slot_blobs = b'', b'', [], [[] for _ in SLOTS]
    for m in meshes:
        indices = np.ascontiguousarray(m['indices'], np.uint32).ravel()
        attrs = m['attrs']
        nv = attrs['position'][1].shape[0]
        fmt = [0xF] * 14; dims = 0; sizes = [0] * 14
        idx_blob = zc.compress(mo.encode_index_buffer(indices, len(indices), nv))
        for slot, name in enumerate(SLOTS):
            if name not in attrs:
                continue
            f, arr = attrs[name]
            arr = np.ascontiguousarray(arr, FORMATS[f])
            if arr.ndim == 1: arr = arr[:, None]
            assert arr.shape[0] == nv and (arr.shape[1] * arr.itemsize) % 4 == 0, name
            fmt[slot] = f; dims |= (arr.shape[1] - 1) << (2 * slot)
            blob = zc.compress(mo.encode_vertex_buffer(arr.view(np.uint8).reshape(nv, -1), nv, arr.shape[1] * arr.itemsize))
            sizes[slot] = len(blob); slot_blobs[slot].append(blob)
        pos = attrs['position'][1].astype(np.float64); lo, hi = pos.min(0), pos.max(0)
        descs += bytes(fmt[i] | (fmt[i + 1] << 4) for i in range(0, 14, 2)) + b'\x00'
        descs += struct.pack('<I', dims) + struct.pack('<14I', *sizes) + struct.pack('<II', len(idx_blob), 0)
        descs += struct.pack('<3I', nv, len(indices), 1)
        tails += struct.pack('<4I', 0, len(indices), 0, nv) + struct.pack('<6f', *((lo + hi) / 2), *((hi - lo) / 2))
        idx_blobs.append(idx_blob)
    payload = b''.join(idx_blobs) + b''.join(b''.join(s) for s in slot_blobs)
    with open(path, 'wb') as f:
        f.write(struct.pack('<HHHI', 1, len(meshes), 0, 3) + descs + tails + payload)

def write(path, indices, attrs, level=19):
    write_all(path, [dict(indices=indices, attrs=attrs)], level)

if __name__ == '__main__':
    import sys
    for i, m in enumerate(read_all(sys.argv[1])):
        print(i, {k: v for k, v in m.items() if k not in ('attrs', 'indices')})
        for k, (f, a) in m['attrs'].items():
            print(f'  {k:13} format {f:2} dims {a.shape[1]} dtype {a.dtype}')
