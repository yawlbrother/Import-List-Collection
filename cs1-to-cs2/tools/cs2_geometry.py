#!/usr/bin/env python3
"""Read and write Cities: Skylines II .Geometry files.

Layout (little-endian, 138-byte header for a single-submesh mesh):
  0   u16 1, u16 1, u16 0, u32 3            (constant in every sample seen)
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
Payload: index stream then each present attribute stream in slot order.
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

def read(path):
    b = open(path, 'rb').read()
    fmt_nibbles = [(b[10 + i // 2] >> (4 * (i % 2))) & 0xF for i in range(14)]
    dims_bits = struct.unpack_from('<I', b, 18)[0]
    sizes = struct.unpack_from('<14I', b, 22)
    nv, ni, nsub = struct.unpack_from('<3I', b, 86)
    if nsub != 1:
        raise NotImplementedError(f'{nsub} submeshes')
    sub = struct.unpack_from('<4I', b, 98)
    bounds = struct.unpack_from('<6f', b, 114)
    frames = _frames(b, HEADER)
    mesh = dict(vertex_count=nv, index_count=ni, submesh=sub, center=bounds[:3], extents=bounds[3:], attrs={})
    mesh['indices'] = mo.decode_index_buffer(ni, 4, frames[0]).astype(np.uint32)
    k = 1
    for slot in range(14):
        if fmt_nibbles[slot] == 0xF:
            continue
        dim = ((dims_bits >> (2 * slot)) & 3) + 1
        dt = np.dtype(FORMATS[fmt_nibbles[slot]])
        stride = dim * dt.itemsize
        raw = mo.decode_vertex_buffer(nv, stride, frames[k]).tobytes(); k += 1
        mesh['attrs'][SLOTS[slot]] = (fmt_nibbles[slot], np.frombuffer(raw, dt).reshape(nv, dim).copy())
    return mesh

# ---------------------------------------------------------------- write
def write(path, indices, attrs, level=19):
    """attrs: {slot_name: (format_id, array[n, dim])} using the raw stored types."""
    indices = np.ascontiguousarray(indices, np.uint32).ravel()
    nv = next(iter(attrs.values()))[1].shape[0]
    fmt = [0xF] * 14; dims = 0; sizes = [0] * 14; payload = []
    zc = zstandard.ZstdCompressor(level=level)
    mo.encode_index_version(1); mo.encode_vertex_version(0)
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
        sizes[slot] = len(blob); payload.append(blob)
    pos = attrs['position'][1].astype(np.float64)
    lo, hi = pos.min(0), pos.max(0)
    h = bytearray()
    h += struct.pack('<HHHI', 1, 1, 0, 3)
    h += bytes(fmt[i] | (fmt[i + 1] << 4) for i in range(0, 14, 2)) + b'\x00'
    h += struct.pack('<I', dims) + struct.pack('<14I', *sizes)
    h += struct.pack('<II', len(idx_blob), 0)
    h += struct.pack('<3I', nv, len(indices), 1) + struct.pack('<4I', 0, len(indices), 0, nv)
    h += struct.pack('<6f', *((lo + hi) / 2), *((hi - lo) / 2))
    assert len(h) == HEADER
    with open(path, 'wb') as f:
        f.write(bytes(h) + idx_blob + b''.join(payload))
    return dict(vertex_count=nv, index_count=len(indices), center=(lo + hi) / 2, extents=(hi - lo) / 2)

if __name__ == '__main__':
    import sys
    m = read(sys.argv[1])
    print({k: v for k, v in m.items() if k not in ('attrs', 'indices')})
    for k, (f, a) in m['attrs'].items():
        print(f'  {k:13} format {f:2} dims {a.shape[1]} dtype {a.dtype}')
