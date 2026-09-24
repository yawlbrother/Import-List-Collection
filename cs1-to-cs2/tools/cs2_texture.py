#!/usr/bin/env python3
"""Read and write Cities: Skylines II plain .Texture files (the non-virtual-texture kind
found in ImportedData and in mods' StreamingData~ folders).

Layout: 18-byte header, then the full BC7 mip chain, largest first.
  u16 5 (version), u16 width, u16 height, u16 1 (depth), u8 mip count,
  u8 Unity GraphicsFormat (108 = RGBA_BC7_SRGB for BaseColor/Emissive,
  109 = RGBA_BC7_UNorm for Normal/MaskMap/ControlMask), then 02 02 01 10 ff ff ff ff.

    python cs2_texture.py FILE.Texture OUT.png   # decode mip 0 to PNG

Needs: pip install etcpak texture2ddecoder pillow numpy
"""
import struct
import numpy as np
from PIL import Image

SRGB, LINEAR = 108, 109
TAIL = bytes.fromhex('02020110ffffffff')

def mip_sizes(w, h):
    out = [(w, h)]
    while w > 1 or h > 1:
        w, h = max(1, w // 2), max(1, h // 2); out.append((w, h))
    return out

def _blocks(w, h):
    return ((w + 3) // 4) * ((h + 3) // 4) * 16

def read(path, flip=True):
    """Returns (PIL RGBA image of mip 0, header dict). flip=True returns it top-row-first."""
    import texture2ddecoder
    b = open(path, 'rb').read()
    ver, w, h, depth, mips, fmt = struct.unpack_from('<HHHHBB', b, 0)
    assert ver == 5 and fmt in (SRGB, LINEAR), (ver, fmt)
    pw, ph = (w + 3) // 4 * 4, (h + 3) // 4 * 4
    raw = texture2ddecoder.decode_bc7(b[18:18 + _blocks(w, h)], pw, ph)   # BGRA
    img = Image.frombytes('RGBA', (pw, ph), raw, 'raw', 'BGRA').crop((0, 0, w, h))
    if flip:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return img, dict(width=w, height=h, mips=mips, format=fmt)

def _encode_bc7(img):
    import etcpak
    w, h = img.size
    pw, ph = (w + 3) // 4 * 4, (h + 3) // 4 * 4
    if (pw, ph) != (w, h):                       # pad by edge replication
        a = np.asarray(img)
        a = np.pad(a, ((0, ph - h), (0, pw - w), (0, 0)), mode='edge')
        img = Image.fromarray(a, 'RGBA')
    return etcpak.compress_bc7(img.tobytes(), pw, ph)

def _max_alpha_mip(a):
    """Halve an RGBA array keeping, per 2x2 block, the texel with the highest alpha (ties: top-left).
    Emissive maps store a light *layer id* in alpha, so averaging (the normal mip filter) turns
    a window's layer into garbage a few mips down and the lights vanish on distant objects."""
    h, w = a.shape[:2]
    if h > 1 and w > 1:
        q = np.stack([a[0::2, 0::2], a[0::2, 1::2], a[1::2, 0::2], a[1::2, 1::2]], 0)
    elif h > 1:
        q = np.stack([a[0::2], a[1::2]], 0)
    else:
        q = np.stack([a[:, 0::2], a[:, 1::2]], 0)
    best = q[..., 3].argmax(0)
    return np.take_along_axis(q, best[None, ..., None], 0)[0]

def write(path, img, srgb, flip=True, mip_filter='lanczos'):
    """img: PIL image in normal top-row-first orientation. Writes the full mip chain.
    mip_filter: 'lanczos' (default) or 'max_alpha' (see _max_alpha_mip; use for emissive maps)."""
    img = img.convert('RGBA')
    if flip:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    w, h = img.size
    sizes = mip_sizes(w, h)
    data = bytearray()
    cur = np.asarray(img)
    for mw, mh in sizes:
        if (mw, mh) != (w, h):
            if mip_filter == 'max_alpha':
                cur = _max_alpha_mip(cur); m = Image.fromarray(np.ascontiguousarray(cur), 'RGBA')
            else:
                m = img.resize((mw, mh), Image.LANCZOS)
        else:
            m = img
        blob = _encode_bc7(m)
        assert len(blob) == _blocks(mw, mh), (len(blob), mw, mh)
        data += blob
    hdr = struct.pack('<HHHHBB', 5, w, h, 1, len(sizes), SRGB if srgb else LINEAR) + TAIL
    with open(path, 'wb') as f:
        f.write(hdr + data)
    return dict(width=w, height=h, mips=len(sizes), bytes=18 + len(data))

if __name__ == '__main__':
    import sys
    img, info = read(sys.argv[1])
    print(info)
    if len(sys.argv) > 2:
        img.save(sys.argv[2])
