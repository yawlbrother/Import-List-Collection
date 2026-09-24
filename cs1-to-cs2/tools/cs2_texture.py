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

def write(path, img, srgb, flip=True):
    """img: PIL image in normal top-row-first orientation. Writes the full mip chain."""
    img = img.convert('RGBA')
    if flip:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    w, h = img.size
    sizes = mip_sizes(w, h)
    data = bytearray()
    for mw, mh in sizes:
        m = img if (mw, mh) == (w, h) else img.resize((mw, mh), Image.LANCZOS)
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
