#!/usr/bin/env python3
"""Tiny software renderer (numpy z-buffer) for icons and preview images.

render(parts, ...) where each part is (V[n,3], N[n,3], UV[n,2], tris[m,3], texture RGB image)
in Unity space (left-handed, +Y up). Texture images are normal top-row-first PNG orientation.
"""
import numpy as np
from PIL import Image

def render(parts, width, height, yaw_deg, pitch_deg, distance, fov_deg, target, bg=(236, 238, 241),
           light=(0.4, 0.8, 0.45), ambient=0.55, gain=1.25):
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    img = np.empty((height, width, 3), np.float32); img[:] = np.array(bg) / 255
    zb = np.full((height, width), np.inf, np.float32)
    L = np.array(light, np.float64); L /= np.linalg.norm(L)
    f = 0.5 * height / np.tan(np.radians(fov_deg) / 2)
    for V, N, UV, T, tex in parts:
        tex = np.asarray(tex.convert('RGB'), np.float32) / 255; th, tw = tex.shape[:2]
        P = np.asarray(V, np.float64) - np.asarray(target, np.float64)
        x = cy * P[:, 0] - sy * P[:, 2]; z = sy * P[:, 0] + cy * P[:, 2]; y = P[:, 1]
        y2 = cp * y - sp * z; z2 = sp * y + cp * z + distance
        sx = width / 2 + f * x / z2; syy = height / 2 - f * y2 / z2
        shade = (ambient + (1 - ambient) * np.clip(np.abs(np.asarray(N) @ L), 0, 1)) * gain
        UV = np.asarray(UV, np.float64)
        for a, b, c in np.asarray(T):
            xs = np.array([sx[a], sx[b], sx[c]]); ys = np.array([syy[a], syy[b], syy[c]])
            x0, x1 = max(int(xs.min()), 0), min(int(xs.max()) + 1, width - 1)
            y0, y1 = max(int(ys.min()), 0), min(int(ys.max()) + 1, height - 1)
            if x0 > x1 or y0 > y1:
                continue
            den = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
            if abs(den) < 1e-9:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            w0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / den
            w1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / den
            w2 = 1 - w0 - w1; m = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not m.any():
                continue
            zz = w0 * z2[a] + w1 * z2[b] + w2 * z2[c]
            sub = zb[y0:y1 + 1, x0:x1 + 1]; m &= zz < sub
            if not m.any():
                continue
            u = w0 * UV[a, 0] + w1 * UV[b, 0] + w2 * UV[c, 0]; v = w0 * UV[a, 1] + w1 * UV[b, 1] + w2 * UV[c, 1]
            tx = (np.mod(u, 1) * (tw - 1)).astype(int); ty = ((1 - np.mod(v, 1)) * (th - 1)).astype(int)
            col = tex[ty, tx] * (w0 * shade[a] + w1 * shade[b] + w2 * shade[c])[..., None]
            sub[m] = zz[m]; img[y0:y1 + 1, x0:x1 + 1][m] = col[m]
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))

def crop_to_content(im, pad=12, square=False):
    from PIL import ImageChops
    bg = Image.new('RGB', im.size, im.getpixel((0, 0)))
    bb = ImageChops.difference(im.convert('RGB'), bg).getbbox()
    if not bb:
        return im
    x0, y0, x1, y1 = bb[0] - pad, bb[1] - pad, bb[2] + pad, bb[3] + pad
    if square:
        s = max(x1 - x0, y1 - y0); cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        x0, x1, y0, y1 = cx - s // 2, cx + s // 2, cy - s // 2, cy + s // 2
    out = Image.new('RGB', (x1 - x0, y1 - y0), im.getpixel((0, 0)))
    out.paste(im.crop((max(x0, 0), max(y0, 0), min(x1, im.width), min(y1, im.height))), (max(-x0, 0), max(-y0, 0)))
    return out
