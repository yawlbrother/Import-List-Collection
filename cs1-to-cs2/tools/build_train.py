#!/usr/bin/env python3
"""Build an installable CS2 multiple-unit train from a CS1 train .crp package.

    python build_train.py ASSET.crp --name SJX40 --title "SJ X40" --out ../dist \
        --front 33:36 --car 0:6 --car 15:18

--front / --car take MESH:LOD entry indices from `crp_tool.py list`. Cars are listed in
consist order behind the front car; the game mirrors the front car onto the back
(m_AddReversedEndCarriage). The same MESH:LOD may appear several times (one carriage
prefab is written and referenced at every position), and the front car's mesh may be
used as a carriage too. Bogies are found from the CS1 wheel data (VehicleInfoGen
m_tyres) and become Wheelset/Axle bones; vertices below the body skirt near a bogie
are skinned to it, wheel discs to their axle.
"""
import argparse, os, shutil, struct, zipfile
import numpy as np
from PIL import Image

import crp_tool as C
import brand as B
import cs2_asset as A
import cs2_geometry as G
import cs2_texture as T
import preview
from build_prop import (did, cs1_materials, material_for_mesh, convert_textures, cs1_mesh_to_cs2,
                        SURFACE_SLOT, write_scripts, DEFAULT_UI_GROUP as PROP_UI_GROUP)

# Vanilla GUIDs copied from a working custom train (Stadler KISS Caltrain pack)
ACTIVITY_BOARD = '55cd31323498ccf4ca4831d41291c0f4'
EFFECT_HEADLIGHT = 'de32f8ad166d2d944b6f069e25675da1'
EFFECT_TRAIN = '76d8c52143e7618448f2ff10af98015c'
BOGIE_TOP_Y = 1.0        # bogie frame and axle boxes stay below this
BOGIE_MARGIN = 0.35      # bogie box reaches this far beyond the outer wheel rims
WHEEL_TOP_Y = 0.72
SURFACE_KEYWORDS = ('_EMISSIVE_PROCEDURAL', '_TANGENTSPACE_OCTO')
WHITE, WARM, RED, BLACK, CAB, AMBER = (1, 1, 1, 1), (1, 0.9974498, 0.745, 1), (1, 0, 0, 1), (0, 0, 0, 1), (0.8, 0.88, 1, 1), (1, 0.6, 0.15, 1)
# ---------------------------------------------------------------- lights
# Lights. The emissive texture's RGB is the light colour and its alpha marks lit texels (255).
# WHICH light a texel belongs to is not in the texture: it is the per-vertex colour attribute of
# the geometry (the game's importer writes the 1-based index of the light into it; the KISS body has
# colour (1,0,0,0) on its 176 window vertices, its headlight mesh (1|2|3,0,0,0) on the lamp quads).
# Without that attribute the shader reads Unity's default white and every lit texel picks the LAST
# light. All four channels get the same index here so it works whichever channel the shader reads.
# Purposes (Game.Prefabs.EmissiveProperties.Purpose): 23 Interior1 = on at night (needs the car's
# InteriorLights flag, which every car gets), 3 Headlight_LowBeam = leading car, 6 RearLight =
# trailing car. Intensity/luminance = KISS values. layerId is only read by the editor.
LIGHT_GROUPS = {   # name -> (light index, texture rgb, purpose, colour, intensity, luminance, layerId)
    'windows':     (1, (255, 235, 200), 23, WARM,  0.05, 0.964706, 255),
    'lamps_red':   (2, (255, 40, 40),   6,  RED,   1.0,  1.0,      51),
    'lamps_white': (3, (255, 255, 255), 3,  WHITE, 1.0,  1.0,      25),
    'cab':         (4, (200, 215, 255), 23, CAB,   0.02, 1.0,      76),
}
# Door indicator lamps live on their own sub-mesh with its own two-entry light list (the KISS pack
# does the same for its headlights), so the body mesh never needs a light index above 4.
# 69 BoardingLightLeft / 70 BoardingLightRight: lit (colour) while boarding on that side, colorOff otherwise.
DOOR_LIGHTS = [('door_left', 69), ('door_right', 70)]
DOOR_RGB, DOOR_PATCH_INDEX = (255, 160, 40), 5
LIT_ALPHA = 255
# Static props get no per-car flags, so their interior lights become DecorativeLight (34: always on,
# invisible by day thanks to auto exposure) and the lamps and door lights stay off (purpose 0).
PROP_PURPOSES = {23: 34, 3: 0, 6: 0, 69: 0, 70: 0}

def light_table(scale=1.0, prop=False):
    """EmissiveProperties multi-light list, in light-index order. scale multiplies the interior
    (window and cab) intensities; --calibrate builds a few cars at different scales."""
    return [(PROP_PURPOSES.get(p, p) if prop else p, c, BLACK, i * (scale if p == 23 else 1.0), l, layer)
            for _, _, p, c, i, l, layer in sorted(LIGHT_GROUPS.values())]

def door_light_table(canary=False):
    """Light list of the door-lamp sub-mesh. canary=True makes the left lamps always-on (purpose 1)
    so a look at the train tells whether the lamp itself renders, independent of the boarding flag."""
    return [(1 if (canary and n == 'door_left') else p, AMBER, BLACK, 0.3, 1.0, 102 + 25 * k)
            for k, (n, p) in enumerate(DOOR_LIGHTS)]

CALIBRATION = (1.0, 0.4, 0.2, 0.1)   # --calibrate: interior intensity scale of the front car and carriage types A, B, C
NOSE_FRACTION = 0.75                 # glass on the last quarter of the car's length is cab glass
DOOR_LAMP = (0.16, 0.08, 3.02)       # door indicator lamp: width (along the car), height, centre height
LOD1_RATIO = 0.4                     # LOD1 keeps this share of LOD0's triangles (KISS: 44 %)
PATCH = 12                           # atlas texels reserved for the door lamp patch

# Line colour: the door leaves take the transport line's colour. ControlMask R marks the texels
# (colour set channel 0), the shader multiplies the base colour by the channel colour, so those
# texels are lightened first (a black door times any colour is black); the variation's own channel
# colour is a dark grey that restores the livery look when the car is not on a line (props, depot).
DOOR_LEAF = (0.72, 0.45, 2.35)       # half width along the car, bottom and top of a door leaf (m)
# More line-coloured elements: a pinstripe along the lower body, painted by world height (every
# side-facing triangle is clipped to the slab and the piece rasterised in UV space, so it runs the
# whole car whatever the atlas layout) and, with a brand band, a short rule under each text block
# (atlas rectangles x0, y0, x1, y1 on the 2048 x 1024 layout).
LINE_STRIPE_Y = (0.62, 0.70)
LINE_BAND_RULES = [(604, 316, 932, 319), (1018, 316, 1346, 319)]
LINE_BASE = (0.35, 160)              # base colour under the mask: base * 0.35 + 160
LINE_DEFAULT = (0.19, 0.19, 0.21, 1) # channel 0 colour off-line: about the livery's dark grey again
RED_FRACTION = 0.4       # bottom part of a combined head/tail lamp lens that glows red
UPPER_LAMP_GAP = 0.8     # lamps this much above the lowest lamp are roof lamps: white only

def dark_components(mask):
    """4-connected components of a boolean mask: list of (pixel mask, bbox x0, y0, x1, y1)."""
    from collections import deque
    H, W = mask.shape; seen = np.zeros_like(mask); out = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        seen[y0, x0] = True; q = deque([(y0, x0)]); pts = []
        while q:
            y, x = q.popleft(); pts.append((y, x))
            for yy, xx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= yy < H and 0 <= xx < W and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True; q.append((yy, xx))
        if len(pts) > 50:
            pts = np.array(pts); comp = np.zeros_like(mask); comp[pts[:, 0], pts[:, 1]] = True
            out.append((comp, pts[:, 1].min(), pts[:, 0].min(), pts[:, 1].max() + 1, pts[:, 0].max() + 1))
    return out

def light_regions(aci, mesh=None):
    """From CS1's ACI map (blue = illumination): the passenger windows the author lit at night (16-40)
    plus the glass they left dark (0: door and end windows), and the lamp lenses (>200). Dark glass
    on the nose of `mesh` (windscreen, cab side windows) is the cab instead of a passenger window."""
    I = np.asarray(aci.convert('RGBA')).astype(int)[..., 2]; H, W = I.shape
    windows = (I >= 16) & (I <= 40); cab = np.zeros((H, W), bool)
    if mesh is not None:
        u, v = uv_pixels(mesh, W, H); t3 = tri_array(mesh); cu, cv = u[t3].mean(1), v[t3].mean(1)
        cz = np.abs(np.asarray(mesh['V'])[t3].mean(1)[:, 2]); nose = NOSE_FRACTION * cz.max()
        for comp, x0, y0, x1, y1 in dark_components(I <= 10):
            sel = (cu >= x0) & (cu < x1) & (cv >= y0) & (cv < y1)
            if sel.any() and cz[sel].mean() > nose:
                cab |= comp
            else:
                windows |= comp
    return {'windows': windows, 'cab': cab, 'lamps': I > 200}

def bbox(mask):
    ys, xs = np.nonzero(mask)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1) if len(xs) else None

def tri_array(m):
    return np.concatenate([np.asarray(t, np.int64).reshape(-1, 3) for t in m['tris']])

def uv_pixels(m, W, H):
    """Vertex UVs as texture pixel coordinates (column, row from the top)."""
    UV = np.asarray(m['UV'], np.float64)
    return UV[:, 0] * (W - 1), (1 - UV[:, 1]) * (H - 1)

def lamp_triangles(m, box, W, H):
    """Indices of the triangles whose UV centroid lies on the lamp lens texels, plus any triangle that
    contains the lens centre (coarse LOD meshes paint the lens onto one big nose polygon)."""
    u, v = uv_pixels(m, W, H); t3 = tri_array(m); cu, cv = u[t3].mean(1), v[t3].mean(1); x0, y0, x1, y1 = box
    inside = (cu >= x0 - 2) & (cu < x1 + 2) & (cv >= y0 - 2) & (cv < y1 + 2)
    px, py = (x0 + x1) / 2, (y0 + y1) / 2
    P = [np.stack([u[t3[:, k]], v[t3[:, k]]], 1) for k in range(3)]
    def edge(p, q): return (q[:, 0] - p[:, 0]) * (py - p[:, 1]) - (q[:, 1] - p[:, 1]) * (px - p[:, 0])
    e = [edge(P[0], P[1]), edge(P[1], P[2]), edge(P[2], P[0])]
    contains = ((e[0] >= 0) & (e[1] >= 0) & (e[2] >= 0)) | ((e[0] <= 0) & (e[1] <= 0) & (e[2] <= 0))
    return np.nonzero(inside | contains)[0]

def lamp_groups(m, tri_idx):
    """Splits lamp triangles into (lower, upper) by world height; roof lamps are white only."""
    if not len(tri_idx):
        return tri_idx, tri_idx
    y = np.asarray(m['V'])[tri_array(m)[tri_idx]].mean(1)[:, 1]
    upper = y > y.min() + UPPER_LAMP_GAP
    return tri_idx[~upper], tri_idx[upper]

def red_rows_from_top(m, tri_idx, W, H):
    """True when texture rows run against world height on the lower lamps, i.e. the top rows of the
    lens texels are the bottom of the lamp, where the red tail light goes. None if undecidable."""
    if len(tri_idx) < 2:
        return None
    vi = np.unique(tri_array(m)[tri_idx]); V = np.asarray(m['V']); _, v = uv_pixels(m, W, H)
    if np.ptp(v[vi]) < 1 or np.ptp(V[vi, 1]) < 1e-3:
        return None
    return bool(np.corrcoef(v[vi], V[vi, 1])[0, 1] > 0)

def uv_coverage(meshes, W, H):
    from PIL import ImageDraw
    mask = Image.new('L', (W, H), 0); d = ImageDraw.Draw(mask)
    for m in meshes:
        u, v = uv_pixels(m, W, H)
        for a, b_, c in tri_array(m):
            d.polygon([(u[a], v[a]), (u[b_], v[b_]), (u[c], v[c])], fill=255)
    return np.asarray(mask) > 0

BAND_STRIP = dict(rows=(292, 328), y=(2.3, 2.7), cols=(570, 1380), shift=-26.5)   # see align_band_strip

def align_band_strip(m, W=2048, H=1024):
    """The CS1 middle cars map the panel strip between the window rows (y 2.34..2.66) to atlas rows
    295..325, the cab car maps the same strip to rows 268..298; below it both map rows 298..330. On the
    band that means the middle car shows the lower part twice and never the upper part. This gives
    the middle car's upper strip the cab car's mapping (its own vertex copies, shifted 26.5 rows up)
    so one band layout reads the same on every car."""
    V = np.asarray(m['V'], np.float64); UV = np.asarray(m['UV'], np.float64); t3 = tri_array(m)
    u = UV[:, 0] * (W - 1); v = (1 - UV[:, 1]) * (H - 1)
    r0, r1 = BAND_STRIP['rows']; y0, y1 = BAND_STRIP['y']; c0, c1 = BAND_STRIP['cols']
    tv, tu, ty, tx = v[t3], u[t3], V[t3][:, :, 1], V[t3][:, :, 0]
    sel = ((tv.min(1) >= r0) & (tv.max(1) <= r1) & (ty.min(1) >= y0) & (ty.max(1) <= y1)
           & (tu.min(1) >= c0) & (tu.max(1) <= c1) & (np.abs(tx).min(1) > 1.2))
    idx = np.nonzero(sel)[0]
    return relocate_triangles(m, idx, 0.0, -BAND_STRIP['shift'] / (H - 1)), len(idx)

def door_leaf_triangles(m, doors):
    """Indices of the body triangles that are door leaves: on the side skin next to a CS1 door
    position, between DOOR_LEAF's bottom and top, facing sideways."""
    V = np.asarray(m['V'], np.float64); N = np.asarray(m['N'], np.float64); t3 = tri_array(m)
    if not len(doors) or not len(t3):
        return np.zeros(0, np.int64)
    hw, y0, y1 = DOOR_LEAF; c = V[t3].mean(1); n = N[t3].mean(1)
    side = (np.abs(n[:, 0]) > 0.6) & (c[:, 1] > y0) & (c[:, 1] < y1)
    near = np.zeros(len(t3), bool)
    for x, _, z in doors:
        skin = np.abs(V[(np.abs(V[:, 2] - z) < hw) & (V[:, 1] > y0) & (V[:, 1] < y1), 0])
        bx = float(skin.max()) if len(skin) else abs(x)
        near |= (np.abs(c[:, 2] - z) < hw) & (np.sign(c[:, 0]) == np.sign(x)) & (np.abs(c[:, 0]) > bx - 0.15)
    return np.nonzero(side & near)[0]

def uv_mask(m, tri_idx, W, H):
    """Texels covered by the given triangles of mesh `m` (bool H x W)."""
    from PIL import ImageDraw
    mask = Image.new('L', (W, H), 0); d = ImageDraw.Draw(mask); u, v = uv_pixels(m, W, H)
    for a, b_, c in tri_array(m)[tri_idx]:
        d.polygon([(u[a], v[a]), (u[b_], v[b_]), (u[c], v[c])], fill=255)
    return np.asarray(mask) > 0

def slab_mask(m, y0, y1, W, H):
    """Texels of the side skin between heights y0 and y1: each side-facing triangle is clipped to the
    slab (Sutherland-Hodgman on y, UVs interpolated) and the clipped polygon filled in UV space."""
    from PIL import ImageDraw
    V = np.asarray(m['V'], np.float64); N = np.asarray(m['N'], np.float64); t3 = tri_array(m)
    u, v = uv_pixels(m, W, H); mask = Image.new('L', (W, H), 0); d = ImageDraw.Draw(mask)
    side = (np.abs(N[t3].mean(1)[:, 0]) > 0.6) & (np.abs(V[t3][:, :, 0]).min(1) > 1.2)
    ys = V[t3][:, :, 1]
    for i in np.nonzero(side & (ys.min(1) < y1) & (ys.max(1) > y0))[0]:
        poly = [(V[k, 1], u[k], v[k]) for k in t3[i]]
        for lo, sign in ((y0, 1.0), (y1, -1.0)):          # keep y >= y0, then y <= y1
            out = []
            for a, b in zip(poly, poly[1:] + poly[:1]):
                ia, ib = sign * (a[0] - lo) >= 0, sign * (b[0] - lo) >= 0
                if ia:
                    out.append(a)
                if ia != ib:
                    t = (lo - a[0]) / (b[0] - a[0]); out.append((lo, a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])))
            poly = out
        if len(poly) >= 3:
            d.polygon([(p[1], p[2]) for p in poly], fill=255)
    return np.asarray(mask) > 0

def paint_line_mask(slots, meshes, doors_by_key, lay, band_rules=False):
    """Marks the door leaves of every mesh on this atlas in ControlMask R and lightens their base
    colour, plus the LINE_STRIPES pinstripe and (band_rules) the LINE_BAND_RULES; lit texels (door
    windows, brand glow) stay as they are. Returns the mask and the triangle counts."""
    W, H = lay['W'], lay['H']; mask = np.zeros((H, W), bool); counts = {}
    for key, m in meshes.items():
        tri = door_leaf_triangles(m, doors_by_key.get(key, ())); counts[str(key)[:8]] = len(tri)
        mask |= uv_mask(m, tri, W, H)
    sc = W / 2048.0
    if sc >= 0.5:
        stripe = np.zeros((H, W), bool); shared = np.zeros((H, W), bool)
        for m in meshes.values():
            stripe |= slab_mask(m, *LINE_STRIPE_Y, W, H)
            V = np.asarray(m['V'], np.float64); N = np.asarray(m['N'], np.float64); t3 = tri_array(m)
            other = np.nonzero((np.abs(N[t3].mean(1)[:, 0]) <= 0.6) | (np.abs(V[t3][:, :, 0]).min(1) <= 1.2))[0]
            shared |= uv_mask(m, other, W, H)      # texels the roof, ends and underframe also sample
        mask |= stripe & ~shared
    for x0, y0, x1, y1 in (LINE_BAND_RULES if band_rules else []):
        if (y1 - y0) * sc >= 2:
            mask[int(y0 * sc):int(y1 * sc), int(x0 * sc):int(x1 * sc)] = True
    mask &= lay['index_map'] == 0
    if mask.any():
        cm = np.array(slots['ControlMask'][0]); cm[mask, 0] = 255; slots['ControlMask'][0].paste(Image.fromarray(cm))
        bc = np.array(slots['BaseColor'][0]).astype(np.float32)
        bc[mask, :3] = bc[mask, :3] * LINE_BASE[0] + LINE_BASE[1]
        slots['BaseColor'][0].paste(Image.fromarray(np.clip(bc, 0, 255).astype(np.uint8)))
    lay['line_mask'] = mask
    return mask, counts

def find_free_rect(cov, w, h, margin=8):
    """Top-left (x, y), 4-aligned, of a w x h texel rectangle no triangle maps to, with `margin` free around it."""
    Wm, Hm = w + 2 * margin, h + 2 * margin; cH, cW = cov.shape
    if Wm > cW or Hm > cH:
        return None
    ii = np.pad(cov.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    S = ii[Hm:, Wm:] - ii[:-Hm, Wm:] - ii[Hm:, :-Wm] + ii[:-Hm, :-Wm]
    for y, x in np.argwhere(S == 0):
        x4, y4 = (x + margin + 3) // 4 * 4, (y + margin + 3) // 4 * 4
        if x4 + w + 4 <= x + Wm and y4 + h + 4 <= y + Hm:
            return int(x4), int(y4)
    return None

def relocate_triangles(m, tri_idx, du, dv):
    """Gives the vertices used by `tri_idx` their own copies with UVs shifted by (du, dv), so those
    triangles sample a relocated texel patch while vertices shared with other triangles stay put."""
    if not len(tri_idx):
        return m
    m = dict(m); t3 = tri_array(m); vi = np.unique(t3[tri_idx]); n = len(m['V'])
    for key in ('V', 'N', 'UV', 'T', 'C'):
        if key in m and len(m[key]) == n:
            arr = np.asarray(m[key]); m[key] = np.concatenate([arr, arr[vi]])
    UV = np.asarray(m['UV'], np.float64).copy(); UV[n:, 0] += du; UV[n:, 1] += dv; m['UV'] = UV
    lut = np.arange(n); lut[vi] = n + np.arange(len(vi)); t3[tri_idx] = lut[t3[tri_idx]]
    m['tris'] = [t3.ravel()]
    return m

def lamp_layout(aci, meshes, ref_key, relocate=True):
    """How the lamp lenses of one texture atlas are lit. `meshes`: {car key: mesh} of every mesh that
    samples this atlas (they share it). Returns a dict: regions, box, red_top, copy (x, y) of a
    white-only duplicate of the lens texels for roof lamps, and per-key roof-lamp triangle indices.
    relocate=False (LOD meshes) only decides the red/white split."""
    regions = light_regions(aci, meshes[ref_key]); H, W = regions['lamps'].shape; box = bbox(regions['lamps'])
    lay = dict(regions=regions, W=W, H=H, box=box, red_top=None, copy=None, upper={})
    if box is None:
        return lay
    lower = {}
    for key, m in meshes.items():
        tri = lamp_triangles(m, box, W, H)
        lower[key], lay['upper'][key] = lamp_groups(m, tri) if relocate else (tri, tri[:0])
    lay['red_top'] = red_rows_from_top(meshes[ref_key], lower[ref_key], W, H)
    cov = uv_coverage(meshes.values(), W, H)
    if any(len(t) for t in lay['upper'].values()):
        x0, y0, x1, y1 = box
        lay['copy'] = find_free_rect(cov, x1 - x0 + 4, y1 - y0 + 4)
        if lay['copy'] is None:
            lay['upper'] = {k: np.zeros(0, np.int64) for k in meshes}
        else:
            cx, cy = lay['copy']; cov[cy - 8:cy + y1 - y0 + 12, cx - 8:cx + x1 - x0 + 12] = True
    lay['door_patch'] = find_free_rect(cov, PATCH, PATCH) if relocate else None
    return lay

def apply_lamp_layout(m, key, lay):
    """Points the roof-lamp triangles of mesh `m` at the white-only lens copy."""
    if lay['copy'] is None or not len(lay['upper'].get(key, ())):
        return m
    x0, y0, _, _ = lay['box']; cx, cy = lay['copy']
    return relocate_triangles(m, lay['upper'][key], (cx + 2 - x0) / (lay['W'] - 1), -(cy + 2 - y0) / (lay['H'] - 1))

PATCH_COLORS = {'BaseColor': (55, 55, 60, 255), 'Normal': (255, 128, 128, 128), 'MaskMap': (0, 0, 0, 200), 'ControlMask': (0, 0, 0, 0)}

def copy_lamp_texels(slots, lay):
    """Duplicates the lens texels (plus a 2 px rim) into the free spot in every non-emissive slot, and
    paints the door lamp patch (dark housing, flat normal, glossy)."""
    for slot, (im, srgb) in slots.items():
        if slot == 'Emissive':
            continue
        if lay['copy'] is not None:
            x0, y0, x1, y1 = lay['box']; cx, cy = lay['copy']
            im.paste(im.crop((x0 - 2, y0 - 2, x1 + 2, y1 + 2)), (cx, cy))
        if lay['door_patch'] is not None:
            px, py = lay['door_patch']; im.paste(PATCH_COLORS[slot], (px, py, px + PATCH, py + PATCH))

def paint_emissive(lay):
    """Emissive texture (RGB = light colour, A = 255 on lit texels, 0 elsewhere) and the light index
    map (uint8 per texel: 0 = unlit, else the light's 1-based index) used for the vertex colours."""
    R = lay['regions']; H, W = R['windows'].shape; out = np.zeros((H, W, 4), np.uint8); idx = np.zeros((H, W), np.uint8)
    lamps, box, red_top = R['lamps'], lay['box'], lay['red_top']
    def paint(mask, group):
        index, rgb = LIGHT_GROUPS[group][:2]; out[mask] = (*rgb, LIT_ALPHA); idx[mask] = index
    paint(R['windows'], 'windows'); paint(R['cab'], 'cab')
    if box is not None:
        x0, y0, x1, y1 = box; rows = np.arange(H)[:, None]
        paint(lamps, 'lamps_white')
        if red_top is True:
            paint(lamps & (rows < y0 + RED_FRACTION * (y1 - y0)), 'lamps_red')
        elif red_top is False:
            paint(lamps & (rows >= y1 - RED_FRACTION * (y1 - y0)), 'lamps_red')
        if lay['copy'] is not None:
            cx, cy = lay['copy']; patch = np.zeros((H, W), bool)
            patch[cy + 2:cy + 2 + (y1 - y0), cx + 2:cx + 2 + (x1 - x0)] = lamps[y0:y1, x0:x1]
            paint(patch, 'lamps_white')
    if lay['door_patch'] is not None:
        px, py = lay['door_patch']; out[py:py + PATCH, px:px + PATCH] = (*DOOR_RGB, LIT_ALPHA)
        idx[py:py + PATCH, px:px + PATCH] = DOOR_PATCH_INDEX   # only the door sub-mesh samples it
    lay['index_map'] = idx
    return Image.fromarray(out, 'RGBA')

def door_lamp_mesh(body, doors, lay):
    """A small separate mesh: one outward-facing quad above every door (CS1 m_doors positions) whose
    UVs point at the door lamp patch; vertex colour R = 1 on the car's left (-x), 2 on its right.
    Triangles are wound the way the body winds its own (cross product vs stored normal), so the quads
    face outward after the same CS1->CS2 conversion; wound the other way they are backface-culled."""
    if lay['door_patch'] is None or not doors:
        return None
    V = np.asarray(body['V'], np.float64); t3 = tri_array(body); N = np.asarray(body['N'], np.float64)
    a, b_, c = V[t3[:, 0]], V[t3[:, 1]], V[t3[:, 2]]
    body_sign = 1.0 if (np.cross(b_ - a, c - a) * N[t3[:, 0]]).sum(1).mean() >= 0 else -1.0
    px, py = lay['door_patch']; W, H = lay['W'], lay['H']
    u0, u1 = (px + 2) / (W - 1), (px + PATCH - 2) / (W - 1); v0, v1 = 1 - (py + 2) / (H - 1), 1 - (py + PATCH - 2) / (H - 1)
    w, h, yc = DOOR_LAMP; nV, nN, nUV, nT, side_of = [], [], [], [], []
    for x, _, z in doors:
        side = 1.0 if x > 0 else -1.0
        near = (np.abs(V[:, 2] - z) < 0.9) & (V[:, 1] > yc - 0.4) & (V[:, 1] < yc + 0.2)
        bx = float(np.abs(V[near, 0]).max()) if near.any() else abs(x)
        base = len(nV)
        for dz, dy, u, v in ((-w / 2, -h / 2, u0, v1), (w / 2, -h / 2, u1, v1), (w / 2, h / 2, u1, v0), (-w / 2, h / 2, u0, v0)):
            nV.append((side * (bx + 0.012), yc + dy, z + dz)); nN.append((side, 0.0, 0.0)); nUV.append((u, v)); side_of.append(side)
        for i, j, k in ((base, base + 1, base + 2), (base, base + 2, base + 3)):
            p0, p1, p2 = (np.array(nV[t]) for t in (i, j, k))
            outward = np.dot(np.cross(p1 - p0, p2 - p0), nN[i]) * body_sign
            nT.append((i, j, k) if outward > 0 else (i, k, j))
    return dict(name='doors', V=np.array(nV), N=np.array(nN), UV=np.array(nUV), T=[], tris=[np.array(nT, np.int64).ravel()],
                side=np.array(side_of))

def simplify_mesh(m, ratio=LOD1_RATIO):
    """A coarser copy of `m` for LOD1: meshoptimizer collapses edges onto existing vertices, so every
    per-vertex attribute (UVs, light-index colours) survives; unused vertices are dropped."""
    import meshoptimizer as mo
    V = np.ascontiguousarray(np.asarray(m['V'], np.float32)); idx = np.ascontiguousarray(tri_array(m).ravel().astype(np.uint32))
    target = max(3, int(len(idx) * ratio) // 3 * 3); dest = np.zeros(len(idx), np.uint32)
    cnt = mo.simplify(dest, idx, V, target_index_count=target, target_error=0.05)
    new = dest[:cnt]; used, remap = np.unique(new, return_inverse=True)
    out = dict(m); out['tris'] = [remap.astype(np.int64)]
    for key in ('V', 'N', 'UV', 'T', 'C'):
        if key in m and len(m[key]) == len(V):
            out[key] = np.asarray(m[key])[used]
    return out

def triangle_light_index(m, index_map):
    """Per-triangle light index: the most common non-zero index among the texels the triangle covers
    in UV space (a body panel triangle that merely contains window texels counts as a window
    triangle: its unlit texels have alpha 0 anyway). 0 when it covers no lit texel."""
    H, W = index_map.shape; u, v = uv_pixels(m, W, H); t3 = tri_array(m); out = np.zeros(len(t3), np.uint8)
    nlights = int(index_map.max()) + 1
    for i, (a, b_, c) in enumerate(t3):
        xs = np.array([u[a], u[b_], u[c]]); ys = np.array([v[a], v[b_], v[c]])
        x0, x1 = max(int(np.floor(xs.min())), 0), min(int(np.ceil(xs.max())), W - 1)
        y0, y1 = max(int(np.floor(ys.min())), 0), min(int(np.ceil(ys.max())), H - 1)
        if x0 > x1 or y0 > y1:
            continue
        sub = index_map[y0:y1 + 1, x0:x1 + 1]
        if not sub.any():
            continue
        den = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
        if abs(den) < 1e-9:
            continue
        gy, gx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        w0 = ((ys[1] - ys[2]) * (gx - xs[2]) + (xs[2] - xs[1]) * (gy - ys[2])) / den
        w1 = ((ys[2] - ys[0]) * (gx - xs[2]) + (xs[0] - xs[2]) * (gy - ys[2])) / den
        inside = (w0 >= -0.02) & (w1 >= -0.02) & (1 - w0 - w1 >= -0.02)
        hit = sub[inside]; hit = hit[hit > 0]
        if len(hit):
            out[i] = np.bincount(hit, minlength=nlights)[1:].argmax() + 1
    return out

def vertex_light_colors(m, index_map):
    """Per-vertex colour attribute (n, 4) uint8: the light index its triangles' lit texels belong to
    (majority vote over the vertex's triangles) in the red channel; 0 on unlit surfaces."""
    t3 = tri_array(m); tri_idx = triangle_light_index(m, index_map); n = len(m['V'])
    nlights = int(index_map.max()) + 1; votes = np.zeros((n, nlights), np.int32)
    for k in range(3):
        np.add.at(votes, (t3[:, k], tri_idx), 1)
    votes[:, 0] = 0
    idx = np.where(votes.sum(1) > 0, votes.argmax(1), 0).astype(np.uint8)
    idx[idx == DOOR_PATCH_INDEX] = 0                          # the body never lights the door patch
    out = np.zeros((n, 4), np.uint8); out[:, 0] = idx      # red channel only, exactly like the KISS geometries
    return out

# ---------------------------------------------------------------- CS1 vehicle data
def vehicle_gen(b, E, mesh_index):
    """m_tyres (x, y, z, radius) of the VehicleInfoGen entry that follows the mesh's material."""
    for j in range(mesh_index + 1, len(E)):
        d = C.entry_bytes(b, E[j])
        if E[j]['type'] == 1 and b'VehicleInfoGen' in d[:100]:
            p = d.find(b'\x07m_tyres'); r = C.Reader(d, p + 8); n = r.i32()
            return [struct.unpack_from('<4f', d, r.p + 16 * k) for k in range(n)]
    raise SystemExit(f'no VehicleInfoGen after mesh {mesh_index}')

def vehicle_data(b, E, mesh_index):
    """doors and light positions from the VehicleInfo GameObject that follows the mesh entry."""
    for e in E[mesh_index + 1:]:
        if e['type'] == 1 and ' - ' not in e['name']:
            d = C.entry_bytes(b, e)
            p = d.find(b'\x07m_doors'); r = C.Reader(d, p + 8); n = r.i32()
            doors = [struct.unpack_from('<3f', d, r.p + 16 * k + 4) for k in range(n)]
            p = d.find(b'\x10m_lightPositions'); r = C.Reader(d, p + 17); n = r.i32()
            lights = [struct.unpack_from('<3f', d, r.p + 12 * k) for k in range(n)]
            return doors, lights
    raise SystemExit(f'no VehicleInfo after mesh {mesh_index}')

def bogies_from_tyres(tyres):
    """-> [(bogie_z, [axle_z, ...]), ...] front bogie first."""
    zs = sorted({round(t[2], 3) for t in tyres})
    front = [z for z in zs if z > 0]; rear = [z for z in zs if z < 0]
    out = []
    for group in (front, rear):
        out.append((float(np.mean(group)), sorted(group, reverse=True)))
    return out, float(np.mean([t[1] for t in tyres])), float(np.mean([t[3] for t in tyres]))

# ---------------------------------------------------------------- skinning
def connected_components(V, T):
    key = {}; canon = np.zeros(len(V), int)
    for i, v in enumerate(map(tuple, np.round(V, 4))):
        canon[i] = key.setdefault(v, i)
    parent = np.arange(len(V))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b, c in canon[T]:
        parent[find(b)] = find(a); parent[find(c)] = find(a)
    return np.array([find(x) for x in canon])

def assign_bones(V, T, bogies, wheel_y):
    """Bone index per vertex: 0 body, then per bogie [wheelset, axle, axle] in bogies order.
    A bogie owns the triangles that lie entirely inside a box around it: below the underframe
    and no further along the car than its outer wheel rims plus a margin. Everything else,
    including the underframe, end skirts and door steps, stays with the body."""
    bone = np.zeros(len(V), np.uint32)
    comp = connected_components(V, T)
    comp_size = np.bincount(comp); comp_ymax = np.full(comp.max() + 1, -9.0)
    np.maximum.at(comp_ymax, comp, V[:, 1])
    is_wheel_part = (comp_size[comp] < 120) & (comp_ymax[comp] < WHEEL_TOP_Y) & (np.abs(V[:, 0]) < 0.9)
    is_shaft = (V[:, 1] < WHEEL_TOP_Y) & (np.abs(V[:, 0]) < 0.70)
    tz, ty = V[T][:, :, 2], V[T][:, :, 1]
    idx = 1
    for bz, axles in bogies:
        half = (max(axles) - min(axles)) / 2 + 0.341 + BOGIE_MARGIN
        tri_in = (np.abs(tz - bz) < half).all(1) & (ty < BOGIE_TOP_Y).all(1)
        near = np.zeros(len(V), bool); near[T[tri_in]] = True
        outside = np.zeros(len(V), bool); outside[T[~tri_in]] = True
        near &= ~outside          # a vertex shared with a body triangle stays with the body
        bone[near] = idx
        for k, az in enumerate(axles):
            at_axle = near & (np.abs(V[:, 2] - az) < 0.45) & (is_wheel_part | is_shaft)
            bone[at_axle] = idx + 1 + k
        idx += 1 + len(axles)
    return bone

def bone_list(prefix, bogies, wheel_y):
    bones = [A.bone(f'{prefix}_RootBone', (0, 0, 0))]
    for n, (bz, axles) in enumerate(bogies, 1):
        wi = len(bones)
        bones.append(A.bone(f'Wheelset{n}', (0, wheel_y, bz), parent=-1, bone_type=15))
        for k, az in enumerate(axles, 1):
            bones.append(A.bone(f'Axle{(n - 1) * len(axles) + k}', (0, wheel_y, az), parent=wi, bone_type=4,
                                parent_world=(0, wheel_y, bz)))
    return bones

# ---------------------------------------------------------------- middle car synthesis
def clip_half(m, keep_sign, cut=0.0, eps=1e-6):
    """Keep the part of a CS1 mesh dict with z*keep_sign >= cut, cutting triangles on that plane,
    then slide the kept part so the cut lands on z = 0."""
    V = np.asarray(m['V'], np.float64); N = np.asarray(m['N'], np.float64); UV = np.asarray(m['UV'], np.float64)
    T = np.concatenate([np.asarray(t, np.int64).reshape(-1, 3) for t in m['tris']])
    d = V[:, 2] * keep_sign - cut; inside = d >= -eps
    nV, nN, nUV, tris = [], [], [], []
    remap = {}; cache = {}
    def keep(i):
        if i not in remap:
            remap[i] = len(nV); nV.append(V[i]); nN.append(N[i]); nUV.append(UV[i])
        return remap[i]
    def cross(i, j):
        k = (min(i, j), max(i, j))
        if k not in cache:
            t = d[i] / (d[i] - d[j]); n = N[i] + (N[j] - N[i]) * t
            cache[k] = len(nV); nV.append(V[i] + (V[j] - V[i]) * t); nN.append(n / max(np.linalg.norm(n), 1e-9))
            nUV.append(UV[i] + (UV[j] - UV[i]) * t)
        return cache[k]
    for a, b, c in T:
        poly = []
        for i, j in ((a, b), (b, c), (c, a)):
            if inside[i]: poly.append(keep(i))
            if inside[i] != inside[j]: poly.append(cross(i, j))
        for k in range(1, len(poly) - 1):
            tris.append((poly[0], poly[k], poly[k + 1]))
    nV = np.asarray(nV); nV[:, 2] -= keep_sign * cut
    return dict(name=m['name'], V=list(nV), N=nN, UV=nUV, T=[], tris=[tuple(np.asarray(tris).ravel())])

def logo_z(m, texture, side=+1, debug=None):
    """z of the largest light-coloured feature on the car side at logo height (the operator logo),
    found by rendering the side head-on against black. Returns None when nothing stands out."""
    V = np.asarray(m['V'], np.float64); N = np.asarray(m['N'], np.float64); UV = np.asarray(m['UV'], np.float64)
    T = np.concatenate([np.asarray(t, np.int64).reshape(-1, 3) for t in m['tris']])
    W, H, dist, fov = 4000, 800, 700.0, 2.2
    im = preview.render([(V, N, UV, T, texture)], W, H, 90 * side, 0, dist, fov, (0, 2.3, 0), bg=(0, 0, 0), ambient=1.0, gain=1.0)
    img = np.asarray(im); f = 0.5 * H / np.tan(np.radians(fov) / 2); ppm = f / dist
    r0, r1 = int(H / 2 - 0.55 * ppm), int(H / 2 + 0.55 * ppm)          # y in [1.75, 2.85]
    count = (img[r0:r1].mean(2) > 75).sum(0)          # logo is mid-grey on a near-black band
    cols = np.where(count >= 3)[0]
    if not len(cols):
        return None
    clusters, start = [], cols[0]
    for a, b_ in zip(cols, cols[1:]):
        if b_ - a > 12: clusters.append((start, a)); start = b_
    clusters.append((start, cols[-1]))
    lo, hi = max(clusters, key=lambda c: count[c[0]:c[1] + 1].sum())
    if count[lo:hi + 1].sum() < 0.04 * ppm * ppm:                         # smaller than 0.04 m² of pixels: not a logo
        return None
    w = count[lo:hi + 1]; centre = (np.arange(lo, hi + 1) * w).sum() / w.sum()
    z = -side * (centre - W / 2) / ppm                                     # screen x runs along -z when looking from +x
    if debug:
        from PIL import ImageDraw
        d = ImageDraw.Draw(im); d.line([(centre, 0), (centre, H)], fill=(255, 0, 0), width=2)
        d.rectangle([lo, r0, hi, r1], outline=(0, 255, 0)); im.crop((W // 2 - 14 * ppm, 0, W // 2 + 14 * ppm, H)).save(debug)
    return float(z)

def merge(a, b):
    off = len(a['V'])
    tris = tuple(a['tris'][0]) + tuple(i + off for i in b['tris'][0])
    return dict(name=a['name'] + '+' + b['name'], V=list(a['V']) + list(b['V']), N=list(a['N']) + list(b['N']),
                UV=list(a['UV']) + list(b['UV']), T=[], tris=[tris])

def splice_middle(b, E, minus, plus):
    """A cabless middle car: the z<0 half of mesh `minus` (flat end at -Z) joined to the z>0 half of
    mesh `plus` (flat end at +Z). Returns (mesh, lod mesh, tyres, doors)."""
    (mi, mlod), (pi_, plod) = minus, plus
    A_, B_ = C.read_mesh(b, E[mi]), C.read_mesh(b, E[pi_])
    # cut through the logo centre on each car (it sits on a window pillar) so the halves join
    # into one logo and one pillar instead of two of each side by side
    tex = C.read_texture(b, E[material_for_mesh(E, cs1_materials(b, E), mi)['_MainTex']])[2]
    za, zb = logo_z(A_, tex), logo_z(B_, tex)
    ca = -za if za is not None and za < 0 else 0.0
    cb = zb if zb is not None and zb > 0 else 0.0
    print(f'middle car: cutting {A_["name"][:20]} at z={-ca:.3f} and {B_["name"][:20]} at z={cb:.3f}')
    mesh = merge(clip_half(A_, -1, ca), clip_half(B_, +1, cb))
    lod = merge(clip_half(C.read_mesh(b, E[mlod]), -1, ca), clip_half(C.read_mesh(b, E[plod]), +1, cb))
    tyres = [(x, y, z + ca, r) for x, y, z, r in vehicle_gen(b, E, mi) if z < -ca] + \
            [(x, y, z - cb, r) for x, y, z, r in vehicle_gen(b, E, pi_) if z > cb]
    doors = [(x, y, z + ca) for x, y, z in vehicle_data(b, E, mi)[0] if z < -ca] + \
            [(x, y, z - cb) for x, y, z in vehicle_data(b, E, pi_)[0] if z > cb]
    return mesh, lod, tyres, doors

# ---------------------------------------------------------------- build
def build(crp, name, title, out, front, cars, speed, capacity, ui_group, middle=None, calibrate=False, units=(1, 1), door_canary=False, brand=None):
    """cars: list of (car key, min count, max count) in consist order; units: (min, max) whole units the game may couple."""
    b, hdr, E = C.parse(crp)
    mats = cs1_materials(b, E)
    root = os.path.join(out, name); shutil.rmtree(root, ignore_errors=True)
    asset_dir = os.path.join(root, 'payload', 'asset'); sub = os.path.join(asset_dir, name)
    icon_dir = os.path.join(root, 'payload', 'TextureAssets')
    for d in (sub, icon_dir):
        os.makedirs(d)
    files = []
    def fname(stem, ext): return f'{stem}_{did(name, stem, ext)}.{ext}'
    def emit(rel, write_fn, cid, base=asset_dir):
        p = os.path.join(base, rel); write_fn(p); A.write_cid(p, cid)
        tag = 'asset' if base == asset_dir else 'TextureAssets'
        files.extend([(tag, rel), (tag, rel + '.cid')])

    # every mesh of the consist, per atlas: '' = LOD0 (and the simplified LOD1), '_LOD2' = the CS1 LOD
    # mesh with its own small textures; all cars share one CS1 texture set
    synthetic = {}          # car key -> (mesh dict, lod mesh dict, tyres, doors)
    if middle:
        synthetic['mid'] = splice_middle(b, E, *middle)
    car_keys = list(dict.fromkeys(spec for spec, _, _ in cars))
    keys = list(dict.fromkeys([front] + car_keys))
    meshes = {'': {}, '_LOD2': {}}
    for key in keys:
        if key in synthetic:
            meshes[''][key], meshes['_LOD2'][key] = synthetic[key][0], synthetic[key][1]
        else:
            meshes[''][key], meshes['_LOD2'][key] = C.read_mesh(b, E[key[0]]), C.read_mesh(b, E[key[1]])

    doors_by_key = {k: (synthetic[k][3] if k in synthetic else vehicle_data(b, E, k[0])[0]) for k in keys}
    tex_cids = {}; layouts = {}
    for level, mi in (('', front[0]), ('_LOD2', front[1])):
        tex_cids[level] = []
        mat = material_for_mesh(E, mats, mi); slots = convert_textures(b, E, mat)
        aci = C.read_texture(b, E[mat['_ACIMap']])[2].resize(slots['BaseColor'][0].size)
        lay = layouts[level] = lamp_layout(aci, meshes[level], front, relocate=(level == ''))
        copy_lamp_texels(slots, lay)
        slots['Emissive'] = (paint_emissive(lay), True)
        if brand:   # operator identity over the SJ crests, before the line mask so its lit texels stay unmasked
            B.apply(slots, lay, B.BRANDS[brand])
        line_mask, line_tris = paint_line_mask(slots, meshes[level], doors_by_key, lay, band_rules=bool(brand))
        print(f'line colour{level or "_LOD0"}: door leaf triangles {line_tris}, {int(line_mask.sum())} texels masked')
        print(f'lights{level or "_LOD0"}: windows {int(lay["regions"]["windows"].sum())} px, cab {int(lay["regions"]["cab"].sum())} px, lamp lens {lay["box"]}, '
              f'red at texel rows {"top" if lay["red_top"] else "bottom" if lay["red_top"] is False else "n/a"}, '
              f'roof-lamp copy at {lay["copy"]}, door lamp patch at {lay["door_patch"]}, roof-lamp triangles ' + str({str(k)[:8]: len(v) for k, v in lay['upper'].items()}))
        for slot, (im, srgb) in slots.items():
            cid = did(name, level, slot)
            emit(fname(f'{name}{level}_{slot}', 'Texture'),
                 lambda p, im=im, srgb=srgb, slot=slot: T.write(p, im, srgb, mip_filter='max_alpha' if slot == 'Emissive' else 'lanczos'), cid)
            tex_cids[level].append((SURFACE_SLOT[slot], cid))
            if level == '' and slot == 'BaseColor': base_img = im
            if level == '' and slot == 'Emissive': emissive_img = im
    tex_cids['_LOD1'] = tex_cids['']; layouts['_LOD1'] = layouts['']

    preview_parts = {}      # car key -> (V, N, UV, tris, bones)
    mesh_cache = {}         # car key -> (train LOD0 render prefab cid, prop LOD0 render prefab cid)
    def build_mesh(stem, key, bogies, wheel_y, doors, scale=1.0):
        """Writes the LOD2 (CS1 LOD), LOD1 (simplified) and LOD0 geometry, surfaces and render prefabs,
        for the train and for the static prop; returns (train, prop) LOD0 render prefab cids."""
        if key in mesh_cache:
            return mesh_cache[key]
        m0 = apply_lamp_layout(meshes[''][key], key, layouts[''])
        if brand:
            m0, n_aligned = align_band_strip(m0, layouts['']['W'], layouts['']['H']); print(f'  {stem}: band strip triangles re-mapped: {n_aligned}')
        variants = {'_LOD2': meshes['_LOD2'][key], '_LOD1': simplify_mesh(m0), '': m0}
        train_lods, prop_lods = [], []
        doors_cid = None
        dm = door_lamp_mesh(m0, doors, layouts[''])
        if dm is not None:
            tris, attrs, lo, hi, area = cs1_mesh_to_cs2(dm); n = len(dm['V'])
            col = np.zeros((n, 4), np.uint8); col[:, 0] = np.where(dm['side'] < 0, 1, 2); attrs['color'] = (2, col)
            attrs['uv1'] = attrs['uv2'] = (1, np.zeros((n, 2), np.float16)); attrs['uv3'] = (0, np.zeros((n, 2), np.float32))
            s = stem + '_Doors'
            geo_cid = did(name, s, 'geometry'); emit(fname(s, 'Geometry'), lambda p: G.write(p, tris, attrs), geo_cid)
            surf_cid = did(name, s, 'surface'); emit(fname(s, 'Surface'), lambda p: A.write_surface(p, tex_cids[''], keywords=SURFACE_KEYWORDS), surf_cid)
            doors_cid = did(name, s, 'renderprefab')
            rp = A.render_prefab(f'{s} Mesh', geo_cid, [surf_cid], lo, hi, area, len(tris), n, components=[A.emissive_properties(door_light_table(door_canary))])
            emit(os.path.join(name, fname(f'{s} Mesh', 'Prefab')), lambda p: A.write_prefab(p, rp), doors_cid)
        for level in ('_LOD2', '_LOD1', ''):
            m = variants[level]
            tris, attrs, lo, hi, area = cs1_mesh_to_cs2(m); n = len(m['V'])
            attrs['color'] = (2, vertex_light_colors(m, layouts[level]['index_map']))
            attrs['uv1'] = attrs['uv2'] = (1, np.zeros((n, 2), np.float16)); attrs['uv3'] = (0, np.zeros((n, 2), np.float32))
            s = stem + level
            surf_cid = did(name, s, 'surface')
            emit(fname(s, 'Surface'), lambda p: A.write_surface(p, tex_cids[level], keywords=SURFACE_KEYWORDS), surf_cid)
            prop_geo = geo_cid = did(name, s, 'geometry')
            train_comps = [A.emissive_properties(light_table(scale))]
            line = [A.color_properties([LINE_DEFAULT, WHITE, WHITE])] if level == '' else []
            train_comps += line
            if level == '':
                # the prop gets the same mesh without bone indices (a static object has no skeleton)
                prop_geo = did(name, s, 'geometry-prop'); emit(fname(s + '_Prop', 'Geometry'), lambda p: G.write(p, tris, attrs), prop_geo)
                V = np.asarray(m['V'], np.float32)
                bones = assign_bones(V, tris.reshape(-1, 3), bogies, wheel_y)
                attrs['blendindices'] = (10, bones[:, None])
                train_comps.append(A.procedural_animation(bone_list(stem, bogies, wheel_y)))
                preview_parts[key] = (V, np.asarray(m['N']), np.asarray(m['UV']), tris.reshape(-1, 3), bones)
            emit(fname(s, 'Geometry'), lambda p: G.write(p, tris, attrs), geo_cid)
            for kind, geo, comps, lods in (('', geo_cid, train_comps, train_lods),
                                           ('_Prop', prop_geo, [A.emissive_properties(light_table(prop=True))] + line, prop_lods)):
                rp_cid = did(name, s + kind, 'renderprefab')
                rp = A.render_prefab(f'{s}{kind} Mesh', geo, [surf_cid], lo, hi, area, len(tris), n,
                                     lod_cids=list(reversed(lods)) if level == '' else [], components=comps)
                emit(os.path.join(name, fname(f'{s}{kind} Mesh', 'Prefab')), lambda p: A.write_prefab(p, rp), rp_cid)
                lods.append(rp_cid)
            print(f'  {s}: {len(tris) // 3} tris, {n} verts')
        mesh_cache[key] = (train_lods[-1], prop_lods[-1], doors_cid)
        return mesh_cache[key]

    # front car mesh first so a carriage that reuses it shares the render prefab
    mi = front[0]
    tyres = vehicle_gen(b, E, mi); bogies, wheel_y, _ = bogies_from_tyres(tyres)
    doors, lights = vehicle_data(b, E, mi)
    mesh_cid, front_prop_cid, front_doors_cid = build_mesh(name, front, bogies, wheel_y, doors)
    prop_cids = {front: (front_prop_cid, f'{title} cab car')}

    # one carriage prefab per distinct mesh, referenced at every consist position it occupies
    car_cid = {}
    for k, car in enumerate(car_keys):
        stem = f'{name}_Car{chr(65 + k)}'
        if car in synthetic:
            ctyres, cdoors = synthetic[car][2], synthetic[car][3]
        else:
            ctyres = vehicle_gen(b, E, car[0]); cdoors, _ = vehicle_data(b, E, car[0])
        cbogies, cwheel_y, _ = bogies_from_tyres(ctyres)
        cmesh_cid, cprop_cid, cdoors_cid = build_mesh(stem, car, cbogies, cwheel_y, cdoors, CALIBRATION[min(k + 1, 3)] if calibrate else 1.0)
        comps = [A.public_transport(capacity), A.vehicle_side_effects(),
                 A.activity_location(ACTIVITY_BOARD, [(np.sign(x) * 1.3, 0.7, z) for x, y, z in cdoors]),
                 A.effect_source([(EFFECT_TRAIN, (0, 4, 0), (0, -1, 0, 0))])]
        cid = did(name, stem, 'car'); car_prefab = A.train_car_prefab(stem, [cmesh_cid] + ([cdoors_cid] if cdoors_cid else []), comps, speed)
        emit(os.path.join(name, fname(stem, 'Prefab')), lambda p: A.write_prefab(p, car_prefab), cid)
        emit(os.path.join(name, f'{fname(stem, "Prefab")[:-7]}_en-US.loc'),
             lambda p: A.write_loc(p, {f'Assets.NAME[{stem}]': f'{title} car {chr(65 + k)}'}), did(name, stem, 'loc'))
        car_cid[car] = cid
        prop_cids.setdefault(car, (cprop_cid, f'{title} {"middle" if car == "mid" else "car " + chr(65 + k)} car'))
    carriages = [(car_cid[spec], 0, lo, hi) for spec, lo, hi in cars]

    V = preview_parts[front][0]; nose_z = float(V[:, 2].max())
    def nose_surface(x, y):
        sel = V[(np.abs(V[:, 0] - x) < 0.15) & (np.abs(V[:, 1] - y) < 0.15)]
        return float(sel[:, 2].max()) + 0.05 if len(sel) else nose_z
    fx = [(EFFECT_HEADLIGHT, (x, y, nose_surface(x, y)), (0, 0, 0, 1)) for x, y, z in lights]
    fx.append((EFFECT_TRAIN, (0, 4, 0), (0, 0, 0, -1)))

    # icon + preview: the longest unit, front car mirrored onto the back like the game does
    parts = []; z0 = 0.0
    order = [preview_parts[front]] + [preview_parts[spec] for spec, lo, hi in cars for _ in range(hi)] + [preview_parts[front]]
    for k, (Vp, Np, UVp, Tp, _) in enumerate(order):
        Vp = Vp.astype(np.float64).copy(); Np = Np.astype(np.float64).copy()
        if k == len(order) - 1: Vp[:, [0, 2]] *= -1; Np[:, [0, 2]] *= -1
        Vp[:, 2] += z0 - Vp[:, 2].max(); z0 = Vp[:, 2].min() - 0.3; parts.append((Vp, Np, UVp, Tp, base_img))
    allV = np.concatenate([p[0] for p in parts]); ctr = (allV.min(0) + allV.max(0)) / 2; size = float(np.ptp(allV, 0).max())
    def car_icon(part, cid):
        Vp = part[0]; im = preview.crop_to_content(preview.render([part], 900, 900, 40, -14, 48, 30, (Vp.min(0) + Vp.max(0)) / 2), square=True)
        emit(fname(cid, 'jpg'), lambda p: im.resize((256, 256), Image.LANCZOS).save(p, 'JPEG', quality=90), cid, base=icon_dir)
        return cid
    icon_cid = car_icon(parts[0], did(name, 'icon'))
    pw = 600 + 300 * len(order)
    preview.crop_to_content(preview.render(parts, pw, pw // 4, 62, -9, size * 1.15, 32, ctr)).save(os.path.join(root, 'preview.png'))
    preview.crop_to_content(preview.render(parts, pw, pw // 8, 90, 0, size * 1.1, 30, ctr)).save(os.path.join(root, 'preview_side.png'))
    lm = layouts['']['line_mask']
    if lm.any():   # the door leaves in a sample line colour, as the game will multiply them
        tinted = np.asarray(base_img.convert('RGB')).astype(np.float32); tinted[lm] *= np.array([0.1, 0.45, 0.9])
        tinted_img = Image.fromarray(tinted.astype(np.uint8)); lparts = [(Vp, Np, UVp, Tp, tinted_img) for Vp, Np, UVp, Tp, _ in parts]
        preview.crop_to_content(preview.render(lparts, pw, pw // 8, 90, 0, size * 1.1, 30, ctr)).save(os.path.join(root, 'preview_line.png'))
    bone_img = bone_preview(preview_parts[front]); bone_img.save(os.path.join(root, 'bones.png'))
    # night previews: base colour dimmed, lit texels in their light colour
    e = np.asarray(emissive_img); night = (np.asarray(base_img.convert('RGB')).astype(np.float32) * 0.22)
    lit = e[..., 3] > 0; night[lit] = e[..., :3][lit]; night_img = Image.fromarray(night.astype(np.uint8))
    nparts = [(Vp, Np, UVp, Tp, night_img) for Vp, Np, UVp, Tp, _ in parts]
    preview.crop_to_content(preview.render(nparts, pw, pw // 4, 62, -9, size * 1.15, 32, ctr, bg=(18, 20, 28), ambient=0.9, gain=1.0)).save(os.path.join(root, 'preview_night.png'))
    fp0 = nparts[0]; fc = (fp0[0].min(0) + fp0[0].max(0)) / 2
    preview.crop_to_content(preview.render([fp0], 900, 700, 152, -8, 34, 30, fc, bg=(18, 20, 28), ambient=0.9, gain=1.0)).save(os.path.join(root, 'preview_front_night.png'))

    comps = [A.public_transport(capacity), A.vehicle_side_effects(),
             A.activity_location(ACTIVITY_BOARD, [(np.sign(x) * 1.3, 0.7, z) for x, y, z in doors]),
             A.ui_object(icon_cid, ui_group), A.effect_source(fx)]
    front_cid = did(name, 'front')
    fp = A.train_front_prefab(name, [mesh_cid] + ([front_doors_cid] if front_doors_cid else []), comps, carriages, speed, reversed_end=True, units=units)
    emit(os.path.join(name, fname(name, 'Prefab')), lambda p: A.write_prefab(p, fp), front_cid)
    emit(os.path.join(name, f'{fname(name, "Prefab")[:-7]}_en-US.loc'),
         lambda p: A.write_loc(p, {f'Assets.NAME[{name}]': title}), did(name, 'loc'))

    # static props: one per car type, placeable like any prop (windows lit at night, lamps off)
    for key, (pcid, ptitle) in prop_cids.items():
        pstem = f'{name}_{"Cab" if key == front else "Mid" if key == "mid" else "Car" + chr(65 + car_keys.index(key))}Prop'
        picon = car_icon(preview_parts[key][:4] + (base_img,), did(name, pstem, 'icon'))
        sop = A.static_object_prefab(pstem, pcid, PROP_UI_GROUP, picon)
        emit(os.path.join(name, fname(pstem, 'Prefab')), lambda p: A.write_prefab(p, sop), did(name, pstem, 'prop'))
        emit(os.path.join(name, f'{fname(pstem, "Prefab")[:-7]}_en-US.loc'),
             lambda p: A.write_loc(p, {f'Assets.NAME[{pstem}]': f'{ptitle} (static)'}), did(name, pstem, 'loc'))

    write_scripts(root, name, title, files, where='It runs on passenger train lines: the game picks it '
                  'at random when a line spawns a new train, or use a vehicle-selector mod to pick it.')
    zpath = os.path.join(out, name + '.zip')
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for dp, _, fs in os.walk(root):
            for f in fs:
                full = os.path.join(dp, f); z.write(full, os.path.join(name, os.path.relpath(full, root)))
    return root, zpath

def bone_preview(part):
    """Side view with vertices coloured by bone: body grey, bogies blue/green, axles red/orange."""
    V, _, _, _, bones = part
    W, H = 1800, 420; sc = W / (np.ptp(V[:, 2]) + 1); img = np.full((H, W, 3), 245, np.uint8)
    palette = np.array([[120, 120, 120], [40, 90, 220], [230, 40, 40], [240, 150, 30], [30, 170, 90], [230, 40, 40], [240, 150, 30]])
    x = ((V[:, 2] - V[:, 2].min() + 0.5) * sc).astype(int); y = (H - 20 - V[:, 1] * sc).astype(int)
    ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    order = np.argsort(bones[ok]); img[y[ok][order], x[ok][order]] = palette[np.minimum(bones[ok][order], 6)]
    return Image.fromarray(img)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('crp'); ap.add_argument('--name', required=True); ap.add_argument('--title', required=True)
    ap.add_argument('--front', required=True, help='MESH:LOD entry indices of the lead car')
    ap.add_argument('--car', action='append', default=[], help="MESH:LOD entry indices, in consist order, or 'mid'; append xMIN-MAX for a count range the game picks from (e.g. midx3-4)")
    ap.add_argument('--units', default='1-1', help='MIN-MAX whole units the game may couple nose to tail (default 1-1)')
    ap.add_argument('--middle', help='MESH:LOD+MESH:LOD: cab car with its flat end at -Z, then one with it at +Z; '
                                     'their flat halves are joined into a cabless middle car usable as --car mid')
    ap.add_argument('--speed', type=int, default=200)
    ap.add_argument('--capacity', type=int, default=95, help='passengers per car (X40: about 95, a 3-car set seats 285)')
    ap.add_argument('--out', default='dist'); ap.add_argument('--ui-group', default=None)
    ap.add_argument('--calibrate', action='store_true', help='carriage types A/B/C get 0.4/0.2/0.1 of the interior light intensity')
    ap.add_argument('--door-canary', action='store_true', help='left door lamps always on (diagnostic)')
    ap.add_argument('--brand', choices=sorted(B.BRANDS), help='paint this operator identity over the SJ branding (mark and wordmark glow at night)')
    a = ap.parse_args()
    pair = lambda s: tuple(int(x) for x in s.split(':'))
    middle = tuple(pair(x) for x in a.middle.split('+')) if a.middle else None
    def car_spec(text):
        spec, _, rng = text.partition('x'); lo, _, hi = rng.partition('-') if rng else ('1', '', '1')
        return (spec if spec == 'mid' else pair(spec), int(lo), int(hi or lo))
    lo, _, hi = a.units.partition('-')
    root, z = build(a.crp, a.name, a.title, a.out, pair(a.front), [car_spec(c) for c in a.car],
                    a.speed, a.capacity, a.ui_group, middle, a.calibrate, (int(lo), int(hi or lo)), a.door_canary, a.brand)
    print('built', root); print('zip  ', z, os.path.getsize(z) // 1024, 'KB')
