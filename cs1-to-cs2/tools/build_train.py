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
import cs2_asset as A
import cs2_geometry as G
import cs2_texture as T
import preview
from build_prop import (did, cs1_materials, material_for_mesh, convert_textures, cs1_mesh_to_cs2,
                        SURFACE_SLOT, write_scripts)

# Vanilla GUIDs copied from a working custom train (Stadler KISS Caltrain pack)
ACTIVITY_BOARD = '55cd31323498ccf4ca4831d41291c0f4'
EFFECT_HEADLIGHT = 'de32f8ad166d2d944b6f069e25675da1'
EFFECT_TRAIN = '76d8c52143e7618448f2ff10af98015c'
BOGIE_TOP_Y = 1.0        # bogie frame and axle boxes stay below this
BOGIE_MARGIN = 0.35      # bogie box reaches this far beyond the outer wheel rims
WHEEL_TOP_Y = 0.72
SURFACE_KEYWORDS = ('_EMISSIVE_PROCEDURAL', '_TANGENTSPACE_OCTO')
WHITE, WARM, RED, BLACK = (1, 1, 1, 1), (1, 0.9974498, 0.745, 1), (1, 0, 0, 1), (0, 0, 0, 1)
# ---------------------------------------------------------------- lights
# Light groups: name -> (texture alpha = layer id, texture rgb, purpose, colour, intensity, luminance).
# The game turns the emissive alpha into a 1-based index into the EmissiveProperties multi-light
# list (KISS uses 25, 51, 76, ... = k * 25.5), so list order and layer ids must agree.
# Purposes (Game.Prefabs.EmissiveProperties.Purpose): 23 Interior1 = on at night on every car,
# 3 Headlight_LowBeam = leading car, 6 RearLight = trailing car. Intensity/luminance = KISS values.
LIGHT_GROUPS = {
    'windows':     (25, (255, 214, 150), 23, WARM,  0.05, 0.964706),
    'lamps_red':   (51, (255, 40, 40),   6,  RED,   1.0,  1.0),
    'lamps_white': (76, (255, 255, 255), 3,  WHITE, 1.0,  1.0),
}
LIGHTS = [(p, c, BLACK, i, l, a) for a, _, p, c, i, l in LIGHT_GROUPS.values()]
RED_FRACTION = 0.4       # bottom part of a combined head/tail lamp lens that glows red
UPPER_LAMP_GAP = 0.8     # lamps this much above the lowest lamp are roof lamps: white only

def light_regions(aci):
    """From CS1's ACI map (blue = illumination): lit passenger windows (16-40) and lamp lenses (>200).
    Illumination 0 (door windows, windscreen, displays) stays dark, as the CS1 author intended."""
    I = np.asarray(aci.convert('RGBA')).astype(int)[..., 2]
    return {'windows': (I >= 16) & (I <= 40), 'lamps': I > 200}

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
    regions = light_regions(aci); H, W = regions['lamps'].shape; box = bbox(regions['lamps'])
    lay = dict(regions=regions, W=W, H=H, box=box, red_top=None, copy=None, upper={})
    if box is None:
        return lay
    lower = {}
    for key, m in meshes.items():
        tri = lamp_triangles(m, box, W, H)
        lower[key], lay['upper'][key] = lamp_groups(m, tri) if relocate else (tri, tri[:0])
    lay['red_top'] = red_rows_from_top(meshes[ref_key], lower[ref_key], W, H)
    if any(len(t) for t in lay['upper'].values()):
        x0, y0, x1, y1 = box
        lay['copy'] = find_free_rect(uv_coverage(meshes.values(), W, H), x1 - x0 + 4, y1 - y0 + 4)
        if lay['copy'] is None:
            lay['upper'] = {k: np.zeros(0, np.int64) for k in meshes}
    return lay

def apply_lamp_layout(m, key, lay):
    """Points the roof-lamp triangles of mesh `m` at the white-only lens copy."""
    if lay['copy'] is None or not len(lay['upper'].get(key, ())):
        return m
    x0, y0, _, _ = lay['box']; cx, cy = lay['copy']
    return relocate_triangles(m, lay['upper'][key], (cx + 2 - x0) / (lay['W'] - 1), -(cy + 2 - y0) / (lay['H'] - 1))

def copy_lamp_texels(slots, lay):
    """Duplicates the lens texels (plus a 2 px rim) into the free spot in every non-emissive slot."""
    if lay['copy'] is None:
        return
    x0, y0, x1, y1 = lay['box']; cx, cy = lay['copy']
    for slot, (im, srgb) in slots.items():
        if slot != 'Emissive':
            im.paste(im.crop((x0 - 2, y0 - 2, x1 + 2, y1 + 2)), (cx, cy))

def paint_emissive(lay):
    """Emissive texture: RGB = light colour, A = layer id; alpha 0 (and black) everywhere else."""
    R = lay['regions']; H, W = R['windows'].shape; out = np.zeros((H, W, 4), np.uint8)
    lamps, box, red_top = R['lamps'], lay['box'], lay['red_top']
    def paint(mask, group):
        layer, rgb = LIGHT_GROUPS[group][:2]; out[mask] = (*rgb, layer)
    paint(R['windows'], 'windows')
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
    return Image.fromarray(out, 'RGBA')

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
def build(crp, name, title, out, front, cars, speed, capacity, ui_group, middle=None):
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

    # every mesh of the consist, per atlas level ('' = LOD0, '_LOD1'); all cars share one CS1 texture set
    synthetic = {}          # car key -> (mesh dict, lod mesh dict, tyres, doors)
    if middle:
        synthetic['mid'] = splice_middle(b, E, *middle)
    keys = list(dict.fromkeys([front] + list(cars)))
    meshes = {'': {}, '_LOD1': {}}
    for key in keys:
        if key in synthetic:
            meshes[''][key], meshes['_LOD1'][key] = synthetic[key][0], synthetic[key][1]
        else:
            meshes[''][key], meshes['_LOD1'][key] = C.read_mesh(b, E[key[0]]), C.read_mesh(b, E[key[1]])

    tex_cids = {}; layouts = {}
    for level, mi in (('', front[0]), ('_LOD1', front[1])):
        tex_cids[level] = []
        mat = material_for_mesh(E, mats, mi); slots = convert_textures(b, E, mat)
        aci = C.read_texture(b, E[mat['_ACIMap']])[2].resize(slots['BaseColor'][0].size)
        lay = layouts[level] = lamp_layout(aci, meshes[level], front, relocate=(level == ''))
        copy_lamp_texels(slots, lay)
        slots['Emissive'] = (paint_emissive(lay), True)
        print(f'lights{level or "_LOD0"}: windows {int(lay["regions"]["windows"].sum())} px, lamp lens {lay["box"]}, '
              f'red at texel rows {"top" if lay["red_top"] else "bottom" if lay["red_top"] is False else "n/a"}, '
              f'roof-lamp copy at {lay["copy"]}, roof-lamp triangles ' + str({str(k)[:8]: len(v) for k, v in lay['upper'].items()}))
        for slot, (im, srgb) in slots.items():
            cid = did(name, level, slot)
            emit(fname(f'{name}{level}_{slot}', 'Texture'),
                 lambda p, im=im, srgb=srgb, slot=slot: T.write(p, im, srgb, mip_filter='max_alpha' if slot == 'Emissive' else 'lanczos'), cid)
            tex_cids[level].append((SURFACE_SLOT[slot], cid))
            if level == '' and slot == 'BaseColor': base_img = im
            if level == '' and slot == 'Emissive': emissive_img = im

    preview_parts = {}      # car key -> (V, N, UV, tris, bones)
    mesh_cache = {}         # car key -> LOD0 render prefab cid
    def build_mesh(stem, key, bogies, wheel_y):
        """Writes LOD1 + LOD0 geometry, surfaces and render prefabs; returns LOD0 render prefab cid."""
        if key in mesh_cache:
            return mesh_cache[key]
        lod_cid = None
        for level in ('_LOD1', ''):
            m = apply_lamp_layout(meshes[level][key], key, layouts[level])
            tris, attrs, lo, hi, area = cs1_mesh_to_cs2(m)
            comps = [A.emissive_properties(LIGHTS)]
            if level == '':
                V = np.asarray(m['V'], np.float32)
                bones = assign_bones(V, tris.reshape(-1, 3), bogies, wheel_y)
                attrs['blendindices'] = (10, bones[:, None])
                comps.append(A.procedural_animation(bone_list(stem, bogies, wheel_y)))
                preview_parts[key] = (V, np.asarray(m['N']), np.asarray(m['UV']), tris.reshape(-1, 3), bones)
            s = stem + level
            geo_cid = did(name, s, 'geometry'); emit(fname(s, 'Geometry'), lambda p: G.write(p, tris, attrs), geo_cid)
            surf_cid = did(name, s, 'surface')
            emit(fname(s, 'Surface'), lambda p: A.write_surface(p, tex_cids[level], keywords=SURFACE_KEYWORDS), surf_cid)
            rp_cid = did(name, s, 'renderprefab')
            rp = A.render_prefab(f'{s} Mesh', geo_cid, [surf_cid], lo, hi, area, len(tris), len(m['V']),
                                 lod_cids=[lod_cid] if lod_cid else [], components=comps)
            emit(os.path.join(name, fname(f'{s} Mesh', 'Prefab')), lambda p: A.write_prefab(p, rp), rp_cid)
            lod_cid = rp_cid
        mesh_cache[key] = lod_cid
        return lod_cid

    # front car mesh first so a carriage that reuses it shares the render prefab
    mi = front[0]
    tyres = vehicle_gen(b, E, mi); bogies, wheel_y, _ = bogies_from_tyres(tyres)
    doors, lights = vehicle_data(b, E, mi)
    mesh_cid = build_mesh(name, front, bogies, wheel_y)

    # one carriage prefab per distinct mesh, referenced at every consist position it occupies
    car_cid = {}
    for k, car in enumerate(dict.fromkeys(cars)):
        stem = f'{name}_Car{chr(65 + k)}'
        if car in synthetic:
            ctyres, cdoors = synthetic[car][2], synthetic[car][3]
        else:
            ctyres = vehicle_gen(b, E, car[0]); cdoors, _ = vehicle_data(b, E, car[0])
        cbogies, cwheel_y, _ = bogies_from_tyres(ctyres)
        cmesh_cid = build_mesh(stem, car, cbogies, cwheel_y)
        comps = [A.public_transport(capacity), A.vehicle_side_effects(),
                 A.activity_location(ACTIVITY_BOARD, [(np.sign(x) * 1.3, 0.7, z) for x, y, z in cdoors]),
                 A.effect_source([(EFFECT_TRAIN, (0, 4, 0), (0, -1, 0, 0))])]
        cid = did(name, stem, 'car'); car_prefab = A.train_car_prefab(stem, cmesh_cid, comps, speed)
        emit(os.path.join(name, fname(stem, 'Prefab')), lambda p: A.write_prefab(p, car_prefab), cid)
        emit(os.path.join(name, f'{fname(stem, "Prefab")[:-7]}_en-US.loc'),
             lambda p: A.write_loc(p, {f'Assets.NAME[{stem}]': f'{title} car {chr(65 + k)}'}), did(name, stem, 'loc'))
        car_cid[car] = cid
    car_cids = [(car_cid[c], 0) for c in cars]

    V = preview_parts[front][0]; nose_z = float(V[:, 2].max())
    def nose_surface(x, y):
        sel = V[(np.abs(V[:, 0] - x) < 0.15) & (np.abs(V[:, 1] - y) < 0.15)]
        return float(sel[:, 2].max()) + 0.05 if len(sel) else nose_z
    fx = [(EFFECT_HEADLIGHT, (x, y, nose_surface(x, y)), (0, 0, 0, 1)) for x, y, z in lights]
    fx.append((EFFECT_TRAIN, (0, 4, 0), (0, 0, 0, -1)))

    # icon + preview: whole consist, front car mirrored onto the back like the game does
    parts = []; z0 = 0.0
    order = [preview_parts[front]] + [preview_parts[c] for c in cars] + [preview_parts[front]]
    for k, (Vp, Np, UVp, Tp, _) in enumerate(order):
        Vp = Vp.astype(np.float64).copy(); Np = Np.astype(np.float64).copy()
        if k == len(order) - 1: Vp[:, [0, 2]] *= -1; Np[:, [0, 2]] *= -1
        Vp[:, 2] += z0 - Vp[:, 2].max(); z0 = Vp[:, 2].min() - 0.3; parts.append((Vp, Np, UVp, Tp, base_img))
    allV = np.concatenate([p[0] for p in parts]); ctr = (allV.min(0) + allV.max(0)) / 2; size = float(np.ptp(allV, 0).max())
    icon = preview.crop_to_content(preview.render([parts[0]], 900, 900, 40, -14, 48, 30, (parts[0][0].min(0) + parts[0][0].max(0)) / 2), square=True)
    icon_cid = did(name, 'icon')
    emit(fname(name, 'jpg'), lambda p: icon.resize((256, 256), Image.LANCZOS).save(p, 'JPEG', quality=90), icon_cid, base=icon_dir)
    pw = 600 + 300 * len(order)
    preview.crop_to_content(preview.render(parts, pw, pw // 4, 62, -9, size * 1.15, 32, ctr)).save(os.path.join(root, 'preview.png'))
    preview.crop_to_content(preview.render(parts, pw, pw // 8, 90, 0, size * 1.1, 30, ctr)).save(os.path.join(root, 'preview_side.png'))
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
    fp = A.train_front_prefab(name, mesh_cid, comps, car_cids, speed, reversed_end=True)
    emit(os.path.join(name, fname(name, 'Prefab')), lambda p: A.write_prefab(p, fp), front_cid)
    emit(os.path.join(name, f'{fname(name, "Prefab")[:-7]}_en-US.loc'),
         lambda p: A.write_loc(p, {f'Assets.NAME[{name}]': title}), did(name, 'loc'))

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
    ap.add_argument('--car', action='append', default=[], help="MESH:LOD entry indices, in consist order, or 'mid'")
    ap.add_argument('--middle', help='MESH:LOD+MESH:LOD: cab car with its flat end at -Z, then one with it at +Z; '
                                     'their flat halves are joined into a cabless middle car usable as --car mid')
    ap.add_argument('--speed', type=int, default=200); ap.add_argument('--capacity', type=int, default=70)
    ap.add_argument('--out', default='dist'); ap.add_argument('--ui-group', default=None)
    a = ap.parse_args()
    pair = lambda s: tuple(int(x) for x in s.split(':'))
    middle = tuple(pair(x) for x in a.middle.split('+')) if a.middle else None
    root, z = build(a.crp, a.name, a.title, a.out, pair(a.front), [c if c == 'mid' else pair(c) for c in a.car],
                    a.speed, a.capacity, a.ui_group, middle)
    print('built', root); print('zip  ', z, os.path.getsize(z) // 1024, 'KB')
