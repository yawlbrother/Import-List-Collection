#!/usr/bin/env python3
"""Build an installable CS2 multiple-unit train from a CS1 train .crp package.

    python build_train.py ASSET.crp --name SJX40 --title "SJ X40" --out ../dist \
        --front 33:36 --car 0:6 --car 15:18

--front / --car take MESH:LOD entry indices from `crp_tool.py list`. Cars are listed in
consist order behind the front car; the game mirrors the front car onto the back
(m_AddReversedEndCarriage). Bogies are found from the CS1 wheel data (VehicleInfoGen
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
BODY_CUT_Y = 1.15        # below this (near a bogie) = bogie frame
BOGIE_ZONE_Z = 5.0       # bogie parts only exist beyond this distance from the car centre
WHEEL_TOP_Y = 0.72

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
    """Bone index per vertex: 0 body, then per bogie [wheelset, axle, axle] in bogies order."""
    bone = np.zeros(len(V), np.uint32)
    comp = connected_components(V, T)
    comp_size = np.bincount(comp); comp_ymax = np.full(comp.max() + 1, -9.0)
    np.maximum.at(comp_ymax, comp, V[:, 1])
    is_wheel_part = (comp_size[comp] < 120) & (comp_ymax[comp] < WHEEL_TOP_Y) & (np.abs(V[:, 0]) < 0.9)
    is_shaft = (V[:, 1] < WHEEL_TOP_Y) & (np.abs(V[:, 0]) < 0.70)
    idx = 1
    for bz, axles in bogies:
        side = np.sign(bz)
        near = (V[:, 1] < BODY_CUT_Y) & (V[:, 2] * side > BOGIE_ZONE_Z)
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

# ---------------------------------------------------------------- build
def build(crp, name, title, out, front, cars, speed, capacity, ui_group):
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

    # shared texture sets: all cars use the same CS1 material
    tex_cids = {}
    for level, mi in (('', front[0]), ('_LOD1', front[1])):
        tex_cids[level] = []
        for slot, (im, srgb) in convert_textures(b, E, material_for_mesh(E, mats, mi)).items():
            cid = did(name, level, slot)
            emit(fname(f'{name}{level}_{slot}', 'Texture'), lambda p, im=im, srgb=srgb: T.write(p, im, srgb), cid)
            tex_cids[level].append((SURFACE_SLOT[slot], cid))
            if level == '' and slot == 'BaseColor': base_img = im

    preview_parts = []
    def build_mesh(stem, mi, lod_mi, bogies, wheel_y):
        """Writes LOD1 + LOD0 geometry, surfaces and render prefabs; returns LOD0 render prefab cid."""
        lod_cid = None
        for level, idx in (('_LOD1', lod_mi), ('', mi)):
            m = C.read_mesh(b, E[idx])
            tris, attrs, lo, hi, area = cs1_mesh_to_cs2(m)
            comps = []
            if level == '':
                V = np.asarray(m['V'], np.float32)
                bones = assign_bones(V, tris.reshape(-1, 3), bogies, wheel_y)
                attrs['blendindices'] = (10, bones[:, None])
                comps.append(A.procedural_animation(bone_list(stem, bogies, wheel_y)))
                preview_parts.append((V, np.asarray(m['N']), np.asarray(m['UV']), tris.reshape(-1, 3), bones))
            s = stem + level
            geo_cid = did(name, s, 'geometry'); emit(fname(s, 'Geometry'), lambda p: G.write(p, tris, attrs), geo_cid)
            surf_cid = did(name, s, 'surface'); emit(fname(s, 'Surface'), lambda p: A.write_surface(p, tex_cids[level]), surf_cid)
            rp_cid = did(name, s, 'renderprefab')
            rp = A.render_prefab(f'{s} Mesh', geo_cid, [surf_cid], lo, hi, area, len(tris), len(m['V']),
                                 lod_cids=[lod_cid] if lod_cid else [], components=comps)
            emit(os.path.join(name, fname(f'{s} Mesh', 'Prefab')), lambda p: A.write_prefab(p, rp), rp_cid)
            lod_cid = rp_cid
        return lod_cid

    car_cids = []
    for n, (mi, lod_mi) in enumerate(cars, 1):
        stem = f'{name}_Car{n}'
        tyres = vehicle_gen(b, E, mi); bogies, wheel_y, _ = bogies_from_tyres(tyres)
        doors, _ = vehicle_data(b, E, mi)
        mesh_cid = build_mesh(stem, mi, lod_mi, bogies, wheel_y)
        comps = [A.public_transport(capacity), A.vehicle_side_effects(),
                 A.activity_location(ACTIVITY_BOARD, [(np.sign(x) * 1.3, 0.7, z) for x, y, z in doors]),
                 A.effect_source([(EFFECT_TRAIN, (0, 4, 0), (0, -1, 0, 0))])]
        cid = did(name, stem, 'car'); car = A.train_car_prefab(stem, mesh_cid, comps, speed)
        emit(os.path.join(name, fname(stem, 'Prefab')), lambda p: A.write_prefab(p, car), cid)
        emit(os.path.join(name, f'{fname(stem, "Prefab")[:-7]}_en-US.loc'),
             lambda p: A.write_loc(p, {f'Assets.NAME[{stem}]': f'{title} car {n + 1}'}), did(name, stem, 'loc'))
        car_cids.append((cid, 0))

    # front car
    mi, lod_mi = front
    tyres = vehicle_gen(b, E, mi); bogies, wheel_y, _ = bogies_from_tyres(tyres)
    doors, lights = vehicle_data(b, E, mi)
    mesh_cid = build_mesh(name, mi, lod_mi, bogies, wheel_y)
    V = preview_parts[-1][0]; nose_z = float(V[:, 2].max())
    def nose_surface(x, y):
        sel = V[(np.abs(V[:, 0] - x) < 0.15) & (np.abs(V[:, 1] - y) < 0.15)]
        return float(sel[:, 2].max()) + 0.05 if len(sel) else nose_z
    fx = [(EFFECT_HEADLIGHT, (x, y, nose_surface(x, y)), (0, 0, 0, 1)) for x, y, z in lights]
    fx.append((EFFECT_TRAIN, (0, 4, 0), (0, 0, 0, -1)))

    # icon + preview: whole consist, front car mirrored onto the back like the game does
    parts = []; z0 = 0.0
    order = [preview_parts[-1]] + preview_parts[:-1] + [preview_parts[-1]]
    for k, (Vp, Np, UVp, Tp, _) in enumerate(order):
        Vp = Vp.astype(np.float64).copy(); Np = Np.astype(np.float64).copy()
        if k == len(order) - 1: Vp[:, [0, 2]] *= -1; Np[:, [0, 2]] *= -1
        Vp[:, 2] += z0 - Vp[:, 2].max(); z0 = Vp[:, 2].min() - 0.3; parts.append((Vp, Np, UVp, Tp, base_img))
    allV = np.concatenate([p[0] for p in parts]); ctr = (allV.min(0) + allV.max(0)) / 2; size = float(np.ptp(allV, 0).max())
    icon = preview.crop_to_content(preview.render([parts[0]], 900, 900, 40, -14, 48, 30, (parts[0][0].min(0) + parts[0][0].max(0)) / 2), square=True)
    icon_cid = did(name, 'icon')
    emit(fname(name, 'jpg'), lambda p: icon.resize((256, 256), Image.LANCZOS).save(p, 'JPEG', quality=90), icon_cid, base=icon_dir)
    preview.crop_to_content(preview.render(parts, 2400, 600, 62, -9, size * 1.15, 32, ctr)).save(os.path.join(root, 'preview.png'))
    bone_img = bone_preview(preview_parts[-1]); bone_img.save(os.path.join(root, 'bones.png'))

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
    ap.add_argument('--car', action='append', default=[], help='MESH:LOD entry indices, in consist order')
    ap.add_argument('--speed', type=int, default=200); ap.add_argument('--capacity', type=int, default=70)
    ap.add_argument('--out', default='dist'); ap.add_argument('--ui-group', default=None)
    a = ap.parse_args()
    pair = lambda s: tuple(int(x) for x in s.split(':'))
    root, z = build(a.crp, a.name, a.title, a.out, pair(a.front), [pair(c) for c in a.car], a.speed, a.capacity, a.ui_group)
    print('built', root); print('zip  ', z, os.path.getsize(z) // 1024, 'KB')
