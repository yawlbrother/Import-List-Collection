#!/usr/bin/env python3
"""Build an installable CS2 prop from one mesh of a CS1 .crp package.

    python build_prop.py ASSET.crp --mesh 33 --lod 36 --name SJX40DisplayTest \
        --title "SJ X40 display (CS1 port test)" --out ../dist

Produces <out>/<name>/ with payload/, install.bat, uninstall.bat, preview.png, README.txt
and a <name>.zip of that folder. The install script copies the asset into the game's
ImportedData folder (where CS2's own asset importer keeps imported assets).
"""
import argparse, hashlib, io, os, shutil, struct, zipfile
import numpy as np
from PIL import Image

import crp_tool as C
import cs2_asset as A
import cs2_geometry as G
import cs2_texture as T
import preview

# Menu group used by the user's own imported ModernCentralStation asset
DEFAULT_UI_GROUP = '1239dee54354f354dbc47d2e5d5ead47'
SLOTS = ['BaseColor', 'ControlMask', 'Emissive', 'MaskMap', 'Normal']
SURFACE_SLOT = {'BaseColor': '_BaseColorMap', 'ControlMask': '_ControlMask', 'Emissive': '_EmissiveColorMap',
                'MaskMap': '_MaskMap', 'Normal': '_NormalMap'}

def did(*parts):
    """Deterministic 32-hex id, so rebuilding the same asset gives the same ids."""
    return hashlib.md5('/'.join(parts).encode()).hexdigest()

# ---------------------------------------------------------------- CS1 side
def cs1_materials(b, E):
    """material entry index -> {slot: texture entry index}"""
    chk = {e['checksum']: i for i, e in enumerate(E)}
    out = {}
    for i, e in enumerate(E):
        if e['type'] != 2:
            continue
        r = C.Reader(b, e['abs'] + 1); r.s(); r.s(); r.s(); tex = {}
        for _ in range(r.i32()):
            kind = r.i32(); pname = r.s()
            if kind in (0, 1): r.p += 16
            elif kind == 2: r.p += 4
            elif kind == 3: r.u8(); tex[pname] = chk.get(r.s())
        out[i] = tex
    return out

def material_for_mesh(E, mats, mesh_index):
    """CS1 stores each mesh's material as the next material entry after it."""
    for j in range(mesh_index + 1, len(E)):
        if j in mats and '_MainTex' in mats[j]:
            return mats[j]
    raise SystemExit(f'no material found after mesh {mesh_index}')

def convert_textures(b, E, tex):
    """CS1 MainTex/XYSMap/ACIMap -> CS2 slot images (PIL, top-row-first)."""
    main = C.read_texture(b, E[tex['_MainTex']])[2].convert('RGBA')
    xys = np.asarray(C.read_texture(b, E[tex['_XYSMap']])[2].convert('RGBA').resize(main.size)).astype(np.int32)
    aci = np.asarray(C.read_texture(b, E[tex['_ACIMap']])[2].convert('RGBA').resize(main.size)).astype(np.int32)
    m = np.asarray(main).astype(np.int32)
    h, w = m.shape[:2]
    def img(r, g, bb, a):
        return Image.fromarray(np.stack([np.broadcast_to(np.asarray(c), (h, w)) for c in (r, g, bb, a)], 2).astype(np.uint8), 'RGBA')
    return {
        # CS1 ACI red = inverted alpha
        'BaseColor': (img(m[..., 0], m[..., 1], m[..., 2], 255 - aci[..., 0]), True),
        # CS2 normals are AG-packed: X in alpha, Y in green, red = 255
        'Normal': (img(255, xys[..., 1], xys[..., 1], xys[..., 0]), False),
        # R metallic, A smoothness (from CS1 specular)
        'MaskMap': (img(0, 0, 0, np.clip(xys[..., 2] * 0.75, 0, 255)), False),
        'ControlMask': (img(0, 0, 0, 0), False),
        'Emissive': (img(0, 0, 0, 255), True),
    }

def cs1_mesh_to_cs2(m):
    V = np.asarray(m['V'], np.float32); N = np.asarray(m['N'], np.float64)
    UV = np.asarray(m['UV'], np.float32); tris = np.concatenate([np.asarray(t, np.uint32) for t in m['tris']])
    Tn = np.asarray(m['T'], np.float64)
    if len(Tn) != len(V):
        Tn = compute_tangents(V, N, UV, tris.reshape(-1, 3))
    attrs = {
        'position': (0, V),
        'normal': (5, G.pack_normals(N)),
        'tangent': (0, G.pack_tangents(Tn).view(np.float32)[:, None]),
        'uv0': (1, UV.astype(np.float16)),
    }
    t3 = tris.reshape(-1, 3).astype(np.int64)
    area = 0.5 * np.linalg.norm(np.cross(V[t3[:, 1]] - V[t3[:, 0]], V[t3[:, 2]] - V[t3[:, 0]]), axis=1).sum()
    return tris, attrs, V.min(0), V.max(0), float(area)

def compute_tangents(V, N, UV, t3):
    t = np.zeros((len(V), 3)); bt = np.zeros((len(V), 3))
    p0, p1, p2 = V[t3[:, 0]], V[t3[:, 1]], V[t3[:, 2]]; u0, u1, u2 = UV[t3[:, 0]], UV[t3[:, 1]], UV[t3[:, 2]]
    e1, e2, d1, d2 = p1 - p0, p2 - p0, u1 - u0, u2 - u0
    den = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]; r = np.where(np.abs(den) > 1e-12, 1 / np.where(den == 0, 1, den), 0)
    sd = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * r[:, None]; td = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) * r[:, None]
    for k in range(3):
        np.add.at(t, t3[:, k], sd); np.add.at(bt, t3[:, k], td)
    t -= N * (N * t).sum(1, keepdims=True)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    w = np.where((np.cross(N, t) * bt).sum(1) < 0, -1.0, 1.0)
    return np.concatenate([t, w[:, None]], 1)

# ---------------------------------------------------------------- build
def build(crp, mesh_index, lod_index, name, title, out, ui_group=DEFAULT_UI_GROUP):
    b, hdr, E = C.parse(crp)
    mats = cs1_materials(b, E)
    root = os.path.join(out, name); shutil.rmtree(root, ignore_errors=True)
    asset_dir = os.path.join(root, 'payload', 'asset'); sub = os.path.join(asset_dir, name)
    icon_dir = os.path.join(root, 'payload', 'TextureAssets')
    for d in (sub, icon_dir):
        os.makedirs(d)
    files = []   # paths relative to payload/asset or payload/TextureAssets, for the uninstaller

    def fname(stem, ext):
        return f'{stem}_{did(name, stem, ext)}.{ext}'

    def emit(path_rel, write_fn, cid, base=asset_dir):
        p = os.path.join(base, path_rel); write_fn(p); A.write_cid(p, cid)
        tag = 'asset' if base == asset_dir else 'TextureAssets'
        files.extend([(tag, path_rel), (tag, path_rel + '.cid')])

    levels = [('', mesh_index)] + ([('_LOD1', lod_index)] if lod_index is not None else [])
    mesh_prefab_cids = {}
    parts_for_preview = None
    for suffix, mi in reversed(levels):          # LOD first so the main mesh can reference it
        stem = name + suffix
        m = C.read_mesh(b, E[mi])
        tris, attrs, lo, hi, area = cs1_mesh_to_cs2(m)
        geo_cid = did(name, stem, 'geometry')
        emit(fname(stem, 'Geometry'), lambda p: G.write(p, tris, attrs), geo_cid)
        tex_cids = []
        for slot, (im, srgb) in convert_textures(b, E, material_for_mesh(E, mats, mi)).items():
            cid = did(name, stem, slot)
            emit(fname(f'{stem}_{slot}', 'Texture'), lambda p, im=im, srgb=srgb: T.write(p, im, srgb), cid)
            tex_cids.append((SURFACE_SLOT[slot], cid))
            if suffix == '' and slot == 'BaseColor':
                parts_for_preview = [(np.asarray(m['V']), np.asarray(m['N']), np.asarray(m['UV']), tris.reshape(-1, 3), im)]
        surf_cid = did(name, stem, 'surface')
        emit(fname(stem, 'Surface'), lambda p: A.write_surface(p, tex_cids), surf_cid)
        rp_cid = did(name, stem, 'renderprefab')
        lods = [mesh_prefab_cids['_LOD1']] if (suffix == '' and '_LOD1' in mesh_prefab_cids) else []
        rp = A.render_prefab(f'{stem} Mesh', geo_cid, [surf_cid], lo, hi, area, len(tris), len(m['V']), lod_cids=lods)
        emit(os.path.join(name, fname(f'{stem} Mesh', 'Prefab')), lambda p: A.write_prefab(p, rp), rp_cid)
        mesh_prefab_cids[suffix] = rp_cid

    # icon + preview
    V = parts_for_preview[0][0]; ctr = (V.min(0) + V.max(0)) / 2; size = float(np.ptp(V, 0).max())
    icon = preview.crop_to_content(preview.render(parts_for_preview, 900, 900, 40, -14, size * 1.9, 30, ctr), square=True)
    icon_cid = did(name, 'icon')
    icon_name = fname(name, 'jpg')
    emit(icon_name, lambda p: icon.resize((256, 256), Image.LANCZOS).save(p, 'JPEG', quality=90), icon_cid, base=icon_dir)
    preview.crop_to_content(preview.render(parts_for_preview, 1600, 700, 55, -10, size * 1.6, 30, ctr)).save(
        os.path.join(root, 'preview.png'))

    obj_cid = did(name, 'object')
    sop = A.static_object_prefab(name, mesh_prefab_cids[''], ui_group, icon_cid)
    emit(os.path.join(name, fname(name, 'Prefab')), lambda p: A.write_prefab(p, sop), obj_cid)
    loc_stem = fname(name, 'x').rsplit('.', 1)[0]
    emit(os.path.join(name, f'{loc_stem}_en-US.loc'), lambda p: A.write_loc(p, {f'Assets.NAME[{name}]': title}),
         did(name, 'loc'))

    write_scripts(root, name, title, files)
    zpath = os.path.join(out, name + '.zip')
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for dp, _, fs in os.walk(root):
            for f in fs:
                full = os.path.join(dp, f); z.write(full, os.path.join(name, os.path.relpath(full, root)))
    return root, zpath

# ---------------------------------------------------------------- Windows scripts
LAUNCHER = r'''<# : batch portion
@echo off
set "CS1PORT_SELF=%~f0"
powershell -NoProfile -ExecutionPolicy Bypass -Command "iex (Get-Content -LiteralPath $env:CS1PORT_SELF -Raw)"
echo.
pause
exit /b
#>
'''

INSTALL = LAUNCHER + r'''
$ErrorActionPreference = 'Stop'
$name  = '__NAME__'
$title = '__TITLE__'
$here  = Split-Path -Parent $env:CS1PORT_SELF
$ud    = Join-Path $env:USERPROFILE 'AppData\LocalLow\Colossal Order\Cities Skylines II'
$imp   = Join-Path $ud 'ImportedData'
Write-Host ""
Write-Host " Installing '$title'"
try {
    if (Get-Process -Name 'Cities2' -ErrorAction SilentlyContinue) { throw 'Cities: Skylines II is running. Close the game first, then run this again.' }
    if (-not (Test-Path -LiteralPath $ud)) { throw "Game data folder not found: $ud" }
    if (-not (Test-Path -LiteralPath $imp)) { New-Item -ItemType Directory -Path $imp | Out-Null }
    # the importer keeps assets in ImportedData\<16 hex chars>\ ; use the existing one
    $target = Get-ChildItem -LiteralPath $imp -Directory | Where-Object { $_.Name -match '^[0-9a-f]{16}$' } |
              Sort-Object { @(Get-ChildItem -LiteralPath $_.FullName -File).Count } -Descending | Select-Object -First 1
    if ($target) { $target = $target.FullName } else { $target = Join-Path $imp '__FALLBACK__'; New-Item -ItemType Directory -Path $target | Out-Null }
    $icons = Join-Path $imp 'TextureAssets'
    if (-not (Test-Path -LiteralPath $icons)) { New-Item -ItemType Directory -Path $icons | Out-Null }
    $log = New-Object System.Collections.Generic.List[string]
    foreach ($pair in @(@('asset', $target), @('TextureAssets', $icons))) {
        $src = Join-Path $here ('payload\' + $pair[0])
        Get-ChildItem -LiteralPath $src -Recurse -File | ForEach-Object {
            $rel  = $_.FullName.Substring($src.Length + 1)
            $dest = Join-Path $pair[1] $rel
            New-Item -ItemType Directory -Path (Split-Path $dest) -Force | Out-Null
            Copy-Item -LiteralPath $_.FullName -Destination $dest -Force
            $log.Add($dest)
        }
    }
    $logDir = Join-Path $ud 'ModsData\CS1Port'
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    $log | Set-Content -LiteralPath (Join-Path $logDir ($name + '.installed.txt')) -Encoding UTF8
    Write-Host (" Copied {0} files into {1}" -f $log.Count, $target)
    Write-Host ""
    Write-Host " Done. Start the game and look in the same menu as your ModernCentralStation asset."
    Write-Host " To remove it later, run uninstall.bat."
} catch {
    Write-Host ""
    Write-Host (" FAILED: " + $_.Exception.Message) -ForegroundColor Red
}
'''

UNINSTALL = LAUNCHER + r'''
$ErrorActionPreference = 'Stop'
$name  = '__NAME__'
$ud    = Join-Path $env:USERPROFILE 'AppData\LocalLow\Colossal Order\Cities Skylines II'
$imp   = Join-Path $ud 'ImportedData'
$known = @(
__FILES__
)
try {
    if (Get-Process -Name 'Cities2' -ErrorAction SilentlyContinue) { throw 'Cities: Skylines II is running. Close the game first, then run this again.' }
    $removed = 0
    # delete exactly the files this package installed, wherever they ended up under ImportedData
    Get-ChildItem -LiteralPath $imp -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $known -contains $_.Name } |
        ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force; $removed++ }
    Get-ChildItem -LiteralPath $imp -Recurse -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -eq $name -and @(Get-ChildItem -LiteralPath $_.FullName -Force).Count -eq 0 } |
        ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force }
    Remove-Item -LiteralPath (Join-Path $ud ('ModsData\CS1Port\' + $name + '.installed.txt')) -Force -ErrorAction SilentlyContinue
    Write-Host (" Removed {0} files." -f $removed)
} catch {
    Write-Host (" FAILED: " + $_.Exception.Message) -ForegroundColor Red
}
'''

def write_scripts(root, name, title, files):
    ps_title = title.replace("'", "''")
    fallback = did(name, 'folder')[:16]
    names = sorted({os.path.basename(rel) for _, rel in files})
    for fn, body in (('install.bat', INSTALL), ('uninstall.bat', UNINSTALL)):
        text = (body.replace('__NAME__', name).replace('__TITLE__', ps_title).replace('__FALLBACK__', fallback)
                .replace('__FILES__', ',\n'.join("    '" + n.replace("'", "''") + "'" for n in names)))
        with open(os.path.join(root, fn), 'w', newline='\r\n', encoding='ascii') as f:
            f.write(text)
    with open(os.path.join(root, 'README.txt'), 'w', newline='\r\n') as f:
        f.write(f'{title}\r\n\r\n1. Close Cities: Skylines II.\r\n2. Double-click install.bat.\r\n'
                f'3. Start the game. The asset "{title}" is in the same menu as your ModernCentralStation.\r\n\r\n'
                'uninstall.bat removes exactly the files install.bat added.\r\n')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('crp'); ap.add_argument('--mesh', type=int, required=True); ap.add_argument('--lod', type=int)
    ap.add_argument('--name', required=True); ap.add_argument('--title', required=True)
    ap.add_argument('--out', default='dist'); ap.add_argument('--ui-group', default=DEFAULT_UI_GROUP)
    a = ap.parse_args()
    root, z = build(a.crp, a.mesh, a.lod, a.name, a.title, a.out, a.ui_group)
    print('built', root); print('zip  ', z, os.path.getsize(z) // 1024, 'KB')
