# CS1 → CS2 asset porting

Personal toolkit for moving Cities: Skylines 1 assets into Cities: Skylines II.

## Tools

| File | What it does |
|---|---|
| `tools/crp_tool.py` | Reads CS1 `.crp` packages: `list`, `info` (metadata + vehicle stats), `extract` (meshes → OBJ, textures → PNG) |
| `tools/cs2_geometry.py` | Reads and writes CS2 `.Geometry` files (meshopt + zstd streams) |
| `tools/cs2_texture.py` | Reads and writes CS2 plain `.Texture` files (BC7 mip chains; `mip_filter='max_alpha'` for emissive maps) |
| `tools/cs2_asset.py` | Writes `.Prefab`, `.Surface`, `.loc`, `.cid` (byte-identical to the game's importer output) |
| `tools/build_prop.py` | CS1 `.crp` mesh → installable CS2 prop package with `install.bat` / `uninstall.bat` |
| `tools/build_train.py` | CS1 train `.crp` → installable CS2 multiple-unit train (bogie/axle bones, doors, consist, night lights: warm windows on every car, white headlamps, red tail lamps) |
| `tools/preview.py` | Small software renderer for icons and previews |
| `tools/cok_tool.py` | Reads CS2 `.cok` packages: `tree` (prefab graph), `prefab` (dump one prefab as clean JSON) |
| `windows/cs2_grab_samples.bat` | Read-only. Zips small reference files (plain .Texture, .Surface, .Geometry, .Prefab from ImportedData and a loose-file mod) to the Desktop, capped at 60 MB |
| `windows/cs2_scout.bat` | Read-only. Run on the gaming PC; writes `cs2_scout_report.txt` to the Desktop describing the CS2 folders, installed packages, game version and Blender install |

`pip install pillow numpy meshoptimizer zstandard etcpak texture2ddecoder`

Build a prop from a CS1 asset (output goes to `dist/`, which is git-ignored because it contains converted third-party assets):

    cd tools
    python build_prop.py ASSET.crp --mesh 33 --lod 36 --name SJX40DisplayTest --title "SJ X40 display (CS1 port test)" --out ../dist

Build a train (front car, then carriages in order; the game mirrors the front car onto the back; a MESH:LOD may repeat):

    python build_train.py ASSET.crp --name SJX40 --title "SJ X40 (CS1 port)" --front 33:36 --car 0:6 --car 15:18 --out ../dist
    python build_train.py ASSET.crp --name SJX40x12 --title "SJ X40 12-car" --front 33:36 \
        --car 0:6 --car 33:36 --car 0:6 --car 33:36 --car 0:6 --car 33:36 --car 0:6 --car 33:36 --car 0:6 --car 15:18 --out ../dist

A cabless middle car can be synthesised from two cab cars whose flat ends face opposite ways (`--middle A:LOD+B:LOD`, A's flat end at -Z, B's at +Z): each is cut through its logo centre (found by rendering the side) and the flat halves are joined, then `--car mid` places it:

    python build_train.py ASSET.crp --name SJX40_4x3 --title "SJ X40 12-car, 4x3" --front 33:36 --middle 15:18+0:6 \
        --car mid --car 0:6 --car 33:36 --car mid --car 0:6 --car 33:36 --car mid --car 0:6 --car 15:18 --car mid --out ../dist

## Format notes

### CS1 `.crp`
- Header: `CRAP`, u16 format, strings package / author, u32 version, string main asset, i32 entry count, i64 data start.
- Entry: string name, string checksum, u32 type, i64 offset, i64 size. Types: 1 GameObject, 2 Material, 3 Texture, 4 Mesh, 80 Locale, 103 CustomAssetMetaData.
- Each object: bool isNull, string type name, string object name, then fields.
- Mesh: vertices (3f), colors (**4 floats**), uv (2f), normals (3f), tangents (4f), bone weights, bind poses, submeshes of i32 indices.
- Texture: bool linear, i32 aniso, i32 length, then a DDS (DXT1/DXT5). **Rows are bottom-up**; flip vertically after decoding.
- Unity is left-handed: mirror X (and reverse winding) when exporting to OBJ/FBX.
- Vehicle maps: `MainTex` = albedo; `XYSMap` = normal X/Y + specular; `ACIMap` = alpha / color mask / illumination.

### CS2 `.cok`
- A plain (stored) zip. Every file has a sibling `.cid` holding its 32-hex asset id; prefabs reference each other with `$fstrref:"CID:<id>"`.
- `.Prefab` is almost-JSON (see `cok_tool.loads`). `.loc` is binary localisation.
- `.Geometry`: **solved, read + write** in `tools/cs2_geometry.py` (full layout in its docstring). 138-byte header, then one zstd frame per stream, each stream meshoptimizer-encoded (index codec v1, vertex codec v0). Typical LOD0 layout: position 3×f32, normal 2×snorm16 octahedral, tangent 32-bit packed octahedral (15+15 bits + bitangent sign), color 4×unorm8, uv0–uv2 2×f16, uv3 2×f32, one uint32 bone index per vertex. Verified by re-encoding all 19 KISS geometries: every attribute is byte-exact after decoding (one zero-area triangle gets its corners reordered by the index codec). `pip install meshoptimizer zstandard numpy`.
- Plain `.Texture` (**solved**, `tools/cs2_texture.py`): 18-byte header (u16 5, u16 w, u16 h, u16 1, u8 mips, u8 Unity GraphicsFormat: 108 = BC7 sRGB for BaseColor/Emissive, 109 = BC7 linear for Normal/MaskMap/ControlMask, then `02 02 01 10 ff ff ff ff`) + full BC7 mip chain. Rows bottom-up. Channel conventions from importer output: Normal is AG-packed (R = 255, G = Y, A = X); MaskMap R = metallic, A = smoothness; ControlMask all 0 = no recolouring; Emissive black = off.
- Plain `.Surface` (**solved**, `tools/cs2_asset.py`): u8 1, u8 0, u8 template (1 standard, 11 glass), 5 × (u32 0, 0xFF), u32 texture count, (u8 len, slot name, 0xFF, 16-byte id), 0xFF, u32 keyword count, keywords. The 16-byte id is the texture's `.cid` with the nibbles of each byte swapped.
- `.loc` (**solved**): u16 1, "English", "en-US", "English", u32 count, (key, value) pairs, u32 0.
- `.Prefab` (**solved**): `$id` counts reference objects, `$type` is `N|Type, Assembly` on first use then `N`, two independent counters. `cs2_asset.dumps` reproduces importer prefabs character for character.
- Where local assets live: `%USERPROFILE%\AppData\LocalLow\Colossal Order\Cities Skylines II\ImportedData\<16 hex>\` holds the binary files flat, with one subfolder per asset for its prefabs and `.loc`. Icons go in `ImportedData\TextureAssets\` and are referenced as `assetdb://global/<icon cid>`.
- `VTTexture`: starts `01 00 ff` + the 16-byte texture GUID that the Surface references, then tile layout info, per-tile block-compressed data in zstd frames. Small mips live in the shared `StreamingData~/TS512_*.MidMips`. Not decoded yet.
- `.Surface` is the material: shader keywords (`_TANGENTSPACE_OCTO`, `_EMISSIVE_PROCEDURAL`) and texture slots `_BaseColorMap`, `_NormalMap`, `_MaskMap`, `_ControlMask`, `_EmissiveColorMap`.
- Textures ship as virtual-texture data (`StreamingData~/VT/*.VTTexture`, `*.VTSurface`, `TS512_*.MidMips`).

### How a CS2 multiple-unit train is built (from the Stadler KISS Caltrain reference)
- `MultipleUnitTrainFrontPrefab`: the lead car. Holds `m_TrackType`, `m_EnergyType`, `m_MaxSpeed`, `m_Acceleration`, `m_Braking`, the ordered `m_Carriages` list (each with `m_Direction`) and `m_AddReversedEndCarriage` (mirrors the lead car onto the back).
- `MultipleUnitTrainCarPrefab`: every other car. Carries its own `PublicTransport` (capacity), `ActivityLocation` (door positions), `EffectSource` (lights, sounds).
- `RenderPrefab`: one mesh. Links a Geometry and a Surface, holds `LodProperties` (LOD1/LOD2 render prefabs) and `ProceduralAnimationProperties`, the bones: a root, one `Wheelset` per bogie (type 15) and `Axle` bones (type 4).
- Bones (`ProceduralAnimationProperties`): list of BoneInfo {name, position (local to parent), rotation, scale, bindPose (inverse of the bone's rest world transform), parentId, m_Type}. Types seen: 0 root, 15 wheelset (swivels on curves), 4 axle (spins). Vertex `blendindices` (one uint32) = index into that list; no blend weights. Only LOD0 carries bones.
- Front prefab: `m_Carriages` (carriage cid + `m_Direction` 0/1) and `m_AddReversedEndCarriage` (mirrors the front car onto the back). `m_MaxSpeed` is km/h.
- Lights: the emissive texture's RGB is the lamp colour and its alpha is a *layer id*: the game turns it into a 1-based index into the `EmissiveProperties` multi-light list, apparently `round(alpha / 25.5)` clamped to the list length (inferred: KISS uses 25, 51, 76, …, and a window on 255 still hit the second of two lights; alpha 0 = no light). Each mapping has a purpose (`Game.Prefabs.EmissiveProperties.Purpose`, read from Game.dll): 23 `Interior1` is on at night on every car of a train, 3 `Headlight_LowBeam` on the leading car, 6 `RearLight` on the trailing car; `color` is the lit colour, `colorOff` only matters for purposes that blend (boarding lights). Intensity/luminance from KISS: windows 0.05 / 0.964706, lamps 1 / 1. Surfaces need the `_EMISSIVE_PROCEDURAL` keyword.
- **Emissive mips must not be averaged.** The normal (Lanczos) mip filter turns a window's layer id into garbage two or three mips down, so only the car nearest the camera lights up and everything further away stays dark. `cs2_texture.write(..., mip_filter='max_alpha')` keeps, per 2×2 block, the texel with the highest alpha.
- The builder lights CS1 illumination 16–40 (the lit passenger windows) as warm interior light on layer 25 and leaves illumination 0 (door windows, windscreen, displays) dark. Lamp lenses (illumination > 200) are split: the top 60 % of the lens texels on layer 76 (white, purpose 3), the bottom 40 % on layer 51 (red, purpose 6), oriented from the mesh (texture rows vs. world height of the lower lamps). Roof lamps are white only: their triangles get duplicated vertices whose UVs point at a copy of the lens texels pasted into free atlas space (`lamp_layout` / `relocate_triangles`).
- Doors = `ActivityLocation` entries with activity `55cd3132…`; headlights = `EffectSource` entries with effect `de32f8ad…`; every car also carries effect `76d8c521…` at (0, 4, 0).
- The KISS pack puts a tiny `kisscaltrain07` "car" (gangway bellows, no seats) between every real car.
- The pack also carries components from the **Better Transit Selector** code mod (`BTS_TrainPack`, `BTS_TrainVariant`). Strip those when using it as a template.
