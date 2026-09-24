# CS1 → CS2 asset porting

Personal toolkit for moving Cities: Skylines 1 assets into Cities: Skylines II.

## Tools

| File | What it does |
|---|---|
| `tools/crp_tool.py` | Reads CS1 `.crp` packages: `list`, `info` (metadata + vehicle stats), `extract` (meshes → OBJ, textures → PNG) |
| `tools/cs2_geometry.py` | Reads and writes CS2 `.Geometry` files (meshopt + zstd streams) |
| `tools/cs2_texture.py` | Reads and writes CS2 plain `.Texture` files (BC7 mip chains) |
| `tools/cs2_asset.py` | Writes `.Prefab`, `.Surface`, `.loc`, `.cid` (byte-identical to the game's importer output) |
| `tools/build_prop.py` | CS1 `.crp` mesh → installable CS2 prop package with `install.bat` / `uninstall.bat` |
| `tools/preview.py` | Small software renderer for icons and previews |
| `tools/cok_tool.py` | Reads CS2 `.cok` packages: `tree` (prefab graph), `prefab` (dump one prefab as clean JSON) |
| `windows/cs2_grab_samples.bat` | Read-only. Zips small reference files (plain .Texture, .Surface, .Geometry, .Prefab from ImportedData and a loose-file mod) to the Desktop, capped at 60 MB |
| `windows/cs2_scout.bat` | Read-only. Run on the gaming PC; writes `cs2_scout_report.txt` to the Desktop describing the CS2 folders, installed packages, game version and Blender install |

`pip install pillow numpy meshoptimizer zstandard etcpak texture2ddecoder`

Build a prop from a CS1 asset (output goes to `dist/`, which is git-ignored because it contains converted third-party assets):

    cd tools
    python build_prop.py ASSET.crp --mesh 33 --lod 36 --name SJX40DisplayTest --title "SJ X40 display (CS1 port test)" --out ../dist

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
- The KISS pack puts a tiny `kisscaltrain07` "car" (gangway bellows, no seats) between every real car.
- The pack also carries components from the **Better Transit Selector** code mod (`BTS_TrainPack`, `BTS_TrainVariant`). Strip those when using it as a template.
