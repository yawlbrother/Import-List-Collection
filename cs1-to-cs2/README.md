# CS1 → CS2 asset porting

Personal toolkit for moving Cities: Skylines 1 assets into Cities: Skylines II.

## Tools

| File | What it does |
|---|---|
| `tools/crp_tool.py` | Reads CS1 `.crp` packages: `list`, `info` (metadata + vehicle stats), `extract` (meshes → OBJ, textures → PNG) |
| `tools/cs2_geometry.py` | Reads and writes CS2 `.Geometry` files (meshopt + zstd streams) |
| `tools/cok_tool.py` | Reads CS2 `.cok` packages: `tree` (prefab graph), `prefab` (dump one prefab as clean JSON) |
| `windows/cs2_scout.bat` | Read-only. Run on the gaming PC; writes `cs2_scout_report.txt` to the Desktop describing the CS2 folders, installed packages, game version and Blender install |

`pip install pillow meshoptimizer zstandard numpy`

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
- `VTTexture`: starts `01 00 ff` + the 16-byte texture GUID that the Surface references, then tile layout info, per-tile block-compressed data in zstd frames. Small mips live in the shared `StreamingData~/TS512_*.MidMips`. Not decoded yet.
- `.Surface` is the material: shader keywords (`_TANGENTSPACE_OCTO`, `_EMISSIVE_PROCEDURAL`) and texture slots `_BaseColorMap`, `_NormalMap`, `_MaskMap`, `_ControlMask`, `_EmissiveColorMap`.
- Textures ship as virtual-texture data (`StreamingData~/VT/*.VTTexture`, `*.VTSurface`, `TS512_*.MidMips`).

### How a CS2 multiple-unit train is built (from the Stadler KISS Caltrain reference)
- `MultipleUnitTrainFrontPrefab`: the lead car. Holds `m_TrackType`, `m_EnergyType`, `m_MaxSpeed`, `m_Acceleration`, `m_Braking`, the ordered `m_Carriages` list (each with `m_Direction`) and `m_AddReversedEndCarriage` (mirrors the lead car onto the back).
- `MultipleUnitTrainCarPrefab`: every other car. Carries its own `PublicTransport` (capacity), `ActivityLocation` (door positions), `EffectSource` (lights, sounds).
- `RenderPrefab`: one mesh. Links a Geometry and a Surface, holds `LodProperties` (LOD1/LOD2 render prefabs) and `ProceduralAnimationProperties`, the bones: a root, one `Wheelset` per bogie (type 15) and `Axle` bones (type 4).
- The KISS pack puts a tiny `kisscaltrain07` "car" (gangway bellows, no seats) between every real car.
- The pack also carries components from the **Better Transit Selector** code mod (`BTS_TrainPack`, `BTS_TrainVariant`). Strip those when using it as a template.
