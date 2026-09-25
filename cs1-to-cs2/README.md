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
| `windows/cs2_import.bat` | The one-click installer: drop it in a folder with the package zips (e.g. `D:\modern central\cs2_import`), double-click, done. Unpacks and installs every zip, replaces earlier versions of the same package cleanly, offers to remove packages whose zip is gone; `cs2_import.bat remove NAME` removes one |
| `windows/cs2_grab_samples.bat` | Read-only. Zips small reference files (plain .Texture, .Surface, .Geometry, .Prefab from ImportedData and a loose-file mod) to the Desktop, capped at 60 MB |
| `windows/cs2_scout.bat` | Read-only. Run on the gaming PC; writes `cs2_scout_report.txt` to the Desktop describing the CS2 folders, installed packages, game version and Blender install |

`pip install pillow numpy meshoptimizer zstandard etcpak texture2ddecoder`

Build a prop from a CS1 asset (output goes to `dist/`, which is git-ignored because it contains converted third-party assets):

    cd tools
    python build_prop.py ASSET.crp --mesh 33 --lod 36 --name SJX40DisplayTest --title "SJ X40 display (CS1 port test)" --out ../dist

Build a train (front car, then carriages in order; the game mirrors the front car onto the back; a MESH:LOD may repeat). `--car SPECxMIN-MAX` gives a carriage a count range and `--units MIN-MAX` lets the game couple that many whole units nose to tail, so one prefab yields several train lengths (here 5, 6, 10, 11 or 12 cars):

    python build_train.py ASSET.crp --name SJX40 --title "SJ X40 (CS1 port)" --front 33:36 --car 0:6 --car 15:18 --out ../dist
    python build_train.py ASSET.crp --name SJX40_MU --title "SJ X40 (CS1 port)" --front 33:36 --middle 15:18+0:6 --car midx3-4 --units 1-2 --out ../dist

Every build also writes a static prop per car type (`<name>_CabProp`, `<name>_MidProp`, …) that shares the train's textures: same mesh without bone indices, interior lights as always-on decorative lights (invisible by day), lamps off.

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
- `.Geometry`: **solved, read + write** in `tools/cs2_geometry.py` (full layout in its docstring). 138-byte header, then one zstd frame per stream, each stream meshoptimizer-encoded (index codec v1, vertex codec v0). Typical LOD0 layout: position 3×f32, normal 2×snorm16 octahedral, tangent 32-bit packed octahedral (15+15 bits + bitangent sign), color 4×unorm8 (**the light index per vertex, see Lights below**), uv0–uv2 2×f16, uv3 2×f32, one uint32 bone index per vertex. Verified by re-encoding all 19 KISS geometries: every attribute is byte-exact after decoding (one zero-area triangle gets its corners reordered by the index codec). `pip install meshoptimizer zstandard numpy`.
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
- Lights, the part that cost the most: the emissive texture's RGB is the light colour and its alpha only marks lit texels (255). **Which light a texel belongs to comes from the geometry's vertex `color` attribute** (4 × uint8, format 2): the game's importer writes the 1-based index into the render prefab's `EmissiveProperties` multi-light list into it. The KISS body has colour (1,0,0,0) on its 176 window vertices and its headlight mesh (1|2|3,0,0,0) on the lamp quads; everything else is (0,0,0,0). A geometry without that attribute gets Unity's default white vertex colour, so every lit texel picks the *last* light in the list: that is why earlier builds glowed blinding white and only on the leading car (the last light was the headlamp, purpose 3, which needs the MainLights flag), and why a list padded with dummies lit nothing at all. `build_train.py` rasterises each triangle in UV space, votes the light index of the lit texels it covers onto its vertices and writes it to all four channels (the wiki's "at most 4 lights per triangle" suggests the channels are slots; using all four sidesteps not knowing which one the shader reads). The prefab's `layerId` is only read by the editor.
- Purposes (`Game.Prefabs.EmissiveProperties.Purpose`, read from Game.dll): 23 `Interior1` = on at night on every car (needs the car's InteriorLights flag, which the train layout job gives every car), 3 `Headlight_LowBeam` = leading car, 6 `RearLight` = trailing car, 1 = always on. `color` is the lit colour; `colorOff` only matters for purposes that blend (boarding lights). Intensity/luminance from KISS: windows 0.05 / 0.964706, lamps 1 / 1. Surfaces need the `_EMISSIVE_PROCEDURAL` keyword.
- Emissive mips must not be averaged (a window's alpha would fade out a few mips down and distant cars would lose their lights): `cs2_texture.write(..., mip_filter='max_alpha')` keeps, per 2×2 block, the texel with the highest alpha.
- Door indicator lights: purposes 69/70 (`BoardingLightLeft/Right`) are lit while the train boards on that side; the game swaps the two for cars it runs reversed, so left is always the mesh's -x side. The CS1 mesh has no lamp geometry for them, so the builder adds a 16 × 8 cm quad above every CS1 door position on a separate sub-mesh (`<stem>_Doors`, the second entry of the car's `m_Meshes`, like the KISS headlight mesh) with its own two-light list: vertex colour R = 1 on the -x side, 2 on +x. The quads sample a small painted patch in free atlas space and are wound the way the body winds its triangles (cross product along the stored normal); wound the other way they are backface-culled and simply never appear. `--door-canary` makes the -x lamps always-on for a look at the train without a platform.
- Line colour: the door leaves take the transport line's colour, like vanilla buses. A `ColorProperties` component on the LOD0 render prefab (one variation, channel 0 bound to the external colour source `Brand` = 0, which the game resolves to the entity's brand, else its current route's colour; `Parent` = 1 would copy the owner's mesh colours, i.e. the depot building's, hue/saturation/value jitter 0) plus a ControlMask whose red channel marks the door-leaf texels (found by geometry: side-facing body triangles next to each CS1 door position, `door_leaf_triangles`). The shader multiplies the base colour by the channel colour, so the masked base texels are lightened to about 165 first, and the variation's own channel colour is the livery's dark grey again for cars that are not on a line and for the static props. A pinstripe along the lower body (0.62..0.70 m up, painted by world height: every side triangle is clipped to the slab and rasterised in UV space, so it runs the whole car on any atlas layout) and, with a brand, a short rule under each band text block take the line colour too. Door windows and brand glow (lit texels) stay unmasked. `preview_line.png` shows the mask in a sample blue.
- Operator identity: `--brand helix|vanta|nexus|okada` (`tools/brand.py`) clears the SJ crests, the "1 klass" lettering, the fleet number and the two vehicle plates (regions measured on the X40 atlas, scaled to the atlas in hand) and paints a mark, wordmark, subtitle and fleet code in the brand colour; the mark and the wordmark also go into the Emissive texture with the windows' light index, so they glow in the brand colour at night. The 8x smaller CS1 LOD atlas only gets the crests cleared and a dot of brand colour. New identities are a mark function plus a BRANDS entry.
- LOD chain: LOD0 (the CS1 mesh with bones), LOD1 (meshoptimizer `simplify` of LOD0 at a 5 % error budget, same textures and vertex colours, no bones, like the KISS LOD1), LOD2 (the CS1 LOD mesh with its own small textures). Only LOD0 carries `LodProperties`.
- The builder lights CS1 illumination 16–40 (the passenger windows the author lit) and the glass the author left dark (illumination 0: door and end windows) as warm interior light; dark glass on the nose quarter of the car (windscreen, cab side windows) becomes a separate cooler, dimmer cab light. `--calibrate` builds carriage types A/B/C at 0.4/0.2/0.1 of the interior intensity to find the right level in one look. Lamp lenses (illumination > 200) are split: the top 60 % of the lens texels white (purpose 3), the bottom 40 % red (purpose 6), oriented from the mesh. Roof lamps are white only: their triangles get duplicated vertices whose UVs point at a copy of the lens texels pasted into free atlas space (`lamp_layout` / `relocate_triangles`).
- Doors = `ActivityLocation` entries with activity `55cd3132…`; headlights = `EffectSource` entries with effect `de32f8ad…`; every car also carries effect `76d8c521…` at (0, 4, 0).
- The KISS pack puts a tiny `kisscaltrain07` "car" (gangway bellows, no seats) between every real car.
- The pack also carries components from the **Better Transit Selector** code mod (`BTS_TrainPack`, `BTS_TrainVariant`). Strip those when using it as a template.
