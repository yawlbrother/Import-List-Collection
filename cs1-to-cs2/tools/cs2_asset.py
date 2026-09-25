#!/usr/bin/env python3
"""Writers for the small CS2 asset files: .Prefab (JSON-ish), .Surface, .loc and .cid.

Formats (all verified against assets made by the game's own importer):
  .cid      32 lowercase hex characters, no newline. Every asset file has one.
  .Surface  u8 1, u8 0, u8 template (1 = standard), 5 x (u32 0, u8 0xFF),
            u32 texture count, per texture: u8 len, name, u8 0xFF, 16-byte id,
            u8 0xFF, u32 keyword count, per keyword: u8 len, name.
            Texture ids are the .cid hex with the two nibbles of every byte swapped.
  .loc      u16 1, str "English", str "en-US", str "English", u32 count,
            (str key, str value) * count, u32 0. Strings are 7-bit-length-prefixed UTF-8.
  .Prefab   JSON with typed objects: reference objects get "$id" (one counter), every
            object gets "$type" written "N|Full.Type, Assembly" on first use and N afterwards
            (a second counter). Asset references are bare tokens: $fstrref:"CID:<cid>".
"""
import json, os, secrets, struct

def new_id():
    return secrets.token_hex(16)

def write_cid(path, cid):
    with open(path + '.cid', 'w', newline='') as f:
        f.write(cid)

def cid_bytes(cid):
    return bytes(int(cid[i + 1] + cid[i], 16) for i in range(0, 32, 2))

def _str7(s):
    b = s.encode('utf-8'); n = len(b); out = bytearray()
    while True:
        c = n & 0x7F; n >>= 7
        out.append(c | (0x80 if n else 0))
        if not n: break
    return bytes(out) + b

# ---------------------------------------------------------------- .Surface
VT_STACKS = (('_BaseColorMap', '_NormalMap', '_MaskMap', '_ControlMask'), ('_EmissiveColorMap',))
VT_TILE = 512   # the game streams textures in 512 px tiles; a stack smaller than a tile on either side aborts start-up

def write_surface(path, textures, template=1, keywords=('_TANGENTSPACE_OCTO',), vt=None):
    """textures: list of (slot name, texture cid) e.g. ('_BaseColorMap', 'ab12...').
    vt: (width, height) or None. None (the default, and the only layout known to render) writes a plain
    surface: the game binds every listed .Texture straight to the material (ManagedBatchSystem.CreateMaterial
    sets each one that is not "handled by virtual texturing").
    With a size the surface gets the virtual-texture block the game's own asset importer writes
    (AssetImportPipeline.ProcessSurfacesForVT): two stacks (1 = BaseColor, Normal, MaskMap, ControlMask;
    2 = Emissive), each u32 width, u32 height and eight 16-byte ids: slots 0-3 the source textures, slots
    4-7 the pre-baked StreamingData~/VT/*.VTTexture tiles for the same maps; after the stacks a nullable
    reference (ff + 16 bytes, or 00) to the surface's *.VTSurface page table. Textures in a stack are then
    streamed from those files instead of being bound, so a surface written with the block but without the
    baked tiles renders nothing (SJX40 v21: visible only at the LOD2 distance, whose surfaces were plain).
    This tool cannot bake the tiles, so the block is experimental: it is written with the source ids only
    and no VTSurface reference, like the vanilla bin prop. A stack smaller than VT_TILE on either side
    aborts the game at start-up ("All sizes need to be bigger than the tileSize!"), so such a surface is
    always written plain. Hash128 ids are stored with the nibbles of each byte swapped relative to the
    .cid text (cid_bytes does that)."""
    if vt and min(vt) < VT_TILE:
        vt = None
    by_slot = dict(textures)
    b = bytearray(b'\x01\x00' + bytes([template]) + b'\x00\x00\x00')     # version, template, 3 zero bytes
    if vt:                                                                # nullable VT block: 01 = present
        w, h = vt; b += b'\x01' + struct.pack('<I', len(VT_STACKS))
        for stack in VT_STACKS:
            b += struct.pack('<II', w, h)
            for k in range(8):
                b += cid_bytes(by_slot[stack[k]]) if k < len(stack) and stack[k] in by_slot else b'\x00' * 16
        b += b'\x00'                                                      # no VTSurface page table
    else:
        b += b'\x00'
    b += b'\xff\x00\x00\x00\x00' * 4 + b'\xff'                              # four empty property lists
    b += struct.pack('<I', len(textures))
    for name, cid in sorted(textures):
        b += bytes([len(name)]) + name.encode() + b'\xff' + cid_bytes(cid)
    b += b'\xff' + struct.pack('<I', len(keywords))
    for k in keywords:
        b += bytes([len(k)]) + k.encode()
    with open(path, 'wb') as f:
        f.write(bytes(b))

# ---------------------------------------------------------------- .loc
def write_loc(path, entries):
    b = bytearray(struct.pack('<H', 1)) + _str7('English') + _str7('en-US') + _str7('English')
    b += struct.pack('<I', len(entries))
    for k, v in entries.items():
        b += _str7(k) + _str7(v)
    b += struct.pack('<I', 0)
    with open(path, 'wb') as f:
        f.write(bytes(b))

# ---------------------------------------------------------------- .Prefab
class Obj:
    """A typed object. ref=True objects get an $id (classes, lists, arrays)."""
    def __init__(self, type_name, fields=None, ref=True):
        self.type_name, self.fields, self.ref = type_name, dict(fields or {}), ref

class Arr:
    """A typed list/array written as $rlength/$rcontent."""
    def __init__(self, type_name, items):
        self.type_name, self.items = type_name, list(items)

class Ref:
    """An asset reference: CID:<hex> or UnityGUID:<hex>."""
    def __init__(self, key):
        self.key = key

class Bare:
    """A value type written as bare numbers (UnityEngine.Color)."""
    def __init__(self, type_name, values):
        self.type_name, self.values = type_name, list(values)

def float3(x, y, z):
    return Obj('Unity.Mathematics.float3, Unity.Mathematics', dict(x=x, y=y, z=z), ref=False)

def quat_identity():
    return Obj('Unity.Mathematics.quaternion, Unity.Mathematics',
               dict(value=Obj('Unity.Mathematics.float4, Unity.Mathematics', dict(x=0, y=0, z=0, w=1), ref=False)), ref=False)

def _num(v):
    if isinstance(v, bool): return 'true' if v else 'false'
    if isinstance(v, int): return str(v)
    if isinstance(v, float):
        if abs(v) <= 1e-6: v = 0.0
        return repr(int(v)) if v.is_integer() else repr(v).replace('e', 'E')   # C# writes 1E-06
    return json.dumps(v, ensure_ascii=False)

def dumps(root):
    ids, types, out = [0], {}, []
    def type_tag(name):
        if name in types:
            return str(types[name])
        types[name] = len(types)
        return json.dumps(f'{types[name]}|{name}')
    def emit(v, ind):
        pad, pad1 = '    ' * ind, '    ' * (ind + 1)
        if isinstance(v, Ref):
            return f'$fstrref:{json.dumps(v.key)}'
        if v is None:
            return 'null'
        if isinstance(v, Obj):
            lines = []
            if v.ref:
                lines.append(f'"$id": {ids[0]}'); ids[0] += 1
            lines.append(f'"$type": {type_tag(v.type_name)}')
            for k, x in v.fields.items():
                lines.append(f'{json.dumps(k)}: {emit(x, ind + 1)}')
            return '{\n' + ',\n'.join(pad1 + l for l in lines) + '\n' + pad + '}'
        if isinstance(v, Arr):
            my_id = ids[0]; ids[0] += 1
            head = [f'"$id": {my_id}', f'"$type": {type_tag(v.type_name)}', f'"$rlength": {len(v.items)}']
            items = [('    ' * (ind + 2)) + emit(x, ind + 2) for x in v.items]
            body = ',\n'.join(pad1 + l for l in head) + ',\n' + pad1 + '"$rcontent": [\n'
            body += (',\n'.join(items) + '\n' if items else '\n') + pad1 + ']'
            return '{\n' + body + '\n' + pad + '}'
        if isinstance(v, Bare):
            lines = [f'"$type": {type_tag(v.type_name)}'] + [_num(x) for x in v.values]
            return '{\n' + ',\n'.join(pad1 + l for l in lines) + '\n' + pad + '}'
        return _num(v)
    return emit(root, 0)

def write_prefab(path, root):
    with open(path, 'w', encoding='utf-8', newline='\r\n') as f:
        f.write(dumps(root))

# ---------------------------------------------------------------- common prefabs
COMPONENT_LIST = 'System.Collections.Generic.List`1[[Game.Prefabs.ComponentBase, Game]], mscorlib'
SURFACE_REF_ARRAY = ('Colossal.IO.AssetDatabase.AssetReference`1[[Colossal.IO.AssetDatabase.SurfaceAsset, '
                     'Colossal.IO.AssetDatabase, Version=0.0.0.0, Culture=neutral, PublicKeyToken=null]][], '
                     'Colossal.IO.AssetDatabase')

def render_prefab(name, geometry_cid, surface_cids, bounds_min, bounds_max, surface_area,
                  index_count, vertex_count, lod_cids=(), components=()):
    comps = list(components)
    if lod_cids:
        comps.append(Obj('Game.Prefabs.LodProperties, Game', dict(
            name='LodProperties', active=True, m_Bias=0, m_ShadowBias=0,
            m_LodMeshes=Arr('Game.Prefabs.RenderPrefab[], Game', [Ref('CID:' + c) for c in lod_cids]))))
    return Obj('Game.Prefabs.RenderPrefab, Game', dict(
        name=name, active=True, version=1, m_prefabFormat=0,
        components=Arr(COMPONENT_LIST, comps),
        m_GeometryAsset=Ref('CID:' + geometry_cid),
        m_SurfaceAssets=Arr(SURFACE_REF_ARRAY, [Ref('CID:' + c) for c in surface_cids]),
        m_Bounds=Obj('Colossal.Mathematics.Bounds3, Colossal.Mathematics',
                     dict(min=float3(*(round(float(x), 6) for x in bounds_min)),
                          max=float3(*(round(float(x), 6) for x in bounds_max))), ref=False),
        m_SurfaceArea=round(float(surface_area), 4), m_IndexCount=int(index_count), m_VertexCount=int(vertex_count),
        m_MeshCount=len(surface_cids), m_IsImpostor=False, m_ManualVTRequired=False))

def static_object_prefab(name, mesh_cid, ui_group_guid, icon_cid, cost=1000):
    return Obj('Game.Prefabs.StaticObjectPrefab, Game', dict(
        name=name, active=True, version=1, m_prefabFormat=0,
        components=Arr(COMPONENT_LIST, [
            Obj('Game.Prefabs.PlaceableObject, Game', dict(name='PlaceableObject', active=True,
                                                           m_ConstructionCost=cost, m_XPReward=0)),
            Obj('Game.Prefabs.UIObject, Game', dict(name='UIObject', active=True,
                                                    m_Group=Ref('UnityGUID:' + ui_group_guid), m_Priority=0,
                                                    m_Icon=f'assetdb://global/{icon_cid}', m_IsDebugObject=False)),
        ]),
        m_Meshes=Arr('Game.Prefabs.ObjectMeshInfo[], Game', [
            Obj('Game.Prefabs.ObjectMeshInfo, Game', dict(m_Mesh=Ref('CID:' + mesh_cid), m_Position=float3(0, 0, 0),
                                                           m_Rotation=quat_identity(), m_RequireState=0))]),
        m_Circular=False))

# ---------------------------------------------------------------- vehicles
def float2(x, y):
    return Obj('Unity.Mathematics.float2, Unity.Mathematics', dict(x=x, y=y), ref=False)

def quat(x, y, z, w):
    return Obj('Unity.Mathematics.quaternion, Unity.Mathematics',
               dict(value=Obj('Unity.Mathematics.float4, Unity.Mathematics', dict(x=x, y=y, z=z, w=w), ref=False)), ref=False)

def vec3(x, y, z):
    return Bare('UnityEngine.Vector3, UnityEngine.CoreModule', [float(x), float(y), float(z)])

def bone(name, world_pos, parent=-1, bone_type=0, parent_world=(0, 0, 0)):
    """A ProceduralAnimationProperties bone at rest. Positions in metres, identity rotation,
    unit scale; bindPose is the inverse of the bone's world transform (a translation by -pos)."""
    wx, wy, wz = (round(float(v), 6) for v in world_pos)
    px, py, pz = (round(float(v), 6) for v in parent_world)
    m = dict(m00=0.0, m10=0.0, m20=0.0, m30=0.0, m01=0.0, m11=0.0, m21=0.0, m31=0.0,
             m02=0.0, m12=0.0, m22=0.0, m32=0.0, m03=-wx, m13=-wy, m23=-wz, m33=1)
    m['m00'] = m['m11'] = m['m22'] = 1
    return Obj('Game.Prefabs.ProceduralAnimationProperties+BoneInfo, Game', dict(
        name=name, position=vec3(round(wx - px, 6), round(wy - py, 6), round(wz - pz, 6)),
        rotation=Bare('UnityEngine.Quaternion, UnityEngine.CoreModule', [0, 0, 0, 1]),
        scale=vec3(1, 1, 1),
        bindPose=Obj('UnityEngine.Matrix4x4, UnityEngine.CoreModule', m, ref=False),
        parentId=parent, m_Type=bone_type, m_Speed=0, m_Acceleration=0, m_ConnectionID=0, m_SourceID=0))

def procedural_animation(bones):
    return Obj('Game.Prefabs.ProceduralAnimationProperties, Game', dict(
        name='ProceduralAnimationProperties', active=True,
        m_Bones=Arr('Game.Prefabs.ProceduralAnimationProperties+BoneInfo[], Game', bones), m_Animations=None))

def public_transport(capacity, transport_type=1):
    return Obj('Game.Prefabs.PublicTransport, Game', dict(
        name='PublicTransport', active=True, m_TransportType=transport_type,
        m_PassengerCapacity=int(capacity), m_Purposes=1, m_MaintenanceRange=1000))

def vehicle_side_effects():
    return Obj('Game.Prefabs.VehicleSideEffects, Game', dict(
        name='VehicleSideEffects', active=True, m_RoadWear=float2(1, 2), m_NoisePollution=float2(0, 10),
        m_AirPollution=float2(0, 0)))

def ui_object(icon_cid, group_guid=None, priority=0):
    return Obj('Game.Prefabs.UIObject, Game', dict(
        name='UIObject', active=True, m_Group=Ref('UnityGUID:' + group_guid) if group_guid else None,
        m_Priority=priority, m_Icon=f'assetdb://global/{icon_cid}', m_IsDebugObject=False))

def activity_location(activity_guid, positions):
    """positions: list of (x, y, z); doors face outward (+x or -x)."""
    locs = []
    for x, y, z in positions:
        rot = quat(0, 0.707106769, 0, 0.707106769) if x < 0 else quat(0, -0.707106769, 0, 0.707106769)
        locs.append(Obj('Game.Prefabs.ActivityLocation+LocationInfo, Game', dict(
            m_Activity=Ref('UnityGUID:' + activity_guid), m_Position=float3(*(round(float(v), 6) for v in (x, y, z))), m_Rotation=rot)))
    return Obj('Game.Prefabs.ActivityLocation, Game', dict(
        name='ActivityLocation', active=True,
        m_Locations=Arr('Game.Prefabs.ActivityLocation+LocationInfo[], Game', locs),
        m_InvertWhen=0, m_AnimatedPropName='', m_RequireAuthorization=False))

def effect_source(effects):
    """effects: list of (effect guid, (x, y, z), (qx, qy, qz, qw))."""
    items = [Obj('Game.Prefabs.EffectSource+EffectSettings, Game', dict(
        m_Effect=Ref('UnityGUID:' + guid), m_PositionOffset=float3(*(round(float(v), 6) for v in pos)), m_Rotation=quat(*rot),
        m_Scale=float3(1, 1, 1), m_Intensity=1, m_ParentMesh=0, m_AnimationIndex=-1)) for guid, pos, rot in effects]
    return Obj('Game.Prefabs.EffectSource, Game', dict(
        name='EffectSource', active=True,
        m_Effects=Arr('System.Collections.Generic.List`1[[Game.Prefabs.EffectSource+EffectSettings, Game]], mscorlib', items),
        m_AnimationCurves=Arr('System.Collections.Generic.List`1[[Game.Prefabs.EffectSource+AnimationProperties, Game]], mscorlib', [])))

def _mesh_list(mesh_cids):
    """One ObjectMeshInfo per render prefab cid (a str or a list): sub-meshes of one object."""
    cids = [mesh_cids] if isinstance(mesh_cids, str) else list(mesh_cids)
    return Arr('Game.Prefabs.ObjectMeshInfo[], Game', [
        Obj('Game.Prefabs.ObjectMeshInfo, Game', dict(m_Mesh=Ref('CID:' + c), m_Position=float3(0, 0, 0),
                                                       m_Rotation=quat_identity(), m_RequireState=0)) for c in cids])

TRAIN_DEFAULTS = dict(m_Circular=False, m_TrackType=1, m_EnergyType=2, m_MaxSpeed=200, m_Acceleration=5,
                      m_Braking=10)

def _train_fields(speed):
    f = dict(TRAIN_DEFAULTS); f['m_MaxSpeed'] = speed
    f.update(m_Turning=float2(90, 10), m_BogieOffset=float2(0, 0), m_AttachOffset=float2(0, 0))
    return f

def train_car_prefab(name, mesh_cid, components, speed=200):
    return Obj('Game.Prefabs.MultipleUnitTrainCarPrefab, Game', dict(
        name=name, active=True, version=1, m_prefabFormat=0,
        components=Arr(COMPONENT_LIST, components), m_Meshes=_mesh_list(mesh_cid), **_train_fields(speed)))

def train_front_prefab(name, mesh_cid, components, carriages, speed=200, reversed_end=True, units=(1, 1)):
    """carriages: list of (cid, direction, min count, max count); the game picks a count in each range
    (VehicleCarriageElement.m_Count is an int2) and couples `units` (min, max) whole units nose to tail."""
    cars = [Obj('Game.Prefabs.MultipleUnitTrainCarriageInfo, Game', dict(
        m_Carriage=Ref('CID:' + c), m_Direction=d, m_MinCount=lo, m_MaxCount=hi)) for c, d, lo, hi in carriages]
    return Obj('Game.Prefabs.MultipleUnitTrainFrontPrefab, Game', dict(
        name=name, active=True, version=1, m_prefabFormat=0,
        components=Arr(COMPONENT_LIST, components), m_Meshes=_mesh_list(mesh_cid), **_train_fields(speed),
        m_MinMultipleUnitCount=units[0], m_MaxMultipleUnitCount=units[1],
        m_Carriages=Arr('Game.Prefabs.MultipleUnitTrainCarriageInfo[], Game', cars),
        m_AddReversedEndCarriage=reversed_end))

COLOR = 'UnityEngine.Color, UnityEngine.CoreModule'

def int3(x, y, z):
    return Obj('Unity.Mathematics.int3, Unity.Mathematics', dict(x=int(x), y=int(y), z=int(z)), ref=False)

def color_properties(colors, external=(True, False, False), source=0):
    """One colour variation for a render prefab: `colors` are the three channel colours (RGBA 0..1)
    multiplied into the texels the ControlMask marks (R = channel 0, G = 1, B = 2). A channel with
    external=True is overwritten from the colour source instead. Game.Rendering.ColorSourceType:
    0 Brand = the entity's brand, else its CurrentRoute's colour (a transport vehicle's line colour),
    else the default brand; 1 Parent = a copy of the owner's mesh colours (a building's sub-objects; for
    a vehicle that is its depot, so never use it for line colour). Variation ranges are 0: no jitter."""
    L = 'System.Collections.Generic.List`1[[Game.Prefabs.ColorProperties+{0}, Game]], mscorlib'
    bind = Arr(L.format('ColorChannelBinding'), [
        Obj('Game.Prefabs.ColorProperties+ColorChannelBinding, Game', dict(m_ChannelId=k, m_CanBeModifiedByExternal=bool(e)))
        for k, e in enumerate(external)])
    sets = Arr(L.format('VariationSet'), [
        Obj('Game.Prefabs.ColorProperties+VariationSet, Game', dict(
            m_Colors=Arr('UnityEngine.Color[], UnityEngine.CoreModule', [Bare(COLOR, list(c)) for c in colors]),
            m_VariationGroup=''))])
    return Obj('Game.Prefabs.ColorProperties, Game', dict(
        name='ColorProperties', active=True, m_ColorVariations=sets, m_ChannelsBinding=bind,
        m_VariationGroups=Arr(L.format('VariationGroup'), []), m_VariationRanges=int3(0, 0, 0),
        m_AlphaRanges=int3(0, 0, 0), m_ExternalColorSource=int(source)))

def emissive_properties(multi):
    """multi: list of (purpose, color, colorOff, intensity, luminance, layerId)."""
    items = [Obj('Game.Prefabs.EmissiveProperties+MultiLightMapping, Game', dict(
        previewState=False, purpose=p, color=Bare(COLOR, list(c)), colorOff=Bare(COLOR, list(off)),
        intensity=i, luminance=l, responseTime=0, animationIndex=-1, layerId=layer)) for p, c, off, i, l, layer in multi]
    return Obj('Game.Prefabs.EmissiveProperties, Game', dict(
        name='EmissiveProperties', active=True, m_SingleLights=None,
        m_MultiLights=Arr('System.Collections.Generic.List`1[[Game.Prefabs.EmissiveProperties+MultiLightMapping, Game]], mscorlib', items),
        m_AnimationCurves=None, m_SignalGroupAnimations=None))
