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
def write_surface(path, textures, template=1, keywords=('_TANGENTSPACE_OCTO',)):
    """textures: list of (slot name, texture cid) e.g. ('_BaseColorMap', 'ab12...')."""
    b = bytearray(b'\x01\x00' + bytes([template]))
    b += b'\x00\x00\x00\x00\xff' * 5
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
                     dict(min=float3(*map(float, bounds_min)), max=float3(*map(float, bounds_max))), ref=False),
        m_SurfaceArea=float(surface_area), m_IndexCount=int(index_count), m_VertexCount=int(vertex_count),
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
    wx, wy, wz = (float(v) for v in world_pos)
    px, py, pz = (float(v) for v in parent_world)
    m = dict(m00=0.0, m10=0.0, m20=0.0, m30=0.0, m01=0.0, m11=0.0, m21=0.0, m31=0.0,
             m02=0.0, m12=0.0, m22=0.0, m32=0.0, m03=-wx, m13=-wy, m23=-wz, m33=1)
    m['m00'] = m['m11'] = m['m22'] = 1
    return Obj('Game.Prefabs.ProceduralAnimationProperties+BoneInfo, Game', dict(
        name=name, position=vec3(wx - px, wy - py, wz - pz),
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
            m_Activity=Ref('UnityGUID:' + activity_guid), m_Position=float3(float(x), float(y), float(z)), m_Rotation=rot)))
    return Obj('Game.Prefabs.ActivityLocation, Game', dict(
        name='ActivityLocation', active=True,
        m_Locations=Arr('Game.Prefabs.ActivityLocation+LocationInfo[], Game', locs),
        m_InvertWhen=0, m_AnimatedPropName='', m_RequireAuthorization=False))

def effect_source(effects):
    """effects: list of (effect guid, (x, y, z), (qx, qy, qz, qw))."""
    items = [Obj('Game.Prefabs.EffectSource+EffectSettings, Game', dict(
        m_Effect=Ref('UnityGUID:' + guid), m_PositionOffset=float3(*map(float, pos)), m_Rotation=quat(*rot),
        m_Scale=float3(1, 1, 1), m_Intensity=1, m_ParentMesh=0, m_AnimationIndex=-1)) for guid, pos, rot in effects]
    return Obj('Game.Prefabs.EffectSource, Game', dict(
        name='EffectSource', active=True,
        m_Effects=Arr('System.Collections.Generic.List`1[[Game.Prefabs.EffectSource+EffectSettings, Game]], mscorlib', items),
        m_AnimationCurves=Arr('System.Collections.Generic.List`1[[Game.Prefabs.EffectSource+AnimationProperties, Game]], mscorlib', [])))

def _mesh_list(mesh_cid):
    return Arr('Game.Prefabs.ObjectMeshInfo[], Game', [
        Obj('Game.Prefabs.ObjectMeshInfo, Game', dict(m_Mesh=Ref('CID:' + mesh_cid), m_Position=float3(0, 0, 0),
                                                       m_Rotation=quat_identity(), m_RequireState=0))])

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

def train_front_prefab(name, mesh_cid, components, carriage_cids, speed=200, reversed_end=True):
    cars = [Obj('Game.Prefabs.MultipleUnitTrainCarriageInfo, Game', dict(
        m_Carriage=Ref('CID:' + c), m_Direction=d, m_MinCount=1, m_MaxCount=1)) for c, d in carriage_cids]
    return Obj('Game.Prefabs.MultipleUnitTrainFrontPrefab, Game', dict(
        name=name, active=True, version=1, m_prefabFormat=0,
        components=Arr(COMPONENT_LIST, components), m_Meshes=_mesh_list(mesh_cid), **_train_fields(speed),
        m_MinMultipleUnitCount=1, m_MaxMultipleUnitCount=1,
        m_Carriages=Arr('Game.Prefabs.MultipleUnitTrainCarriageInfo[], Game', cars),
        m_AddReversedEndCarriage=reversed_end))
