#!/usr/bin/env python3
"""Read Cities: Skylines 1 asset packages (.crp).

    python crp_tool.py list    ASSET.crp          # every entry in the package
    python crp_tool.py info    ASSET.crp          # metadata + vehicle stats
    python crp_tool.py extract ASSET.crp OUTDIR   # meshes -> OBJ, textures -> PNG

Needs: pip install pillow
"""
import io, os, re, struct, sys

# ---------------------------------------------------------------- reading
class Reader:
    def __init__(self, buf, pos=0):
        self.b, self.p = buf, pos
    def u8(self):  v = self.b[self.p]; self.p += 1; return v
    def u16(self): v = struct.unpack_from('<H', self.b, self.p)[0]; self.p += 2; return v
    def i32(self): v = struct.unpack_from('<i', self.b, self.p)[0]; self.p += 4; return v
    def u32(self): v = struct.unpack_from('<I', self.b, self.p)[0]; self.p += 4; return v
    def i64(self): v = struct.unpack_from('<q', self.b, self.p)[0]; self.p += 8; return v
    def f32(self): v = struct.unpack_from('<f', self.b, self.p)[0]; self.p += 4; return v
    def s(self):   # .NET BinaryWriter string: 7-bit length prefix + UTF-8
        n = sh = 0
        while True:
            c = self.u8(); n |= (c & 0x7f) << sh; sh += 7
            if not c & 0x80: break
        v = self.b[self.p:self.p + n].decode('utf-8', 'replace'); self.p += n; return v
    def array(self, fmt):
        n = self.i32(); sz = struct.calcsize(fmt)
        v = [struct.unpack_from('<' + fmt, self.b, self.p + i * sz) for i in range(n)]
        self.p += n * sz; return v

# Entry type ids seen in the wild (ColossalFramework Package.AssetType + game user types)
TYPES = {1: 'GameObject', 2: 'Material', 3: 'Texture', 4: 'Mesh', 50: 'Data', 80: 'Locale',
         103: 'CustomAssetMetaData'}

def parse(path):
    b = open(path, 'rb').read()
    if b[:4] != b'CRAP':
        raise SystemExit(f'{path}: not a CS1 .crp package')
    r = Reader(b, 4)
    hdr = dict(format=r.u16(), package=r.s(), author=r.s(), pkg_version=r.u32(), main_asset=r.s())
    n, start = r.i32(), r.i64()
    entries = []
    for _ in range(n):
        e = dict(name=r.s(), checksum=r.s(), type=r.u32(), off=r.i64(), size=r.i64())
        e['abs'] = start + e['off']
        entries.append(e)
    return b, hdr, entries

def entry_bytes(b, e):
    return b[e['abs']:e['abs'] + e['size']]

# Each serialized object starts with: bool isNull, string typeName, string objectName
def read_mesh(b, e):
    r = Reader(b, e['abs']); r.u8(); r.s(); name = r.s()
    m = dict(name=name, V=r.array('3f'), C=r.array('4f'), UV=r.array('2f'), N=r.array('3f'),
             T=r.array('4f'), BW=r.array('4f4i'), BP=r.array('16f'))
    m['tris'] = []
    for _ in range(r.i32()):
        k = r.i32(); m['tris'].append(struct.unpack_from('<%di' % k, b, r.p)); r.p += 4 * k
    m['ok'] = r.p == e['abs'] + e['size']
    return m

def read_texture(b, e):
    """Returns (name, linear, PIL.Image). CS1 stores rows bottom-up, so we flip."""
    from PIL import Image
    r = Reader(b, e['abs']); r.u8(); r.s(); name = r.s()
    linear = r.u8(); r.i32(); n = r.i32()
    img = Image.open(io.BytesIO(b[r.p:r.p + n]))
    return name, linear, img.transpose(Image.FLIP_TOP_BOTTOM)

SIMPLE = {'System.Single': 'f', 'System.Int32': 'i', 'System.Boolean': '?', 'System.UInt32': 'I',
          'System.Int16': 'h', 'System.Byte': 'B', 'System.UInt16': 'H', 'System.Int64': 'q', 'System.UInt64': 'Q'}

def scan_fields(d):
    """Best-effort scan of serialized component fields: (fieldName, typeName, value)."""
    out = []
    pat = rb'[\x10-\x7f]([A-Za-z][\w.+\[\]`]*), ([\w.-]+), Version=[\d.]+, Culture=neutral, PublicKeyToken=\w+'
    for m in re.finditer(pat, d):
        t = m.group(1).decode(); p = m.end()
        n = d[p] if p < len(d) else 0; name = d[p + 1:p + 1 + n]
        if not re.fullmatch(rb'[A-Za-z_]\w*', name or b'-'):
            continue
        name = name.decode(); p += 1 + n
        try:
            if t in SIMPLE: v = struct.unpack_from('<' + SIMPLE[t], d, p)[0]
            elif t == 'System.String': v = Reader(d, p).s()
            elif '+' in t and not t.endswith('[]') and 'Package+Asset' not in t: v = struct.unpack_from('<i', d, p)[0]
            else: v = None
        except struct.error:
            v = None
        out.append((name, t, v))
    return out

# ---------------------------------------------------------------- commands
def cmd_list(path):
    b, h, E = parse(path)
    print(h, f'{len(E)} entries')
    for i, e in enumerate(E):
        print(f"{i:3} {TYPES.get(e['type'], e['type']):>20} {e['size']:>9}  {e['name']}")

VEHICLE_TYPES = {1: 'Car', 2: 'Metro', 4: 'Train', 8: 'Ship', 16: 'Plane', 32: 'Bicycle', 64: 'Tram',
                 128: 'Helicopter', 4096: 'Monorail'}
INTERESTING = ('m_class', 'm_vehicleType', 'm_maxSpeed', 'm_acceleration', 'm_braking', 'm_turning',
               'm_passengerCapacity', 'm_transportInfo', 'm_useColorVariations', 'm_Thumbnail')

def cmd_info(path):
    b, h, E = parse(path)
    print(f"Package {h['package']}  main asset '{h['main_asset']}'  author '{h['author']}'")
    for e in E:
        d = entry_bytes(b, e)
        if e['type'] == 103:
            ts = re.search(rb'timeStamp.(\d{4}-\d\d-\d\dT[\d:.+-]+)', d)
            print(f"\n[{e['name']}] saved {ts.group(1).decode() if ts else '?'}")
        if e['type'] == 1 and b'VehicleInfo+VehicleType' in d and b'm_maxSpeed' in d:
            print(f"\n[{e['name']}]")
            ai = re.findall(rb'[\x10-\x7f]([A-Za-z]\w*AI), Assembly-CSharp', d)
            print('   AI:', ', '.join(sorted({a.decode() for a in ai})))
            for name, t, v in scan_fields(d):
                if name in INTERESTING:
                    if name == 'm_vehicleType': v = VEHICLE_TYPES.get(v, v)
                    if name in ('m_class', 'm_transportInfo'):
                        i = d.find(bytes([len(name)]) + name.encode()); v = Reader(d, i + 1 + len(name)).s()
                    print(f'   {name:22} {v}')
            i = d.find(b'\nm_trailers')
            if i >= 0:
                r = Reader(d, i + 11)
                for _ in range(r.i32()):
                    print(f'   trailer {r.s()}  probability {r.i32()}  invert {r.i32()}')
    print('\nMeshes:')
    for e in E:
        if e['type'] == 4:
            m = read_mesh(b, e); zs = [v[2] for v in m['V']]
            print(f"   {m['name'][:45]:45} verts {len(m['V']):6}  tris {sum(map(len, m['tris'])) // 3:6}  length {max(zs) - min(zs):.2f} m")

def cmd_extract(path, outdir):
    b, h, E = parse(path)
    os.makedirs(outdir, exist_ok=True)
    tex_names, mesh_i = {}, 0
    for i, e in enumerate(E):
        if e['type'] == 3:
            name, linear, img = read_texture(b, e)
            fn = f"{i:02d}{name or '_lod'}.png"; img.save(os.path.join(outdir, fn))
            tex_names[e['checksum']] = fn
            print('texture', fn, img.size)
    main = next((v for k, v in tex_names.items() if 'MainTex' in v), None)
    with open(os.path.join(outdir, 'materials.mtl'), 'w') as f:
        f.write('newmtl main\nKd 1 1 1\n' + (f'map_Kd {main}\n' if main else ''))
    for i, e in enumerate(E):
        if e['type'] != 4:
            continue
        m = read_mesh(b, e)
        safe = re.sub(r'[^\w.-]+', '_', m['name'])
        fn = os.path.join(outdir, f'{i:02d}_{safe}.obj')
        with open(fn, 'w') as f:
            f.write(f"# {m['name']} from {os.path.basename(path)}\n")
            f.write('# Unity left-handed -> right-handed: X mirrored, winding reversed. Meters, +Z = forward.\n')
            f.write('mtllib materials.mtl\nusemtl main\n')
            for x, y, z in m['V']: f.write('v %.5f %.5f %.5f\n' % (-x, y, z))
            for u, v in m['UV']: f.write('vt %.5f %.5f\n' % (u, v))
            for x, y, z in m['N']: f.write('vn %.5f %.5f %.5f\n' % (-x, y, z))
            for t in m['tris']:
                for k in range(0, len(t), 3):
                    a, bb, c = (x + 1 for x in t[k:k + 3])
                    f.write(f'f {a}/{a}/{a} {c}/{c}/{c} {bb}/{bb}/{bb}\n')
        print('mesh', os.path.basename(fn), 'ok' if m['ok'] else 'PARSE MISMATCH')

if __name__ == '__main__':
    if len(sys.argv) < 3 or sys.argv[1] not in ('list', 'info', 'extract'):
        raise SystemExit(__doc__)
    {'list': cmd_list, 'info': cmd_info, 'extract': cmd_extract}[sys.argv[1]](*sys.argv[2:])
