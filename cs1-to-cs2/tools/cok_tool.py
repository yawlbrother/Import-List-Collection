#!/usr/bin/env python3
"""Inspect Cities: Skylines II asset packages (.cok = plain zip).

    python cok_tool.py tree   PACKAGE.cok      # prefab graph: train -> cars -> meshes -> files
    python cok_tool.py prefab PACKAGE.cok NAME # dump one prefab as clean JSON

CS2 .Prefab files are almost-JSON; load() patches the two non-JSON quirks
($fstrref:"..." references and objects holding bare values, e.g. Color).
"""
import json, re, sys, zipfile

def loads(text):
    text = text.lstrip('﻿')
    text = re.sub(r'\$fstrref:"([^"]*)"', lambda m: json.dumps('ref:' + m.group(1)), text)
    lines, stack = [], []
    for i, ln in enumerate(text.split('\n')):
        s = ln.strip()
        if stack and stack[-1] == '{' and re.fullmatch(r'-?[\d.eE+-]+,?|true,?|false,?', s):
            ln = ln.replace(s, f'"$v{i}": {s}')
        if s.endswith('{') or s.endswith('['): stack.append(s[-1])
        if s[:1] in '}]' and stack: stack.pop()
        lines.append(ln)
    return json.loads('\n'.join(lines))

def clean(o):
    if isinstance(o, dict):
        if '$rcontent' in o: return [clean(x) for x in o['$rcontent']]
        vals = [v for k, v in o.items() if k.startswith('$v')]
        if vals: return vals
        return {k: clean(v) for k, v in o.items() if k != '$id'}
    if isinstance(o, list): return [clean(x) for x in o]
    return o

def type_name(d):
    t = d.get('$type', '')
    return t.split('|', 1)[-1].split(',')[0].split('.')[-1] if isinstance(t, str) else str(t)

def refs(o, acc):
    if isinstance(o, dict): [refs(v, acc) for v in o.values()]
    elif isinstance(o, list): [refs(v, acc) for v in o]
    elif isinstance(o, str) and o.startswith('ref:'): acc.append(o[4:])
    return acc

class Package:
    def __init__(self, path):
        self.z = zipfile.ZipFile(path)
        self.names = self.z.namelist()
        self.cid = {self.z.read(n).decode().strip(): n[:-4] for n in self.names if n.endswith('.cid')}
    def prefab(self, name):
        return clean(loads(self.z.read(name).decode('utf-8')))

def cmd_tree(path):
    pk = Package(path); seen = set()
    def show(name, depth):
        pad = '  ' * depth
        if not name.endswith('.Prefab'):
            print(f'{pad}{name}  ({pk.z.getinfo(name).file_size} B)'); return
        d = pk.prefab(name)
        comps = [c.get('name') for c in d.get('components', [])]
        print(f"{pad}[{type_name(d)}] {d.get('name')}  {comps}")
        if name in seen: return
        seen.add(name)
        for r in dict.fromkeys(refs(d, [])):
            kind, _, key = r.partition(':')
            if kind == 'CID' and key in pk.cid: show(pk.cid[key], depth + 1)
            else: print(f'{pad}  -> external {r}')
    referenced = set()
    for n in pk.names:
        if n.endswith('.Prefab'):
            for r in refs(pk.prefab(n), []):
                if r.startswith('CID:') and r[4:] in pk.cid: referenced.add(pk.cid[r[4:]])
    for n in sorted(pk.names):
        if n.endswith('.Prefab') and n not in referenced:
            show(n, 0); print()

def cmd_prefab(path, name):
    pk = Package(path)
    match = [n for n in pk.names if n.endswith('.Prefab') and name.lower() in n.lower()]
    for n in match:
        print(f'===== {n}'); print(json.dumps(pk.prefab(n), indent=1))

if __name__ == '__main__':
    if len(sys.argv) < 3 or sys.argv[1] not in ('tree', 'prefab'):
        raise SystemExit(__doc__)
    {'tree': cmd_tree, 'prefab': cmd_prefab}[sys.argv[1]](*sys.argv[2:])
