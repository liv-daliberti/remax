"""Pure finite-budget metrics and explicitly scoped outcome equivalences."""
from collections import Counter
import ast
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import tempfile


def sha(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def file_sha(p):
    d=hashlib.sha256()
    with Path(p).open('rb') as h:
        for b in iter(lambda:h.read(1<<20),b''):d.update(b)
    return d.hexdigest()


def atomic_new(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=p.parent,prefix='.'+p.name)
    try:
        with os.fdopen(fd,'w') as h:json.dump(x,h,sort_keys=True,indent=2,allow_nan=False);h.write('\n');h.flush();os.fsync(h.fileno())
        os.link(tmp,p)
    finally:os.unlink(tmp)


def read_lines(p):
    return [json.loads(l) for l in Path(p).open() if l.strip()]


def counts(keys):
    return Counter(k for k in keys if k is not None)


def miss(n,c,k):
    if any(type(v)is not int for v in (n,c,k)) or not 0<=c<=n or not 0<=k<=n:raise ValueError('invalid finite sample counts')
    return math.comb(n-c,k)/math.comb(n,k)


def portfolio(pools,budgets):
    if len(pools)!=len(budgets) or not pools:raise ValueError('invalid portfolio')
    cs=[counts(p) for p in pools];allkeys=set().union(*cs)
    d=sum(1-math.prod(miss(len(p),c.get(key,0),k) for p,c,k in zip(pools,cs,budgets)) for key in allkeys)
    p=1-math.prod(miss(len(pool),sum(c.values()),k) for pool,c,k in zip(pools,cs,budgets))
    return {'pass8':p,'distinct8':d,'extra8':d-p,'correct':sum(k*sum(c.values())/len(pool) if pool else 0 for pool,c,k in zip(pools,cs,budgets))}


def collision(keys):
    c=counts(keys);n=sum(c.values())
    return sum(v*(v-1) for v in c.values())/(n*(n-1)) if n>=2 else None


def cross_metrics(a,b):
    ca,cb=counts(a),counts(b);ra,rb=sum(ca.values()),sum(cb.values())
    cab=sum(v*cb.get(k,0) for k,v in ca.items())/(ra*rb) if ra and rb else None
    aa,bb=collision(a),collision(b)
    return {'cross_collision':cab,'collision_a':aa,'collision_b':bb,
            'squared_distance':aa+bb-2*cab if aa is not None and bb is not None else None,
            'shared':len(set(ca)&set(cb)), 'jaccard':len(set(ca)&set(cb))/len(set(ca)|set(cb)) if ca or cb else None,
            'correct_a':ra,'correct_b':rb}


def coarse_key(key,spec):
    if key is None:return None
    if not isinstance(key,str):raise ValueError('canonical key must be a string')
    v=spec['verifier']
    if v=='graph_coloring':
        if not key.startswith('graph_coloring:'):raise ValueError('wrong graph key')
        raw=key.split(':',1)[1]
        if not raw.isdigit() or len(raw)!=spec['n']:raise ValueError('invalid graph vector')
        values=tuple(map(int,raw));anchors={int(c) for c in spec.get('partial_colors',[]) if c is not None}
        colors=(1,2,3)
        if any(c not in colors for c in values):raise ValueError('unsupported graph colors')
        candidates=[]
        for perm in itertools.permutations(colors):
            mapping=dict(zip(colors,perm))
            if any(mapping[c]!=c for c in anchors):continue
            candidates.append(''.join(str(mapping[c]) for c in values))
        return 'graph_color_orbit:'+min(candidates)
    if v=='python_factor_function':
        if not key.startswith('python_factor:'):raise ValueError('wrong python key')
        ds=[int(x) for x in key.split(':',1)[1].split(',')];ns=spec['cases']
        if len(ds)!=len(ns) or any(type(n)is not int or not 1<d<n or n%d for n,d in zip(ns,ds)):raise ValueError('invalid factor vector')
        return 'python_factor_pairs:'+','.join(f'{min(d,n//d)}*{max(d,n//d)}' for n,d in zip(ns,ds))
    if v=='countdown':
        if not key.startswith('countdown:'):raise ValueError('wrong countdown key')
        def parse(node):
            if isinstance(node,ast.Constant) and type(node.value)is int:return ('num',node.value)
            if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name) or node.keywords:raise ValueError('invalid canonical AST')
            name=node.func.id
            if name not in ('add','mul','sub','div','neg') or len(node.args)!=(1 if name=='neg' else 2):raise ValueError('invalid operator')
            args=[parse(n) for n in node.args]
            if name in ('add','mul'):
                parts=[]
                for a in args:parts.extend(a[1:]) if a[0]==name else parts.append(a)
                return (name,*sorted(parts,key=repr))
            return (name,*args)
        return 'countdown_ac:'+json.dumps(parse(ast.parse(key.split(':',1)[1],mode='eval').body),separators=(',',':'))
    if v in ('pantry_plan','mathir_algebra','mathir_action_menu') or v.startswith('mathir'):return key
    raise ValueError('unsupported verifier '+v)
