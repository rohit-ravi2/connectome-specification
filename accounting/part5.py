import csv, numpy as np
from collections import deque
R=list(csv.DictReader(open("data/processed/dev_stabilization_program/master_directed_chemical_pairs.csv")))
neur=sorted(set([r['src'] for r in R]+[r['tgt'] for r in R])); nix={n:i for i,n in enumerate(neur)}; n=len(neur)
# adult undirected graph (union of a7/a8 chemical)
edges=[(r['src'],r['tgt']) for r in R if r['a7']=='1' or r['a8']=='1']
core=[(r['src'],r['tgt']) for r in R if r['a7']=='1' and r['a8']=='1']
halo=[(r['src'],r['tgt']) for r in R if (r['a7']=='1')!=(r['a8']=='1')]
# match core vs halo on contact-area bin (log_area) and endpoint degree
la={(r['src'],r['tgt']):float(r['log_area']) for r in R}
def gleff(excl):
    adj=[[] for _ in range(n)]
    ex=set(excl)
    for s,t in edges:
        if (s,t) in ex: continue
        adj[nix[s]].append(nix[t]); adj[nix[t]].append(nix[s])
    tot=0.0
    for s in range(n):
        d=[-1]*n; d[s]=0; q=deque([s])
        while q:
            u=q.popleft()
            for v in adj[u]:
                if d[v]<0: d[v]=d[u]+1; q.append(v)
        tot+=sum(1/d[t] for t in range(n) if d[t]>0)
    return tot/(n*(n-1))
base=gleff([])
rng=np.random.default_rng(0)
# contact-area-matched sampling: bin core & halo by log_area quartile, sample equal counts per bin
def matched(pool_a,pool_b,k):
    qa=np.quantile([la[e] for e in pool_a],[.25,.5,.75])
    def binz(e): return int(np.digitize(la[e],qa))
    from collections import defaultdict
    ba=defaultdict(list); bb=defaultdict(list)
    for e in pool_a: ba[binz(e)].append(e)
    for e in pool_b: bb[binz(e)].append(e)
    sa,sb=[],[]
    for b in range(4):
        m=min(len(ba[b]),len(bb[b]),k//4)
        if m>0:
            sa+=[ba[b][i] for i in rng.choice(len(ba[b]),m,replace=False)]
            sb+=[bb[b][i] for i in rng.choice(len(bb[b]),m,replace=False)]
    return sa,sb
sc,sh=matched(core,halo,240)
dc=base-gleff(sc); dh=base-gleff(sh)
print("PART 5 — functional matched knockout (contact-area-matched core vs halo)")
print(f"  baseline global efficiency={base:.4f}  (matched n={len(sc)} each)")
print(f"  remove core (matched):  efficiency drop={dc:.5f}")
print(f"  remove halo (matched):  efficiency drop={dh:.5f}")
print(f"  -> core is {'MORE' if dc>dh*1.1 else 'NOT more'} load-bearing after contact-area matching (ratio {dc/max(dh,1e-9):.2f}x)")
