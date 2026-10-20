import csv, math, re, numpy as np
from collections import defaultdict
D="../data/celegans/data/connectome"
R=list(csv.DictReader(open("data/processed/specification_accounting/master_edge_table.csv")))
def h(p): return 0.0 if p<=0 or p>=1 else -p*math.log2(p)-(1-p)*math.log2(1-p)
print("\n=== Phase 2: MASTER TABLE AUDIT ===")
# chemical vs electrical separation (rebuild by type)
presC=defaultdict(set); presE=defaultdict(set); synC=defaultdict(lambda:defaultdict(int))
with open(f"{D}/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        ds=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1)); pr=(min(r['pre'],r['post']),max(r['pre'],r['post']))
        if r['type']=='electrical': presE[pr].add(ds)
        else: presC[pr].add(ds); synC[pr][ds]+=int(r['synapses'])
repro_all=[(r['src'],r['tgt']) for r in R if r['reproducible']=='1']
def both(pr,pres): return 7 in pres.get(pr,set()) and 8 in pres.get(pr,set())
rc=sum(1 for pr in repro_all if both(pr,presC)); re_=sum(1 for pr in repro_all if both(pr,presE))
mixed=sum(1 for pr in repro_all if both(pr,presC) and both(pr,presE))
print(f"  reproducible edges (mixed defn used in Round2): {len(repro_all)}")
print(f"    of these, reproducible-CHEMICAL: {rc}  reproducible-ELECTRICAL(gap-jn): {re_}  both: {mixed}")
print(f"  *** FLAW: strong-core MIXES chemical + gap junctions; {re_} gap-junction pairs in 'strong-core' ***")
# threshold sensitivity (chemical, synmin)
for thr in (1,2,4):
    n=sum(1 for pr in presC if min(synC[pr].get(7,0),synC[pr].get(8,0))>=thr)
    print(f"  reproducible chemical synmin>={thr}: {n}")
# strong-core/halo exclusivity + self pairs
rep=sum(int(r['reproducible']) for r in R); halo=sum(int(r['halo']) for r in R)
overlap=sum(1 for r in R if r['reproducible']=='1' and r['halo']=='1')
print(f"  strong-core/halo overlap (should be 0): {overlap}  | self-pairs: {sum(1 for r in R if r['src']==r['tgt'])}")
print(f"  directedness: table is UNDIRECTED (pre/post collapsed) — chemical directionality LOST (flaw for source-NT/directed claims)")

print("\n=== Phase 3: NOISE FLOOR AUDIT (variants + bootstrap CI) ===")
def floor_cond(rows,ya,yb):
    a=np.array([int(r[ya]) for r in rows]); b=np.array([int(r[yb]) for r in rows]); fl=0.0
    for v in(0,1):
        m=a==v
        if m.sum(): fl+=m.mean()*h(b[m].mean())
    return fl
allp=R; cont=[r for r in R if r['contact']=='1']
def boot(rows,ya,yb,n=200):
    idx=np.arange(len(rows)); vals=[]
    rng=np.random.default_rng(0)
    for _ in range(n):
        s=rng.choice(idx,len(idx),replace=True); rs=[rows[i] for i in s]; vals.append(floor_cond(rs,ya,yb))
    return np.percentile(vals,[2.5,97.5])
print(f"  {'variant':38s} {'entropy':>8} {'95% CI':>18}")
for nm,rows in [("H(ad8|ad7) all pairs",allp),("H(ad8|ad7) contacting",cont)]:
    f1=floor_cond(rows,'ad7','ad8'); f2=floor_cond(rows,'ad8','ad7'); ci=boot(rows,'ad7','ad8')
    print(f"  {nm:38s} {f1:8.4f} [{ci[0]:.4f},{ci[1]:.4f}]  (asym: {f1:.3f}/{f2:.3f}, sym {(f1+f2)/2:.3f})")
# all-pairs triviality check: entropy among contact NON-edges
nonc=[r for r in R if r['contact']=='0']
a8n=np.array([int(r['ad8']) for r in nonc])
print(f"  *** all-pairs floor is LOW because {len(nonc)} non-contacting pairs are ~always non-edges "
      f"(ad8 rate={a8n.mean():.3f}, H={h(a8n.mean()):.3f}) — the all-pairs 'closure' is largely contact FILTERING ***")
