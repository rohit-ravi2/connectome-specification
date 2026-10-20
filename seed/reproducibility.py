"""How reproducible is the C. elegans connectome across same-stage isogenic animals?
Splits edges into reproducible CORE vs VARIABLE per stage (Witvliet 2020). Stdlib only."""
import csv, re
from collections import defaultdict

CSV = "../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv"
stage = {1:'L1',2:'L1',3:'L1',4:'L1',5:'L2',6:'L3',7:'adult',8:'adult'}

rows = []
with open(CSV) as f:
    for r in csv.DictReader(f):
        ds = int(re.search(r'witvliet_2020_(\d)', r['source_file']).group(1))
        pre, post, typ = r['pre'], r['post'], r['type']
        edge = (min(pre,post), max(pre,post), 'e') if typ=='electrical' else (pre,post,'c')
        rows.append((ds, edge, typ))

allds = sorted(set(r[0] for r in rows))
neurons = set(x for r in rows for x in r[1][:2])
print(f"total edge-records: {len(rows)}   animals(datasets): {allds}   neurons: {len(neurons)}\n")

for stg, dss in [('adult',[7,8]), ('L1',[1,2,3,4])]:
    n = len(dss)
    edge_animals = defaultdict(set)
    for ds, edge, typ in rows:
        if ds in dss:
            edge_animals[edge].add(ds)
    total = len(edge_animals)
    counts = [len(v) for v in edge_animals.values()]
    dist = {k: counts.count(k) for k in range(1, n+1)}
    core = dist[n]; variable = total - core
    print(f"=== {stg}  ({n} animals: {dss}) ===")
    print(f"  distinct edges (union): {total}")
    for k in range(1, n+1):
        print(f"    in exactly {k}/{n}: {dist[k]:5d}  ({100*dist[k]/total:4.1f}%)")
    print(f"  CORE (all {n}):   {core:5d}  ({100*core/total:4.1f}%)  <- reproducible (rules+seed)")
    print(f"  VARIABLE (<{n}):  {variable:5d}  ({100*variable/total:4.1f}%)  <- noise/plasticity")
    for t,lab in [('c','chemical'),('e','electrical')]:
        et = {e:v for e,v in edge_animals.items() if e[2]==t}
        if et:
            c = sum(1 for v in et.values() if len(v)==n)
            print(f"    {lab:10s}: {len(et):5d} edges, core {100*c/len(et):4.1f}%")
    print()
