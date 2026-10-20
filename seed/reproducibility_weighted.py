"""Reproducibility vs connection STRENGTH (adult pair 7,8). Stdlib only.
Tests whether the variable half is just weak edges (Witvliet's finding)."""
import csv, re
from collections import defaultdict

CSV = "../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv"
edge_syn = defaultdict(dict)   # edge -> {ds: synapses}
with open(CSV) as f:
    for r in csv.DictReader(f):
        ds = int(re.search(r'witvliet_2020_(\d)', r['source_file']).group(1))
        if ds not in (7, 8): continue
        pre, post, typ = r['pre'], r['post'], r['type']
        edge = (min(pre,post), max(pre,post), 'e') if typ=='electrical' else (pre,post,'c')
        edge_syn[edge][ds] = edge_syn[edge].get(ds, 0) + int(r['synapses'])

edges = list(edge_syn)
def strength(e): return max(edge_syn[e].values())   # max synapses across the 2 animals
def in_both(e):  return len(edge_syn[e]) == 2

total_edges = len(edges)
core = [e for e in edges if in_both(e)]
tot_syn = sum(sum(edge_syn[e].values()) for e in edges)
core_syn = sum(sum(edge_syn[e].values()) for e in core)

print(f"adult pair (7,8): {total_edges} distinct edges")
print(f"  edge-count reproducibility (present in both):   {100*len(core)/total_edges:4.1f}%")
print(f"  SYNAPSE-WEIGHTED reproducibility (synapses on shared edges): {100*core_syn/tot_syn:4.1f}%\n")

print("  reproducibility vs connection strength (max synapses):")
bins = [(1,1),(2,2),(3,3),(4,6),(7,10),(11,10**9)]
for lo,hi in bins:
    grp = [e for e in edges if lo <= strength(e) <= hi]
    if grp:
        rep = 100*sum(in_both(e) for e in grp)/len(grp)
        lab = f"{lo}" if lo==hi else (f"{lo}+" if hi>10**8 else f"{lo}-{hi}")
        print(f"    strength {lab:5s} syn: {len(grp):5d} edges   present-in-both {rep:5.1f}%")

# what fraction of edges are singletons (1 synapse), and their reproducibility
singles = [e for e in edges if strength(e)==1]
print(f"\n  1-synapse edges: {len(singles)} ({100*len(singles)/total_edges:.0f}% of all edges), "
      f"reproducibility {100*sum(in_both(e) for e in singles)/len(singles):.1f}%")
strong = [e for e in edges if strength(e)>=4]
print(f"  >=4-synapse edges: {len(strong)} ({100*len(strong)/total_edges:.0f}% of all edges), "
      f"reproducibility {100*sum(in_both(e) for e in strong)/len(strong):.1f}%")
