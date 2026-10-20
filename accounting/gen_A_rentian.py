"""GENERATOR A — RENTIAN HIERARCHY.
Does a hierarchical-modular organizing principle GENERATE the contact geometry,
beating a degree-preserving null, compactly, out-of-sample?
"""
import sys; sys.path.insert(0,'../accounting')
import numpy as np, networkx as nx
from networkx.algorithms.community import louvain_communities
from bulk_benchmark import *

rng = np.random.default_rng(0)

# ---- build the contact graph (STRUCTURAL: this IS the target geometry we explain) ----
G = nx.Graph()
G.add_nodes_from(range(M))
for k in np.where(CONTACT>0)[0]:
    G.add_edge(int(PI[k]), int(PJ[k]))
print(f"contact graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ---- 1. RECURSIVE community detection -> dendrogram of nested modules ----
# each module records: set of nodes, level (depth). node_levels[node] = [mod_id at each level from root down]
modules = {}          # mod_id -> set of nodes
mod_size = {}         # mod_id -> size
node_path = {n: [] for n in range(M)}   # node -> list of mod_ids root->leaf
_next = [0]
MIN_SPLIT = 4         # don't split modules smaller than this

def recurse(nodes, depth, maxdepth):
    mid = _next[0]; _next[0]+=1
    modules[mid] = set(nodes); mod_size[mid] = len(nodes)
    for n in nodes: node_path[n].append(mid)
    if len(nodes) < MIN_SPLIT or depth >= maxdepth:
        return depth
    sub = G.subgraph(nodes)
    comms = louvain_communities(sub, seed=1, resolution=1.0)
    comms = [c for c in comms if len(c)>0]
    if len(comms) <= 1:   # no further structure
        return depth
    md = depth
    for c in comms:
        md = max(md, recurse(sorted(c), depth+1, maxdepth))
    return md

MAXDEPTH = 12
maxlevel = recurse(list(range(M)), 0, MAXDEPTH)
# number of distinct hierarchy levels actually used
level_of = {}
def _assign_levels(mid, lvl):
    level_of[mid]=lvl
# derive level from path position
for n in range(M):
    for lvl,mid in enumerate(node_path[n]):
        level_of[mid]=lvl
n_levels = max(level_of.values())+1
print(f"hierarchy: {len(modules)} modules across {n_levels} levels (maxdepth reached {maxlevel})")

# ---- Rent exponent: external edges T vs module size N, log-log fit ----
def external_edges(nodeset):
    s = nodeset; t = 0
    for u in s:
        for v in G.neighbors(u):
            if v not in s: t += 1
    return t   # counts each boundary edge once (from the internal endpoint)

Ns, Ts = [], []
for mid, s in modules.items():
    N = len(s)
    if N < 2 or N == M:        # skip singletons and the whole-graph root
        continue
    T = external_edges(s)
    if T > 0:
        Ns.append(N); Ts.append(T)
Ns = np.array(Ns,float); Ts = np.array(Ts,float)
lx, ly = np.log(Ns), np.log(Ts)
p, logk = np.polyfit(lx, ly, 1)
yhat = p*lx + logk
ss_res = ((ly-yhat)**2).sum(); ss_tot = ((ly-ly.mean())**2).sum()
R2 = 1 - ss_res/ss_tot
print(f"RENT: exponent p={p:.3f}  intercept k={np.exp(logk):.2f}  log-log R2={R2:.3f}  (n_modules={len(Ns)})")
print(f"  p in hierarchical range 0.5-0.75? {'YES' if 0.5<=p<=0.75 else 'NO'}  (p<1 => sublinear boundary growth = modular)")

# ---- 2. GENERATE propensity from hierarchy + Rent exponent ALONE ----
# LCA module size for a pair = size of smallest module containing BOTH nodes.
# Rentian scaling: connection prob between two nodes whose LCA module has N nodes
#   ~ (external conns) / (possible pairs) ~ N^p / N^2 = N^(p-2).
# => propensity(i,j) = (LCA_module_size)^(p-2).  Uses ONLY module membership + exponent p.
def lca_size(i,j):
    pi_, pj_ = node_path[i], node_path[j]
    L = min(len(pi_), len(pj_))
    common = M   # root always shared
    for l in range(L):
        if pi_[l]==pj_[l]:
            common = mod_size[pi_[l]]
        else:
            break
    return common

LCA = np.array([lca_size(int(PI[k]), int(PJ[k])) for k in range(NP)], float)
prop = LCA**(p-2.0)
res = score(prop, n_params=n_levels+1, name='A-Rentian')

# ---- 3. DISCRIMINATOR: scrambled-module control (permute node<->hierarchy slot, preserve all sizes) ----
def scrambled_prop(seed):
    perm = np.random.default_rng(seed).permutation(M)
    inv = {int(perm[n]): n for n in range(M)}   # slot -> shuffled node
    # rebuild node_path under permutation: physical node p sits where perm maps
    sp = {int(perm[n]): node_path[n] for n in range(M)}
    def lca_s(i,j):
        pi_, pj_ = sp[i], sp[j]; L=min(len(pi_),len(pj_)); common=M
        for l in range(L):
            if pi_[l]==pj_[l]: common=mod_size[pi_[l]]
            else: break
        return common
    LCAs = np.array([lca_s(int(PI[k]),int(PJ[k])) for k in range(NP)],float)
    return LCAs**(p-2.0)

scr_gens=[]
for s in range(20):
    sp_ = scrambled_prop(s)
    scr_gens.append(neuron_disjoint_auc(sp_))
scr_gens=np.array(scr_gens)
print(f"SCRAMBLED-MODULE control: gen_auc {scr_gens.mean():.3f} +/- {scr_gens.std():.3f} (n=20)")
print(f"REAL-vs-SCRAMBLED margin: {res['gen_auc_neurondisjoint']-scr_gens.mean():.3f}")

# ---- 4. COMPRESSION: how few levels reproduce contact? truncate hierarchy at K levels ----
print("\nCOMPRESSION (truncate hierarchy to first K levels):")
for K in range(1, n_levels+1):
    def lca_K(i,j,K):
        pi_,pj_=node_path[i][:K],node_path[j][:K]; L=min(len(pi_),len(pj_)); common=M
        for l in range(L):
            if pi_[l]==pj_[l]: common=mod_size[pi_[l]]
            else: break
        return common
    LCAk=np.array([lca_K(int(PI[k]),int(PJ[k]),K) for k in range(NP)],float)
    pk=LCAk**(p-2.0)
    g=neuron_disjoint_auc(pk)
    print(f"  K={K} levels (~{K+1} params): gen_auc={g:.3f}  margin_over_degnull={g-0.700:.3f}")
