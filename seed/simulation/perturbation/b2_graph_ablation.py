"""
Part B2 Test 2-3: GRAPH-ABLATION perturbation of the adult chemical connectome.

FRAMING: This tests the FUNCTIONAL SUFFICIENCY / topological consequence of removing
each edge class from the graph. It is NOT a test of developmental mechanism or causality.
"Load-bearing" here = graph-topological load, i.e. how much a class of edges carries
information-routing capacity relative to a degree/contact-matched random set of edges.

Base network  : strong_core1 == 1  (adult_syn_min >= 1) -> reproducible adult chemical
                directed graph, 1469 edges among ~180-200 neurons.
Nulls per class: >=20 matched random ablations of identical size, under 3 matching schemes
                (plain-random, contact-area-matched, endpoint-degree-matched).
Effect size   : z = (delta_class - mean_null) / std_null  per metric, with bootstrap CI
                on the null and empirical p (one-sided, class more disruptive).
"""
import os, numpy as np, pandas as pd, networkx as nx
from numpy.random import default_rng

RNG = default_rng(20260701)
OUT = "../seed/data/processed/mechanistic_perturbation"
TBL = f"{OUT}/dev_edge_trajectory_table.csv"
N_NULL = 30          # matched random ablations per class per scheme (>=20 required)
os.makedirs(OUT, exist_ok=True)

d = pd.read_csv(TBL)

# ---------------------------------------------------------------- neuron modality
# modality is intrinsic to a neuron; derive a single label per neuron.
mod_rows = pd.concat([
    d[['pre','pre_mod']].rename(columns={'pre':'n','pre_mod':'m'}),
    d[['post','post_mod']].rename(columns={'post':'n','post_mod':'m'}),
])
MOD = mod_rows.groupby('n')['m'].agg(lambda s: s.value_counts().index[0]).to_dict()

# ---------------------------------------------------------------- base graph
core = d[d.strong_core1 == 1].copy()
G0 = nx.DiGraph()
for n in set(core.pre) | set(core.post):
    G0.add_node(n, mod=MOD.get(n, 'inter'))
for _, r in core.iterrows():
    G0.add_edge(r.pre, r.post, syn=r.adult_syn_min, area=r.contact_area)
NODES = list(G0.nodes())
N = len(NODES)
print(f"Base adult-core graph: {N} nodes, {G0.number_of_edges()} edges")

SENS = [n for n in NODES if G0.nodes[n]['mod'] == 'sensory']
MOTOR = [n for n in NODES if G0.nodes[n]['mod'] == 'motor']
print(f"sensory={len(SENS)} motor={len(MOTOR)} inter={sum(1 for n in NODES if G0.nodes[n]['mod']=='inter')} command={sum(1 for n in NODES if G0.nodes[n]['mod']=='command')}")

# ---------------------------------------------------------------- metrics
def directed_efficiency_and_paths(G):
    """global efficiency (mean 1/d over ordered reachable pairs, 0 if unreachable),
    and mean shortest path over reachable pairs."""
    nodes = list(G.nodes())
    n = len(nodes)
    tot_inv = 0.0
    tot_d = 0.0
    n_reach = 0
    spl = dict(nx.all_pairs_shortest_path_length(G))
    for u in nodes:
        du = spl.get(u, {})
        for v, dist in du.items():
            if v == u:
                continue
            tot_inv += 1.0 / dist
            tot_d += dist
            n_reach += 1
    denom = n * (n - 1)
    geff = tot_inv / denom if denom else 0.0
    msp = tot_d / n_reach if n_reach else 0.0
    return geff, msp, n_reach

def sm_reachability(G):
    """fraction of (sensory, motor) ordered pairs with a directed path."""
    if not SENS or not MOTOR:
        return np.nan
    reached = 0
    total = 0
    for s in SENS:
        if s not in G:
            continue
        rs = nx.descendants(G, s)
        for m in MOTOR:
            if m == s:
                continue
            total += 1
            if m in rs:
                reached += 1
    return reached / total if total else np.nan

def largest_scc(G):
    return max((len(c) for c in nx.strongly_connected_components(G)), default=0)

def communicability_total(G):
    """total communicability = sum of exp(A) on the (binary) adjacency, undirected proxy."""
    UG = G.to_undirected()
    order = list(UG.nodes())
    A = nx.to_numpy_array(UG, nodelist=order)
    from scipy.linalg import expm
    # scale to keep expm bounded
    lam = np.abs(np.linalg.eigvalsh(A)).max()
    E = expm(A / (lam if lam > 0 else 1.0))
    return float(E.sum())

def betweenness_vec(G, nodelist):
    bc = nx.betweenness_centrality(G, normalized=True)
    return np.array([bc.get(n, 0.0) for n in nodelist])

def metrics(G):
    geff, msp, nreach = directed_efficiency_and_paths(G)
    return dict(global_efficiency=geff,
                mean_shortest_path=msp,
                n_reachable_pairs=nreach,
                largest_scc=largest_scc(G),
                sm_reachability=sm_reachability(G),
                communicability=communicability_total(G))

BASE_BC = betweenness_vec(G0, NODES)
BASE = metrics(G0)
print("BASELINE:", {k: round(v, 4) for k, v in BASE.items()})

# edges as list with attributes for matching
EDGES = list(G0.edges(data=True))
EIDX = {(u, v): i for i, (u, v, _) in enumerate(EDGES)}
AREA = np.array([e['area'] for _, _, e in EDGES])
OUTDEG = dict(G0.out_degree())
INDEG = dict(G0.in_degree())
EDGE_DEGKEY = np.array([OUTDEG[u] + INDEG[v] for u, v, _ in EDGES])

def ablate(edge_set):
    G = G0.copy()
    G.remove_edges_from([(u, v) for (u, v) in edge_set])
    return G

def apply_and_measure(edge_set):
    G = ablate(edge_set)
    m = metrics(G)
    bc = betweenness_vec(G, NODES)
    m['betweenness_L1_shift'] = float(np.abs(bc - BASE_BC).sum())
    return m

# ---------------------------------------------------------------- matched nulls
def bin_index(values, k=8):
    # quantile bins; returns array of bin ids
    qs = np.quantile(values, np.linspace(0, 1, k + 1))
    qs[0] -= 1e-9; qs[-1] += 1e-9
    return np.clip(np.digitize(values, qs) - 1, 0, k - 1)

AREA_BIN = bin_index(AREA)
DEG_BIN = bin_index(EDGE_DEGKEY.astype(float))
ALL_EI = np.arange(len(EDGES))

def sample_matched(class_ei, scheme):
    """return indices into EDGES for one matched null of the same size as class_ei."""
    k = len(class_ei)
    # Null = random set of k edges from the WHOLE network (same size). For classes
    # smaller than half the graph we exclude the class itself (cleaner contrast);
    # for large classes (>half) we must draw from all edges.
    pool = np.setdiff1d(ALL_EI, class_ei)
    if len(pool) < k:
        pool = ALL_EI
    if scheme == 'random':
        return RNG.choice(pool, size=k, replace=False)
    binvec = AREA_BIN if scheme == 'contact' else DEG_BIN
    target_counts = pd.Series(binvec[class_ei]).value_counts().to_dict()
    picks = []
    for b, cnt in target_counts.items():
        cand = pool[binvec[pool] == b]
        if len(cand) >= cnt:
            picks.extend(RNG.choice(cand, size=cnt, replace=False).tolist())
        else:  # bin too small -> take all + backfill from full pool
            picks.extend(cand.tolist())
            need = cnt - len(cand)
            rest = np.setdiff1d(pool, np.array(picks, dtype=int))
            picks.extend(RNG.choice(rest, size=need, replace=False).tolist())
    return np.array(picks[:k], dtype=int)

METRIC_KEYS = ['global_efficiency', 'mean_shortest_path', 'largest_scc',
               'sm_reachability', 'communicability', 'betweenness_L1_shift',
               'n_reachable_pairs']
# direction of "more disruptive": efficiency/scc/reachability/communic/reachpairs DOWN;
# mean_shortest_path and betweenness_shift UP.
LOWER_IS_MORE_DISRUPTIVE = {'global_efficiency', 'largest_scc', 'sm_reachability',
                            'communicability', 'n_reachable_pairs'}

# ---------------------------------------------------------------- edge classes
def eiset(mask):
    sub = core[mask.values]
    return np.array([EIDX[(r.pre, r.post)] for r in sub.itertuples()])

# hub edges: endpoints in top-decile total degree
totdeg = {n: OUTDEG[n] + INDEG[n] for n in NODES}
hub_thr = np.quantile(list(totdeg.values()), 0.90)
hubset = {n for n in NODES if totdeg[n] >= hub_thr}
core = core.reset_index(drop=True)

CLASSES = {}
CLASSES['strong_core2 (>=2 syn)'] = eiset(core.strong_core2 == 1)
CLASSES['strong_core4 (>=4 syn)'] = eiset(core.strong_core4 == 1)
CLASSES['weak_core (exactly 1 syn)'] = eiset((core.strong_core1 == 1) & (core.strong_core2 == 0))
CLASSES['preadult_persistent (>=2 groups)'] = eiset(core.n_preadult_groups >= 2)
CLASSES['transient_in_core (0 preadult groups)'] = eiset(core.n_preadult_groups == 0)
CLASSES['stable_class'] = eiset(core.classification == 'stable')
CLASSES['post_embryonic_integration'] = eiset(core.classification == 'post-embryonic brain integration')
CLASSES['dev_strengthened'] = eiset(core.classification == 'developmentally dynamic (strengthened)')
CLASSES['hub_edges'] = np.array([i for i, (u, v, _) in enumerate(EDGES) if u in hubset or v in hubset])
CLASSES['NT_GABA (inhibitory)'] = eiset(core.pre_nt == 'GABA')
CLASSES['NT_ACh'] = eiset(core.pre_nt == 'ACh')
CLASSES['NT_Glu'] = eiset(core.pre_nt == 'Glu')
CLASSES['NT_mono'] = eiset(core.pre_nt == 'mono')
CLASSES['sensory->inter'] = eiset((core.pre_mod == 'sensory') & (core.post_mod == 'inter'))
CLASSES['inter->motor'] = eiset((core.pre_mod == 'inter') & (core.post_mod == 'motor'))
CLASSES['sensory->motor(direct)'] = eiset((core.pre_mod == 'sensory') & (core.post_mod == 'motor'))

# ---------------------------------------------------------------- run
records = []
rank_rows = []
for cname, cei in CLASSES.items():
    k = len(cei)
    if k < 3:
        print(f"  skip {cname}: only {k} edges")
        continue
    edge_pairs = [(EDGES[i][0], EDGES[i][1]) for i in cei]
    cm = apply_and_measure(edge_pairs)
    for scheme in ['random', 'contact', 'degree']:
        null_vals = {mk: [] for mk in METRIC_KEYS}
        for _ in range(N_NULL):
            nei = sample_matched(cei, scheme)
            npairs = [(EDGES[i][0], EDGES[i][1]) for i in nei]
            nm = apply_and_measure(npairs)
            for mk in METRIC_KEYS:
                null_vals[mk].append(nm[mk])
        for mk in METRIC_KEYS:
            nv = np.array(null_vals[mk], float)
            mu, sd = nv.mean(), nv.std(ddof=1)
            cval = cm[mk]
            delta_class = cval - BASE.get(mk, 0.0)
            delta_null = nv - BASE.get(mk, 0.0)
            z = (delta_class - delta_null.mean()) / (delta_null.std(ddof=1) + 1e-12)
            # one-sided empirical p: class MORE disruptive than matched null
            if mk in LOWER_IS_MORE_DISRUPTIVE:
                p = (np.sum(nv <= cval) + 1) / (N_NULL + 1)  # class lower (more disruptive)
                more_disruptive = cval < mu
            else:
                p = (np.sum(nv >= cval) + 1) / (N_NULL + 1)
                more_disruptive = cval > mu
            ci = np.percentile(delta_null, [2.5, 97.5])
            records.append(dict(
                edge_class=cname, n_edges=k, null_scheme=scheme, metric=mk,
                baseline=round(BASE.get(mk, 0.0), 5), class_value=round(cval, 5),
                class_delta=round(delta_class, 5),
                null_delta_mean=round(delta_null.mean(), 5),
                null_delta_ci_lo=round(ci[0], 5), null_delta_ci_hi=round(ci[1], 5),
                z_vs_null=round(float(z), 3), emp_p_more_disruptive=round(float(p), 4),
                more_disruptive_than_null=bool(more_disruptive)))
    print(f"  done {cname} (k={k})")

res = pd.DataFrame(records)
res.to_csv(f"{OUT}/B2_graph_ablation_results.csv", index=False)
print(f"\nwrote B2_graph_ablation_results.csv: {len(res)} rows")

# ---------------------------------------------------------------- importance ranking
# Rank classes by a composite load-bearing score = mean |z| across the two most
# functionally meaningful metrics (sm_reachability, global_efficiency) under the
# CONTACT-matched null (the strictest geometry-aware control), signed so that
# positive = more disruptive than matched-random.
key_metrics = ['sm_reachability', 'global_efficiency', 'communicability', 'largest_scc']
imp = []
for cname in CLASSES:
    sub = res[(res.edge_class == cname) & (res.null_scheme == 'contact')]
    if sub.empty:
        continue
    km = sub[sub.metric.isin(key_metrics)]
    # signed z: more disruptive -> positive contribution
    signed = []
    for _, r in km.iterrows():
        s = -r.z_vs_null if r.metric in LOWER_IS_MORE_DISRUPTIVE else r.z_vs_null
        signed.append(s)
    reach = sub[sub.metric == 'sm_reachability']
    geff = sub[sub.metric == 'global_efficiency']
    imp.append(dict(
        edge_class=cname,
        n_edges=int(sub.n_edges.iloc[0]),
        frac_of_core=round(int(sub.n_edges.iloc[0]) / G0.number_of_edges(), 3),
        composite_load_score=round(float(np.mean(signed)), 3),
        sm_reach_z=float(reach.z_vs_null.iloc[0]) if len(reach) else np.nan,
        sm_reach_p=float(reach.emp_p_more_disruptive.iloc[0]) if len(reach) else np.nan,
        geff_z=float(geff.z_vs_null.iloc[0]) if len(geff) else np.nan,
        geff_p=float(geff.emp_p_more_disruptive.iloc[0]) if len(geff) else np.nan,
    ))
impdf = pd.DataFrame(imp).sort_values('composite_load_score', ascending=False)
# per-edge load = composite / n_edges normalized (efficiency density)
impdf['load_per_edge'] = (impdf.composite_load_score / impdf.n_edges * 100).round(4)
impdf.to_csv(f"{OUT}/B2_edgeclass_importance.csv", index=False)
print("\n=== EDGE-CLASS IMPORTANCE (contact-matched null, higher = more load-bearing) ===")
print(impdf.to_string(index=False))
