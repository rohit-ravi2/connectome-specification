#!/usr/bin/env python
"""Part A2: cross-dataset / cross-system generalization of the
'rich contact topology + strong-core scaffold' pattern.

Outputs to data/processed/mechanistic_perturbation/:
  A2_cross_dataset_results.csv   (adult robustness + sex transfer)
  A2_cross_system_results.csv    (cross-species graph-level primitives)
  A2_dataset_mapping_quality.csv (mapping quality per dataset)
"""
import os, re, warnings
import numpy as np
import pandas as pd
import networkx as nx

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)

CONN = "../data/celegans/data/connectome"
DROS = "../data/celegans/data/drosophila"
PLAT = "../data/celegans/data/platynereis/Platynereis_3D_connectome_2024"
OUT  = "../seed/data/processed/mechanistic_perturbation"
os.makedirs(OUT, exist_ok=True)

# ======================================================================
# LOADERS (chemical synapses, directed, weight = synapse count)
# ======================================================================
def load_white(which="A"):
    f = f"{CONN}/white_MoW/aconnectome_white_1986_{which}.csv"
    df = pd.read_csv(f, sep=r"\s+", engine="python")
    df = df[df["type"] == "chemical"]
    d = {}
    for _, r in df.iterrows():
        d[(str(r["pre"]).strip(), str(r["post"]).strip())] = float(r["synapses"])
    return d

def _load_cook_adj(sheet):
    f = f"{CONN}/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx"
    raw = pd.read_excel(f, sheet_name=sheet, header=None, engine="openpyxl")
    # Find header row/col: the adjacency block has neuron names on row and col.
    # Cook layout: first 2-3 rows metadata; find row where many string cells appear.
    # Strategy: locate the row and column indices holding neuron labels.
    # The matrix typically starts with labels at row idx 2 and col idx 2.
    # Detect by scanning for the largest square block of numeric data.
    arr = raw.values
    nr, nc = arr.shape
    # header row = first row with >20 non-null string entries after some offset
    hdr_row = None
    for i in range(min(6, nr)):
        strs = sum(isinstance(x, str) for x in arr[i])
        if strs > 20:
            hdr_row = i; break
    hdr_col = None
    for j in range(min(6, nc)):
        strs = sum(isinstance(arr[i, j], str) for i in range(nr))
        if strs > 20:
            hdr_col = j; break
    if hdr_row is None or hdr_col is None:
        raise RuntimeError(f"could not locate header in {sheet}")
    cols = [str(x).strip() for x in arr[hdr_row, hdr_col+1:]]
    rows = [str(arr[i, hdr_col]).strip() for i in range(hdr_row+1, nr)]
    block = arr[hdr_row+1:, hdr_col+1:]
    d = {}
    for ri, pre in enumerate(rows):
        if pre in ("None", "nan", ""): continue
        for ci, post in enumerate(cols):
            if post in ("None", "nan", ""): continue
            v = block[ri, ci]
            try: v = float(v)
            except (TypeError, ValueError): continue
            if np.isfinite(v) and v > 0:
                d[(pre, post)] = v
    return d

def load_cook_herm(): return _load_cook_adj("hermaphrodite chemical")
def load_cook_male(): return _load_cook_adj("male chemical")

def load_witvliet_adult():
    f = f"{CONN}/witvliet2020/witvliet_combined_edges.csv"
    df = pd.read_csv(f)
    df = df[df["type"] == "chemical"]
    adult = df[df["source_file"].isin(["witvliet_2020_7.xlsx", "witvliet_2020_8.xlsx"])]
    d = {}
    for _, r in adult.iterrows():
        k = (str(r["pre"]).strip(), str(r["post"]).strip())
        d[k] = d.get(k, 0.0) + float(r["synapses"])
    return d

# ======================================================================
# CROSS-DATASET / SEX helpers
# ======================================================================
def nodes_of(d):
    s = set()
    for a, b in d: s.add(a); s.add(b)
    return s

def restrict(d, keep):
    return {(a, b): w for (a, b), w in d.items() if a in keep and b in keep}

def jaccard(e1, e2):
    s1, s2 = set(e1), set(e2)
    u = len(s1 | s2)
    return len(s1 & s2) / u if u else np.nan

def pair_overlap(dA, dB, name):
    """Restrict to shared nodes; report Jaccard, high vs low weight conservation,
    Spearman weight corr on shared edges."""
    shared = nodes_of(dA) & nodes_of(dB)
    a = restrict(dA, shared); b = restrict(dB, shared)
    jac = jaccard(a.keys(), b.keys())
    # high vs low weight conservation (relative to A)
    if len(a) >= 10:
        ws = np.array(sorted(a.values()))
        hi_thr = np.quantile(ws, 0.67)
        lo_thr = np.quantile(ws, 0.33)
        hi = [k for k, w in a.items() if w >= hi_thr]
        lo = [k for k, w in a.items() if w <= lo_thr]
        hi_cons = np.mean([k in b for k in hi]) if hi else np.nan
        lo_cons = np.mean([k in b for k in lo]) if lo else np.nan
    else:
        hi_cons = lo_cons = np.nan
    # weight correlation on shared edges
    shared_e = set(a) & set(b)
    if len(shared_e) >= 10:
        from scipy.stats import spearmanr
        xa = [a[k] for k in shared_e]; xb = [b[k] for k in shared_e]
        rho = spearmanr(xa, xb).correlation
    else:
        rho = np.nan
    return dict(comparison=name, n_shared_nodes=len(shared),
                n_edges_A=len(a), n_edges_B=len(b),
                jaccard=jac, hi_weight_conservation=hi_cons,
                lo_weight_conservation=lo_cons, weight_spearman=rho,
                hi_minus_lo=hi_cons - lo_cons if np.isfinite(hi_cons) and np.isfinite(lo_cons) else np.nan)

# ======================================================================
# CROSS-SYSTEM graph-level primitives
# ======================================================================
def edgelist_to_digraph(d):
    G = nx.DiGraph()
    for (a, b), w in d.items():
        if a == b: continue
        G.add_edge(a, b, weight=float(w))
    return G

def load_winding_larval():
    f = f"{DROS}/winding2023/Supplementary-Data-S1/all-all_connectivity_matrix.csv"
    m = pd.read_csv(f, index_col=0)
    m.columns = m.index.astype(str)  # square, align
    A = m.values.astype(float)
    idx = list(m.index.astype(str))
    d = {}
    nz = np.argwhere(A > 0)
    for i, j in nz:
        if i == j: continue
        d[(idx[i], idx[j])] = float(A[i, j])
    return d

def _extract_r_vector(text, name):
    m = re.search(rf'\b{name} = c\(', text)
    if not m: return None
    i = m.end()
    depth = 1; j = i
    while depth > 0 and j < len(text):
        c = text[j]
        if c == '(': depth += 1
        elif c == ')': depth -= 1
        j += 1
    body = text[i:j-1]
    nums = re.findall(r'-?\d+\.?\d*', body)
    return np.array([float(x) for x in nums])

def load_platynereis():
    f = f"{PLAT}/supplements/connectome_graph.txt"
    s = open(f).read()
    k = s.find("edges = structure")
    seg = s[k:]
    frm = _extract_r_vector(seg, "from")
    to  = _extract_r_vector(seg, "to")
    wt  = _extract_r_vector(seg, "weight")
    if wt is None: wt = _extract_r_vector(seg, "value")
    n = min(len(frm), len(to))
    if wt is None or len(wt) < n:
        wt = np.ones(n)
    d = {}
    for a, b, w in zip(frm[:n].astype(int), to[:n].astype(int), wt[:n]):
        if a == b: continue
        key = (f"n{a}", f"n{b}")
        d[key] = d.get(key, 0.0) + float(w)
    return d

def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() == 0: return np.nan
    idx = np.arange(1, n+1)
    return (np.sum((2*idx - n - 1) * x)) / (n * x.sum())

def rich_club_norm(G_und, nrand=5, seed=0):
    """Max normalized rich-club coefficient over k, vs degree-preserving swap null."""
    G = nx.Graph(G_und)
    G.remove_edges_from(nx.selfloop_edges(G))
    if G.number_of_edges() < 10: return np.nan, np.nan
    try:
        phi = nx.rich_club_coefficient(G, normalized=False)
    except Exception:
        return np.nan, np.nan
    # null
    rng2 = np.random.RandomState(seed)
    rand_phis = []
    E = G.number_of_edges()
    for _ in range(nrand):
        Gr = G.copy()
        try:
            nx.double_edge_swap(Gr, nswap=min(10*E, 200000),
                                max_tries=min(100*E, 2000000), seed=rng2.randint(1e9))
        except Exception:
            pass
        try:
            rand_phis.append(nx.rich_club_coefficient(Gr, normalized=False))
        except Exception:
            pass
    if not rand_phis: return np.nan, np.nan
    ks = sorted(phi.keys())
    ratios = []
    for k in ks:
        rp = np.mean([rp.get(k, np.nan) for rp in rand_phis])
        if rp and np.isfinite(rp) and rp > 0:
            ratios.append(phi[k] / rp)
    if not ratios: return np.nan, np.nan
    # report the robust high-k regime: max over upper half of k range
    ratios = np.array(ratios)
    return float(np.nanmax(ratios)), float(np.nanmedian(ratios))

def core_halo_enrichment(G, nperm=200, seed=0):
    """Do strong (top-20% weight) edges preferentially connect high-degree hubs?
    metric = mean(deg_u*deg_v over top-weight edges) / mean over all edges.
    Null: shuffle edge weights across the fixed topology."""
    Gu = nx.Graph()
    for a, b, dd in G.edges(data=True):
        w = dd["weight"]
        if Gu.has_edge(a, b):
            Gu[a][b]["weight"] += w
        else:
            Gu.add_edge(a, b, weight=w)
    deg = dict(Gu.degree())
    edges = list(Gu.edges(data="weight"))
    if len(edges) < 20: return np.nan, np.nan
    w = np.array([e[2] for e in edges])
    dp = np.array([deg[e[0]] * deg[e[1]] for e in edges])
    thr = np.quantile(w, 0.80)
    def stat(weights):
        top = weights >= np.quantile(weights, 0.80)
        return dp[top].mean() / dp.mean()
    obs = stat(w)
    rs = np.random.RandomState(seed)
    null = np.array([stat(rs.permutation(w)) for _ in range(nperm)])
    z = (obs - null.mean()) / (null.std() + 1e-12)
    return float(obs), float(z)

def system_primitives(d, name):
    G = edgelist_to_digraph(d)
    Gu = G.to_undirected()
    Gu.remove_edges_from(nx.selfloop_edges(Gu))
    deg = np.array([d_ for _, d_ in Gu.degree()])
    strength = np.array([s for _, s in Gu.degree(weight="weight")])
    ew = np.array([dd["weight"] for _, _, dd in G.edges(data=True)])
    gini_deg = gini(deg)
    gini_str = gini(strength)
    gini_ew = gini(ew)
    top10_frac = np.sort(ew)[::-1][:max(1, int(0.1*len(ew)))].sum() / ew.sum()
    rc_max, rc_med = rich_club_norm(Gu, nrand=5, seed=0)
    ch_obs, ch_z = core_halo_enrichment(G, nperm=200, seed=0)
    return dict(system=name, n_nodes=Gu.number_of_nodes(), n_edges=G.number_of_edges(),
                weighted=bool(np.any(ew != ew[0])),
                gini_degree=gini_deg, gini_strength=gini_str, gini_edgeweight=gini_ew,
                top10pct_edge_weight_frac=top10_frac,
                richclub_norm_max=rc_max, richclub_norm_median=rc_med,
                coreHalo_hubEnrichment=ch_obs, coreHalo_z=ch_z,
                heavy_tail_shared=bool(gini_str > 0.30 and top10_frac > 0.30),
                richclub_shared=bool(np.isfinite(rc_max) and rc_max > 1.1),
                coreHalo_shared=bool(np.isfinite(ch_z) and ch_z > 3))

# ======================================================================
# RUN
# ======================================================================
print("loading worm datasets...")
mapping_rows = []
cross_rows = []

white_A   = load_white("A")
white_who = load_white("whole")
cook_h    = load_cook_herm()
cook_m    = load_cook_male()
witv      = load_witvliet_adult()

worm_sets = {"White1986_A": white_A, "White1986_whole": white_who,
             "Cook2019_herm": cook_h, "Cook2019_male": cook_m,
             "Witvliet2020_adult": witv}
for nm, d in worm_sets.items():
    print(f"  {nm}: {len(nodes_of(d))} nodes, {len(d)} chem edges")

# ---- mapping quality: node vocabulary overlap across the harmonized adult sets
adult_sets = {"White1986_A": white_A, "Cook2019_herm": cook_h, "Witvliet2020_adult": witv}
ref_nodes = nodes_of(cook_h)
for nm, d in adult_sets.items():
    nn = nodes_of(d)
    inter = nn & ref_nodes
    mapping_rows.append(dict(dataset=nm, kind="worm_adult_chemical",
        n_nodes=len(nn), n_edges=len(d),
        node_overlap_with_Cook_herm=len(inter),
        node_jaccard_with_Cook_herm=len(inter)/len(nn | ref_nodes),
        frac_nodes_mapped=len(inter)/len(nn),
        mapping_quality="GOOD (shared WormAtlas nomenclature)"))
mapping_rows.append(dict(dataset="Cook2019_male", kind="worm_male_chemical",
    n_nodes=len(nodes_of(cook_m)), n_edges=len(cook_m),
    node_overlap_with_Cook_herm=len(nodes_of(cook_m) & ref_nodes),
    node_jaccard_with_Cook_herm=len(nodes_of(cook_m)&ref_nodes)/len(nodes_of(cook_m)|ref_nodes),
    frac_nodes_mapped=len(nodes_of(cook_m)&ref_nodes)/len(nodes_of(cook_m)),
    mapping_quality="GOOD within shared neurons; male has sex-specific extras"))
mapping_rows.append(dict(dataset="Varshney2011_connectome", kind="worm_adult_chemical",
    n_nodes=0, n_edges=0, node_overlap_with_Cook_herm=0, node_jaccard_with_Cook_herm=0,
    frac_nodes_mapped=0,
    mapping_quality="ABSENT: folder contains Yemini2013 behavioral DB (NIHMS496349), NOT connectome adjacency"))

# ---- TEST 1: adult robustness (pairwise) ----
print("test 1: adult robustness")
cross_rows.append({**pair_overlap(cook_h, white_A, "Cook_herm_vs_White_A"), "test":"adult_robustness"})
cross_rows.append({**pair_overlap(cook_h, witv,   "Cook_herm_vs_Witvliet_adult"), "test":"adult_robustness"})
cross_rows.append({**pair_overlap(white_A, witv,  "White_A_vs_Witvliet_adult"), "test":"adult_robustness"})

# strong-core analog: edges present in ALL THREE adult datasets on shared nodes
shared3 = nodes_of(cook_h) & nodes_of(white_A) & nodes_of(witv)
ch = restrict(cook_h, shared3); wa = restrict(white_A, shared3); wv = restrict(witv, shared3)
common_edges = set(ch) & set(wa) & set(wv)
# are common (reproducible) edges higher weight than dataset-unique?
def mean_w(edges, dd):
    vals=[dd[e] for e in edges if e in dd]
    return np.mean(vals) if vals else np.nan
alle = set(ch)|set(wa)|set(wv)
uniq = alle - common_edges
cross_rows.append(dict(test="strong_core_analog", comparison="edges_in_all3_adult",
    n_shared_nodes=len(shared3), n_edges_A=len(common_edges), n_edges_B=len(alle),
    jaccard=len(common_edges)/len(alle) if alle else np.nan,
    hi_weight_conservation=mean_w(common_edges, ch),
    lo_weight_conservation=mean_w(uniq, ch),
    weight_spearman=np.nan,
    hi_minus_lo=mean_w(common_edges, ch)-mean_w(uniq, ch)))

# ---- TEST 2: sex transfer ----
print("test 2: sex transfer")
cross_rows.append({**pair_overlap(cook_h, cook_m, "Cook_herm_vs_male"), "test":"sex_transfer"})

pd.DataFrame(cross_rows).to_csv(f"{OUT}/A2_cross_dataset_results.csv", index=False)
pd.DataFrame(mapping_rows).to_csv(f"{OUT}/A2_dataset_mapping_quality.csv", index=False)
print("wrote cross_dataset + mapping_quality")

# ---- TEST 3: cross-system ----
print("test 3: cross-system primitives")
sys_rows = []
sys_rows.append(system_primitives(cook_h, "C.elegans_adult_hermaphrodite"))
print("  worm done")
try:
    dros = load_winding_larval()
    print(f"  drosophila larval: {len(nodes_of(dros))} nodes, {len(dros)} edges")
    sys_rows.append(system_primitives(dros, "Drosophila_larval_Winding2023"))
except Exception as e:
    sys_rows.append(dict(system="Drosophila_larval_Winding2023", n_nodes=0, n_edges=0,
                         mapping_quality=f"BLOCKED: {e}"))
try:
    plat = load_platynereis()
    print(f"  platynereis: {len(nodes_of(plat))} nodes, {len(plat)} edges")
    sys_rows.append(system_primitives(plat, "Platynereis_larval_Veraszto2024"))
except Exception as e:
    import traceback; traceback.print_exc()
    sys_rows.append(dict(system="Platynereis_larval_Veraszto2024", n_nodes=0, n_edges=0,
                         mapping_quality=f"BLOCKED: {e}"))

pd.DataFrame(sys_rows).to_csv(f"{OUT}/A2_cross_system_results.csv", index=False)
print("wrote cross_system")

# mapping quality for cross-system
for nm, note in [("Drosophila_larval_Winding2023",
                  "graph-level ONLY: node IDs are skeleton IDs, NO node homology to worm"),
                 ("Platynereis_larval_Veraszto2024",
                  "graph-level ONLY: parsed from R igraph dump; NO node homology to worm")]:
    mapping_rows.append(dict(dataset=nm, kind="cross_species_graph",
        n_nodes=0, n_edges=0, node_overlap_with_Cook_herm=0, node_jaccard_with_Cook_herm=0,
        frac_nodes_mapped=0, mapping_quality=note))
pd.DataFrame(mapping_rows).to_csv(f"{OUT}/A2_dataset_mapping_quality.csv", index=False)
print("DONE")
