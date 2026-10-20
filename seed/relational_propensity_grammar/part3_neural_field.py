#!/usr/bin/env python3
"""PART 3 — NEURAL FIELD / CONTINUOUS DEVELOPMENTAL COORDINATES.

Model per-neuron wiring propensity as a CONTINUOUS field over DEVELOPMENTAL
COORDINATES (birth time, nerve-ring 3D position, lineage), independent of adult
adjacency. Discriminating question: is the latent structure SMOOTH in
developmental coordinates (-> developmental PROCESS) or DISCRETE/CATEGORICAL
(-> molecular ADDRESS code)?

Firewall: score against DISPERSION RATIO and held-out cross-animal BRIER on a
NEURON-DISJOINT split. AUC reported but not success.
Non-circularity: coordinates measured independently of adult adjacency.
Target = adult edge presence (W7/W8 Witvliet); features never derived from the
adult connectome graph. Contact area is the substrate (allowed).
"""
import json, csv, os, math
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = os.path.expanduser("../seed")
OUT = f"{ROOT}/data/processed/relational_propensity_grammar"
SKEL = f"{ROOT}/data/external/mechanistic_frontier/witvliet2021_developmental/data/skeletons"
BIRTH = f"{ROOT}/data/external/mechanistic_frontier/neuron_birth_times_sulston/neuron_birth_times.csv"
THETA = f"{OUT}/latent_theta_table.csv"
TREE = "../lineage/jtauber_lineage/cells.json"
RELTAB = "../data/celegans/data/lineage/merged_lineage_table.csv"
NPZ = "../data/tier5/connectome.npz"
rng_global = np.random.default_rng(0)
def _r(x): return '' if (x is None or (isinstance(x, float) and np.isnan(x))) else round(float(x), 4)

# ---------------- load pairs / target / contact ----------------
pairs, contact, w7, w8, thetaL1 = [], [], [], [], []
for r in csv.DictReader(open(THETA)):
    pairs.append((r['pre'], r['post']))
    contact.append(float(r['contact_area']))
    w7.append(int(r['w7'])); w8.append(int(r['w8']))
    thetaL1.append(float(r['theta_L1']))
contact = np.array(contact); w7 = np.array(w7); w8 = np.array(w8)
thetaL1 = np.array(thetaL1)
y_any = ((w7 == 1) | (w8 == 1)).astype(int)
neurons = sorted(set(a for a, b in pairs) | set(b for a, b in pairs))
print(f"pairs {len(pairs)} | neurons {len(neurons)} | edge rate any {y_any.mean():.3f} "
      f"W7 {w7.mean():.3f} W8 {w8.mean():.3f}")

# ---------------- COORDINATE 1: birth time ----------------
birth = {r['neuron']: float(r['birth_time_min']) for r in csv.DictReader(open(BIRTH))}

# ---------------- COORDINATE 2: 3D nerve-ring position + arbor extent ----------------
def skel_feats(path):
    d = json.load(open(path))
    out = {}
    for n, v in d.items():
        C = np.array(list(v['coords'].values()), float)
        if C.ndim != 2 or len(C) == 0:
            continue
        cen = C.mean(0)
        ext = math.sqrt(((C - cen) ** 2).sum(1).mean())   # radius of gyration
        out[n] = (cen, ext, float(v.get('length', np.nan)))
    return out
sk7 = skel_feats(f"{SKEL}/Dataset7_skeletons.json")
sk8 = skel_feats(f"{SKEL}/Dataset8_skeletons.json")
pos = {}; extent = {}; arborlen = {}
for n in neurons:
    cs, es, ls = [], [], []
    for sk in (sk7, sk8):
        if n in sk:
            cs.append(sk[n][0]); es.append(sk[n][1]); ls.append(sk[n][2])
    pos[n] = np.mean(cs, 0); extent[n] = float(np.mean(es)); arborlen[n] = float(np.mean(ls))

# ---------------- COORDINATE 3: lineage (jtauber tree path) ----------------
tree = json.load(open(TREE))
lpath = {}
def walk(node, p):
    ch = node.get('children') or []
    if not ch:
        lpath[node['name']] = p; return
    for i, c in enumerate(ch):
        walk(c, p + str(i))
walk(tree, '')
# relatedness table fallback for the 2 missing (AVFL/AVFR)
rel = {}
for r in csv.DictReader(open(RELTAB)):
    rel[(r['NeuronA'], r['NeuronB'])] = float(r['Relatedness'])
    rel[(r['NeuronB'], r['NeuronA'])] = float(r['Relatedness'])

def lineage_dist(a, b):
    """tree distance from binary lineage path; fallback to relatedness table."""
    if a in lpath and b in lpath:
        pa, pb = lpath[a], lpath[b]
        k = 0
        for x, y in zip(pa, pb):
            if x == y: k += 1
            else: break
        return (len(pa) - k) + (len(pb) - k)   # edges between leaves in lineage tree
    # fallback: map relatedness (shared-ancestor depth) to a pseudo-distance
    rv = rel.get((a, b), np.nan)
    if np.isnan(rv):
        return np.nan
    return 40.0 - rv   # larger relatedness -> smaller distance (monotone proxy)
# fill missing lineage distances with median later
print(f"lineage: {sum(n in lpath for n in neurons)}/{len(neurons)} neurons in jtauber tree")

# ---------------- PER-PAIR developmental-coordinate features ----------------
def pair_feats(a, b):
    ba, bb = birth[a], birth[b]
    birth_diff = abs(ba - bb)
    birth_mean = 0.5 * (ba + bb)
    soma_dist = float(np.linalg.norm(pos[a] - pos[b]))
    ext_a, ext_b = extent[a], extent[b]
    overlap = soma_dist / (ext_a + ext_b + 1e-9)          # <1 ~ arbors overlap
    ld = lineage_dist(a, b)
    return dict(logc=math.log1p(contact_of(a, b)),
                birth_diff=birth_diff, birth_mean=birth_mean,
                soma_dist=soma_dist, overlap=overlap,
                ext_a=ext_a, ext_b=ext_b,
                arborlen_a=arborlen[a], arborlen_b=arborlen[b],
                lineage=ld)
# contact lookup
_cmap = {(a, b): c for (a, b), c in zip(pairs, contact)}
def contact_of(a, b): return _cmap[(a, b)]

FEATNAMES = ['logc', 'birth_diff', 'birth_mean', 'soma_dist', 'overlap',
             'ext_a', 'ext_b', 'arborlen_a', 'arborlen_b', 'lineage']
Fd = [pair_feats(a, b) for a, b in pairs]
X = np.array([[f[k] for k in FEATNAMES] for f in Fd], float)
# impute missing lineage (col index) with median
li = FEATNAMES.index('lineage')
med = np.nanmedian(X[:, li]); X[np.isnan(X[:, li]), li] = med
CONT = X[:, [FEATNAMES.index(k) for k in
             ['birth_diff', 'birth_mean', 'soma_dist', 'overlap', 'ext_a', 'ext_b', 'lineage']]]
LOGC = X[:, [FEATNAMES.index('logc')]]
print("feature matrix", X.shape, "| continuous-coord block", CONT.shape)

# ---------------- neuron-disjoint CV fold assignment ----------------
def fold_map(seed=0, k=5):
    rng = np.random.default_rng(seed)
    f = np.array_split(rng.permutation(neurons), k)
    return {n: i for i, ff in enumerate(f) for n in ff}

def ndcv_predict(build_fit, y, seed=0, k=5):
    """build_fit(Xtr,ytr)->model with predict_proba; neuron-disjoint (both endpoints in fold)."""
    nf = fold_map(seed, k)
    p = np.full(len(y), np.nan)
    for kk in range(k):
        te = np.array([nf[a] == kk and nf[b] == kk for a, b in pairs])
        tr = np.array([nf[a] != kk and nf[b] != kk for a, b in pairs])
        if te.sum() < 10 or y[tr].sum() < 5: continue
        m = build_fit(tr, kk)
        p[te] = m(te)
    return p

# ---------------- MODEL builders ----------------
def make_contact_only(y):
    def bf(tr, kk):
        sc = StandardScaler().fit(LOGC[tr])
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(LOGC[tr]), y[tr])
        return lambda te: clf.predict_proba(sc.transform(LOGC[te]))[:, 1]
    return bf

def make_smooth(y, gamma=0.25, ncomp=200, seed=0):
    """SMOOTH field: Nystroem RBF kernel over standardized coords + contact (genuinely smooth)."""
    feat = np.hstack([LOGC, CONT])
    def bf(tr, kk):
        sc = StandardScaler().fit(feat[tr])
        ny = Nystroem(kernel='rbf', gamma=gamma, n_components=ncomp,
                      random_state=seed).fit(sc.transform(feat[tr]))
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(ny.transform(sc.transform(feat[tr])), y[tr])
        return lambda te: clf.predict_proba(ny.transform(sc.transform(feat[te])))[:, 1]
    return bf

def make_linear_coords(y):
    """linear (non-kernel) coords + contact — smoothest possible baseline."""
    feat = np.hstack([LOGC, CONT])
    def bf(tr, kk):
        sc = StandardScaler().fit(feat[tr])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(feat[tr]), y[tr])
        return lambda te: clf.predict_proba(sc.transform(feat[te]))[:, 1]
    return bf

# CATEGORICAL baseline: discrete per-neuron cluster (KMeans on neuron coords) -> block model.
# per-neuron coordinate vector for clustering (birth, xyz-pos, extent, lineage-depth)
neu_coord = {}
for n in neurons:
    depth = len(lpath[n]) if n in lpath else np.nan
    neu_coord[n] = np.array([birth[n], pos[n][0], pos[n][1], pos[n][2], extent[n],
                             depth if not np.isnan(depth) else 0.0])
NCarr = np.array([neu_coord[n] for n in neurons])
NCidx = {n: i for i, n in enumerate(neurons)}

def make_categorical(y, Kc=8, ncomp=200, seed=0):
    """CATEGORICAL / address-code baseline: KMeans -> block (cluster_i x cluster_j) one-hot + contact.
    Kc^2 block params matched roughly to smooth ncomp (Kc=8 -> 64; also class-block version below)."""
    def bf(tr, kk):
        # cluster only on training neurons (leakage-safe), assign test by nearest centroid
        trneu = sorted(set(a for i, (a, b) in enumerate(pairs) if tr[i]) |
                       set(b for i, (a, b) in enumerate(pairs) if tr[i]))
        sc = StandardScaler().fit(np.array([neu_coord[n] for n in trneu]))
        km = KMeans(Kc, n_init=5, random_state=seed).fit(sc.transform(np.array([neu_coord[n] for n in trneu])))
        def cl(n): return int(km.predict(sc.transform(neu_coord[n].reshape(1, -1)))[0])
        clmap = {n: cl(n) for n in neurons}
        def blockfeat(idx):
            B = np.zeros((idx.sum(), Kc * Kc))
            sub = [(a, b) for i, (a, b) in enumerate(pairs) if idx[i]]
            for r, (a, b) in enumerate(sub):
                B[r, clmap[a] * Kc + clmap[b]] = 1.0
            return B
        Btr = np.hstack([LOGC[tr], blockfeat(tr)])
        scb = StandardScaler(with_mean=False).fit(Btr)
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(scb.transform(Btr), y[tr])
        def pred(te):
            Bte = np.hstack([LOGC[te], blockfeat(te)])
            return clf.predict_proba(scb.transform(Bte))[:, 1]
        return pred
    return bf

# functional cell-class address baseline (name -> class, e.g. AVAL/AVAR -> AVA)
import re
def cell_class(n):
    m = re.match(r'^([A-Z0-9]+?)[LR]?$', n)
    base = re.sub(r'[LR]?[0-9]*$', '', n)
    return base if base else n
classes = sorted(set(cell_class(n) for n in neurons))
cidx = {c: i for i, c in enumerate(classes)}
print(f"functional cell-classes: {len(classes)}")

# ---------------- run all models on y_any + per-animal ----------------
def dispersion(p, ya, yb, mask):
    obs = np.abs(ya[mask] - yb[mask]).sum()
    pred = (2 * p[mask] * (1 - p[mask])).sum()
    return obs / pred, obs, pred

def eval_model(name, builder_any, builder7=None, builder8=None):
    p_any = ndcv_predict(builder_any(y_any), y_any)
    m = ~np.isnan(p_any)
    auc = roc_auc_score(y_any[m], p_any[m])
    brier = brier_score_loss(y_any[m], p_any[m])
    # dispersion: predict each animal separately, average propensity
    b7 = builder7(w7) if builder7 else builder_any(w7)
    b8 = builder8(w8) if builder8 else builder_any(w8)
    p7 = ndcv_predict(b7, w7); p8 = ndcv_predict(b8, w8)
    pavg = np.nanmean(np.vstack([p7, p8]), 0); md = ~np.isnan(pavg)
    ratio, obs, pred = dispersion(pavg, w7, w8, md)
    print(f"  {name:28s} AUC {auc:.3f} Brier {brier:.4f} dispersion {ratio:.3f} (n={m.sum()})")
    return dict(name=name, auc=auc, brier=brier, dispersion=ratio, p_any=p_any, mask=m)

print("\n=== MODEL COMPARISON (neuron-disjoint CV) ===")
res = {}
res['contact_only'] = eval_model('contact-only', lambda y: make_contact_only(y))
res['linear_coords'] = eval_model('linear coords+contact', lambda y: make_linear_coords(y))
res['smooth'] = eval_model('SMOOTH field (RBF g=.25)', lambda y: make_smooth(y, gamma=0.25))
res['categorical'] = eval_model('CATEGORICAL block (K=8)', lambda y: make_categorical(y, Kc=8))

# ---------------- SMOOTHNESS STATISTIC 1: RBF bandwidth sweep ----------------
print("\n=== SMOOTHNESS STAT 1: RBF bandwidth sweep (AUC vs gamma) ===")
print("  degenerate-small optimum -> lookup/discrete; moderate optimum -> smooth field")
gammas = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
sweep = []
for g in gammas:
    p = ndcv_predict(make_smooth(y_any, gamma=g), y_any)
    m = ~np.isnan(p); a = roc_auc_score(y_any[m], p[m])
    sweep.append((g, a)); print(f"  gamma={g:5.2f}  AUC={a:.3f}")
best_g = max(sweep, key=lambda t: t[1])[0]

# ---------------- SMOOTHNESS STATISTIC 2: categorical granularity sweep ----------------
print("\n=== SMOOTHNESS STAT 2: categorical K sweep (does finer discretization help?) ===")
catsweep = []
for K in [3, 5, 8, 12, 20]:
    p = ndcv_predict(make_categorical(y_any, Kc=K), y_any)
    m = ~np.isnan(p); a = roc_auc_score(y_any[m], p[m])
    catsweep.append((K, a)); print(f"  K={K:3d}  block AUC={a:.3f}")

# ---------------- SMOOTHNESS STATISTIC 3: Moran's I of neuron propensity over coord space ----------------
# per-neuron mean predicted propensity (from smooth field) vs coordinate-space kNN graph.
print("\n=== SMOOTHNESS STAT 3: Moran's I of per-neuron out-propensity over developmental coord space ===")
p_smooth = res['smooth']['p_any']
neu_prop = defaultdict(list)
for i, (a, b) in enumerate(pairs):
    if not np.isnan(p_smooth[i]):
        neu_prop[a].append(p_smooth[i])
prop = np.array([np.mean(neu_prop[n]) if neu_prop[n] else np.nan for n in neurons])
valid = ~np.isnan(prop)
# also observed out-degree rate as a model-free propensity
obs_prop = np.array([np.mean([y_any[i] for i, (a, b) in enumerate(pairs) if a == n]) for n in neurons])
scN = StandardScaler().fit_transform(NCarr[:, [0, 1, 2, 3, 4, 5]])  # birth,xyz,extent,depth
def morans_I(vals, coords, kk=8):
    from scipy.spatial import cKDTree
    v = vals.copy(); ok = ~np.isnan(v)
    v = v[ok]; co = coords[ok]
    tree = cKDTree(co)
    n = len(v); z = v - v.mean()
    W = 0.0; num = 0.0
    d, idx = tree.query(co, k=kk + 1)
    for i in range(n):
        for j in idx[i, 1:]:
            num += z[i] * z[j]; W += 1
    den = (z ** 2).sum()
    I = (n / W) * (num / den)
    # permutation null
    perm = []
    rng = np.random.default_rng(1)
    for _ in range(500):
        zp = rng.permutation(z); nu = 0.0
        for i in range(n):
            for j in idx[i, 1:]:
                nu += zp[i] * zp[j]
        perm.append((n / W) * (nu / den))
    perm = np.array(perm)
    p = (np.sum(perm >= I) + 1) / (len(perm) + 1)
    return I, perm.mean(), perm.std(), p
I_s, mu_s, sd_s, p_s = morans_I(prop, scN)
I_o, mu_o, sd_o, p_o = morans_I(obs_prop, scN)
print(f"  smooth-field propensity: Moran's I={I_s:.3f} null {mu_s:.3f}+-{sd_s:.3f}  p={p_s:.4f}")
print(f"  observed out-degree rate: Moran's I={I_o:.3f} null {mu_o:.3f}+-{sd_o:.3f}  p={p_o:.4f}")

# ---------------- SMOOTHNESS STAT 4: multi-seed sign stability of smooth-vs-categorical ----------------
print("\n=== SMOOTHNESS STAT 4: multi-seed smooth(g=0.1) vs categorical(K=8,K=12) vs contact ===")
seed_rows = []
for sd in range(5):
    pc = ndcv_predict(make_contact_only(y_any), y_any, seed=sd); mc = ~np.isnan(pc)
    ps = ndcv_predict(make_smooth(y_any, gamma=0.1, seed=sd), y_any, seed=sd)
    pk8 = ndcv_predict(make_categorical(y_any, Kc=8, seed=sd), y_any, seed=sd)
    pk12 = ndcv_predict(make_categorical(y_any, Kc=12, seed=sd), y_any, seed=sd)
    ac = roc_auc_score(y_any[mc], pc[mc]); asm = roc_auc_score(y_any[mc], ps[mc])
    ak8 = roc_auc_score(y_any[mc], pk8[mc]); ak12 = roc_auc_score(y_any[mc], pk12[mc])
    seed_rows.append((sd, ac, asm, ak8, ak12))
    print(f"  seed{sd}: contact {ac:.3f} smooth {asm:.3f} catK8 {ak8:.3f} catK12 {ak12:.3f} "
          f"| smooth-catbest {asm-max(ak8,ak12):+.3f}")
sr = np.array([r[1:] for r in seed_rows])
print(f"  MEAN: contact {sr[:,0].mean():.3f} smooth {sr[:,1].mean():.3f} "
      f"catK8 {sr[:,2].mean():.3f} catK12 {sr[:,3].mean():.3f}")
print(f"  smooth - best-categorical (per seed): {(sr[:,1]-sr[:,2:].max(1))}")

# ---------------- save per-pair field table ----------------
os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/part3_field.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["pre", "post", "contact_area", "birth_diff", "soma_dist", "overlap",
                "lineage_dist", "w7", "w8", "theta_adult",
                "p_contact_only", "p_smooth_field", "p_categorical"])
    for i, (a, b) in enumerate(pairs):
        w.writerow([a, b, round(contact[i], 1),
                    round(X[i, FEATNAMES.index('birth_diff')], 1),
                    round(X[i, FEATNAMES.index('soma_dist')], 1),
                    round(X[i, FEATNAMES.index('overlap')], 3),
                    round(X[i, li], 1), w7[i], w8[i], round((w7[i] + w8[i]) / 2, 2),
                    _r(res['contact_only']['p_any'][i]),
                    _r(res['smooth']['p_any'][i]),
                    _r(res['categorical']['p_any'][i])])
print(f"\nfield table -> {OUT}/part3_field.csv")

# ---------------- summary verdict numbers ----------------
dA = res['smooth']['auc'] - res['categorical']['auc']
print("\n=== SUMMARY ===")
print(f"  contact-only:  AUC {res['contact_only']['auc']:.3f} Brier {res['contact_only']['brier']:.4f} disp {res['contact_only']['dispersion']:.3f}")
print(f"  SMOOTH field:  AUC {res['smooth']['auc']:.3f} Brier {res['smooth']['brier']:.4f} disp {res['smooth']['dispersion']:.3f}")
print(f"  CATEGORICAL:   AUC {res['categorical']['auc']:.3f} Brier {res['categorical']['brier']:.4f} disp {res['categorical']['dispersion']:.3f}")
print(f"  SMOOTH - CATEGORICAL dAUC = {dA:+.4f}  (best RBF gamma={best_g})")
print(f"  Brier improvement field vs contact = {res['contact_only']['brier']-res['smooth']['brier']:+.4f}")

with open(f"{OUT}/part3_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "AUC", "Brier", "dispersion_ratio"])
    for k in ['contact_only', 'linear_coords', 'smooth', 'categorical']:
        r = res[k]; w.writerow([r['name'], round(r['auc'], 4), round(r['brier'], 4), round(r['dispersion'], 4)])
    w.writerow([])
    w.writerow(["rbf_gamma", "AUC"])
    for g, a in sweep: w.writerow([g, round(a, 4)])
    w.writerow([])
    w.writerow(["categorical_K", "AUC"])
    for K, a in catsweep: w.writerow([K, round(a, 4)])
    w.writerow([])
    w.writerow(["moran_target", "I", "null_mean", "null_sd", "p"])
    w.writerow(["smooth_field_propensity", round(I_s, 4), round(mu_s, 4), round(sd_s, 4), round(p_s, 4)])
    w.writerow(["observed_outdegree", round(I_o, 4), round(mu_o, 4), round(sd_o, 4), round(p_o, 4)])
print(f"summary -> {OUT}/part3_summary.csv")
