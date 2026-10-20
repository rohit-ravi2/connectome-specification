#!/usr/bin/env python3
"""PART 2 — GNCA AS DEVELOPMENTAL TRAJECTORY.

Repurpose a Graph Neural Cellular Automaton from a static edge-predictor into a
DEVELOPMENTAL TRAJECTORY / local growth rule on the growing contact hypergraph.
Train stage_t -> stage_{t+1} wiring state from stage_t state + LOCAL contact
neighbourhood (message passing over the per-stage contact graph). Then roll the
learned rule forward as a STOCHASTIC generative process many times and ask
whether the one-field canalization signatures EMERGE UN-FIT:
   (a) bimodal theta  (per-pair present-fraction concentrated near 0 and 1)
   (b) under-dispersion (observed cross-animal disagreement / predicted 2p(1-p) < 1),
       real reference ~0.47 (4 L1 replicates) / ~0.54 (W7/W8 contact+NT+topo model).

FIREWALL (verbatim discipline in this run):
 1. Score generative claim against DISPERSION RATIO (obs disagreement / 2p(1-p) of a
    NEURON-DISJOINT CV propensity) + held-out cross-animal BRIER. High reconstruction
    AUC is EXPLICITLY NOT success (prior GNCA r=0.987 was topology RE-DESCRIPTION in-sample).
 2. Observation-window control: NO stage/confirmation-count feature is scored. Canalization
    scored only via cross-animal reproducibility of the GENERATED ensemble.
 3. Non-circularity: rule inputs = per-stage CONTACT + node NT + contact-degree ONLY.
    Never the adult adjacency being predicted. Reconstruction of the connectome is READOUT.
 4. Predict-before-fit: pre-registration + numeric flip threshold printed at top.
 5. Label failed-with-current-data vs biologically-false.
"""
import csv, re, os, math, sys
import numpy as np
from collections import defaultdict

np.random.seed(0)
import torch
torch.manual_seed(0)

C = "../data/celegans/data"
ROOT = os.path.expanduser("../seed")
OUT = f"{ROOT}/data/processed/relational_propensity_grammar"
EDGES = f"{C}/connectome/witvliet2020/witvliet_combined_edges.csv"
MASTER = f"{ROOT}/data/processed/contact_decomposition/contact_decomposition_master.csv"
NPZ = "../data/tier5/connectome.npz"
os.makedirs(OUT, exist_ok=True)

# ============================================================ PRE-REGISTRATION
PREREG = """
=== PRE-REGISTERED PREDICTION (before any fit) ===
The GNCA growth rule is trained ONLY to predict per-stage edge presence from
per-stage contact + node context (message passing). Dispersion / theta-bimodality
are NEVER fit targets, so any match is emergent.

Prediction: contact dominates (all prior branches), so the rule reconstructs adult
wiring at AUC ~0.80-0.88 (NOT success). I predict theta-bimodality WILL emerge
(contact strongly separates edge/no-edge -> saturated p), but the DISPERSION RATIO
will NOT reach the real ~0.47 -- because under-dispersion is model-inadequacy of a
smooth CV propensity, and a contact+NT+topology CV model refit on GNCA output will
recover the GNCA's (contact-driven) sharpness -> ratio near 1. i.e. emergent
canalization FAILS on the hard criterion.

FLIP THRESHOLD (binding):
  SUCCESS  iff generated-ensemble dispersion ratio in [0.35, 0.60]  (brackets real 0.47)
           AND theta bimodality fraction (theta<0.1 or >0.9) >= 0.80
           BOTH arising un-fit.
  NEGATIVE otherwise. Ratio > 0.70 OR bimodality < 0.70 => rule RECONSTRUCTS but
           canalization does NOT emerge => READOUT not SPECIFIER.
"""
print(PREREG)

# ============================================================ DATA
# per-animal directed CHEMICAL edges
animal_edges = defaultdict(set)
for r in csv.DictReader(open(EDGES)):
    if r['type'] == 'electrical':
        continue
    a = int(re.search(r'witvliet_2020_(\d)', r['source_file']).group(1))
    animal_edges[a].add((r['pre'], r['post']))
L1_animals = [1, 2, 3, 4]
STAGE_ANIMALS = {'L1': [1, 2, 3, 4], 'L2': [5], 'L3': [6], 'adult': [7, 8]}
W7, W8 = animal_edges[7], animal_edges[8]

# undirected per-stage contact trajectory from master
import pandas as pd
m = pd.read_csv(MASTER)
CSTAGES = ['c_L1', 'c_L2', 'c_L3', 'c_adult']
# node universe = neurons in any contact pair
nodes = sorted(set(m.na) | set(m.nb))
N = len(nodes)
idx = {n: i for i, n in enumerate(nodes)}

# symmetric per-stage contact matrices (raw area) -> log1p later
Cmat = {s: np.zeros((N, N), dtype=np.float32) for s in ['L1', 'L2', 'L3', 'adult']}
for _, row in m.iterrows():
    i, j = idx[row.na], idx[row.nb]
    for s, col in zip(['L1', 'L2', 'L3', 'adult'], CSTAGES):
        v = float(row[col])
        if v > 0:
            Cmat[s][i, j] = v
            Cmat[s][j, i] = v

# NT from NPZ (authoritative), one-hot
d = np.load(NPZ, allow_pickle=True)
def ntn(x):
    for k, v in [("Acetyl", "ACh"), ("Glu", "Glu"), ("GABA", "GABA"),
                 ("Dopamine", "mono"), ("Serotonin", "mono"),
                 ("Octopamine", "mono"), ("Tyramine", "mono")]:
        if k in x:
            return v
    return "Unk"
NT = {a: ntn(str(b)) for a, b in zip([str(x) for x in d["names"]],
                                     [str(x) for x in d["nt_primary"]])}
NTCATS = ["ACh", "Glu", "GABA", "mono", "Unk"]
ntoh = np.zeros((N, len(NTCATS)), dtype=np.float32)
for n in nodes:
    ntoh[idx[n], NTCATS.index(NT.get(n, "Unk"))] = 1.0

# directed OPPORTUNITY pairs = both directions where the unordered pair is ever in contact
any_contact = (Cmat['L1'] + Cmat['L2'] + Cmat['L3'] + Cmat['adult'])
pairs = []
for i in range(N):
    for j in range(N):
        if i != j and any_contact[i, j] > 0:
            pairs.append((i, j))
pairs = np.array(pairs)
P = len(pairs)
pi, pj = pairs[:, 0], pairs[:, 1]
print(f"nodes={N}  directed opportunity pairs={P}")

# per-stage DIRECTED soft targets = fraction of that stage's animals with the edge
def stage_target(stage):
    animals = STAGE_ANIMALS[stage]
    y = np.zeros(P, dtype=np.float32)
    for k, (i, j) in enumerate(pairs):
        a, b = nodes[i], nodes[j]
        y[k] = sum(1 for an in animals if (a, b) in animal_edges[an]) / len(animals)
    return y
Y = {s: stage_target(s) for s in ['L1', 'L2', 'L3', 'adult']}
for s in ['L1', 'L2', 'L3', 'adult']:
    print(f"  stage {s}: mean edge presence over opp pairs = {Y[s].mean():.4f}")

# per-stage contact area on directed pairs (log1p), and contact-degree node feature per stage
logC = {}
degC = {}
for s in ['L1', 'L2', 'L3', 'adult']:
    logC[s] = np.log1p(Cmat[s][pi, pj]).astype(np.float32)
    degC[s] = np.log1p(Cmat[s].sum(1)).astype(np.float32)  # node contact degree

# ============================================================ GNCA MODEL
dev = 'cpu'
Cmat_t = {s: torch.tensor(Cmat[s], device=dev) for s in Cmat}
# row-normalized contact adjacency (message weights) per stage
Cnorm_t = {}
for s in Cmat_t:
    A = torch.log1p(Cmat_t[s])
    rs = A.sum(1, keepdim=True)
    rs[rs == 0] = 1.0
    Cnorm_t[s] = A / rs
ntoh_t = torch.tensor(ntoh, device=dev)
degC_t = {s: torch.tensor(degC[s], device=dev).unsqueeze(1) for s in degC}
pi_t = torch.tensor(pi, device=dev)
pj_t = torch.tensor(pj, device=dev)
logC_t = {s: torch.tensor(logC[s], device=dev).unsqueeze(1) for s in logC}

D = 24
K_MP = 2  # message-passing steps (locality radius)

class GNCA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        node_in = len(NTCATS) + 1  # NT one-hot + contact degree
        self.enc = torch.nn.Linear(node_in, D)
        self.msg = torch.nn.Linear(D, D)
        self.upd = torch.nn.Linear(2 * D, D)
        # edge readout: [h_a, h_b, logC_ab, prev_state] -> logit
        self.edge = torch.nn.Sequential(
            torch.nn.Linear(2 * D + 2, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))

    def node_states(self, stage):
        x = torch.cat([ntoh_t, degC_t[stage]], dim=1)
        h = torch.tanh(self.enc(x))
        A = Cnorm_t[stage]
        for _ in range(K_MP):
            msn = A @ torch.tanh(self.msg(h))
            h = torch.tanh(self.upd(torch.cat([h, msn], dim=1)))
        return h

    def edge_logits(self, stage, prev_state):
        h = self.node_states(stage)
        ha = h[pi_t]; hb = h[pj_t]
        feat = torch.cat([ha, hb, logC_t[stage], prev_state.unsqueeze(1)], dim=1)
        return self.edge(feat).squeeze(1)

model = GNCA().to(dev)
opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)

# training transitions: (target_stage, prev_stage_state).
# growth: empty -> L1 -> L2 -> L3 -> adult, teacher-forced prev = previous soft target
Yt = {s: torch.tensor(Y[s], device=dev) for s in Y}
zero = torch.zeros(P, device=dev)
transitions = [('L1', zero), ('L2', Yt['L1']), ('L3', Yt['L2']), ('adult', Yt['L3'])]
bce = torch.nn.BCEWithLogitsLoss()

EPOCHS = 400
for ep in range(EPOCHS):
    model.train(); opt.zero_grad(); loss = 0.0
    for stage, prev in transitions:
        logit = model.edge_logits(stage, prev)
        loss = loss + bce(logit, Yt[stage])
    loss.backward(); opt.step()
    if ep % 100 == 0 or ep == EPOCHS - 1:
        print(f"  epoch {ep:3d} loss {loss.item():.4f}")

# ============================================================ RECONSTRUCTION (READOUT, not success)
from sklearn.metrics import roc_auc_score, brier_score_loss
model.eval()
with torch.no_grad():
    p_adult = torch.sigmoid(model.edge_logits('adult', Yt['L3'])).cpu().numpy()
    p_L1 = torch.sigmoid(model.edge_logits('L1', zero)).cpu().numpy()
y_adult_any = np.array([1 if (nodes[i], nodes[j]) in W7 or (nodes[i], nodes[j]) in W8 else 0
                        for i, j in pairs])
auc_adult = roc_auc_score(y_adult_any, p_adult)
print(f"\n[READOUT — NOT SUCCESS] adult reconstruction AUC (teacher-forced) = {auc_adult:.3f}")

# ============================================================ STOCHASTIC ROLLOUT
def rollout(nsamp, temperature=1.0):
    """Roll the rule forward as a stochastic generative process. Returns per-stage
    binary ensembles [nsamp x P]. State carried forward (sampled), fully generative
    (no teacher forcing). Node context recomputed per stage from that stage's contact."""
    ens = {s: np.zeros((nsamp, P), dtype=np.int8) for s in ['L1', 'L2', 'L3', 'adult']}
    with torch.no_grad():
        for r in range(nsamp):
            prev = torch.zeros(P, device=dev)
            for stage in ['L1', 'L2', 'L3', 'adult']:
                logit = model.edge_logits(stage, prev)
                p = torch.sigmoid(logit / temperature)
                samp = (torch.rand(P, device=dev) < p).float()
                ens[stage][r] = samp.cpu().numpy().astype(np.int8)
                prev = samp
    return ens

NSAMP = 300
ens = rollout(NSAMP)
theta_gen_L1 = ens['L1'].mean(0)
theta_gen_ad = ens['adult'].mean(0)

def bimodal_frac(theta):
    return float(np.mean((theta < 0.1) | (theta > 0.9)))

print("\n=== EMERGENT theta (generated ensemble) ===")
print(f"  L1    theta bimodality frac (<0.1 | >0.9) = {bimodal_frac(theta_gen_L1):.3f}")
print(f"  adult theta bimodality frac              = {bimodal_frac(theta_gen_ad):.3f}")
# real L1 present-count over 4 animals -> bimodality
real_k = np.array([sum(1 for an in L1_animals if (nodes[i], nodes[j]) in animal_edges[an])
                   for i, j in pairs])
real_theta_L1 = real_k / 4.0
print(f"  REAL  L1 theta bimodality frac           = {bimodal_frac(real_theta_L1):.3f} "
      f"(k-hist {np.bincount(real_k, minlength=5).tolist()})")

# ============================================================ DISPERSION RATIO
# neuron-disjoint CV propensity model (contact+NT+topology) — SAME family as real pipeline.
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# static features for CV model (contact area adult + NT-pair + contact degrees + common-neighbors)
Bnb = (Cmat['adult'] > 0).astype(np.float32)
degn = Bnb.sum(1)
ntpairs = sorted(set((NT.get(nodes[i], 'Unk'), NT.get(nodes[j], 'Unk')) for i, j in pairs))
ntki = {k: t for t, k in enumerate(ntpairs)}
def cvfeats():
    F = []
    for i, j in pairs:
        cn = int(((Bnb[i] > 0) & (Bnb[j] > 0)).sum())
        oh = [0] * len(ntpairs); oh[ntki[(NT.get(nodes[i], 'Unk'), NT.get(nodes[j], 'Unk'))]] = 1
        F.append([np.log1p(Cmat['adult'][i, j]), cn, np.log1p(degn[i]), np.log1p(degn[j])] + oh)
    return np.array(F, np.float32)
Xcv = cvfeats()

neu = sorted(set(pi.tolist()) | set(pj.tolist()))
def ndcv_proba(y, seed=0):
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(neu), 5)
    nf = {n: f for f, ff in enumerate(folds) for n in ff}
    p = np.full(len(y), np.nan)
    for k in range(5):
        te = np.array([nf[i] == k and nf[j] == k for i, j in pairs])
        tr = np.array([nf[i] != k and nf[j] != k for i, j in pairs])
        if te.sum() < 10 or y[tr].sum() < 5:
            continue
        clf = make_pipeline(StandardScaler(with_mean=False),
                            GradientBoostingClassifier(random_state=seed, n_estimators=80))
        clf.fit(Xcv[tr], y[tr]); p[te] = clf.predict_proba(Xcv[te])[:, 1]
    return p

def dispersion_ratio(ya, yb, seed=0):
    """Mirror real: fit CV propensity on each of two animals, average, ratio."""
    pa = ndcv_proba(ya, seed); pb = ndcv_proba(yb, seed + 1)
    p = np.nanmean(np.vstack([pa, pb]), axis=0)
    mask = ~np.isnan(p)
    obs = np.abs(ya[mask] - yb[mask]).sum()
    pred = (2 * p[mask] * (1 - p[mask])).sum()
    return obs / pred, obs, pred, p, mask

# --- REAL baseline in THIS pipeline (apples-to-apples) ---
y7 = np.array([1 if (nodes[i], nodes[j]) in W7 else 0 for i, j in pairs])
y8 = np.array([1 if (nodes[i], nodes[j]) in W8 else 0 for i, j in pairs])
r_real, obs_r, pred_r, p_real, mreal = dispersion_ratio(y7, y8)
brier_real = brier_score_loss(y8[mreal], p_real[mreal])
print("\n=== DISPERSION RATIO (adult W7/W8) ===")
print(f"  REAL  (this pipeline): ratio={r_real:.3f}  obs={obs_r:.0f} pred={pred_r:.0f}  brier={brier_real:.3f}")

# --- REAL L1 4-replicate dispersion (mirror gate's 0.47) ---
def l1_multi_dispersion(labels_list, seed=0):
    # average pairwise ratio over the animal pairs, fit p per animal
    import itertools
    ratios = []
    ps = [ndcv_proba(l, seed + t) for t, l in enumerate(labels_list)]
    for (ta, ya), (tb, yb) in itertools.combinations(list(enumerate(labels_list)), 2):
        p = np.nanmean(np.vstack([ps[ta], ps[tb]]), axis=0); mask = ~np.isnan(p)
        obs = np.abs(ya[mask] - yb[mask]).sum(); pred = (2 * p[mask] * (1 - p[mask])).sum()
        ratios.append(obs / pred)
    return float(np.mean(ratios)), ratios
L1_labels = [np.array([1 if (nodes[i], nodes[j]) in animal_edges[an] else 0 for i, j in pairs])
             for an in L1_animals]
r_real_L1, _ = l1_multi_dispersion(L1_labels)
print(f"  REAL  L1 4-replicate mean pairwise ratio = {r_real_L1:.3f}  (gate reference ~0.47)")

# --- GENERATED ensemble dispersion (mirror exactly: two synthetic animals) ---
def gen_dispersion(stage, npairs_of_animals=8, seed=0):
    rng = np.random.default_rng(seed)
    E = ens[stage]
    ratios = []; briers = []
    for t in range(npairs_of_animals):
        a, b = rng.choice(NSAMP, 2, replace=False)
        ya, yb = E[a].astype(int), E[b].astype(int)
        if ya.sum() < 5 or yb.sum() < 5:
            continue
        r, obs, pred, p, mask = dispersion_ratio(ya, yb, seed=t)
        ratios.append(r); briers.append(brier_score_loss(yb[mask], p[mask]))
    return float(np.mean(ratios)), float(np.std(ratios)), float(np.mean(briers)), ratios

r_gen_ad, sd_gen_ad, brier_gen_ad, ratios_ad = gen_dispersion('adult', 8)
r_gen_L1, sd_gen_L1, brier_gen_L1, ratios_L1 = gen_dispersion('L1', 8)
print("\n=== EMERGENT DISPERSION RATIO (GNCA rollout, un-fit) ===")
print(f"  GENERATED adult: ratio={r_gen_ad:.3f} +/- {sd_gen_ad:.3f}  brier={brier_gen_ad:.3f}")
print(f"  GENERATED L1   : ratio={r_gen_L1:.3f} +/- {sd_gen_L1:.3f}  brier={brier_gen_L1:.3f}")

# --- control: TRUE generative dispersion (should be ~1.0 by construction: independent Bernoulli) ---
def true_gen_dispersion(stage):
    E = ens[stage]; theta = E.mean(0)
    # expected pairwise disagreement of two draws vs 2 theta(1-theta): identity => ratio 1
    obs = np.mean([np.abs(E[a] - E[b]).mean() for a, b in
                   zip(np.random.default_rng(1).integers(0, NSAMP, 30),
                       np.random.default_rng(2).integers(0, NSAMP, 30))])
    pred = np.mean(2 * theta * (1 - theta))
    return obs / pred
print(f"\n[control] TRUE generative dispersion (obs vs 2theta(1-theta), independent) "
      f"adult={true_gen_dispersion('adult'):.3f} (must be ~1.0: confirms under-dispersion,"
      f" if any, is CV-model inadequacy not process correlation)")

# ============================================================ VERDICT
success = (0.35 <= r_gen_L1 <= 0.60) and (bimodal_frac(theta_gen_L1) >= 0.80)
verdict = "SUCCESS (emergent canalization)" if success else "NEGATIVE (reconstructs, no emergent canalization)"
print("\n" + "=" * 70)
print(f"VERDICT: {verdict}")
print(f"  criterion: L1 gen ratio in [0.35,0.60]? {0.35 <= r_gen_L1 <= 0.60} (={r_gen_L1:.3f})")
print(f"            L1 gen theta bimodality>=0.80? {bimodal_frac(theta_gen_L1)>=0.80} (={bimodal_frac(theta_gen_L1):.3f})")
print("=" * 70)

# ============================================================ SAVE CSV
rows = [
    ["metric", "value", "note"],
    ["prereg_flip_success", "ratio_L1 in [0.35,0.60] AND bimodal_L1>=0.80", "binding"],
    ["n_nodes", N, ""],
    ["n_directed_opp_pairs", P, ""],
    ["mp_steps", K_MP, "locality radius"],
    ["hidden_dim", D, ""],
    ["recon_adult_AUC", round(auc_adult, 4), "READOUT not success"],
    ["real_theta_L1_bimodal_frac", round(bimodal_frac(real_theta_L1), 4), "target ~0.90"],
    ["gen_theta_L1_bimodal_frac", round(bimodal_frac(theta_gen_L1), 4), "emergent"],
    ["gen_theta_adult_bimodal_frac", round(bimodal_frac(theta_gen_ad), 4), "emergent"],
    ["real_dispersion_W7W8_thispipe", round(r_real, 4), "apples-to-apples baseline"],
    ["real_dispersion_L1_4rep", round(r_real_L1, 4), "gate ref ~0.47"],
    ["gen_dispersion_adult", round(r_gen_ad, 4), "emergent, un-fit"],
    ["gen_dispersion_adult_sd", round(sd_gen_ad, 4), ""],
    ["gen_dispersion_L1", round(r_gen_L1, 4), "emergent, un-fit"],
    ["gen_dispersion_L1_sd", round(sd_gen_L1, 4), ""],
    ["gen_brier_adult", round(brier_gen_ad, 4), "held-out cross-animal"],
    ["gen_brier_L1", round(brier_gen_L1, 4), ""],
    ["real_brier_W8", round(brier_real, 4), ""],
    ["true_gen_dispersion_adult", round(true_gen_dispersion('adult'), 4), "~1.0 control"],
    ["bimodal_theta_emerges_unfit", "Y" if bimodal_frac(theta_gen_L1) >= 0.80 else "N", ""],
    ["dispersion_emerges_unfit", "Y" if 0.35 <= r_gen_L1 <= 0.60 else "N", ""],
    ["VERDICT", verdict, ""],
]
with open(f"{OUT}/part2_gnca.csv", "w", newline="") as f:
    csv.writer(f).writerows(rows)
print(f"\nsaved -> {OUT}/part2_gnca.csv")

# stash a few numbers for the doc
np.savez(f"{OUT}/part2_gnca_arrays.npz",
         theta_gen_L1=theta_gen_L1, theta_gen_ad=theta_gen_ad,
         real_theta_L1=real_theta_L1, ratios_ad=np.array(ratios_ad), ratios_L1=np.array(ratios_L1))
print("done.")
