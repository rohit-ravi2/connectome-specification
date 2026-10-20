"""GENERATOR D — MINIMAL SUFFICIENT STATISTIC / compression benchmark.
Low-rank (G)RDPG embedding of the CONTACT adjacency. This is the honest control:
it IS allowed to see contact. It sets the bar A/B/C must beat at matched params.
Params = rank k. Out-of-sample (neuron-disjoint) placement so gen_auc isn't memorization.
"""
import sys; sys.path.insert(0, '../accounting')
from bulk_benchmark import *
import numpy as np
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0)

def grdpg_embed(A, k):
    """GRDPG adjacency spectral embedding: top-k by |eigenvalue|.
    Returns Y (n x k) latent positions and S (k,) sign vector.
    Reconstruction M = Y diag(S) Y^T."""
    w, V = np.linalg.eigh(A)                 # ascending eigenvalues
    order = np.argsort(-np.abs(w))[:k]       # top-k by magnitude
    lam = w[order]; U = V[:, order]
    S = np.sign(lam)
    Y = U * np.sqrt(np.abs(lam))[None, :]    # n x k
    return Y, S

def insample_prop(k):
    """Full in-sample rank-k reconstruction propensity over all PAIRS."""
    Y, S = grdpg_embed(CADJ, k)
    Mrec = (Y * S[None, :]) @ Y.T
    return Mrec[PI, PJ]

def oos_prop(k):
    """Neuron-disjoint out-of-sample propensity. For each fold f, embed train
    (other folds) neurons, place fold-f neurons via their train-contact profile,
    then score within-fold-f pairs in the SAME train basis. Cross-fold pairs
    filled with in-sample reconstruction (not used by gen_auc)."""
    prop = insample_prop(k).copy()           # baseline for cross-fold pairs
    for f in range(5):
        te = np.where(FOLD == f)[0]           # test neuron indices
        tr = np.where(FOLD != f)[0]           # train neuron indices
        if len(te) < 2:
            continue
        A_tr = CADJ[np.ix_(tr, tr)]
        Ytr, S = grdpg_embed(A_tr, k)         # train basis
        # OOS placement: predicted contact t->train j = y_t . S . Ytr[j]
        # B = Ytr diag(S)  (n_tr x k); observed a_t = CADJ[te, tr]
        B = Ytr * S[None, :]
        Bpinv = np.linalg.pinv(B)             # k x n_tr
        A_te_tr = CADJ[np.ix_(te, tr)]        # n_te x n_tr observed contacts
        Yte = (Bpinv @ A_te_tr.T).T           # n_te x k  latent for test neurons
        # fill within-fold-f pairs with test-basis inner products
        pos = {n: i for i, n in enumerate(te)}
        for pk in np.where(PFOLD == f)[0]:
            i, j = PI[pk], PJ[pk]
            prop[pk] = float(Yte[pos[i]] @ (S * Yte[pos[j]]))
    return prop

print("=== GENERATOR D: low-rank contact-embedding compression benchmark ===")
ranks = [1, 2, 3, 4, 6, 8, 12, 16]
results = {}
for k in ranks:
    p_oos = oos_prop(k)
    d = score(p_oos, n_params=k, name=f'D-embed-rank{k}')
    # in-sample reference
    p_in = insample_prop(k)
    d['full_insample_auc'] = round(roc_auc_score(CONTACT, p_in), 3)
    d['cook_insample_auc'] = round(roc_auc_score(YC, p_in), 3)
    results[k] = d

print("\n=== SPATIAL MINIMAL MODEL (raw soma distance, 1-2 params) ===")
# soma distance from POS; some neurons may be NaN
d_ij = np.linalg.norm(POS[PI] - POS[PJ], axis=1)
valid = ~np.isnan(d_ij)
print(f"valid soma-distance pairs: {valid.sum()}/{len(d_ij)} (NaN pos -> filled median)")
d_fill = np.where(valid, d_ij, np.nanmedian(d_ij))
# 1-param: propensity = -distance (closer -> more contact)
score(-d_fill, n_params=1, name='D-spatial-negdist')
# 2-param: gaussian/exponential kernel exp(-d/tau); pick tau maximizing full AUC (still ~1 shape param)
best = None
for tau in np.linspace(d_fill[valid].std()*0.1, d_fill[valid].std()*3, 40):
    a = roc_auc_score(CONTACT, np.exp(-d_fill / tau))
    if best is None or a > best[1]:
        best = (tau, a)
tau = best[0]
score(np.exp(-d_fill / tau), n_params=2, name=f'D-spatial-expkernel(tau={tau:.0f})')

print("\n=== SUMMARY: AUC-vs-rank curve ===")
print(f"{'rank':>4} {'gen_nd':>7} {'margin':>7} {'cook':>6} {'full_in':>8} {'cook_in':>8}")
for k in ranks:
    d = results[k]
    print(f"{k:>4} {d['gen_auc_neurondisjoint']:>7} {d['margin_over_degree']:>7} "
          f"{d['cook_synapse_auc']:>6} {d['full_insample_auc']:>8} {d['cook_insample_auc']:>8}")
print(f"\ndegree-null neuron-disjoint AUC (the bar to clear) = {DEGNULL_ND:.3f}")
