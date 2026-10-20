#!/usr/bin/env python
"""Part C1 — Directionality / compartment polarity via ACTUAL synapse positions.

QUESTION: Does SUBCELLULAR position (pre-distal / post-proximal compartment polarity)
explain directed chemical synapse choice among high-contact pairs, beyond cell-class identity?

Prior: contact_decomposition Part D found pre sites more DISTAL than post (polarity p=3.5e-9);
self_avoidance C1 found direction predictable from cell identity at degree-free AUC 0.75.
Here we use the ACTUAL per-synapse positions and test the INCREMENTAL value of position/polarity
beyond identity, under neuron-disjoint CV.
"""
import json, numpy as np, pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

RNG = np.random.RandomState(20260702)
BASE = 'data/processed'
WITV = '../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv'

# ---------------------------------------------------------------- load
co = pd.read_csv(f'{BASE}/self_avoidance_apposition/high_contact_residual_master.csv')
sp = pd.read_csv(f'{BASE}/subcellular_tiling/synapse_position_table.csv')  # D8 per pre->post
sk = json.load(open(f'{BASE}/subcellular_tiling/_skeleton_coords.json'))
ps = json.load(open(f'{BASE}/subcellular_tiling/_presyn_sites.json'))
w = pd.read_csv(WITV)

# adult = witvliet ds7 + ds8
ad = w[w.source_file.isin(['witvliet_2020_7.xlsx', 'witvliet_2020_8.xlsx'])]
chem = ad[ad.type == 'chemical'].groupby(['pre', 'post']).synapses.sum()
elec = ad[ad.type == 'electrical'].groupby(['pre', 'post']).synapses.sum()
CHEM = chem.to_dict()
GAP = set()
for (a, b), s in elec.items():
    GAP.add(frozenset((a, b)))

# neuron -> NT / modality (from both endpoints of cohort)
NT, MOD = {}, {}
for r in co.itertuples():
    NT.setdefault(r.na, r.nt_a); NT.setdefault(r.nb, r.nt_b)
    MOD.setdefault(r.na, r.mod_a); MOD.setdefault(r.nb, r.mod_b)
def nt_of(_co, n):
    return NT.get(n, 'Unk')
def co_mod(_co, n):
    return MOD.get(n, 'inter')

# synaptic degrees (adult chemical) + gap degree
chem_out, chem_in, gap_deg = {}, {}, {}
for (a, b), s in CHEM.items():
    chem_out[a] = chem_out.get(a, 0) + 1
    chem_in[b] = chem_in.get(b, 0) + 1
for fs in GAP:
    for n in fs:
        gap_deg[n] = gap_deg.get(n, 0) + 1

# ---------------------------------------------------------------- skeleton / relpos machinery
SKTREE, SKDR, MAXDR = {}, {}, {}
for n, v in sk.items():
    if len(v['dr']) == 0:
        continue
    xyz = np.asarray(v['xyz'], float)
    dr = np.asarray(v['dr'], float)
    SKTREE[n] = cKDTree(xyz)
    SKDR[n] = dr
    MAXDR[n] = dr.max() if dr.max() > 0 else 1.0

def relpos_at(neuron, xyz):
    """relative dist_to_root (0=proximal/soma, 1=distal) of neuron's nearest node to xyz."""
    if neuron not in SKTREE:
        return np.nan
    d, i = SKTREE[neuron].query(xyz)
    return SKDR[neuron][i] / MAXDR[neuron]

# per-neuron OUTPUT DOMAIN (relpos of all its presynaptic sites, from presyn_sites.json)
out_dom_mean, out_dom_std = {}, {}
for n, sites in ps.items():
    if n not in MAXDR:
        continue
    rp = [s[4] / MAXDR[n] for s in sites]  # s = [partner,x,y,z,dist_root]
    if rp:
        out_dom_mean[n] = float(np.mean(rp))
        out_dom_std[n] = float(np.std(rp))

# ---------------------------------------------------------------- C1.1 directional pair table
def category(a, b):
    ab = CHEM.get((a, b), 0)
    ba = CHEM.get((b, a), 0)
    g = frozenset((a, b)) in GAP
    if ab > 0 and ba > 0:
        return 'reciprocal', ab, ba, g
    if ab > 0:
        return 'forward', ab, ba, g
    if ba > 0:
        return 'backward', ab, ba, g
    if g:
        return 'gap_only', ab, ba, g
    return 'nonsyn', ab, ba, g

rows = []
for r in co.itertuples():
    cat, ab, ba, g = category(r.na, r.nb)
    rows.append(dict(na=r.na, nb=r.nb, cat=cat, n_ab=ab, n_ba=ba, has_gap=int(g),
                     c_adult=r.c_adult, nt_a=r.nt_a, nt_b=r.nt_b, mod_a=r.mod_a, mod_b=r.mod_b,
                     same_nt=r.same_nt, same_mod=r.same_mod))
P = pd.DataFrame(rows)
c11 = P.cat.value_counts().to_dict()
print('C1.1 categories:', c11)

# ---------------------------------------------------------------- C1.2 per-synapse polarity (pre vs post relpos)
# For each D8 directed synapse: pre bouton relpos on pre arbor; post apposed-node relpos on post arbor.
pol = []
for r in sp.itertuples():
    xyz = np.array([r.pre_x, r.pre_y, r.pre_z])
    rpre = r.pre_dist_root / MAXDR.get(r.pre, np.nan) if r.pre in MAXDR else np.nan
    rpost = relpos_at(r.post, xyz)
    pol.append((r.pre, r.post, rpre, rpost))
POL = pd.DataFrame(pol, columns=['pre', 'post', 'r_pre', 'r_post']).dropna()
# per-neuron: as-pre mean relpos vs as-post mean relpos (identity-controlled paired test, reproduce Part D)
aspre = POL.groupby('pre').r_pre.mean()
aspost = POL.groupby('post').r_post.mean()
both = sorted(set(aspre.index) & set(aspost.index))
pre_v = aspre.loc[both].values
post_v = aspost.loc[both].values
w_stat, w_p = stats.wilcoxon(pre_v, post_v)
print(f'C1.2 polarity (n={len(both)} neurons): as-pre relpos={pre_v.mean():.3f} '
      f'as-post relpos={post_v.mean():.3f}  paired Wilcoxon p={w_p:.2e}')
# per-synapse level
syn_pre_gt_post = float((POL.r_pre > POL.r_post).mean())
sw = stats.wilcoxon(POL.r_pre, POL.r_post)
print(f'C1.2 per-synapse: r_pre>r_post in {syn_pre_gt_post:.3f} of {len(POL)} synapses; '
      f'medians pre={POL.r_pre.median():.3f} post={POL.r_post.median():.3f} p={sw.pvalue:.2e}')

# ---------------------------------------------------------------- C1.3 DIRECTION PREDICTION
# unidirectional pairs, with D8 synapse position available in the observed direction.
# Build per-directed-pair mean site relpos for BOTH endpoints at the observed synapse location.
# For pre->post pair: r_pre_at_site (pre arbor), r_post_at_site (post arbor).
site = POL.groupby(['pre', 'post']).agg(r_pre=('r_pre', 'mean'), r_post=('r_post', 'mean')).reset_index()
SITE = {(r.pre, r.post): (r.r_pre, r.r_post) for r in site.itertuples()}

MODRANK = {'sensory': 0, 'inter': 1, 'command': 2}
def mod_flags(m):
    return dict(sens=int(m == 'sensory'), cmd=int(m == 'command'), modrank=MODRANK.get(m, 1))

uni = P[P.cat.isin(['forward', 'backward'])].copy()
recs = []
for r in uni.itertuples():
    a, b = r.na, r.nb
    fwd = r.cat == 'forward'          # forward => a is presynaptic
    pre_n, post_n = (a, b) if fwd else (b, a)
    if (pre_n, post_n) not in SITE:
        continue                       # need D8 position for the observed synapse
    if pre_n not in MAXDR or post_n not in MAXDR:
        continue
    rpre, rpost = SITE[(pre_n, post_n)]
    # randomize orientation N1/N2 so label is not tied to alphabetical order
    if RNG.rand() < 0.5:
        n1, n2, y = pre_n, post_n, 1   # N1 is pre
        r1_site, r2_site = rpre, rpost
    else:
        n1, n2, y = post_n, pre_n, 0   # N1 is post
        r1_site, r2_site = rpost, rpre
    fa, fb = mod_flags(co_mod(co, n1)), mod_flags(co_mod(co, n2))
    recs.append(dict(
        n1=n1, n2=n2, y=y,
        # ---- POSITION / POLARITY features (antisymmetric N1-N2) ----
        d_relpos_site=r1_site - r2_site,                       # core: who is more distal at contact
        d_out_domain=out_dom_mean.get(n1, np.nan) - out_dom_mean.get(n2, np.nan),
        # ---- IDENTITY degree-free ----
        d_sens=fa['sens'] - fb['sens'],
        d_cmd=fa['cmd'] - fb['cmd'],
        d_modrank=fa['modrank'] - fb['modrank'],
        d_mono=int(nt_of(co, n1) in ('DA', '5HT', 'OA', 'TA')) - int(nt_of(co, n2) in ('DA', '5HT', 'OA', 'TA')),
        # ---- IDENTITY degree ----
        d_chem_out=chem_out.get(n1, 0) - chem_out.get(n2, 0),
        d_chem_in=chem_in.get(n1, 0) - chem_in.get(n2, 0),
        d_gap_deg=gap_deg.get(n1, 0) - gap_deg.get(n2, 0),
    ))
D = pd.DataFrame(recs).dropna(subset=['d_relpos_site'])
D['d_out_domain'] = D['d_out_domain'].fillna(0.0)  # missing output-domain -> neutral difference
print(f'C1.3 direction set: {len(D)} unidirectional pairs with D8 position')

# neuron-disjoint 5-fold: both endpoints of a test pair held out of training
neurons = sorted(set(D.n1) | set(D.n2))
fold = {n: RNG.randint(5) for n in neurons}
D['f1'] = D.n1.map(fold); D['f2'] = D.n2.map(fold)

FSETS = {
    'ID_degfree': ['d_sens', 'd_cmd', 'd_modrank', 'd_mono'],
    'ID_full':    ['d_sens', 'd_cmd', 'd_modrank', 'd_mono', 'd_chem_out', 'd_chem_in', 'd_gap_deg'],
    'POS_only':   ['d_relpos_site', 'd_out_domain'],
    'POS_site_only': ['d_relpos_site'],
    # site-only (genuine subcellular polarity, NOT identity) added to identity:
    'ID_degfree+site': ['d_sens', 'd_cmd', 'd_modrank', 'd_mono', 'd_relpos_site'],
    'ID_full+site': ['d_sens', 'd_cmd', 'd_modrank', 'd_mono', 'd_chem_out', 'd_chem_in', 'd_gap_deg', 'd_relpos_site'],
    # full position (site + out-domain, out-domain is partly identity/source-ness):
    'ID_degfree+POS': ['d_sens', 'd_cmd', 'd_modrank', 'd_mono', 'd_relpos_site', 'd_out_domain'],
    'ID_full+POS': ['d_sens', 'd_cmd', 'd_modrank', 'd_mono', 'd_chem_out', 'd_chem_in', 'd_gap_deg', 'd_relpos_site', 'd_out_domain'],
}

def cv_auc(feats, n_rep=1):
    aucs = []
    for k in range(5):
        te = D[(D.f1 == k) & (D.f2 == k)]
        tr = D[(D.f1 != k) & (D.f2 != k)]
        if len(te) < 8 or te.y.nunique() < 2:
            continue
        Xtr = tr[feats].values; Xte = te[feats].values
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(sc.transform(Xtr), tr.y)
        p = clf.predict_proba(sc.transform(Xte))[:, 1]
        aucs.append(roc_auc_score(te.y, p))
    return np.mean(aucs), np.std(aucs), len(aucs)

# single-feature (no training): more-distal-at-contact = presynaptic
single_auc = roc_auc_score(D.y, D.d_relpos_site)
# permutation null for the standalone polarity direction signal
perm = np.array([roc_auc_score(RNG.permutation(D.y.values), D.d_relpos_site.values) for _ in range(2000)])
single_p = float((np.abs(perm - 0.5) >= abs(single_auc - 0.5)).mean())
print(f'C1.3 single fixed-rule (more distal at contact = pre): AUC={single_auc:.3f} (n={len(D)}) '
      f'perm p={single_p:.4f} null={perm.mean():.3f}±{perm.std():.3f}')

# likelihood-ratio test: does d_relpos_site add beyond full identity (pooled, in-sample)?
import statsmodels.api as smapi
def llf(feats):
    X = smapi.add_constant(StandardScaler().fit_transform(D[feats].values))
    return smapi.Logit(D.y.values, X).fit(disp=0).llf
base_feats = ['d_sens', 'd_cmd', 'd_modrank', 'd_mono', 'd_chem_out', 'd_chem_in', 'd_gap_deg']
ll0 = llf(base_feats); ll1 = llf(base_feats + ['d_relpos_site'])
lr = 2 * (ll1 - ll0); lr_p = float(stats.chi2.sf(lr, 1))
print(f'C1.3 LR test add d_relpos_site to ID_full: chi2={lr:.2f} p={lr_p:.2e}')

results = {}
for name, feats in FSETS.items():
    m, s, nf = cv_auc(feats)
    results[name] = (m, s, nf)
    print(f'  {name:18s} neuron-disjoint AUC = {m:.3f} ± {s:.3f}  (folds={nf})')

incr_site_degfree = results['ID_degfree+site'][0] - results['ID_degfree'][0]
incr_site_full = results['ID_full+site'][0] - results['ID_full'][0]
incr_degfree = results['ID_degfree+POS'][0] - results['ID_degfree'][0]
incr_full = results['ID_full+POS'][0] - results['ID_full'][0]
print(f'  INCREMENTAL site(pure subcellular) beyond ID_degfree = {incr_site_degfree:+.3f}')
print(f'  INCREMENTAL site(pure subcellular) beyond ID_full    = {incr_site_full:+.3f}')
print(f'  INCREMENTAL full-POS beyond ID_degfree = {incr_degfree:+.3f}')
print(f'  INCREMENTAL full-POS beyond ID_full    = {incr_full:+.3f}')

# ---------------------------------------------------------------- same-NT avoids reciprocity?
chempairs = P[P.cat.isin(['forward', 'backward', 'reciprocal'])].copy()
recip = (chempairs.cat == 'reciprocal').astype(int)
ct = pd.crosstab(chempairs.same_nt, recip)
# OR: recip among same_nt vs diff_nt
a = ct.loc[1, 1] if (1 in ct.index and 1 in ct.columns) else 0  # same_nt & recip
b = ct.loc[1, 0] if 1 in ct.index else 0
c = ct.loc[0, 1] if 0 in ct.index else 0
d = ct.loc[0, 0] if 0 in ct.index else 0
OR = (a * d) / (b * c) if b * c > 0 else np.nan
_, fp = stats.fisher_exact([[a, b], [c, d]])
recip_same = a / (a + b) if (a + b) else np.nan
recip_diff = c / (c + d) if (c + d) else np.nan
print(f'same-NT reciprocity: same_nt recip frac={recip_same:.3f} vs diff_nt={recip_diff:.3f} '
      f'OR={OR:.3f} Fisher p={fp:.3g}')

# ---------------------------------------------------------------- write master result csv
D_out = D.copy()
D_out['pred_single'] = (D_out.d_relpos_site > 0).astype(int)
D_out.to_csv(f'{BASE}/subcellular_tiling/partC1_directionality_polarity_results.csv', index=False)
print('wrote partC1_directionality_polarity_results.csv rows', len(D_out))

# stash summary for the doc
summary = dict(
    c11=c11,
    polarity_neuron=dict(n=len(both), aspre=float(pre_v.mean()), aspost=float(post_v.mean()), p=float(w_p)),
    polarity_syn=dict(n=len(POL), frac_pre_gt_post=syn_pre_gt_post,
                      med_pre=float(POL.r_pre.median()), med_post=float(POL.r_post.median()), p=float(sw.pvalue)),
    n_direction=len(D),
    single_rule_auc=float(single_auc),
    single_rule_perm_p=single_p,
    lr_site_beyond_idfull=dict(chi2=float(lr), p=lr_p),
    cv=results,
    incr_site_degfree=float(incr_site_degfree),
    incr_site_full=float(incr_site_full),
    incr_degfree=float(incr_degfree),
    incr_full=float(incr_full),
    recip=dict(OR=float(OR), p=float(fp), same=float(recip_same), diff=float(recip_diff),
               a=int(a), b=int(b), c=int(c), d=int(d)),
)
json.dump(summary, open(f'{BASE}/subcellular_tiling/_partC1_summary.json', 'w'), indent=2)
print('DONE')
