"""
Part D: developmental geometry / co-growth analysis.
Hypothesis: early synapse establishment is driven by co-developmental opportunity
(birth-time proximity + lineage proximity), enriched AT FIXED contact topology.

Outputs to data/processed/developmental_window_expression/:
  partD_birth_lineage_gradients.csv   (D1)
  partD_contact_generation.csv        (D3)
  partD_cogrowth_model_results.csv    (D4/D5/D6)
Doc written separately.
"""
import numpy as np, pandas as pd, json, warnings
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
rng = np.random.default_rng(42)

OUT = 'data/processed/developmental_window_expression/'
MASTER = 'data/processed/upstream_early_establishment/upstream_master_table.csv'
TARGETS = ['T_L1_repro', 'T_preadult_persist', 'T_first_L1']
CONTACT = ['contact_area', 'contact_cos', 'contact_cn']

df = pd.read_csv(MASTER)
df['birth_diff'] = (df.pre_birth - df.post_birth).abs()
df['min_birth'] = df[['pre_birth', 'post_birth']].min(axis=1)   # later-born neuron gates availability
df['max_birth'] = df[['pre_birth', 'post_birth']].max(axis=1)
# lineage-derived co-dev features; impute lin_dist median + indicator
df['lin_missing'] = df.lin_dist.isna().astype(int)
df['lin_dist_i'] = df.lin_dist.fillna(df.lin_dist.median())
df['same_founder'] = df.same_founder.fillna(df.same_founder.mode()[0])
DEV = ['birth_diff', 'max_birth', 'lin_dist_i', 'same_founder', 'lin_missing']
NT_COLS = []
# NT one-hot (pre/post)
for c in ['pre_nt', 'post_nt']:
    d = pd.get_dummies(df[c], prefix=c)
    df = pd.concat([df, d], axis=1)
    NT_COLS += list(d.columns)

# ---- neuron-disjoint CV: partition neurons into K folds; test pairs = both endpoints in test block ----
def neuron_folds(df, K=5, seed=0):
    neurons = sorted(set(df.pre) | set(df.post))
    r = np.random.default_rng(seed)
    perm = r.permutation(neurons)
    fold_of = {n: i % K for i, n in enumerate(perm)}
    return fold_of

def disjoint_cv_auc(df, feat_cols, target, K=5, seed=0, model='logit', return_preds=False):
    """train on pairs w/ both endpoints in train neuron blocks; test on pairs w/ both in held-out block."""
    fold_of = neuron_folds(df, K, seed)
    pf = df.pre.map(fold_of).values; qf = df.post.map(fold_of).values
    X = df[feat_cols].astype(float).values; y = df[target].values
    aucs, aps, lls, all_yt, all_yp = [], [], [], [], []
    for k in range(K):
        test = (pf == k) & (qf == k)
        train = (pf != k) & (qf != k)
        if test.sum() < 8 or y[train].sum() < 3 or (y[train] == 0).sum() < 3:
            continue
        if len(np.unique(y[test])) < 2:
            continue
        sc = StandardScaler().fit(X[train])
        Xtr, Xte = sc.transform(X[train]), sc.transform(X[test])
        if model == 'logit':
            m = LogisticRegression(max_iter=2000, C=1.0)
        else:
            m = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=seed)
        m.fit(Xtr, y[train])
        p = m.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(y[test], p))
        aps.append(average_precision_score(y[test], p))
        lls.append(log_loss(y[test], p, labels=[0, 1]))
        all_yt.append(y[test]); all_yp.append(p)
    if return_preds:
        return (np.concatenate(all_yt), np.concatenate(all_yp)) if all_yt else (np.array([]), np.array([]))
    return np.mean(aucs), np.mean(aps), np.mean(lls), len(aucs)

# ==================== D1: birth/lineage gradient stratified by contact-area tercile ====================
print('=== D1 gradients ===')
df['area_tercile'] = pd.qcut(df.contact_area, 3, labels=['low', 'mid', 'high'])
grad_rows = []
# birth_diff bins (co-natal vs staggered) and lin_dist bins
df['bd_bin'] = pd.cut(df.birth_diff, [-1, 0, 40, 120, 1e5], labels=['0(co-natal)', '1-40', '41-120', '>120'])
df['ld_bin'] = pd.cut(df.lin_dist_i, [0, 12, 15, 18, 100], labels=['<=12', '13-15', '16-18', '>18'])
for ter in ['low', 'mid', 'high']:
    sub = df[df.area_tercile == ter]
    for feat, binc in [('birth_diff', 'bd_bin'), ('lin_dist', 'ld_bin')]:
        for b in sub[binc].cat.categories:
            s2 = sub[sub[binc] == b]
            if len(s2) < 10:
                continue
            row = {'test': 'D1', 'area_tercile': ter, 'feature': feat, 'bin': b, 'n': len(s2)}
            for t in TARGETS:
                row['P_' + t] = round(s2[t].mean(), 4)
            grad_rows.append(row)
# also overall gradient (unstratified) + spearman
from scipy.stats import spearmanr
for feat in ['birth_diff', 'lin_dist_i']:
    for t in TARGETS:
        rho, pv = spearmanr(df[feat], df[t], nan_policy='omit')
        grad_rows.append({'test': 'D1_spearman', 'area_tercile': 'all', 'feature': feat, 'bin': 'corr_vs_' + t,
                          'n': len(df), 'P_' + t: round(rho, 4), 'spearman_p': round(pv, 4)})
# partial: within-tercile spearman (fixed contact)
for ter in ['low', 'mid', 'high']:
    sub = df[df.area_tercile == ter]
    for feat in ['birth_diff', 'lin_dist_i']:
        for t in TARGETS:
            rho, pv = spearmanr(sub[feat], sub[t], nan_policy='omit')
            grad_rows.append({'test': 'D1_spearman_fixedcontact', 'area_tercile': ter, 'feature': feat,
                              'bin': 'corr_vs_' + t, 'n': len(sub), 'P_' + t: round(rho, 4), 'spearman_p': round(pv, 4)})
grad = pd.DataFrame(grad_rows)
grad.to_csv(OUT + 'partD_birth_lineage_gradients.csv', index=False)
print(grad.to_string())

# ==================== D3: contact generation — does dev predict contact topology? ====================
print('\n=== D3 contact generation (neuron-disjoint CV) ===')
def disjoint_cv_reg(df, feat_cols, target, K=5, seed=0):
    fold_of = neuron_folds(df, K, seed)
    pf = df.pre.map(fold_of).values; qf = df.post.map(fold_of).values
    X = df[feat_cols].astype(float).values; y = df[target].astype(float).values
    r2s = []
    from sklearn.metrics import r2_score
    for k in range(K):
        test = (pf == k) & (qf == k); train = (pf != k) & (qf != k)
        if test.sum() < 8 or train.sum() < 30:
            continue
        sc = StandardScaler().fit(X[train])
        m = Ridge(alpha=1.0).fit(sc.transform(X[train]), y[train])
        r2s.append(r2_score(y[test], m.predict(sc.transform(X[test]))))
    return np.mean(r2s), len(r2s)
cg_rows = []
dev_for_contact = ['birth_diff', 'max_birth', 'lin_dist_i', 'same_founder']
for ctgt in CONTACT + ['contact_cn']:
    r2s = [disjoint_cv_reg(df, dev_for_contact, ctgt, seed=s)[0] for s in range(5)]
    rho_bd, _ = spearmanr(df.birth_diff, df[ctgt])
    rho_ld, _ = spearmanr(df.lin_dist_i, df[ctgt])
    cg_rows.append({'contact_target': ctgt, 'cv_r2_mean': round(np.mean(r2s), 4), 'cv_r2_std': round(np.std(r2s), 4),
                    'spearman_birthdiff': round(rho_bd, 4), 'spearman_lindist': round(rho_ld, 4)})
cg = pd.DataFrame(cg_rows).drop_duplicates('contact_target')
cg.to_csv(OUT + 'partD_contact_generation.csv', index=False)
print(cg.to_string())

# ==================== D4/D5/D6 model results ====================
print('\n=== D4 residual + D6 cogrowth (neuron-disjoint CV, multi-seed) ===')
SEEDS = range(5)
model_rows = []
feature_sets = {
    'contact_only': CONTACT,
    'dev_only': DEV,
    'nt_only': NT_COLS,
    'contact+dev': CONTACT + DEV,
    'contact+dev+nt(cogrowth)': CONTACT + DEV + NT_COLS,
}
res = {}
for t in TARGETS:
    res[t] = {}
    for name, fc in feature_sets.items():
        for model in ['logit', 'gb']:
            aucs, aps, lls = [], [], []
            for s in SEEDS:
                a, ap, ll, nf = disjoint_cv_auc(df, fc, t, seed=s, model=model)
                aucs.append(a); aps.append(ap); lls.append(ll)
            res[t][(name, model)] = (np.array(aucs), np.array(aps), np.array(lls))
            model_rows.append({'test': 'D4/D6', 'target': t, 'feature_set': name, 'model': model,
                               'auc_mean': round(np.mean(aucs), 4), 'auc_std': round(np.std(aucs), 4),
                               'ap_mean': round(np.mean(aps), 4), 'logloss_mean': round(np.mean(lls), 4),
                               'n_seeds': len(aucs)})
# delta AUC (contact+dev vs contact_only) with bootstrap-ish CI across seeds
for t in TARGETS:
    for model in ['logit', 'gb']:
        base = res[t][('contact_only', model)][0]
        aug = res[t][('contact+dev', model)][0]
        d = aug - base
        model_rows.append({'test': 'D4_deltaAUC', 'target': t, 'feature_set': 'contact+dev - contact', 'model': model,
                           'auc_mean': round(d.mean(), 4), 'auc_std': round(d.std(), 4),
                           'ap_mean': round(np.percentile(d, 5), 4), 'logloss_mean': round(np.percentile(d, 95), 4),
                           'n_seeds': len(d)})
        cog = res[t][('contact+dev+nt(cogrowth)', model)][0]
        dc = cog - base
        model_rows.append({'test': 'D6_deltaAUC', 'target': t, 'feature_set': 'cogrowth - contact', 'model': model,
                           'auc_mean': round(dc.mean(), 4), 'auc_std': round(dc.std(), 4),
                           'ap_mean': round(np.percentile(dc, 5), 4), 'logloss_mean': round(np.percentile(dc, 95), 4),
                           'n_seeds': len(dc)})

# ==================== D5: nulls (>=20 shuffles) ====================
print('\n=== D5 nulls (node-level birth shuffle + pair-level lineage shuffle) ===')
neurons = sorted(set(df.pre) | set(df.post))
b1 = df[['pre', 'pre_birth']].rename(columns={'pre': 'n', 'pre_birth': 'b'})
b2 = df[['post', 'post_birth']].rename(columns={'post': 'n', 'post_birth': 'b'})
birth_map = pd.concat([b1, b2]).drop_duplicates('n').set_index('n')['b'].to_dict()
NSHUF = 25
null_delta = {t: {m: [] for m in ['logit', 'gb']} for t in TARGETS}
# also null for D4 gradient: spearman birth_diff vs target at fixed contact (use logit delta as principal)
for i in range(NSHUF):
    dfx = df.copy()
    # node-level birth shuffle: permute birth across neurons, preserve graph+degree+modality
    perm_b = rng.permutation(list(birth_map.values()))
    nb = dict(zip(neurons, perm_b))
    dfx['pre_birth'] = dfx.pre.map(nb); dfx['post_birth'] = dfx.post.map(nb)
    dfx['birth_diff'] = (dfx.pre_birth - dfx.post_birth).abs()
    dfx['max_birth'] = dfx[['pre_birth', 'post_birth']].max(axis=1)
    # pair-level lineage shuffle (preserves contact graph edges; breaks lineage-target link)
    pidx = rng.permutation(len(dfx))
    dfx['lin_dist_i'] = df['lin_dist_i'].values[pidx]
    dfx['same_founder'] = df['same_founder'].values[pidx]
    dfx['lin_missing'] = df['lin_missing'].values[pidx]
    for t in TARGETS:
        for model in ['logit']:  # gb too slow x25; do logit for null, gb once below
            a_base, *_ = disjoint_cv_auc(dfx, CONTACT, t, seed=0, model=model)
            a_aug, *_ = disjoint_cv_auc(dfx, CONTACT + DEV, t, seed=0, model=model)
            null_delta[t][model].append(a_aug - a_base)
    if (i + 1) % 5 == 0:
        print(f'  null shuffle {i+1}/{NSHUF}')

# compare real delta (logit, seed0 to match null) vs null distribution
for t in TARGETS:
    nd = np.array(null_delta[t]['logit'])
    real = res[t][('contact+dev', 'logit')][0][0] - res[t][('contact_only', 'logit')][0][0]  # real delta AUC, seed0
    z = (real - nd.mean()) / (nd.std() + 1e-9)
    p_emp = (np.sum(nd >= real) + 1) / (len(nd) + 1)
    model_rows.append({'test': 'D5_null', 'target': t, 'feature_set': 'real_delta_vs_null', 'model': 'logit',
                       'auc_mean': round(real, 4), 'auc_std': round(nd.mean(), 4),
                       'ap_mean': round(nd.std(), 4), 'logloss_mean': round(z, 3), 'n_seeds': len(nd)})
    model_rows[-1]['null_p_emp'] = round(p_emp, 4)
    print(f'  {t}: real_delta={real:.4f} null_mean={nd.mean():.4f}±{nd.std():.4f} z={z:.2f} p_emp={p_emp:.4f}')

mres = pd.DataFrame(model_rows)
mres.to_csv(OUT + 'partD_cogrowth_model_results.csv', index=False)
print('\n=== model results table ===')
print(mres.to_string())

# summary json for doc
summary = {'n_pairs': len(df), 'n_neurons': len(neurons),
           'target_rates': {t: round(df[t].mean(), 3) for t in TARGETS},
           'same_founder_rate': round(df.same_founder.mean(), 3)}
with open(OUT + 'partD_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print('\nDONE. Wrote 3 CSVs + summary to', OUT)
