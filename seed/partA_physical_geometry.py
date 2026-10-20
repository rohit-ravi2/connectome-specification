"""
Part A - Physical contact geometry decomposition (C. elegans wiring).
Question: which physical contact metric matters -- absolute area vs relative
allocation vs rank vs saturation?

A1 metric horse-race (univariate + degree-controlled, neuron-disjoint CV AUC)
A2 rank/competition (top-k, rank vs raw area)
A3 saturation / calibration (decile P(synapse|area), isotonic threshold)
A4 contact efficiency residuals (high-contact-nonsynaptic vs low-contact-synaptic)
A5 directionality (which direction; reciprocity; NT/modality)

Outputs:
  data/processed/contact_decomposition/partA_physical_geometry_results.csv
  data/processed/contact_decomposition/partA_contact_efficiency_residuals.csv
"""
import pandas as pd, numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

RNG = np.random.default_rng(0)
OUT = 'data/processed/contact_decomposition'
M = pd.read_csv(f'{OUT}/contact_decomposition_master.csv')

# ---------------------------------------------------------------- feature build
# targets
M['syn_adult'] = (M.s_adult > 0).astype(int)          # contact->synapse conversion (adult)
M['syn_any']   = (M[['s_L1','s_L2','s_L3','s_adult']] > 0).any(axis=1).astype(int)

# global percentile of adult / L1 area (monotonic transform of raw -> same univariate AUC)
M['pctile_adult'] = M.c_adult.rank(pct=True)
M['pctile_L1']    = M.c_L1.rank(pct=True)

# within-neuron competitive rank & normalized frac at a given stage, treating na & nb symmetrically
def add_rank_frac(df, area_col, tag):
    # total contact area per neuron at this stage
    tot = Counter()
    for a, b, c in zip(df.na, df.nb, df[area_col]):
        tot[a] += c; tot[b] += c
    # collect partner areas per neuron to compute rank (1 = largest)
    part = defaultdict(list)   # neuron -> list of (area, idx)
    for i,(a,b,c) in enumerate(zip(df.na, df.nb, df[area_col])):
        part[a].append((c,i)); part[b].append((c,i))
    rank_of = {}   # (neuron, idx) -> rank(1=largest), and degree
    ndeg = {}
    for n, lst in part.items():
        lst_sorted = sorted(lst, key=lambda t: -t[0])
        ndeg[n] = len(lst_sorted)
        for r,(c,i) in enumerate(lst_sorted, start=1):
            rank_of[(n,i)] = r
    frac_a = np.zeros(len(df)); frac_b = np.zeros(len(df))
    rk_a = np.zeros(len(df)); rk_b = np.zeros(len(df))
    rkn_a = np.zeros(len(df)); rkn_b = np.zeros(len(df))
    for i,(a,b,c) in enumerate(zip(df.na, df.nb, df[area_col])):
        frac_a[i] = c/tot[a] if tot[a]>0 else 0.0
        frac_b[i] = c/tot[b] if tot[b]>0 else 0.0
        rk_a[i] = rank_of[(a,i)]; rk_b[i] = rank_of[(b,i)]
        rkn_a[i] = rank_of[(a,i)]/ndeg[a]; rkn_b[i] = rank_of[(b,i)]/ndeg[b]
    df[f'frac_a_{tag}'] = frac_a
    df[f'frac_b_{tag}'] = frac_b
    df[f'min_frac_{tag}'] = np.minimum(frac_a, frac_b)
    df[f'max_frac_{tag}'] = np.maximum(frac_a, frac_b)
    # rank: use BEST (lowest) rank across the two endpoints = strongest competitive position
    df[f'minrank_{tag}'] = np.minimum(rk_a, rk_b)
    df[f'maxrank_{tag}'] = np.maximum(rk_a, rk_b)
    df[f'minrank_norm_{tag}'] = np.minimum(rkn_a, rkn_b)
    df[f'deg_a_{tag}'] = df.na.map(ndeg)
    df[f'deg_b_{tag}'] = df.nb.map(ndeg)

add_rank_frac(M, 'c_adult', 'ad')
add_rank_frac(M, 'c_L1', 'l1')

# ------------------------------------------------------- neuron-disjoint folds
neurons = sorted(set(M.na) | set(M.nb))
fold_of = {n: RNG.integers(0,5) for n in neurons}
M['fa'] = M.na.map(fold_of); M['fb'] = M.nb.map(fold_of)

def neurondisjoint_auc_univar(feat, target, higher_is_pos=True):
    """Pooled test-edge AUC where both endpoints are in the held-out fold.
    Univariate monotonic feature -> no training needed, just pool held-out edges."""
    ys, xs = [], []
    for k in range(5):
        te = M[(M.fa==k) & (M.fb==k)]
        if te[target].nunique() < 2: continue
        ys.append(te[target].values); xs.append(te[feat].values)
    y = np.concatenate(ys); x = np.concatenate(xs)
    if not higher_is_pos: x = -x
    m = ~np.isnan(x)
    return roc_auc_score(y[m], x[m]), len(y)

def global_auc_univar(feat, target, higher_is_pos=True):
    x = M[feat].values.copy(); y = M[target].values
    if not higher_is_pos: x = -x
    m = ~np.isnan(x)
    return roc_auc_score(y[m], x[m])

def neurondisjoint_auc_model(feats, target):
    """5-fold neuron-disjoint logistic regression; pooled held-out AUC."""
    preds, ys = [], []
    for k in range(5):
        tr = M[(M.fa!=k) & (M.fb!=k)]
        te = M[(M.fa==k) & (M.fb==k)]
        if te[target].nunique() < 2 or tr[target].nunique() < 2: continue
        Xtr = tr[feats].values; Xte = te[feats].values
        # standardize on train
        mu = Xtr.mean(0); sd = Xtr.std(0); sd[sd==0]=1
        Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, tr[target].values)
        preds.append(clf.predict_proba(Xte)[:,1]); ys.append(te[target].values)
    y=np.concatenate(ys); p=np.concatenate(preds)
    return roc_auc_score(y,p), len(y)

# ------------------------------------------------------------------- A1 + A2
TARGETS = {
 'T_first_L1':'T_first_L1',          # EARLY (synapse present already at L1)
 'T_preadult_persist':'T_preadult_persist',
 'adult_strong_core1':'adult_strong_core1',
 'syn_adult':'syn_adult',            # conversion synapse|contact (adult)
 'syn_any':'syn_any',
}

# metric families. For EARLY target we use L1-stage geometry; else adult-stage.
adult_metrics = {
 'raw_c_adult':('c_adult',True),
 'log_c_adult':('logc_adult',True),          # == raw AUC (monotone)
 'pctile_adult':('pctile_adult',True),        # == raw AUC (monotone)
 'frac_src(a)':('frac_a_ad',True),
 'frac_tgt(b)':('frac_b_ad',True),
 'min_frac':('min_frac_ad',True),
 'max_frac':('max_frac_ad',True),
 'minrank(best,lower=better)':('minrank_ad',False),
 'minrank_norm':('minrank_norm_ad',False),
}
l1_metrics = {
 'raw_c_L1':('c_L1',True),
 'log_c_L1':('logc_L1',True),
 'pctile_L1':('pctile_L1',True),
 'frac_src(a)':('frac_a_l1',True),
 'frac_tgt(b)':('frac_b_l1',True),
 'min_frac':('min_frac_l1',True),
 'max_frac':('max_frac_l1',True),
 'minrank(best,lower=better)':('minrank_l1',False),
 'minrank_norm':('minrank_norm_l1',False),
}

rows=[]
for tname, tcol in TARGETS.items():
    mets = l1_metrics if tname=='T_first_L1' else adult_metrics
    stage = 'L1' if tname=='T_first_L1' else 'adult'
    for mlabel,(fcol,hip) in mets.items():
        g = global_auc_univar(fcol, tcol, hip)
        nd,ntest = neurondisjoint_auc_univar(fcol, tcol, hip)
        rows.append(dict(analysis='A1_univar', target=tname, stage=stage,
                         metric=mlabel, feature=fcol,
                         global_auc=round(g,4), nd_cv_auc=round(nd,4), n_test=ntest))

res = pd.DataFrame(rows)

# ---- A1 degree control: does absolute area / rank add beyond node degree? ----
ctrl_rows=[]
for tname,tcol in TARGETS.items():
    stg = 'l1' if tname=='T_first_L1' else 'ad'
    deg=[f'deg_a_{stg}',f'deg_b_{stg}']
    area = 'c_L1' if tname=='T_first_L1' else 'c_adult'
    frac=[f'frac_a_{stg}',f'frac_b_{stg}']
    rank=[f'minrank_norm_{stg}']
    combos={
      'degree_only':deg,
      'degree+area':deg+[area],
      'degree+frac':deg+frac,
      'degree+rank':deg+rank,
      'degree+area+frac+rank':deg+[area]+frac+rank,
    }
    for cl,feats in combos.items():
        a,n=neurondisjoint_auc_model(feats,tcol)
        ctrl_rows.append(dict(analysis='A1_degctrl',target=tname,model=cl,
                              nd_cv_auc=round(a,4),n_test=n))
ctrl=pd.DataFrame(ctrl_rows)

# --------------------------------------- A2 top-k: are top contacts synaptic?
topk_rows=[]
for tname,tcol in [('syn_adult','syn_adult')]:
    stg='ad'
    for k in [1,2,3,5,10]:
        top = (M[f'minrank_{stg}']<=k)
        rate_top = M.loc[top, tcol].mean()
        rate_rest = M.loc[~top, tcol].mean()
        topk_rows.append(dict(analysis='A2_topk',target=tname,k=k,
                              n_top=int(top.sum()),
                              syn_rate_top=round(rate_top,4),
                              syn_rate_rest=round(rate_rest,4)))
topk=pd.DataFrame(topk_rows)

# --------------------------------------- A3 saturation / decile calibration
sat_rows=[]
cont = M[M.c_adult>0].copy()   # restrict to pairs actually in adult contact
cont['dec']=pd.qcut(cont.c_adult,10,labels=False,duplicates='drop')
for d,g in cont.groupby('dec'):
    sat_rows.append(dict(analysis='A3_decile',decile=int(d),
        n=len(g), area_lo=round(g.c_adult.min(),0), area_hi=round(g.c_adult.max(),0),
        area_med=round(g.c_adult.median(),0),
        p_syn_adult=round(g.syn_adult.mean(),4)))
sat=pd.DataFrame(sat_rows)
# isotonic fit of P(syn) vs area to check monotone saturation
iso=IsotonicRegression(out_of_bounds='clip')
iso.fit(cont.c_adult.values, cont.syn_adult.values)
xs=np.quantile(cont.c_adult,np.linspace(0.05,0.95,19))
iso_pred=iso.predict(xs)
iso_df=pd.DataFrame(dict(analysis='A3_isotonic',area=np.round(xs,0),
                         p_syn_iso=np.round(iso_pred,4)))

# minimum contact area: smallest area among synaptic adult pairs, and syn rate in bottom deciles
min_syn_area = cont.loc[cont.syn_adult==1,'c_adult'].min()

# --------------------------------------- A4 efficiency residuals
# efficiency = synapses per unit adult contact area
eff = M[M.c_adult>0].copy()
eff['syn_per_area'] = eff.s_adult / eff.c_adult
# high-contact-nonsynaptic: contact_no_syn==1 & big area ; low-contact-synaptic: small area & syn
hi_thresh = eff.c_adult.quantile(0.90)
lo_thresh = eff.c_adult.quantile(0.10)
hcns = eff[(eff.contact_no_syn==1)&(eff.c_adult>=hi_thresh)].copy()
lcs  = eff[(eff.syn_adult==1)&(eff.c_adult<=lo_thresh)].copy()
resid_rows=[]
for label,sub in [('high_contact_nonsynaptic',hcns),('low_contact_synaptic',lcs)]:
    resid_rows.append(dict(analysis='A4_resid_summary',group=label,n=len(sub),
        med_area=round(sub.c_adult.median(),0),
        med_frac_src=round(sub.frac_src.median(),4),
        med_minrank=round(sub['minrank_ad'].median(),1),
        med_lin_dist=round(sub.lin_dist.median(),3) if sub.lin_dist.notna().any() else np.nan,
        frac_both_born_preL1=round(sub.both_born_preL1.mean(),3)))
resid_sum=pd.DataFrame(resid_rows)
# save the actual residual pairs
resid_pairs = pd.concat([hcns.assign(resid_group='high_contact_nonsynaptic'),
                         lcs.assign(resid_group='low_contact_synaptic')])
resid_cols=['na','nb','resid_group','c_adult','frac_src','frac_tgt','minrank_ad',
            's_adult','contact_no_syn','pre_nt','post_nt','pre_mod','post_mod',
            'lin_dist','both_born_preL1','contact_persist']
resid_pairs[resid_cols].to_csv(f'{OUT}/partA_contact_efficiency_residuals.csv',index=False)

# --------------------------------------- A5 directionality
# Need directed synapse info. Master is undirected/summed. We test what we can:
#  (a) reciprocity enrichment at high contact: fraction of synaptic pairs that are
#      bidirectional cannot be resolved from summed data -> use s_adult counts &
#      compare synaptic-pair contact area vs non-synaptic contact area (already A3).
#  (b) does source vs target NT/modality bias whether a contact converts?
dir_rows=[]
# DATA LIMITATION: pre_nt/post_nt/pre_mod/post_mod are populated ONLY for pairs that
# have a synapse (pre_nt.notna() == syn_any exactly). So "conversion rate by NT" is a
# tautology and DIRECTION is not recoverable from the summed undirected master.
# What we CAN test: among synaptic pairs, does contact area grade synapse COUNT (strength)?
from scipy.stats import spearmanr
syn = M[M.syn_adult==1]
rho,p = spearmanr(syn.c_adult, syn.s_adult)
dir_rows.append(dict(analysis='A5_area_vs_strength',
    note='among adult-synaptic pairs: contact area vs synapse count',
    n=len(syn), spearman_rho=round(rho,4), pval=f'{p:.2e}'))
# does NT-identity add anything to area for STRENGTH among synaptic pairs? (descriptive)
for cat,g in syn.groupby('pre_nt'):
    if len(g)<30: continue
    dir_rows.append(dict(analysis='A5_strength_by_nt',note=f'pre_nt={cat}',
        n=len(g), spearman_rho=round(spearmanr(g.c_adult,g.s_adult)[0],3),
        pval=round(g.s_adult.mean(),2)))
dir_df=pd.DataFrame(dir_rows)
# reciprocity proxy: at high vs low contact, rate of adult synapse presence
recip_rows=[]
q=M.c_adult.rank(pct=True)
for lab,mask in [('top10pct_area',q>=0.9),('bottom50pct_area',q<0.5)]:
    g=M[mask]
    recip_rows.append(dict(analysis='A5_recip_proxy',bin=lab,n=len(g),
        p_syn_adult=round(g.syn_adult.mean(),4),
        med_syn_count=round(g.s_adult.median(),2),
        mean_syn_count=round(g.s_adult.mean(),3)))
recip=pd.DataFrame(recip_rows)

# ------------------------------------------------------------------ assemble
allres = pd.concat([res, ctrl, topk, sat, iso_df, resid_sum, dir_df, recip],
                   ignore_index=True)
allres.to_csv(f'{OUT}/partA_physical_geometry_results.csv',index=False)

# ----------------------------------------------------------------- print report
pd.set_option('display.width',200,'display.max_rows',200,'display.max_columns',30)
print("\n================ A1 UNIVARIATE METRIC HORSE-RACE ================")
for t in TARGETS:
    sub=res[res.target==t].sort_values('global_auc',ascending=False)
    print(f"\n--- target={t} (stage={sub.stage.iloc[0]}) ---")
    print(sub[['metric','global_auc','nd_cv_auc','n_test']].to_string(index=False))
print("\n================ A1 DEGREE-CONTROL (neuron-disjoint LR) ================")
print(ctrl.to_string(index=False))
print("\n================ A2 TOP-K (adult syn) ================")
print(topk.to_string(index=False))
print("\n================ A3 SATURATION / DECILE CALIBRATION (adult contact) ================")
print(sat.to_string(index=False))
print(f"min adult contact area among synaptic pairs: {min_syn_area:.0f}")
print("\nisotonic P(syn|area):")
print(iso_df.to_string(index=False))
print("\n================ A4 EFFICIENCY RESIDUALS ================")
print(resid_sum.to_string(index=False))
print("\n================ A5 DIRECTIONALITY ================")
print(recip.to_string(index=False))
print(dir_df.to_string(index=False))
print("\nWROTE:", f'{OUT}/partA_physical_geometry_results.csv',
      f'{OUT}/partA_contact_efficiency_residuals.csv')
