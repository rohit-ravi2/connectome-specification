"""Part C: (C1) directionality of chemical edges given undirected contact;
(C2) low-contact-strong synapses (real vs measurement artifact).

Chemical and electrical kept SEPARATE. Neuron-disjoint CV for C1. Contact/degree matching.
"""
import csv, re, math, json
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT   = '../seed'
EDGES  = '../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv'
MASTER = f'{ROOT}/data/processed/contact_decomposition/contact_decomposition_master.csv'
NPZ    = '../data/celegans/experiments/nb55_archetype_dynamics/data/upstream_artifacts/connectome.npz'
NB28   = '../data/celegans/New Notebooks/data_derived/nb28_neuron_profile.csv'
OUT    = f'{ROOT}/data/processed/self_avoidance_apposition'

# ---------------------------------------------------------------- load edges
stagemap = {1:'L1',2:'L1',3:'L1',4:'L1',5:'L2',6:'L3',7:'adult',8:'adult'}
# directed chemical per stage (max over animals in stage); adult per-dataset for reproducibility
chem_stage = defaultdict(lambda: defaultdict(int))   # (pre,post)->stage->max syn
chem_adult_ds = defaultdict(dict)                     # (pre,post)->ds(7/8)->syn
elec_adult = defaultdict(int)                         # undirected electrical adult (max ds7/8)
for r in csv.DictReader(open(EDGES)):
    ds = int(re.search(r'witvliet_2020_(\d)', r['source_file']).group(1))
    k = (r['pre'], r['post']); s = int(r['synapses']); st = stagemap[ds]
    if r['type'] == 'chemical':
        chem_stage[k][st] = max(chem_stage[k][st], s)
        if ds in (7,8): chem_adult_ds[k][ds] = s
    else:  # electrical
        if ds in (7,8): elec_adult[k] = max(elec_adult[k], s)

def cadult(a,b):   # directed adult chemical (max over ds7,8)
    d = chem_adult_ds.get((a,b),{}); return max(d.values()) if d else 0
def cadult_ds(a,b):  # (ds7,ds8)
    d = chem_adult_ds.get((a,b),{}); return d.get(7,0), d.get(8,0)
def cstage(a,b,st):
    return chem_stage.get((a,b),{}).get(st,0)
def gap_undirected(a,b):
    return max(elec_adult.get((a,b),0), elec_adult.get((b,a),0))

# ---------------------------------------------------------------- neuron attrs
z = np.load(NPZ, allow_pickle=True)
names = list(z['names']); nt_prim = {n:str(t) for n,t in zip(names, z['nt_primary'])}
klass  = {n:str(t) for n,t in zip(names, z['klass'])}
nb = pd.read_csv(NB28).set_index('neuron')
def attr(n, col, default=0.0):
    return float(nb.at[n,col]) if n in nb.index and col in nb.columns else default
def sattr(n, col, default=''):
    return str(nb.at[n,col]) if n in nb.index and col in nb.columns else default

MODRANK = {'sensory':0,'inter':1,'motor':2,'modulatory':1,'muscle':3}
COMMAND = {'AVAL','AVAR','AVBL','AVBR','AVDL','AVDR','AVEL','AVER','PVCL','PVCR'}

# ---------------------------------------------------------------- master
M = pd.read_csv(MASTER)
M = M[M.c_adult > 0].copy()   # contacting pairs (adult)
print(f'[master] {len(M)} adult-contacting pairs')

def modrank(mod): return MODRANK.get(str(mod), 1)

# =====================================================================
# C1 — DIRECTIONALITY
# =====================================================================
print('\n=== C1 DIRECTIONALITY ===')
rows=[]
for _,r in M.iterrows():
    a,b = r['na'], r['nb']
    n_ab, n_ba = cadult(a,b), cadult(b,a)     # directed adult chem
    if n_ab==0 and n_ba==0:
        cat='nonsyn'
    elif n_ab>0 and n_ba>0:
        cat='reciprocal'
    elif n_ab>0:
        cat='forward'      # a->b only  (a is presynaptic)
    else:
        cat='backward'     # b->a only  (b is presynaptic)
    rows.append(dict(na=a,nb=b,c_adult=r['c_adult'],logc=math.log1p(r['c_adult']),
        n_ab=n_ab,n_ba=n_ba,cat=cat, has_gap=int(gap_undirected(a,b)>0),
        nt_a=nt_prim.get(a,''),nt_b=nt_prim.get(b,''),
        mod_a=sattr(a,'is_sensory'),  # placeholder, use modrank from master mod below
        modrank_a=modrank(r.get('pre_mod')), modrank_b=modrank(r.get('post_mod')),
        same_nt=int(nt_prim.get(a,'x')==nt_prim.get(b,'y')),
        chem_out_a=attr(a,'chem_out'),chem_out_b=attr(b,'chem_out'),
        chem_in_a=attr(a,'chem_in'),chem_in_b=attr(b,'chem_in'),
        gap_deg_a=attr(a,'gap_deg'),gap_deg_b=attr(b,'gap_deg'),
        sens_a=attr(a,'is_sensory'),sens_b=attr(b,'is_sensory'),
        cmd_a=int(a in COMMAND),cmd_b=int(b in COMMAND),
        mono_a=attr(a,'is_monoamine'),mono_b=attr(b,'is_monoamine')))
C1 = pd.DataFrame(rows)
syn = C1[C1.cat!='nonsyn'].copy()
print('categories:', C1.cat.value_counts().to_dict())

# --- (i) reciprocal vs unidirectional: contact area
recip = syn[syn.cat=='reciprocal']; uni = syn[syn.cat.isin(['forward','backward'])]
mwu = stats.mannwhitneyu(recip.c_adult, uni.c_adult, alternative='greater')
print(f'  reciprocal median c={recip.c_adult.median():.0f} (n={len(recip)}) vs '
      f'uni median c={uni.c_adult.median():.0f} (n={len(uni)}); MWU(recip>uni) p={mwu.pvalue:.2e}')

# --- (ii) DIRECTION prediction among unidirectional (neuron-disjoint CV)
U = syn[syn.cat.isin(['forward','backward'])].copy().reset_index(drop=True)
# label: y=1 if na is presynaptic (forward), 0 if backward. na<nb alphabetically => ~50/50, unbiased.
U['y'] = (U.cat=='forward').astype(int)
# asymmetric features: value(na) - value(nb)  (flip sign under swap => encodes direction)
U['d_chem_out'] = U.chem_out_a - U.chem_out_b
U['d_chem_in']  = U.chem_in_a  - U.chem_in_b
U['d_gap_deg']  = U.gap_deg_a  - U.gap_deg_b
U['d_modrank']  = U.modrank_a  - U.modrank_b     # >0 => na more motor-like
U['d_sens']     = U.sens_a     - U.sens_b
U['d_cmd']      = U.cmd_a       - U.cmd_b
U['d_mono']     = U.mono_a     - U.mono_b
FEATS      = ['d_chem_out','d_chem_in','d_gap_deg','d_modrank','d_sens','d_cmd','d_mono']
FEATS_NODEG= ['d_modrank','d_sens','d_cmd','d_mono']   # degree-free identity only
print(f'  unidirectional n={len(U)}, y-base(forward frac)={U.y.mean():.3f}')

def neuron_disjoint_cv(df, feats, K=5, seed=0):
    rng = np.random.default_rng(seed)
    neurons = sorted(set(df.na)|set(df.nb))
    fold = {n:int(rng.integers(K)) for n in neurons}
    oof_pred = np.full(len(df), np.nan);
    for k in range(K):
        test_neu = {n for n,f in fold.items() if f==k}
        te = df[df.na.isin(test_neu) & df.nb.isin(test_neu)]     # both held out
        tr = df[~df.na.isin(test_neu) & ~df.nb.isin(test_neu)]   # neither in test
        if len(te)==0 or tr.y.nunique()<2: continue
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(tr[feats].values, tr.y.values)
        oof_pred[te.index.values] = clf.predict_proba(te[feats].values)[:,1]
    m = ~np.isnan(oof_pred)
    if m.sum()<10 or len(set(df.y.values[m]))<2: return np.nan, int(m.sum())
    return roc_auc_score(df.y.values[m], oof_pred[m]), int(m.sum())

U = U.reset_index(drop=True)
def cv_auc(feats):
    a=[]; nn=0
    for s in range(10):
        auc,n = neuron_disjoint_cv(U, feats, K=5, seed=s)
        if not np.isnan(auc): a.append(auc); nn=n
    return float(np.mean(a)), float(np.std(a)), nn
dir_auc, dir_auc_sd, n = cv_auc(FEATS)
dir_auc_nodeg, dir_auc_nodeg_sd, _ = cv_auc(FEATS_NODEG)
print(f'  DIRECTION prediction (neuron-disjoint 5-fold, ALL asym feats incl global degree): '
      f'AUC={dir_auc:.3f} +/- {dir_auc_sd:.3f} (n_test~{n}, 10 seeds)')
print(f'  DIRECTION prediction (DEGREE-FREE identity only: modality/sensory/command/monoamine): '
      f'AUC={dir_auc_nodeg:.3f} +/- {dir_auc_nodeg_sd:.3f}')
# single-feature AUCs (in-sample, sign-aware)
sf={}
for f in FEATS:
    a = roc_auc_score(U.y, U[f]); sf[f]=round(max(a,1-a),3)
print('  single-feature |AUC-0.5| ranking:', dict(sorted(sf.items(), key=lambda x:-x[1])))

# --- (iii) modality flow: among unidirectional, is PRE more sensory (lower modrank) than POST?
def pre_post_modrank(r):
    if r['cat']=='forward': return r['modrank_a'], r['modrank_b']   # pre=na
    else: return r['modrank_b'], r['modrank_a']                     # pre=nb
pp = U.apply(lambda r: pre_post_modrank(r), axis=1, result_type='expand')
U['pre_mr'], U['post_mr'] = pp[0], pp[1]
ff = U[U.pre_mr != U.post_mr]
frac_ff = (ff.pre_mr < ff.post_mr).mean()   # sensory->inter->motor feedforward
bt = stats.binomtest(int((ff.pre_mr<ff.post_mr).sum()), len(ff), 0.5)
print(f'  modality feed-forward (pre more sensory than post): frac={frac_ff:.3f} '
      f'(n={len(ff)} rank-differing, binom p={bt.pvalue:.2e})')

# --- (iv) same-NT directionality / avoidance
for lab, sub in [('same-NT', C1[C1.same_nt==1]), ('diff-NT', C1[C1.same_nt==0])]:
    tot=len(sub); ns=(sub.cat=='nonsyn').mean(); rc=(sub.cat=='reciprocal').mean()
    syn_sub=sub[sub.cat!='nonsyn']; rc_syn=(syn_sub.cat=='reciprocal').mean() if len(syn_sub) else np.nan
    print(f'  {lab}: n={tot} nonsyn={ns:.3f} reciprocal(all)={rc:.3f} reciprocal|synaptic={rc_syn:.3f}')

# save C1
C1.to_csv(f'{OUT}/partC1_directionality_results.csv', index=False)

# =====================================================================
# C2 — LOW-CONTACT STRONG
# =====================================================================
print('\n=== C2 LOW-CONTACT STRONG ===')
D = M.copy()
D['s_dir_adult'] = [max(cadult(a,b),cadult(b,a)) for a,b in zip(D.na,D.nb)]  # directed max adult chem
D['ds7'] = [max(cadult_ds(a,b)[0],cadult_ds(b,a)[0]) for a,b in zip(D.na,D.nb)]
D['ds8'] = [max(cadult_ds(a,b)[1],cadult_ds(b,a)[1]) for a,b in zip(D.na,D.nb)]
D['both_animals'] = ((D.ds7>0)&(D.ds8>0)).astype(int)   # == strong_core1 reproducibility
D['syn_per_area'] = D.s_adult / D.c_adult
# contacting terciles by c_adult
q1,q2 = D.c_adult.quantile([1/3,2/3])
D['ctercile'] = np.where(D.c_adult<=q1,'low',np.where(D.c_adult<=q2,'mid','high'))
print(f'  contact terciles cut at c_adult={q1:.0f}/{q2:.0f}')

# STRONG def: directed adult >=3 synapses OR reproducible >=2 in both animals
D['strong'] = ((D.s_dir_adult>=3) | ((D.ds7>=2)&(D.ds8>=2))).astype(int)
lcs = D[(D.ctercile=='low') & (D.strong==1)].copy()
print(f'  strong synapses total={D.strong.sum()}; low-contact-strong={len(lcs)} '
      f'({100*len(lcs)/max(1,D.strong.sum()):.0f}% of strong)')

# stage-mismatch: was contact higher pre-adult?
lcs['c_preadult_max'] = lcs[['c_L1','c_L2','c_L3']].max(axis=1)
lcs['contact_shrank'] = (lcs.c_preadult_max > 2*lcs.c_adult).astype(int)  # earlier contact >2x adult
# artifact flags
lcs['single_synapse'] = (lcs.s_dir_adult==1).astype(int)
lcs['one_animal_only']= (lcs.both_animals==0).astype(int)
lcs['low_persist']    = (lcs.contact_persist<=1).astype(int)
lcs['high_efficiency']= (lcs.syn_per_area > D.syn_per_area.quantile(0.9)).astype(int)
lcs['command_pair']   = [int((a in COMMAND) or (b in COMMAND)) for a,b in zip(lcs.na,lcs.nb)]
lcs['both_command']   = [int((a in COMMAND) and (b in COMMAND)) for a,b in zip(lcs.na,lcs.nb)]
lcs['same_nt']        = [int(nt_prim.get(a,'x')==nt_prim.get(b,'y')) for a,b in zip(lcs.na,lcs.nb)]
lcs['has_gap']        = [int(gap_undirected(a,b)>0) for a,b in zip(lcs.na,lcs.nb)]

# ROBUST vs MEASUREMENT-LIMITED
lcs['robust'] = ((lcs.both_animals==1) & (lcs.s_dir_adult>=2)).astype(int)
lcs['meas_limited'] = ((lcs.one_animal_only==1) | (lcs.single_synapse==1)).astype(int)
nrob=lcs.robust.sum(); nml=lcs.meas_limited.sum()
print(f'  ROBUST (both adult animals & >=2 syn): {nrob}/{len(lcs)} = {100*nrob/len(lcs):.0f}%')
print(f'  measurement-limited (single-syn or one-animal): {nml}/{len(lcs)} = {100*nml/len(lcs):.0f}%')
print(f'  contact shrank from pre-adult (>2x): {lcs.contact_shrank.sum()} '
      f'({100*lcs.contact_shrank.mean():.0f}%)')
print(f'  high syn/area (top-decile efficiency): {lcs.high_efficiency.sum()} '
      f'({100*lcs.high_efficiency.mean():.0f}%)')
print(f'  command-interneuron involved: {lcs.command_pair.sum()} '
      f'(both-command: {lcs.both_command.sum()}); same-NT: {lcs.same_nt.sum()}; gap-coupled: {lcs.has_gap.sum()}')

# enrichment: command involvement in low-contact-strong vs all strong
allstrong=D[D.strong==1]
cmd_lcs = lcs.command_pair.mean()
cmd_all = np.mean([int((a in COMMAND) or (b in COMMAND)) for a,b in zip(allstrong.na,allstrong.nb)])
# 2x2: command x (low-contact | mid/high) among strong
allstrong=allstrong.copy()
allstrong['cmd']=[int((a in COMMAND) or (b in COMMAND)) for a,b in zip(allstrong.na,allstrong.nb)]
allstrong['low']=(allstrong.ctercile=='low').astype(int)
ct=pd.crosstab(allstrong.cmd, allstrong.low)
try:
    orr,pf=stats.fisher_exact(ct.values)
except Exception:
    orr,pf=np.nan,np.nan
print(f'  command enrichment in low-contact vs mid/high (among strong): '
      f'lcs {cmd_lcs:.2f} vs allstrong {cmd_all:.2f}; Fisher OR={orr:.2f} p={pf:.2e}')

lcs_out = lcs[['na','nb','c_adult','ctercile','s_dir_adult','ds7','ds8','both_animals',
    'syn_per_area','c_L1','c_L2','c_L3','c_preadult_max','contact_shrank','contact_persist',
    'first_synapse_stage','single_synapse','one_animal_only','low_persist','high_efficiency',
    'command_pair','both_command','same_nt','has_gap','robust','meas_limited']].sort_values(
    ['robust','s_dir_adult'], ascending=[False,False])
lcs_out.to_csv(f'{OUT}/partC2_low_contact_strong_edges.csv', index=False)

# ---- dump summary json for docs
summary = dict(
  c1=dict(n_contacting=len(M), categories=C1.cat.value_counts().to_dict(),
      recip_median_c=float(recip.c_adult.median()), uni_median_c=float(uni.c_adult.median()),
      recip_gt_uni_p=float(mwu.pvalue), n_recip=int(len(recip)), n_uni=int(len(uni)),
      direction_auc=dir_auc, direction_auc_sd=dir_auc_sd,
      direction_auc_nodeg=dir_auc_nodeg, direction_auc_nodeg_sd=dir_auc_nodeg_sd,
      n_unidirectional=int(len(U)),
      single_feat_auc=sf, modality_ff_frac=float(frac_ff), modality_ff_p=float(bt.pvalue),
      n_rankdiff=int(len(ff))),
  c2=dict(n_lowcontact_strong=int(len(lcs)), n_strong=int(D.strong.sum()),
      robust=int(nrob), meas_limited=int(nml), pct_robust=float(100*nrob/len(lcs)),
      contact_shrank=int(lcs.contact_shrank.sum()), high_eff=int(lcs.high_efficiency.sum()),
      command_pairs=int(lcs.command_pair.sum()), both_command=int(lcs.both_command.sum()),
      cmd_OR=float(orr), cmd_p=float(pf)))
json.dump(summary, open(f'{OUT}/partC_summary.json','w'), indent=2)
print('\n[saved] partC1_directionality_results.csv, partC2_low_contact_strong_edges.csv, partC_summary.json')
