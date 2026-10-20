"""
Part D (subcellular / spatial contact meaning) -- WHERE is the synapse and does it matter?

Data:
  Witvliet 2021 Dataset8 (adult) per-synapse JSON  -> pre/post, size, size_proportion_per_section,
     pre_tid/post_tid -> 3D coords (z = 30nm EM section axis = AP position) + dist_to_root (compartment).
  Cook 2019 SI3 synapse lists (N2U adult) -> #sites + section-extent per pair (NO absolute position).
  Master contact_decomposition_master.csv -> contact area, synapse counts, persistence, strong-core.

Outputs (data/processed/contact_decomposition/):
  partD_spatial_synapse_results.csv   per-edge spatial features + master merge (D1/D2)
  partD_subcellular_residuals.csv     D3/D4 residual-CV + D5 compartment/directionality stats
Doc written separately.
"""
import json, warnings
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
rng = np.random.default_rng(42)

BASE = 'data/external/mechanistic_frontier/witvliet2021_developmental'
OUT  = 'data/processed/contact_decomposition/'
MASTER = OUT + 'contact_decomposition_master.csv'
SECTION_NM = 30.0            # EM section thickness
CLUSTER_GAP_NM = 300.0       # >=10 sections gap -> separate spatial cluster along AP

# ---------------------------------------------------------------- load Witvliet D8
syn = json.load(open(f'{BASE}/synapses/Dataset8_synapses.json'))
sk  = json.load(open(f'{BASE}/data/skeletons/Dataset8_skeletons.json'))

def coord(neuron, tid):
    c = sk.get(neuron, {}).get('coords', {}).get(str(tid))
    return np.array(c, float) if c is not None else None
def relpos(neuron, tid):
    """position along arbor: dist_to_root(tid)/max_dist_to_root  (0=root/soma-side .. 1=far tip)."""
    v = sk.get(neuron)
    if v is None: return np.nan
    d = v.get('dist_to_root', {})
    dt = d.get(str(tid))
    if dt is None: return np.nan
    mx = max(d.values()) if d else np.nan
    return dt / mx if mx and mx > 0 else np.nan

# per (pre->post_i) synaptic site records
rows = []
for s in syn:
    pre = s['pre']; ptid = s['pre_tid']; size = float(s['size'])
    nsec = len(s['size_proportion_per_section'])
    pc = coord(pre, ptid)
    z_pre = pc[2] if pc is not None else np.nan
    rp_pre = relpos(pre, ptid)
    for post, w, ttid in zip(s['post'], s['post_weights'], s['post_tid']):
        rp_post = relpos(post, ttid)
        rows.append(dict(pre=pre, post=post, weight=float(w), ssize=size, size_w=size*float(w),
                         nsec=nsec, z=z_pre, rp_pre=rp_pre, rp_post=rp_post, ttid=ttid))
sr = pd.DataFrame(rows)
print(f'[load] {len(syn)} synapse objects -> {len(sr)} pre->post site records; '
      f'{sr[["pre","post"]].drop_duplicates().shape[0]} directed pairs')

# ---------------------------------------------------------------- per-edge spatial features
def z_entropy(zvals, binw=10*SECTION_NM):
    if len(zvals) < 2: return 0.0
    lo, hi = min(zvals), max(zvals)
    if hi == lo: return 0.0
    nb = max(2, int((hi-lo)/binw)+1)
    h, _ = np.histogram(zvals, bins=nb)
    p = h[h>0]/h.sum()
    return float(-(p*np.log(p)).sum())
def n_clusters(zvals, gap=CLUSTER_GAP_NM):
    z = np.sort(np.asarray(zvals, float)); z = z[~np.isnan(z)]
    if len(z)==0: return 0
    return int(1 + (np.diff(z) > gap).sum())

feat = []
for (pre, post), g in sr.groupby(['pre','post']):
    zv = g.z.dropna().values
    feat.append(dict(
        na=pre, nb=post,
        n_sites=len(g),                          # polyadic site count for this pair
        syn_weight=g.weight.sum(),
        size_sum=g.size_w.sum(), size_mean=g.ssize.mean(),
        size_cv=(g.ssize.std()/g.ssize.mean()) if g.ssize.mean()>0 else 0.0,
        sec_extent_mean=g.nsec.mean(),           # physical size in # sections
        z_mean=np.nanmean(g.z), z_std=(np.nanstd(zv) if len(zv)>1 else 0.0),
        z_range=(zv.max()-zv.min()) if len(zv)>1 else 0.0,
        z_entropy=z_entropy(zv), n_clusters=n_clusters(zv),
        rp_pre_mean=np.nanmean(g.rp_pre), rp_post_mean=np.nanmean(g.rp_post),
    ))
feat = pd.DataFrame(feat)

# ---------------------------------------------------------------- neuron AP profiles (arbor)
neu = {}
for n, v in sk.items():
    cs = np.array(list(v['coords'].values()), float)
    if len(cs)==0: continue
    neu[n] = dict(z_c=cs[:,2].mean(), z_lo=cs[:,2].min(), z_hi=cs[:,2].max(),
                  z_span=cs[:,2].max()-cs[:,2].min())
neu = pd.DataFrame(neu).T

# ---------------------------------------------------------------- merge master
m = pd.read_csv(MASTER)
deg_out = m[m.c_adult>0].groupby('na').size().rename('pre_deg')
deg_in  = m[m.c_adult>0].groupby('nb').size().rename('post_deg')
m = m.merge(deg_out, left_on='na', right_index=True, how='left') \
     .merge(deg_in,  left_on='nb', right_index=True, how='left')
m[['pre_deg','post_deg']] = m[['pre_deg','post_deg']].fillna(0)
for side, col in [('na','pre'),('nb','post')]:
    for f in ['z_c','z_lo','z_hi','z_span']:
        m[f'{col}_{f}'] = m[side].map(neu[f])
# AP overlap of the two arbors (nm of shared section range)
m['z_overlap'] = (np.minimum(m.pre_z_hi,m.post_z_hi) - np.maximum(m.pre_z_lo,m.post_z_lo)).clip(lower=0)
m['z_absdiff'] = (m.pre_z_c - m.post_z_c).abs()

d = m.merge(feat, on=['na','nb'], how='left')
d['has_syn'] = (d.s_adult>0).astype(int)
d.to_csv(OUT+'partD_spatial_synapse_results.csv', index=False)
print(f'[save] partD_spatial_synapse_results.csv  {d.shape}')

# ================================================================ RESULTS
res = {}     # scalar summary rows
def add(section, name, **kw): res.setdefault(section, []).append(dict(metric=name, **kw))

syn_edges = d[d.n_sites.notna()].copy()   # edges with mapped synapse sites (D8)
print(f'\n[D8] {len(syn_edges)} directed edges carry synapse sites')

# ---------------- D1: spatial distribution: core vs non-core --------------------
print('\n=== D1: synapse spatial distribution per edge (strong-core vs rest) ===')
core = syn_edges[syn_edges.adult_strong_core1==1]
rest = syn_edges[syn_edges.adult_strong_core1==0]
for met in ['n_sites','z_std','z_range','z_entropy','n_clusters','sec_extent_mean','size_mean']:
    a, b = core[met].dropna(), rest[met].dropna()
    if len(a)>3 and len(b)>3:
        U,p = stats.mannwhitneyu(a,b,alternative='two-sided')
        print(f'  {met:16s} core={a.median():.1f}  rest={b.median():.1f}  p={p:.2e}')
        add('D1', met, core_med=float(a.median()), rest_med=float(b.median()),
            core_mean=float(a.mean()), rest_mean=float(b.mean()), n_core=len(a), n_rest=len(b), mwu_p=float(p))
# per-site spatial spread (normalize cluster count by sites) -> are core edges MORE clustered?
syn_edges['clusters_per_site'] = syn_edges.n_clusters/syn_edges.n_sites
for grp,lbl in [(syn_edges.adult_strong_core1==1,'core'),(syn_edges.adult_strong_core1==0,'rest')]:
    sub=syn_edges[grp]
    print(f'  clusters/site {lbl}: {sub.clusters_per_site.mean():.3f}')

# ---------------- D2: synapse density per contact area --------------------------
print('\n=== D2: synapse density vs contact area ===')
cc = d[d.c_adult>0].copy()
sp = stats.spearmanr(cc.c_adult, cc.s_adult)
print(f'  spearman(c_adult, s_adult) all contacting edges: rho={sp.correlation:.3f} p={sp.pvalue:.1e}  n={len(cc)}')
add('D2','spearman_contact_vs_syn', rho=float(sp.correlation), p=float(sp.pvalue), n=len(cc))
# density among synaptic edges
se = cc[cc.s_adult>0].copy(); se['density']= se.s_adult/(se.c_adult/1e6)
print(f'  synapse density (syn per 1e6 nm^2 contact): median={se.density.median():.3f} '
      f'IQR=[{se.density.quantile(.25):.3f},{se.density.quantile(.75):.3f}]')
# within-neuron: do higher-contact partners get more synapses? (identity-controlled)
rhos=[]
for pre,g in cc.groupby('na'):
    if g.na.size>=6 and g.s_adult.nunique()>1 and g.c_adult.nunique()>1:
        rhos.append(stats.spearmanr(g.c_adult,g.s_adult).correlation)
rhos=[r for r in rhos if not np.isnan(r)]
print(f'  within-neuron spearman(contact,syn): mean={np.mean(rhos):.3f} '
      f'(median={np.median(rhos):.3f}, n_neurons={len(rhos)}, frac>0={np.mean(np.array(rhos)>0):.2f})')
tt=stats.wilcoxon(rhos) if len(rhos)>5 else None
add('D2','within_neuron_contact_syn_rho', mean=float(np.mean(rhos)), median=float(np.median(rhos)),
    n=len(rhos), frac_pos=float(np.mean(np.array(rhos)>0)), wilcoxon_p=float(tt.pvalue) if tt else np.nan)

# ---------------- D3: AP position predicts contact->synapse conversion ----------
print('\n=== D3: does AP position of the contact predict contact->synapse conversion? ===')
conv = d[(d.c_adult>0) & d.pre_z_c.notna() & d.post_z_c.notna()].copy()
conv['pair_z'] = (conv.pre_z_c+conv.post_z_c)/2   # AP band proxy of the contact
for f in ['pair_z','z_overlap','z_absdiff']:
    r=stats.spearmanr(conv[f], conv.has_syn)
    print(f'  spearman({f:10s}, has_syn) = {r.correlation:+.3f}  p={r.pvalue:.1e}')
    add('D3', f'conv_vs_{f}', rho=float(r.correlation), p=float(r.pvalue), n=len(conv))

# ---------------- D4: positional residuals, neuron-disjoint CV -------------------
print('\n=== D4: positional residuals over contact+topology (neuron-disjoint CV) ===')
X = d[(d.c_adult>0) & d.pre_z_c.notna() & d.post_z_c.notna()].copy()
X['logc']=np.log1p(X.c_adult)
BASE_F   = ['logc','pre_deg','post_deg']
SPAT_F   = ['pair_z','z_overlap','z_absdiff','pre_z_c','post_z_c','pre_z_span','post_z_span']
X['pair_z']=(X.pre_z_c+X.post_z_c)/2
y = X.has_syn.values
# neuron-disjoint folds: assign every neuron to a fold; test edges w/ BOTH endpoints held out
neurons = pd.unique(pd.concat([X.na,X.nb]))
fold_of = {n:i for i,n in enumerate(rng.permutation(neurons))}
K=5
X['fa']=X.na.map(lambda n: fold_of[n]%K); X['fb']=X.nb.map(lambda n: fold_of[n]%K)
def cv_auc(feats):
    yt_all, yp_all = [], []
    for k in range(K):
        te = (X.fa==k)&(X.fb==k); tr = (X.fa!=k)&(X.fb!=k)
        if te.sum()<8 or y[te.values].sum()<2 or y[tr.values].sum()<2: continue
        sc=StandardScaler().fit(X.loc[tr,feats])
        clf=LogisticRegression(max_iter=2000,C=1.0)
        clf.fit(sc.transform(X.loc[tr,feats]), y[tr.values])
        p=clf.predict_proba(sc.transform(X.loc[te,feats]))[:,1]
        yt_all.append(y[te.values]); yp_all.append(p)
    yt=np.concatenate(yt_all); yp=np.concatenate(yp_all)
    return roc_auc_score(yt,yp), len(yt)
auc_b,nb_ = cv_auc(BASE_F)
auc_s,ns_ = cv_auc(BASE_F+SPAT_F)
print(f'  baseline (logc+deg)        AUC={auc_b:.4f}  (n_test={nb_})')
print(f'  +spatial (AP pos/overlap)  AUC={auc_s:.4f}  delta={auc_s-auc_b:+.4f}')
add('D4','neuron_disjoint_cv', auc_base=float(auc_b), auc_spatial=float(auc_s),
    delta=float(auc_s-auc_b), n_test=int(ns_))

# ---------------- D5: compartment + directionality from location ----------------
print('\n=== D5: compartment (dist_to_root) & directionality ===')
# (a) within-neuron: are OUTPUT sites (neuron as pre) at different relative compartment than INPUT sites (as post)?
out_rp = sr.groupby('pre').rp_pre.mean()
in_rp  = sr.groupby('post').rp_post.mean()
comp = pd.concat([out_rp.rename('out_rp'), in_rp.rename('in_rp')], axis=1).dropna()
w = stats.wilcoxon(comp.out_rp, comp.in_rp)
print(f'  per-neuron output rp mean={comp.out_rp.mean():.3f}  input rp mean={comp.in_rp.mean():.3f}  '
      f'wilcoxon p={w.pvalue:.2e}  n={len(comp)}')
add('D5','compartment_out_vs_in_relpos', out_mean=float(comp.out_rp.mean()),
    in_mean=float(comp.in_rp.mean()), wilcoxon_p=float(w.pvalue), n=len(comp))
# (b) directionality vs AP: for reciprocally-CONTACTING pairs, is pre more anterior/posterior than post?
X['pre_ant'] = (X.pre_z_c < X.post_z_c).astype(int)   # pre is anterior (smaller z)
syn_only = X[X.has_syn==1]
frac_ant = syn_only.pre_ant.mean()
bt = stats.binomtest(int(syn_only.pre_ant.sum()), len(syn_only), 0.5)
print(f'  among synaptic edges: pre anterior-to-post in {frac_ant:.3f}  (binom p={bt.pvalue:.2e}, n={len(syn_only)})')
add('D5','directionality_pre_anterior_frac', frac=float(frac_ant), p=float(bt.pvalue), n=len(syn_only))
# (c) does AP direction add to predicting edge orientation among reciprocal contact pairs?
#    reciprocal contact pair (A,B): both c_adult>0; is the synaptic direction the anterior->posterior one?
recip = X.merge(X, left_on=['na','nb'], right_on=['nb','na'], suffixes=('','_r'))
recip = recip[recip.na<recip.nb]  # unordered pair once
# among pairs where exactly one direction is synaptic, does that direction go ant->post?
one_dir = recip[(recip.has_syn+recip.has_syn_r)==1]
if len(one_dir)>5:
    # direction that is synaptic
    syn_ant = np.where(one_dir.has_syn==1, one_dir.pre_ant, 1-one_dir.pre_ant)
    fr=syn_ant.mean(); btr=stats.binomtest(int(syn_ant.sum()),len(syn_ant),0.5)
    print(f'  polarized pairs: synaptic direction is anterior->posterior in {fr:.3f} (p={btr.pvalue:.2e}, n={len(one_dir)})')
    add('D5','polarized_pair_ant_to_post', frac=float(fr), p=float(btr.pvalue), n=len(one_dir))

# ---------------- Cook cross-check (D1 #sites & extent, NO position) -------------
print('\n=== Cook SI3 cross-check (#sites + section-extent; no absolute position) ===')
ck = pd.read_excel('../data/celegans/data/connectome/cook2019/SI 3 Synapse lists.xlsx',
                   sheet_name='hermaphrodite', header=0)
ck.columns=[str(c).strip() for c in ck.columns]
ck = ck[(ck['EM series']=='N2U') & (ck['type']=='chemical')].copy()
ck['sections']=pd.to_numeric(ck['sec-tions'],errors='coerce')
csite = ck.groupby(['pre','post']).agg(cook_nsites=('sections','size'),
                                       cook_sec_mean=('sections','mean'),
                                       cook_sec_sum=('sections','sum')).reset_index()
cm = feat.merge(csite, left_on=['na','nb'], right_on=['pre','post'], how='inner')
if len(cm)>10:
    r1=stats.spearmanr(cm.n_sites,cm.cook_nsites); r2=stats.spearmanr(cm.sec_extent_mean,cm.cook_sec_mean)
    print(f'  n_sites Witvliet vs Cook: rho={r1.correlation:.3f} (n={len(cm)})')
    print(f'  sec-extent Witvliet vs Cook: rho={r2.correlation:.3f}')
    add('Cook','witvliet_vs_cook_nsites', rho=float(r1.correlation), n=len(cm))
    add('Cook','witvliet_vs_cook_secextent', rho=float(r2.correlation), n=len(cm))

# ---------------------------------------------------------------- write residual/stat table
out_rows=[]
for sec,items in res.items():
    for it in items:
        out_rows.append(dict(analysis=sec, **it))
rdf=pd.DataFrame(out_rows)
rdf.to_csv(OUT+'partD_subcellular_residuals.csv', index=False)
print(f'\n[save] partD_subcellular_residuals.csv  {rdf.shape}')
print('\nDONE')
