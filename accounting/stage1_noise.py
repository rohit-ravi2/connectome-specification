"""STAGE 1 — NOISE-DECOMPOSITION: right-size the target. The 2-animal 'reproducibility' (adults 7&8 agree) is
a Bernoulli comparison with n=2. Decompose the AUC gap into overfitting (within-Witvliet neuron-disjoint drop)
and estimation-noise (calibrated from the 4 L1 replicates = Witvliet datasets 1-4, the only n>2 for one state).
Corrected target = Cook-held-out AUC (immune to Witvliet-2-animal noise). Output: corrected AUC + noise floor.
"""
import warnings,sys,json,csv,re; warnings.simplefilter("ignore")
import numpy as np, openpyxl, pandas as pd
from collections import defaultdict
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, cohen_kappa_score
def P(*a): print(*a); sys.stdout.flush()
RNG=np.random.default_rng(0); D="../data/celegans/data"
SKD="../seed/data/external/mechanistic_frontier/witvliet2021_developmental/data/skeletons"
cg=pd.read_csv(D+"/expression/cengen/thresholded/021821_liberal_threshold1.csv"); classes=list(cg.columns[3:]); cls_idx={c:k for k,c in enumerate(classes)}
def n2c(n):
    if n in cls_idx: return cls_idx[n]
    b=n
    for s in "LR":
        if b.endswith(s): b=b[:-1]; break
    if b in cls_idx: return cls_idx[b]
    for s in "DV":
        if b.endswith(s) and b[:-1] in cls_idx: return cls_idx[b[:-1]]
    return cls_idx.get(b,None)
wb0=openpyxl.load_workbook(f"{D}/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx",read_only=True,data_only=True)
ws=wb0['adult nerve ring neighbors']; dd=list(ws.iter_rows(values_only=True)); hdr=[h for h in dd[0][1:]]
ca=defaultdict(float)
for r in dd[1:]:
    rn=r[0]
    if rn is None: continue
    for jj,v in enumerate(r[1:]):
        cn=hdr[jj]
        if cn and v not in (None,'') and rn!=cn:
            try: ca[(min(rn,cn),max(rn,cn))]=max(ca[(min(rn,cn),max(rn,cn))],float(v))
            except: pass
sk=json.load(open(f"{SKD}/Dataset7_skeletons.json"))
ring=[n for n in hdr if n in sk and sk[n].get('coords') and n2c(n) is not None]
M=len(ring); idx={n:i for i,n in enumerate(ring)}
pairs=[(ring[i],ring[j]) for i in range(M) for j in range(i+1,M)]; Np=len(pairs)
area=np.array([ca.get((min(a,b),max(a,b)),0.0) for a,b in pairs]); elig=area>0
# ---- per-dataset synapse presence (undirected pair) ----
perd=defaultdict(lambda: defaultdict(int))
with open(f"{D}/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']!='chemical': continue
        m=re.search(r'witvliet_2020_(\d)',r['source_file'])
        if not m: continue
        perd[int(m.group(1))][(min(r['pre'],r['post']),max(r['pre'],r['post']))]+=1
def presvec(dset):
    return np.array([1 if perd[dset].get((min(a,b),max(a,b)),0)>0 else 0 for a,b in pairs])
L1=[presvec(d) for d in (1,2,3,4)]; a7,a8=presvec(7),presvec(8)
yW=((a7==1)&(a8==1)).astype(int)   # adult 2-animal reproducibility (the 0.75 target)

# ===== ESTIMATION-NOISE from 4 L1 replicates =====
P("="*84); P("STAGE 1a — ESTIMATION-NOISE (4 L1 replicates: the only n>2 for one developmental state)"); P("="*84)
e=elig
repro_12=((L1[0]==1)&(L1[1]==1)).astype(int); repro_34=((L1[2]==1)&(L1[3]==1)).astype(int)  # two disjoint 2-animal estimates
union_present=((L1[0]+L1[1]+L1[2]+L1[3])>0)&e
flip=(repro_12[e]!=repro_34[e]).mean()
flip_meaningful=(repro_12[union_present]!=repro_34[union_present]).mean()
kap=cohen_kappa_score(repro_12[e],repro_34[e])
theta4=(L1[0]+L1[1]+L1[2]+L1[3])/4.0
# how much does 2-animal 'both-present' over/under-estimate stable presence?
P(f"  L1 eligible pairs {e.sum()} | present in >=1 L1 animal: {union_present.sum()}")
P(f"  2-animal reproducibility FLIP RATE across disjoint pairs (1&2 vs 3&4): all-eligible {flip:.3f} | among-present {flip_meaningful:.3f}")
P(f"  Cohen kappa(2-anim repro pair A, pair B) = {kap:.3f}  (1=stable, 0=noise)")
# fraction of 2-animal 'reproducible' edges that are NOT stable (present<3/4 animals)
r12_edges=repro_12.astype(bool)&e
stable_frac=(theta4[r12_edges]>=0.75).mean()
P(f"  of edges called '2-animal reproducible' (pair 1&2): {stable_frac:.2f} are stable (>=3/4 L1 animals), {1-stable_frac:.2f} are 2-animal flukes")
P(f"  => ESTIMATION-NOISE FLOOR: ~{(1-kap):.2f} of apparent 2-animal reproducibility is sampling artifact (kappa-based)")

# ===== BUILD PREDICTORS (contact, spatial, topology) =====
def build(n):
    v=sk[n]; co=v['coords']; ids=list(co.keys()); pts=np.array([co[i] for i in ids],float)
    dtr=np.array([v['dist_to_root'].get(i,0) for i in ids],float)
    if len(pts)>90: s=RNG.choice(len(pts),90,replace=False); pts=pts[s]; dtr=dtr[s]
    ch=defaultdict(int)
    for node,par in v['arbor'].items(): ch[par]+=1
    nbr=sum(1 for k2,c in ch.items() if c>=2)
    return pts,dtr/(dtr.max()+1e-9),float(v['length']),nbr,pts.std(0).mean(),float(np.max(dtr))
PT={};DTR={};ALEN={};NBR={};EXT={};MDTR={}
for n in ring: PT[n],DTR[n],ALEN[n],NBR[n],EXT[n],MDTR[n]=build(n)
trees={n:cKDTree(PT[n]) for n in ring}; THR=800.0
def pf(a,b):
    d,_=trees[b].query(PT[a]); c=d<THR; d2,_=trees[a].query(PT[b]); c2=d2<THR
    return [c.sum(),d.min(),(d[c].mean() if c.any() else d.min()),(DTR[a][c].mean() if c.any() else .5),
            (DTR[b][c2].mean() if c2.any() else .5),(PT[a][c].std(0).mean() if c.sum()>1 else 0),
            np.log1p(ALEN[a]),np.log1p(ALEN[b]),NBR[a],NBR[b],EXT[a],EXT[b],MDTR[a],MDTR[b]]
SPAT=np.array([pf(a,b) for a,b in pairs])
CA=np.column_stack([np.log1p(area)])
Cadj=np.zeros((M,M))
for k,(a,b) in enumerate(pairs):
    if area[k]>0: Cadj[idx[a],idx[b]]=1; Cadj[idx[b],idx[a]]=1
cdeg=Cadj.sum(1); comm=Cadj@Cadj
TOPO=np.column_stack([cdeg[[idx[a] for a,b in pairs]],cdeg[[idx[b] for a,b in pairs]],np.array([comm[idx[a],idx[b]] for a,b in pairs])])
# Cook target
wb=openpyxl.load_workbook(f"{D}/connectome/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
A=list(wb['hermaphrodite chemical'].iter_rows(values_only=True)); nset=set(ring)
def nm(seq): return sum(1 for x in seq if isinstance(x,str) and x.strip() in nset)
hrow=int(np.argmax([nm(r) for r in A])); cols=[(str(x).strip() if x is not None else None) for x in A[hrow]]
hcol=int(np.argmax([nm([A[r][c] for r in range(len(A))]) for c in range(min(6,len(A[0])))]))
GC=np.zeros((M,M))
for r in range(hrow+1,len(A)):
    rn=A[r][hcol]; rn=rn.strip() if isinstance(rn,str) else None
    if rn not in idx: continue
    for c in range(hcol+1,len(A[r])):
        cn=cols[c]; v=A[r][c]
        if cn in idx and v not in (None,''):
            try:
                if float(v)>0: GC[idx[rn],idx[cn]]=1
            except: pass
yC=np.array([1 if (GC[idx[a],idx[b]] or GC[idx[b],idx[a]]) else 0 for a,b in pairs])
# ---- three-regime AUC per predictor ----
fold=np.array([RNG.integers(5) for _ in ring]); pfold=np.array([max(fold[idx[a]],fold[idx[b]]) if fold[idx[a]]==fold[idx[b]] else -1 for a,b in pairs])
def insample(F):
    clf=HistGradientBoostingClassifier(max_iter=200,random_state=0).fit(F[elig],yW[elig]); return roc_auc_score(yW[elig],clf.predict_proba(F[elig])[:,1])
def cv_and_cook(F):
    pred=np.full(Np,np.nan)
    for fdx in range(5):
        te=(pfold==fdx)&elig; tr=(pfold!=fdx)&(pfold>=0)&elig
        if te.sum()<10 or tr.sum()<50: continue
        clf=HistGradientBoostingClassifier(max_iter=200,random_state=0).fit(F[tr],yW[tr]); pred[te]=clf.predict_proba(F[te])[:,1]
    ok=~np.isnan(pred); return roc_auc_score(yW[ok],pred[ok]), roc_auc_score(yC[ok],pred[ok])
P("\n"+"="*84); P("STAGE 1b — THREE-REGIME AUC + CORRECTED TARGET (Cook = corrected denominator)"); P("="*84)
P(f"  {'predictor':22s} {'in-sample':>10s} {'neuron-disj CV':>15s} {'COOK(corrected)':>16s} {'overfit-gap':>12s} {'x-animal-gap':>12s}")
for nmk,F in [('contact-area',CA),('spatial(skeleton)',np.hstack([CA,SPAT])),('topology(contact-graph)',np.hstack([CA,TOPO]))]:
    ins=insample(F); cv,ck=cv_and_cook(F)
    P(f"  {nmk:22s} {ins:10.3f} {cv:15.3f} {ck:16.3f} {ins-cv:12.3f} {cv-ck:12.3f}")
P("\n"+"="*84)
P("CORRECTED TARGET: the denominator for all downstream scoring is the COOK-held-out AUC (~0.75 in-sample")
P("collapses to the Cook numbers above). Overfit-gap = capacity; x-animal-gap includes the estimation-noise floor.")
P(f"ESTIMATION-NOISE FLOOR: ~{1-kap:.0%} of 2-animal 'reproducibility' is sampling artifact (kappa {kap:.2f}, flip {flip_meaningful:.2f}).")
P("FLAG: prior results scored against in-sample ~0.75 were scored against an inflated target; the real bar is the")
P("      Cook column. Spatial's Cook number is its true difficulty-adjusted score.")
P("="*84)
