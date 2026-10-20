"""PART 1 — CROSS-PHYLUM UNIVERSALITY (Drosophila larva, Winding 2023).

C. elegans finding: geometry codes the EDGE (who connects); a whole-cell identity
scalar codes the ARROW (which way), and that scalar = functional sensory->motor
trophic hierarchy h (computed label-free via MacKay trophic-level lstsq).

Questions:
 (2) compute trophic h per neuron (label-free)
 (3) does h recover annotated cell-type ordering (sensory=source, DN/motor=sink)?
 (4) what predicts DIRECTION of an edge: degree, cell-type label, h?
 (5) Dros F0~0.635 vs Celegans 0.407 — less hierarchical, or DEEPER (more layers)?
 (6) does edge-prediction separate from direction-prediction? degree confound?
"""
import warnings,sys; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import lsqr
from scipy.stats import spearmanr
np.random.seed(0)
def P(*a): print(*a); sys.stdout.flush()
DRO="../data/celegans/data/drosophila/winding2023/Supplementary-Data-S1"
OUT="../accounting/principles"

# ---------- load ----------
M=pd.read_csv(f"{DRO}/all-all_connectivity_matrix.csv",index_col=0)
M.index=M.index.astype(str); M.columns=M.columns.astype(str)
# matrix orientation: rows=source(pre), cols=target(post)? check via inputs/outputs later; Winding convention row->col
W=M.values.astype(float); ids0=list(M.index)
P(f"raw connectivity matrix {M.shape}  total synapses {W.sum():.0f}")

deg=W.sum(0)+W.sum(1); keep=deg>0
W=W[keep][:,keep]; ids=[ids0[i] for i in range(len(ids0)) if keep[i]]; n=W.shape[0]
P(f"connected neurons {n}  directed edges {(W>0).sum()}  (density {(W>0).mean():.4f})")

# ---------- annotations ----------
ann=pd.read_csv(f"{DRO}/annotations.csv")
skid2type={}; skid2mod={}
for _,row in ann.iterrows():
    for col in ['left_id','right_id']:
        v=str(row[col])
        if v!='no pair':
            skid2type[v]=row['celltype']; skid2mod[v]=str(row['additional_annotations'])
ctype=np.array([skid2type.get(s,'NA') for s in ids])
P(f"annotated {np.sum(ctype!='NA')}/{n}")

# expected coarse hierarchy rank (source=0 ... sink=high). label-derived ONLY for validation.
rank_map={
 'sensory':0,'ascending':0,          # inputs
 'PN':1,'PN-somato':1,'LN':1,         # 2nd order / local
 'LHN':2,'KC':2,'MBIN':2,             # 3rd order
 'MBON':3,'MB-FFN':3,                 # MB output
 'CN':4,'MB-FBN':4,                   # convergence/feedback
 'pre-DN-SEZ':5,'pre-DN-VNC':5,       # pre-output
 'DN-SEZ':6,'DN-VNC':6,'RGN':6,       # OUTPUT / sink
}
exp_rank=np.array([rank_map.get(t,np.nan) for t in ctype])

# ---------- trophic level h (MacKay, label-free) ----------
Ws=csr_matrix(W)
indeg=np.array(W.sum(0)).ravel(); outdeg=np.array(W.sum(1)).ravel()
u=indeg+outdeg; v=indeg-outdeg
L=diags(u)-(Ws+Ws.T)
A=L+csr_matrix(np.ones((1,n))).T@csr_matrix(np.ones((1,n)))/n
h=lsqr(A,v,atol=1e-10,btol=1e-10,iter_lim=5000)[0]
h=h-h.min()
P(f"\n== TROPHIC LEVEL h ==  range {h.max():.2f}  std {h.std():.2f}")

# incoherence F0
r,c=Ws.nonzero(); wv=np.array(W[r,c]).ravel()
F0=float((wv*(h[c]-h[r]-1)**2).sum()/wv.sum())
F0n=[float((wv*(h[np.random.permutation(n)][c]-h[np.random.permutation(n)][r]-1)**2).sum()/wv.sum()) for _ in range(20)]
up=(wv*(h[c]>h[r])).sum()/wv.sum()
P(f"F0 incoherence = {F0:.3f}   shuffled-h null = {np.mean(F0n):.2f}±{np.std(F0n):.2f}   (Celegans 0.407)")
P(f"forward (up-hierarchy) flow fraction = {up:.3f}   (Celegans 0.88)")

# ---------- (3) does h recover cell-type ordering? label-free h vs label rank ----------
m=~np.isnan(exp_rank)
rho,pv=spearmanr(h[m],exp_rank[m])
P(f"\n== h vs expected cell-type rank ==  Spearman rho={rho:.3f} (p={pv:.1e}, n={m.sum()})")
P("  mean trophic h by cell type (sorted):")
rows=[]
for t in pd.unique(ctype):
    if t=='NA':continue
    sel=ctype==t
    rows.append((t,sel.sum(),h[sel].mean(),h[sel].std(),rank_map.get(t,np.nan)))
for t,ns,mn,sd,er in sorted(rows,key=lambda x:x[2]):
    P(f"    {t:14s} n={ns:4d}  h={mn:5.2f}±{sd:4.2f}  exp_rank={er}")

# specifically sensory vs DN output
sens=h[ctype=='sensory']; dn=h[np.isin(ctype,['DN-VNC','DN-SEZ'])]
P(f"\n  sensory h mean={sens.mean():.2f}  DN(output) h mean={dn.mean():.2f}  gap={dn.mean()-sens.mean():.2f}")
# AUC: does h rank sensory below DN?
from itertools import product
auc=np.mean([hs<hd for hs in sens for hd in dn])
P(f"  P(h_sensory < h_DN) = {auc:.3f}  (label-free h separating source from sink)")

# ---------- (5) less hierarchical vs deeper? decompose ----------
# depth = spread of h among the FORWARD backbone. Normalize F0 by variance available.
# A DAG that is deep but coherent has F0~0. High F0 = genuine feedback/incoherence.
# Compare: fraction of edge weight that is "backward" (h_post < h_pre) = true incoherence.
back=(wv*(h[c]<h[r])).sum()/wv.sum()
P(f"\n== depth vs incoherence decomposition ==")
P(f"  h dynamic range (depth proxy) = {h.max():.2f}   Celegans reference typically ~4-6")
P(f"  backward-flow weight fraction (true feedback) = {back:.3f}")
P(f"  forward but where |Δh|!=1 contributes to F0 too. mean forward Δh = {np.average((h[c]-h[r])[h[c]>h[r]], weights=wv[h[c]>h[r]]):.2f}")
# F0 = <(Δh-1)^2>. Decompose into backward part vs forward-spread part
dh=h[c]-h[r]
f0_back=float((wv[dh<0]*(dh[dh<0]-1)**2).sum()/wv.sum())
f0_fwd=float((wv[dh>=0]*(dh[dh>=0]-1)**2).sum()/wv.sum())
P(f"  F0 contribution: backward edges={f0_back:.3f}  forward edges={f0_fwd:.3f}  (backward frac of F0={f0_back/F0:.2f})")

# ---------- (4)+(6) EDGE prediction vs DIRECTION prediction ----------
P(f"\n== EDGE vs DIRECTION prediction ==")
# EDGE: among all ordered pairs, is there an edge? predictors: degree product (topology).
# sample negatives
rng=np.random.default_rng(1)
pos=list(zip(r,c))
posset=set(pos)
neg=[]
while len(neg)<len(pos):
    i=rng.integers(n); j=rng.integers(n)
    if i!=j and (i,j) not in posset: neg.append((i,j))
def feats_edge(pairs):
    X=[]
    for i,j in pairs:
        X.append([np.log1p(outdeg[i]),np.log1p(indeg[j]),np.log1p(outdeg[i])*np.log1p(indeg[j])])
    return np.array(X)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
Xe=np.vstack([feats_edge(pos),feats_edge(neg)]); ye=np.r_[np.ones(len(pos)),np.zeros(len(neg))]
pe=cross_val_predict(LogisticRegression(max_iter=500),Xe,ye,cv=5,method='predict_proba')[:,1]
P(f"  EDGE (exists?) from DEGREE only: AUC={roc_auc_score(ye,pe):.3f}  [pure topology/degree confound]")

# DIRECTION: for each UNORDERED connected pair with asymmetry, predict which way is dominant.
# build pair dict
pairw={}
for a,b,w in zip(r,c,wv):
    key=(min(a,b),max(a,b))
    pairw.setdefault(key,[0,0])
    if a<b: pairw[key][0]+=w
    else:   pairw[key][1]+=w
dir_pairs=[(i,j,fw,bw) for (i,j),(fw,bw) in pairw.items() if fw!=bw]
P(f"  reciprocal/asymmetric pairs for direction test: {len(dir_pairs)}")
# label y=1 if forward (i->j) dominant. features from h, degree, celltype-rank
yh=[]; Xh_h=[]; Xh_deg=[]; Xh_rank=[]
for i,j,fw,bw in dir_pairs:
    y=1 if fw>bw else 0
    yh.append(y)
    Xh_h.append([h[i]-h[j]])
    Xh_deg.append([np.log1p(outdeg[i])-np.log1p(outdeg[j]),np.log1p(indeg[i])-np.log1p(indeg[j])])
    er_i=exp_rank[i] if not np.isnan(exp_rank[i]) else 0
    er_j=exp_rank[j] if not np.isnan(exp_rank[j]) else 0
    Xh_rank.append([er_i-er_j])
yh=np.array(yh)
for name,X in [('trophic h (label-free)',np.array(Xh_h)),
               ('degree asym (label-free)',np.array(Xh_deg)),
               ('celltype rank (labels)',np.array(Xh_rank))]:
    ph=cross_val_predict(LogisticRegression(max_iter=500),X,yh,cv=5,method='predict_proba')[:,1]
    P(f"  DIRECTION (which way) from {name:28s}: AUC={roc_auc_score(yh,ph):.3f}")
# combined
Xall=np.hstack([np.array(Xh_h),np.array(Xh_deg)])
pa=cross_val_predict(LogisticRegression(max_iter=500),Xall,yh,cv=5,method='predict_proba')[:,1]
P(f"  DIRECTION from h+degree combined: AUC={roc_auc_score(yh,pa):.3f}")

# does h add over degree for direction? and is h itself just degree?
rho_hd,_=spearmanr(h, indeg-outdeg)
P(f"\n  h vs (indeg-outdeg) Spearman={rho_hd:.3f}  [h partly a degree-imbalance restatement — CONFOUND note]")
P(f"  h vs total-degree Spearman={spearmanr(h,indeg+outdeg)[0]:.3f}")

# save
np.savez(f"{OUT}/A1-crossphylum-out.npz",h=h,ids=np.array(ids),ctype=ctype,exp_rank=exp_rank,
         indeg=indeg,outdeg=outdeg,F0=F0)
P("\n[saved A1-crossphylum-out.npz]")
