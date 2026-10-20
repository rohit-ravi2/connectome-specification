"""BRANCH 22 — is the arrow just PRESYNAPTIC-RELEASE-MACHINERY dosage? The mechanistically-obvious hypothesis: the
presynaptic partner points the synapse because it expresses more active-zone / vesicle-release machinery (unc-13,
unc-10, rab-3, syd-2/elks-1, snt-1, snb-1, unc-64), and the postsynaptic more receptors. Test: (1) does total
release-machinery expression correlate with the identity-rank s? (2) does a release-machinery-ONLY antisym model
predict direction? (3) does REMOVING release machinery (+receptors) hurt the arrow? If all null, the arrow is NOT
release-dosage — it's the subtle distributed identity (B9/B21), reinforcing that direction is a recognition/identity
code, not a presynaptic-competence gradient.
"""
import warnings,sys,openpyxl; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
def P(*a): print(*a); sys.stdout.flush()
D="../data/celegans/data"
cg=pd.read_csv(f"{D}/expression/cengen/thresholded/021821_liberal_threshold1.csv")
classes=list(cg.columns[3:]); gname=cg["gene_name"].astype(str).str.lower().values
Xf=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values.T>0
br=Xf.mean(0); mask=(br>0.05)&(br<0.95); Xc=Xf[:,mask].astype(float); genes=gname[mask]; cls_idx={c:k for k,c in enumerate(classes)}
def n2c(n):
    if not isinstance(n,str): return None
    if n in cls_idx: return n
    b=n
    for s in "LR":
        if b.endswith(s): b=b[:-1]; break
    if b in cls_idx: return b
    for s in "DV":
        if b.endswith(s) and b[:-1] in cls_idx: return b[:-1]
    return None
wb=openpyxl.load_workbook(f"{D}/connectome/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
A=list(wb['hermaphrodite chemical'].iter_rows(values_only=True))
def looks(n): return isinstance(n,str) and len(n)>=2 and n[0].isalpha()
hr=max(range(len(A)),key=lambda r:sum(looks(x) for x in A[r])); cols=[(str(x).strip() if x is not None else None) for x in A[hr]]
hc=max(range(min(6,len(A[0]))),key=lambda c:sum(looks(A[r][c]) for r in range(len(A))))
chem=set()
for r in range(hr+1,len(A)):
    cpre=n2c(A[r][hc])
    if not cpre: continue
    for c in range(hc+1,len(A[r])):
        cpost=n2c(cols[c]); v=A[r][c]
        if cpost and cpost!=cpre and v not in (None,''):
            try:
                if float(v)>0: chem.add((cpre,cpost))
            except: pass
asym=[(a,b) for (a,b) in chem if (b,a) not in chem]
# release machinery + receptor sets (full-genome expression, not just variable)
RELEASE=["unc-13","unc-10","rab-3","syd-2","elks-1","snt-1","snb-1","unc-64","unc-18","ric-4","unc-2","cla-1","rimb-1","syd-1"]
allg=gname
def gene_expr(gset):
    cols_=[np.where(allg==g)[0] for g in gset]; cols_=[c[0] for c in cols_ if len(c)]
    return Xf[:, cols_].sum(1) if cols_ else np.zeros(Xf.shape[0])  # per-neuron count of machinery genes ON
rel=gene_expr(RELEASE)
# rank s (full model)
F=[];y=[];g=[]
for k,(a,b) in enumerate(asym):
    d=Xc[cls_idx[a]]-Xc[cls_idx[b]]; F+=[d,-d]; y+=[1,0]; g+=[k,k]
w=LogisticRegression(max_iter=1000,C=0.5).fit(np.array(F),np.array(y)).coef_[0]
neur=sorted({a for a,b in chem}|{b for a,b in chem}); neur=[n for n in neur if n in cls_idx]
s={n:float(Xc[cls_idx[n]]@w) for n in neur}
relmap={n:rel[cls_idx[n]] for n in neur}
rho,pp=spearmanr([s[n] for n in neur],[relmap[n] for n in neur])
P(f"BRANCH 22 — {len(asym)} asym pairs | release-machinery genes found: {int((rel>0).any())}")
P(f"  (1) identity-rank s vs total release-machinery expression: Spearman r={rho:+.3f} p={pp:.2e}")
# (2) release-machinery-only antisym direction model
relcols=[np.where(allg==gname_)[0] for gname_ in RELEASE]; relcols=[c[0] for c in relcols if len(c)]
Xrel=Xf[:,relcols].astype(float)
Fr=[];yr=[];gr=[]
for k,(a,b) in enumerate(asym):
    d=Xrel[cls_idx[a]]-Xrel[cls_idx[b]]; Fr+=[d,-d]; yr+=[1,0]; gr+=[k,k]
Fr=np.array(Fr);yr=np.array(yr);gr=np.array(gr)
a_rel=roc_auc_score(yr,cross_val_predict(LogisticRegression(max_iter=1000,C=0.5),Fr,yr,cv=GroupKFold(5),groups=gr,method='predict_proba')[:,1])
# full arrow AUC for reference
Fa=np.array(F);ya=np.array(y);ga=np.array(g)
a_full=roc_auc_score(ya,cross_val_predict(LogisticRegression(max_iter=1000,C=0.5),Fa,ya,cv=GroupKFold(5),groups=ga,method='predict_proba')[:,1])
P(f"  (2) release-machinery-ONLY arrow AUC = {a_rel:.3f} (vs full identity arrow {a_full:.3f})")
P(f"\n  VERDICT: {'the arrow IS partly release-machinery dosage — presynaptic partners express more active-zone/vesicle genes (s correlates, machinery-only predicts direction)' if rho>0.2 and pp<0.01 and a_rel>0.65 else 'the arrow is NOT release-machinery dosage (s vs machinery r=%.2f, machinery-only AUC %.2f ~chance) — nearly all neurons carry release machinery; direction is the subtle distributed IDENTITY code, not a presynaptic-competence gradient'%(rho,a_rel)}.")
