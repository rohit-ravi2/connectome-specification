"""BRANCH 10 — is the arrow a 1D global RANK, and does it EQUAL the functional hierarchy? The linear antisym model
is inherently a rank: it scores each neuron s_n = w.expr_n and predicts a->b iff s_a>s_b. Two tests:
(1) RANK vs PAIR-SPECIFIC: compare the linear-antisym (pure global rank, can only express s_a-s_b) to a NONLINEAR
    joint model (HistGBM on [expr_a || expr_b], can capture pair-specific direction). If GBM>>linear, direction
    carries non-rank structure; if ~equal, the arrow is a pure 1D identity rank.
(2) RANK vs HIERARCHY: does the identity-rank s_n correlate with the trophic level h_n (functional sensory->motor
    axis)? If yes, identity encodes each neuron's hierarchical position and synapses flow down it — unifying the
    arrow (identity) with the functional hierarchy (dynamics), even though Branch 7 showed it's NOT birth-order.
"""
import warnings,sys,openpyxl; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
def P(*a): print(*a); sys.stdout.flush()
D="../data/celegans/data"
cg=pd.read_csv(f"{D}/expression/cengen/thresholded/021821_liberal_threshold1.csv")
classes=list(cg.columns[3:]); Xf=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values.T>0
br=Xf.mean(0); Xc=Xf[:,(br>0.05)&(br<0.95)].astype(float); cls_idx={c:k for k,c in enumerate(classes)}
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
Ed=[]
for r in range(hr+1,len(A)):
    cpre=n2c(A[r][hc])
    if not cpre: continue
    for c in range(hc+1,len(A[r])):
        cpost=n2c(cols[c]); v=A[r][c]
        if cpost and cpost!=cpre and v not in (None,''):
            try:
                if float(v)>0: Ed.append((cpre,cpost,float(v)))
            except: pass
chem=set((a,b) for a,b,w in Ed); asym=[(a,b) for (a,b) in chem if (b,a) not in chem]
# (1) rank (linear antisym) vs pair-specific (nonlinear joint)
Fd=[];Fj=[];y=[];g=[]
for k,(a,b) in enumerate(asym):
    ea,eb=Xc[cls_idx[a]],Xc[cls_idx[b]]
    Fd+=[ea-eb,eb-ea]; Fj+=[np.concatenate([ea,eb]),np.concatenate([eb,ea])]; y+=[1,0]; g+=[k,k]
Fd=np.array(Fd);Fj=np.array(Fj);y=np.array(y);g=np.array(g)
gk=GroupKFold(5)
a_rank=roc_auc_score(y,cross_val_predict(LogisticRegression(max_iter=1000,C=0.5),Fd,y,cv=gk,groups=g,method='predict_proba')[:,1])
a_joint=roc_auc_score(y,cross_val_predict(HistGradientBoostingClassifier(max_iter=200,random_state=0),Fj,y,cv=gk,groups=g,method='predict_proba')[:,1])
P(f"BRANCH 10 — {len(asym)} asym pairs")
P(f"\n  (1) RANK vs PAIR-SPECIFIC:")
P(f"    linear-antisym (pure 1D identity rank) AUC = {a_rank:.3f}")
P(f"    nonlinear joint [expr_a||expr_b] (pair-specific) AUC = {a_joint:.3f}  (Δ {a_joint-a_rank:+.3f})")
P(f"    -> {'arrow is ~PURE 1D identity rank (pair-specific adds little)' if a_joint-a_rank<0.03 else 'direction carries PAIR-SPECIFIC structure beyond a global rank'}")
# per-neuron identity-rank score s_n = w.expr (fit on all asym)
w=LogisticRegression(max_iter=1000,C=0.5).fit(Fd,y).coef_[0]
neur=sorted({a for a,b,w_ in Ed}|{b for a,b,w_ in Ed})
s={n:float(Xc[cls_idx[n]]@w) for n in neur}
# how much of the directed connectome flows DOWN the identity-rank?
down=np.mean([1 if s[a]>s[b] else 0 for a,b,w_ in Ed])
downw=np.average([1 if s[a]>s[b] else 0 for a,b,w_ in Ed],weights=[w_ for a,b,w_ in Ed])
P(f"\n    fraction of directed edges flowing DOWN the identity-rank (s_pre>s_post): {down:.3f} (weight-wtd {downw:.3f})")
# (2) rank vs functional hierarchy (trophic)
idx={n:i for i,n in enumerate(neur)}; M=len(neur); W=np.zeros((M,M))
for a,b,w_ in Ed: W[idx[a],idx[b]]+=w_
indeg=W.sum(0); outdeg=W.sum(1)
h=np.linalg.lstsq(np.diag(indeg+outdeg)-(W+W.T)+np.ones((M,M))/M, indeg-outdeg, rcond=None)[0]; h=h-h.min()
sv=np.array([s[n] for n in neur])
rho,pp=spearmanr(sv,h)
P(f"\n  (2) identity-RANK s_n vs functional TROPHIC level h_n: Spearman r={rho:+.3f} p={pp:.1e}")
P(f"     (positive s = presynaptic-source identity; low h = sensory-source) -> expect NEGATIVE if identity-rank = hierarchy")
# degree control
def resid(a,c): b=np.polyfit(c,a,1); return a-np.polyval(b,c)
deg=indeg+outdeg; rc,pc=spearmanr(resid(sv,deg),resid(h,deg))
P(f"     degree-controlled partial r={rc:+.3f} p={pc:.1e}")
P(f"\n  VERDICT: {'the arrow is a ~1D identity RANK that IS the functional hierarchy — identity encodes each neurons pre/post rank and synapses flow down it (|r|>0.3, degree-robust); unifies arrow(identity)+hierarchy(dynamics), distinct from birth-order' if abs(rc)>0.3 and pc<0.01 else 'identity-rank predicts direction but is NOT simply the trophic hierarchy (r=%.2f) — the pairwise arrow is finer than the global functional axis'%rc}.")
