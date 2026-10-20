"""BRANCH 21 — is the identity-rank the PRIMARY AXIS of transcriptomic diversity? The arrow is a whole-transcriptome
scalar s_n = w.expr_n. Is w just recovering a top principal component of neuronal expression — i.e., is 'presynaptic-
source identity' literally the main axis along which neuron types differ? Correlate s with the top PCs of the CeNGEN
variable-gene matrix (over the SAME neurons/genes), and measure how much of s is captured by the leading PCs. If
s aligns with PC1/PC2, the connectome's direction axis = the dominant axis of neuronal type variation.
"""
import warnings,sys,openpyxl; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from scipy.stats import spearmanr,pearsonr
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
F=[];y=[]
for a,b in asym:
    d=Xc[cls_idx[a]]-Xc[cls_idx[b]]; F+=[d,-d]; y+=[1,0]
w=LogisticRegression(max_iter=1000,C=0.5).fit(np.array(F),np.array(y)).coef_[0]
# neurons that appear in the connectome, their expression + rank
neur=sorted({a for a,b in chem}|{b for a,b in chem}); neur=[n for n in neur if n in cls_idx]
Xn=np.array([Xc[cls_idx[n]] for n in neur]); s=Xn@w
# PCA on the neuron x variable-gene matrix
pca=PCA(n_components=10).fit(Xn); PCs=pca.transform(Xn); ev=pca.explained_variance_ratio_
P(f"BRANCH 21 — {len(neur)} connectome neurons x {Xn.shape[1]} variable genes; PCA")
P(f"  top-10 explained variance: {', '.join(f'{e:.2f}' for e in ev)}")
P(f"\n  identity-rank s vs expression PCs (Pearson |r|):")
best=0;bi=-1
for k in range(6):
    r,_=pearsonr(s,PCs[:,k]); rs,_=spearmanr(s,PCs[:,k])
    P(f"    PC{k+1} (var {ev[k]:.2f}): pearson r={r:+.3f} spearman={rs:+.3f}")
    if abs(r)>best: best=abs(r); bi=k
# how much of s do leading PCs explain (regress s on PC1..k)?
from numpy.linalg import lstsq
for K in [1,2,3,5,10]:
    Xk=np.c_[np.ones(len(s)),PCs[:,:K]]; coef=lstsq(Xk,s,rcond=None)[0]; pred=Xk@coef
    r2=1-((s-pred)**2).sum()/((s-s.mean())**2).sum()
    P(f"    s explained by top-{K} PCs: R2={r2:.3f}")
P(f"\n  best single PC = PC{bi+1} (|r|={best:.2f}, explains {ev[bi]:.2f} of expression variance)")
P(f"\n  VERDICT: {'the identity-rank IS ~the primary axis of transcriptomic diversity — s aligns with a leading expression PC (|r|>0.6): the connectomes direction axis is literally the dominant axis of neuronal type variation' if best>0.6 and bi<2 else ('s is a DISTRIBUTED combination not any single PC (best |r|=%.2f with PC%d) — the direction axis is a specific low-variance identity direction, not the top axis of type diversity'%(best,bi+1))}.")
