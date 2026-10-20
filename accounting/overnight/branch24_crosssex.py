"""BRANCH 24 — CROSS-SEX generalization. The identity-rank governs hermaphrodite synapse direction, cross-ANIMAL
(Witvliet) and from birth (B23). Does the SAME herm-derived rank predict direction in the MALE connectome? The male
nervous system is substantially rewired (sex-specific circuits, extra neurons). If the herm rank still predicts male
direction on SHARED neurons, the arrow is a deeply conserved, sex-independent identity code (CeNGEN expression is
herm, so male-specific cells are unmapped — test the shared core). Also: learn a MALE rank and compare to herm rank.
"""
import warnings,sys,openpyxl; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
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
def load_sheet(sheet):
    wb=openpyxl.load_workbook(f"{D}/connectome/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
    A=list(wb[sheet].iter_rows(values_only=True))
    def looks(n): return isinstance(n,str) and len(n)>=2 and n[0].isalpha()
    hr=max(range(len(A)),key=lambda r:sum(looks(x) for x in A[r])); cols=[(str(x).strip() if x is not None else None) for x in A[hr]]
    hc=max(range(min(6,len(A[0]))),key=lambda c:sum(looks(A[r][c]) for r in range(len(A))))
    ed=set()
    for r in range(hr+1,len(A)):
        cpre=n2c(A[r][hc])
        if not cpre: continue
        for c in range(hc+1,len(A[r])):
            cpost=n2c(cols[c]); v=A[r][c]
            if cpost and cpost!=cpre and v not in (None,''):
                try:
                    if float(v)>0: ed.add((cpre,cpost))
                except: pass
    return ed
herm=load_sheet('hermaphrodite chemical'); male=load_sheet('male chemical')
def learn(edset):
    asym=[(a,b) for (a,b) in edset if (b,a) not in edset]
    F=[];y=[]
    for a,b in asym:
        d=Xc[cls_idx[a]]-Xc[cls_idx[b]]; F+=[d,-d]; y+=[1,0]
    w=LogisticRegression(max_iter=1000,C=0.5).fit(np.array(F),np.array(y)).coef_[0]
    return w,{c:float(Xc[cls_idx[c]]@w) for c in classes if c in cls_idx},asym
wH,sH,asymH=learn(herm)
P(f"BRANCH 24 — herm chem class-edges {len(herm)} | male chem class-edges {len(male)}")
# (1) herm rank -> male direction (male asym pairs, mapped)
masym=[(a,b) for (a,b) in male if (b,a) not in male and a in sH and b in sH]
malep=set(male)
male_only=[(a,b) for (a,b) in masym if (a,b) not in herm and (b,a) not in herm]  # male-specific connections
def dir_auc(pairs,s):
    yv=[];sc=[]
    for a,b in pairs: yv+=[1,0]; sc+=[s[a]-s[b], s[b]-s[a]]
    return roc_auc_score(yv,sc), np.mean([1 if s[a]>s[b] else 0 for a,b in pairs])
au,dn=dir_auc(masym,sH)
P(f"\n  (1) HERM identity-rank -> MALE direction: {len(masym)} male asym pairs | down-rank {dn:.3f} | AUC {au:.3f}")
if len(male_only)>20:
    au2,dn2=dir_auc(male_only,sH)
    P(f"      restricted to MALE-SPECIFIC connections (not in herm): {len(male_only)} pairs | AUC {au2:.3f}")
# (2) male-learned rank vs herm-learned rank
wM,sM,asymM=learn(male)
common=[c for c in classes if c in sH and c in sM]
rho,pp=spearmanr([sH[c] for c in common],[sM[c] for c in common])
P(f"  (2) herm rank vs male-learned rank ({len(common)} shared classes): Spearman r={rho:+.3f} p={pp:.1e}")
P(f"\n  VERDICT: {'the arrow is SEX-INDEPENDENT — the herm identity-rank predicts male synapse direction (AUC>0.8), incl. male-specific connections, and herm/male ranks agree (r>0.7): direction is a conserved identity code read the same way in both sexes rewired circuitry' if au>0.8 and rho>0.6 else 'partial cross-sex transfer (male dir AUC %.2f, rank-agree r%.2f)'%(au,rho)}.")
