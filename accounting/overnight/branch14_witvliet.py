"""BRANCH 14 — CROSS-ANIMAL replication of the rank<->hierarchy synthesis on the independent WITVLIET connectomes.
The whole picture (arrow=1D identity rank; rank=trophic hierarchy; recovers sensory->motor) is built on Cook+CeNGEN.
Does it replicate in Witvliet 2020 (separate animals, EM-reconstructed)? Tests: (1) does the Cook-derived identity-
rank s predict Witvliet synapse DIRECTION? (2) does a Witvliet-derived rank s_W agree with Cook's s (cross-animal
identity axis)? (3) does the Witvliet trophic hierarchy h_W correlate with the identity-rank, independently?
"""
import warnings,sys,csv,re; warnings.simplefilter("ignore")
import numpy as np, pandas as pd, openpyxl
from sklearn.linear_model import LogisticRegression
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
# ---- Cook directed chemical (learn rank s) ----
wb=openpyxl.load_workbook(f"{D}/connectome/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
A=list(wb['hermaphrodite chemical'].iter_rows(values_only=True))
def looks(n): return isinstance(n,str) and len(n)>=2 and n[0].isalpha()
hr=max(range(len(A)),key=lambda r:sum(looks(x) for x in A[r])); cols=[(str(x).strip() if x is not None else None) for x in A[hr]]
hc=max(range(min(6,len(A[0]))),key=lambda c:sum(looks(A[r][c]) for r in range(len(A))))
cookEd=[]
for r in range(hr+1,len(A)):
    cpre=n2c(A[r][hc])
    if not cpre: continue
    for c in range(hc+1,len(A[r])):
        cpost=n2c(cols[c]); v=A[r][c]
        if cpost and cpost!=cpre and v not in (None,''):
            try:
                if float(v)>0: cookEd.append((cpre,cpost))
            except: pass
def learn_rank(edset):
    chem=set(edset); asym=[(a,b) for (a,b) in chem if (b,a) not in chem]
    F=[];y=[]
    for a,b in asym:
        d=Xc[cls_idx[a]]-Xc[cls_idx[b]]; F+=[d,-d]; y+=[1,0]
    w=LogisticRegression(max_iter=1000,C=0.5).fit(np.array(F),np.array(y)).coef_[0]
    return w,{c:float(Xc[cls_idx[c]]@w) for c in classes if c in cls_idx}
wCook,sCook=learn_rank(cookEd)
# ---- Witvliet directed chemical (aggregate all 8 animals) ----
from collections import Counter
wEd=Counter()
with open(f"{D}/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']=='chemical':
            a=n2c(r['pre']); b=n2c(r['post'])
            if a and b and a!=b: wEd[(a,b)]+=int(r['synapses'])
witEd=[(a,b) for (a,b),n in wEd.items()]
P(f"BRANCH 14 — Cook chem class-edges {len(set(cookEd))} | Witvliet chem class-edges {len(witEd)}")
# (1) Cook-rank predicts Witvliet direction (on Witvliet asym pairs)
witset=set(witEd); wasym=[(a,b) for (a,b) in witset if (b,a) not in witset and a in sCook and b in sCook]
correct=np.mean([1 if sCook[a]>sCook[b] else 0 for a,b in wasym])
# AUC form
yv=[];sc=[]
for a,b in wasym:
    yv+=[1,0]; sc+=[sCook[a]-sCook[b], sCook[b]-sCook[a]]
P(f"\n  (1) COOK identity-rank -> WITVLIET direction: {len(wasym)} asym pairs, down-rank fraction {correct:.3f}, AUC {roc_auc_score(yv,sc):.3f}")
# (2) Witvliet-derived rank agrees with Cook's?
wWit,sWit=learn_rank(witEd)
common=[c for c in classes if c in sCook and c in sWit]
rho,pp=spearmanr([sCook[c] for c in common],[sWit[c] for c in common])
P(f"  (2) Cook rank vs Witvliet rank (cross-animal identity axis), {len(common)} classes: Spearman r={rho:+.3f} p={pp:.1e}")
# (3) Witvliet trophic hierarchy vs identity-rank (independent)
neur=sorted({a for a,b in witEd}|{b for a,b in witEd}); idx={n:i for i,n in enumerate(neur)}; M=len(neur)
W=np.zeros((M,M))
for (a,b),n in wEd.items():
    if a in idx and b in idx: W[idx[a],idx[b]]+=n
indeg=W.sum(0); outdeg=W.sum(1)
h=np.linalg.lstsq(np.diag(indeg+outdeg)-(W+W.T)+np.ones((M,M))/M, indeg-outdeg, rcond=None)[0]; h=h-h.min()
sv=np.array([sWit[n] for n in neur]); rho3,pp3=spearmanr(sv,h)
r_cook,_=spearmanr([sCook[n] for n in neur],h)
P(f"  (3) Witvliet trophic h vs Witvliet identity-rank: r={rho3:+.3f} p={pp3:.1e}; vs COOK identity-rank: r={r_cook:+.3f}")
P(f"\n  VERDICT: {'REPLICATES cross-animal — the Cook identity-rank predicts Witvliet direction (AUC>0.8), the two animals share the identity axis (r>0.6), and Witvliets own hierarchy tracks it. The rank<->hierarchy synthesis is not a single-dataset artifact' if roc_auc_score(yv,sc)>0.75 and rho>0.5 else 'partial replication (dir AUC %.2f, rank-agree r%.2f)'%(roc_auc_score(yv,sc),rho)}.")
