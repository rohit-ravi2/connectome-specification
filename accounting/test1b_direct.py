"""TEST 1b — DIRECT prediction discriminator (confirms the loading result without the |loading|
metric). For each gene set: restrict CeNGEN to those genes, PCA->5 comps (capacity-matched),
endpoint sum+diff, predict FP beyond nonlinear degree. Compare modulatory(167) vs identity(59)
vs random matched-size draws. Question: does modulatory-expression predict FP better than
identity-expression, and does either beat random matched-size?
"""
import warnings; warnings.simplefilter("ignore")
import numpy as np, pandas as pd, re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
RNG=np.random.default_rng(1)
SC="../work"
d=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); m=np.load(f"{SC}/mode0.npz",allow_pickle=True)
I,J=d["I"],d["J"]; neurons=[str(x) for x in d["neurons"]]; logC=d["base"][:,0]; FP=m["FP"]; N=len(neurons)
deg=np.zeros(N)
for a,b in zip(I,J): deg[a]+=1; deg[b]+=1
cg=pd.read_csv("../data/celegans/data/expression/cengen/thresholded/021821_liberal_threshold1.csv")
genes=cg["gene_name"].astype(str).str.lower().values
classes=list(cg.columns[3:]); Mx=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values.T  # class x gene
cls_idx={c:k for k,c in enumerate(classes)}
def n2c(n):
    if n in cls_idx: return cls_idx[n]
    b=n
    for s in "LR":
        if b.endswith(s): b=b[:-1]; break
    if b in cls_idx: return cls_idx[b]
    for s in "DV":
        if b.endswith(s) and b[:-1] in cls_idx: return cls_idx[b[:-1]]
    return cls_idx.get(b,None)
c_of=np.array([n2c(n) if n2c(n) is not None else -1 for n in neurons])
has=c_of>=0; idxh=np.where(has)[0]
dI,dJ=deg[I],deg[J]; lo=np.minimum(dI,dJ); hi=np.maximum(dI,dJ)
Xbase=np.c_[logC,np.log1p(lo),np.log1p(hi),lo,hi,lo**2,hi**2,np.sqrt(lo),np.sqrt(hi),lo*hi]
elig=has[I]&has[J]; Xb=Xbase[elig]; Ie=I[elig]; y=FP[elig].astype(int)
gkf=GroupKFold(5)
def cv_auc(X):
    pred=np.zeros(len(y))
    for tr,te in gkf.split(X,y,Ie):
        pred[te]=LogisticRegression(max_iter=3000,class_weight="balanced",C=0.5).fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
    return roc_auc_score(y,pred)
base=cv_auc(Xb); print(f"base FP-AUC={base:.4f}  elig edges={elig.sum()}")

def setfeat(gmask,p=5):
    sub=Mx[:,gmask]
    if sub.shape[1]<p: return None
    Z=StandardScaler().fit_transform(sub)
    sc=PCA(p,random_state=0).fit_transform(Z)          # class x p
    G=np.zeros((N,p)); G[has]=sc[c_of[has]]
    G[has]=StandardScaler().fit_transform(G[has])
    s=G[I]+G[J]; dif=np.abs(G[I]-G[J]); return np.c_[s[elig],dif[elig]]
def auc_of(gmask,p=5):
    F=setfeat(gmask,p)
    return None if F is None else cv_auc(np.c_[Xb,F])

mod_mask=np.array([bool(re.match(r'^(ins|nlp|flp|npr|mihp)-',g)) for g in genes])
IDENT=["unc-119","rab-3","snb-1","ric-4","unc-104","ida-1","snt-1","unc-10","unc-64","unc-31",
 "unc-17","cha-1","eat-4","unc-25","unc-47","unc-46","tph-1","cat-1","cat-2","cat-4","bas-1","tbh-1","tdc-1",
 "ceh-10","ceh-14","ceh-36","ttx-3","ttx-1","unc-3","unc-4","unc-30","unc-42","unc-86","mec-3","lin-11",
 "cog-1","die-1","fax-1","ast-1","lim-6","vab-3","ceh-43","unc-39","ceh-2","zag-1","mls-2","ceh-32",
 "osm-9","ocr-2","tax-2","tax-4","che-2","osm-6","mec-4","mec-10","del-1","egl-46","sem-4","lin-39"]
id_mask=np.array([g in set(IDENT) for g in genes]) & ~mod_mask
nM,nD=int(mod_mask.sum()),int(id_mask.sum())
aM=auc_of(mod_mask); aD=auc_of(id_mask)
print(f"\n  modulatory({nM} genes->5PC):  FP-AUC={aM:.4f}  Δ={aM-base:+.4f}")
print(f"  identity  ({nD} genes->5PC):  FP-AUC={aD:.4f}  Δ={aD-base:+.4f}")
allidx=np.arange(len(genes))
def rand_null(size,R=60):
    out=[]
    for _ in range(R):
        gm=np.zeros(len(genes),bool); gm[RNG.choice(allidx,size,replace=False)]=True
        a=auc_of(gm)
        if a is not None: out.append(a)
    return np.array(out)
nulM=rand_null(nM); nulD=rand_null(nD)
zM=(aM-nulM.mean())/(nulM.std()+1e-9); zD=(aD-nulD.mean())/(nulD.std()+1e-9)
print(f"\n  random(size {nM}) FP-AUC = {nulM.mean():.4f}±{nulM.std():.4f}  -> modulatory z={zM:+.2f}  (p={(nulM>=aM).mean():.3f})")
print(f"  random(size {nD}) FP-AUC = {nulD.mean():.4f}±{nulD.std():.4f}  -> identity   z={zD:+.2f}  (p={(nulD>=aD).mean():.3f})")
print("\n"+"="*70)
if zM>=2 and aM>aD: print("  DIRECT: MODULATORY predicts FP beyond degree & beats identity")
elif zD>=2 and aD>aM and zM<2: print("  DIRECT: IDENTITY predicts FP; modulatory does not (type-based)")
elif zM<2 and zD<2: print("  DIRECT: neither beats random matched-size -> not attributable to either set")
else: print(f"  DIRECT: mixed (zMod={zM:+.2f} aMod={aM:.4f} | zId={zD:+.2f} aId={aD:.4f})")
print("="*70)
