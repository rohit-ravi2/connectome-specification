"""DECISIVE BRIDGE — gene-PCs (transcriptome) not archetype (connectivity-derived).
Archetype bridging the residual is partly circular: archetype is DEFINED from multiplex degree,
so 'residual concentrates on an archetype' can be re-described degree. Gene-PCs are computed
from CeNGEN expression ALONE — independent of the connectome. If endpoint gene-PCs predict the
deterministic residual BEYOND a nonlinear contact-degree baseline, with a calibrated
degree-stratified permutation null, the modulatory/transcriptional axis genuinely reaches the
edge residual (non-circular). This is the real test of the Nb31 -> residual bridge.
"""
import warnings; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
RNG=np.random.default_rng(0)
SC="../work"
d=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); m=np.load(f"{SC}/mode0.npz",allow_pickle=True)
I,J=d["I"],d["J"]; neurons=[str(x) for x in d["neurons"]]; logC=d["base"][:,0]
FP,FN,DC=m["FP"],m["FN"],m["det_core"]; N=len(neurons); Np=len(I)
deg=np.zeros(N)
for a,b in zip(I,J): deg[a]+=1; deg[b]+=1

# --- CeNGEN class x gene -> gene-PCs (transcriptome only) ---
cg=pd.read_csv("../data/celegans/data/expression/cengen/thresholded/021821_liberal_threshold1.csv")
classes=list(cg.columns[3:]); M=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values.T  # class x gene
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
Ms=StandardScaler().fit_transform(M)
NPC=20; scores=PCA(n_components=NPC,random_state=0).fit_transform(Ms)   # class x NPC
# per-neuron gene-PC vector (NaN if class missing)
G=np.full((N,NPC),np.nan)
for i,n in enumerate(neurons):
    c=n2c(n)
    if c is not None: G[i]=scores[c]
has=~np.isnan(G[:,0]); idx=np.where(has)[0]
print(f"neurons with gene-PCs: {has.sum()}/{N}  | PCs={NPC}")
Gz=G.copy(); Gz[~has]=0.0
Gs=StandardScaler().fit_transform(Gz[has]); Gfull=np.zeros((N,NPC)); Gfull[has]=Gs

# --- edge features: nonlinear degree baseline + endpoint gene-PC (sum & |diff|) ---
dI,dJ=deg[I],deg[J]; lo=np.minimum(dI,dJ); hi=np.maximum(dI,dJ)
Xbase=np.c_[logC, np.log1p(lo),np.log1p(hi), lo,hi, lo**2,hi**2, np.sqrt(lo),np.sqrt(hi), lo*hi]
elig=has[I]&has[J]                                 # both endpoints have gene-PCs
print(f"edges with both endpoints gene-mapped: {elig.sum()}/{Np}")
def gfeat(Gm):
    s=Gm[I]+Gm[J]; dif=np.abs(Gm[I]-Gm[J]); return np.c_[s,dif]
gkf=GroupKFold(5)
def cv_auc(X,y,groups):
    pred=np.zeros(len(y))
    for tr,te in gkf.split(X,y,groups):
        pred[te]=LogisticRegression(max_iter=3000,class_weight="balanced",C=0.5).fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
    return roc_auc_score(y,pred)
# restrict to gene-eligible edges for a fair base-vs-full comparison
Xb=Xbase[elig]; Ie=I[elig]
qc=pd.qcut(deg[has],10,labels=False,duplicates="drop")
print("\n"+"="*78); print("GENE-PC BRIDGE: endpoint gene-PCs -> residual beyond nonlinear degree"); print("="*78)
for name,mask in [("det_core",DC),("FP",FP),("FN",FN)]:
    y=mask[elig].astype(int)
    ab=cv_auc(Xb,y,Ie); af=cv_auc(np.c_[Xb,gfeat(Gfull)[elig]],y,Ie)
    nul=[]
    for _ in range(50):
        perm=np.arange(N)
        p=idx.copy()
        for q in np.unique(qc):                    # permute gene-PC assignment within degree deciles
            sel=np.where(qc==q)[0]; src=idx[sel]; perm[src]=RNG.permutation(src)
        Gp=Gfull[perm]
        nul.append(cv_auc(np.c_[Xb,gfeat(Gp)[elig]],y,Ie))
    nul=np.array(nul); z=(af-nul.mean())/(nul.std()+1e-9)
    print(f"  [{name:8s}] base={ab:.4f} +genePC={af:.4f} Δ={af-ab:+.4f} | null={nul.mean():.4f}±{nul.std():.4f} z={z:+.2f}"+("  <== NON-CIRCULAR BRIDGE" if z>=2 else "  (dead)"))
