"""TEST 1 — WHICH PCs CARRY THE FP BRIDGE: modulatory vs identity discriminator.
FP/avoidance residual has non-circular gene-PC signal (z=+3.15). Is it neuromodulatory
(insulin/peptide 'Master Control' axis) or generic cell-type identity (already-known type-based
self-avoidance)? Rank 20 gene-PCs by FP-predictiveness (add-one Δ AUC over nonlinear degree),
form per-gene FP-relevance = Σ_k w_k|loading[k,g]|, then compare modulatory vs identity vs random.

PRE-REGISTERED: FP-relevance(modulatory) > FP-relevance(identity) > FP-relevance(random);
modulatory beats random z>=2 AND modulatory beats identity (perm p<0.05).  Identity-wins =
identity>modulatory & mod~random.  Unresolvable = mod~identity~random (report honestly).
"""
import warnings; warnings.simplefilter("ignore")
import numpy as np, pandas as pd, re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
RNG=np.random.default_rng(0)
SC="../work"
d=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); m=np.load(f"{SC}/mode0.npz",allow_pickle=True)
I,J=d["I"],d["J"]; neurons=[str(x) for x in d["neurons"]]; logC=d["base"][:,0]
FP=m["FP"]; N=len(neurons)
deg=np.zeros(N)
for a,b in zip(I,J): deg[a]+=1; deg[b]+=1

# --- CeNGEN gene-PCs (identical construction to gene_bridge) ---
cg=pd.read_csv("../data/celegans/data/expression/cengen/thresholded/021821_liberal_threshold1.csv")
genes=cg["gene_name"].astype(str).str.lower().values
classes=list(cg.columns[3:]); Mx=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values.T
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
Ms=StandardScaler().fit_transform(Mx)
NPC=20; pca=PCA(NPC,random_state=0).fit(Ms); scores=pca.transform(Ms); comps=pca.components_  # 20 x nGenes
G=np.full((N,NPC),np.nan)
for i,n in enumerate(neurons):
    c=n2c(n)
    if c is not None: G[i]=scores[c]
has=~np.isnan(G[:,0]); Gz=G.copy(); Gz[~has]=0.0
Gs=StandardScaler().fit_transform(Gz[has]); Gfull=np.zeros((N,NPC)); Gfull[has]=Gs

# --- edge setup ---
dI,dJ=deg[I],deg[J]; lo=np.minimum(dI,dJ); hi=np.maximum(dI,dJ)
Xbase=np.c_[logC,np.log1p(lo),np.log1p(hi),lo,hi,lo**2,hi**2,np.sqrt(lo),np.sqrt(hi),lo*hi]
elig=has[I]&has[J]; Xb=Xbase[elig]; Ie=I[elig]; y=FP[elig].astype(int)
gkf=GroupKFold(5)
def cv_auc(X):
    pred=np.zeros(len(y))
    for tr,te in gkf.split(X,y,Ie):
        pred[te]=LogisticRegression(max_iter=3000,class_weight="balanced",C=0.5).fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
    return roc_auc_score(y,pred)
def ep(vec): s=vec[I]+vec[J]; dif=np.abs(vec[I]-vec[J]); return np.c_[s[elig],dif[elig]]

print("="*80); print("STEP A — rank 20 gene-PCs by FP-predictiveness (add-one Δ AUC over degree)"); print("="*80)
base=cv_auc(Xb); print(f"  base (nonlinear degree) FP-AUC = {base:.4f}")
dlt=np.zeros(NPC)
for k in range(NPC):
    dlt[k]=cv_auc(np.c_[Xb,ep(Gfull[:,k])])-base
order=np.argsort(-dlt)
for k in order: print(f"    PC{k:02d}  Δ={dlt[k]:+.4f}")
w=np.clip(dlt,0,None); w=w/ (w.sum()+1e-12)
topPCs=order[:5]; print(f"  top FP PCs: {list(topPCs)}  (weights {[round(w[k],3) for k in topPCs]})")

print("\n"+"="*80); print("STEP B — per-gene FP-relevance = Σ_k w_k |loading[k,g]|; modulatory vs identity vs random"); print("="*80)
rel=(w[:,None]*np.abs(comps)).sum(0)          # per-gene FP-relevance
# gene sets
mod_mask=np.array([bool(re.match(r'^(ins|nlp|flp|npr|mihp)-',g)) for g in genes])
IDENT=[ # generic cell-type / fate / terminal-selector identity markers (NON-modulatory)
 "unc-119","rab-3","snb-1","ric-4","unc-104","ida-1","snt-1","unc-10","unc-64","unc-31",
 "unc-17","cha-1","eat-4","unc-25","unc-47","unc-46","tph-1","cat-1","cat-2","cat-4","bas-1","tbh-1","tdc-1",
 "ceh-10","ceh-14","ceh-36","ttx-3","ttx-1","unc-3","unc-4","unc-30","unc-42","unc-86","mec-3","lin-11",
 "cog-1","die-1","fax-1","ast-1","lim-6","vab-3","ceh-43","unc-39","ceh-2","zag-1","mls-2","ceh-32",
 "osm-9","ocr-2","tax-2","tax-4","che-2","osm-6","mec-4","mec-10","del-1","egl-46","sem-4","lin-39"]
id_mask=np.array([g in set(IDENT) for g in genes]) & ~mod_mask
nM=int(mod_mask.sum()); nD=int(id_mask.sum())
print(f"  modulatory genes present: {nM} | identity genes present: {nD}")
relM=rel[mod_mask]; relD=rel[id_mask]
mM,mD=relM.mean(),relD.mean(); mAll=rel.mean()
print(f"  mean FP-relevance:  modulatory={mM:.4e}   identity={mD:.4e}   all-genes={mAll:.4e}")

# random matched-size null for each set
def rand_null(size,reps=5000):
    idx=np.arange(len(rel)); out=np.empty(reps)
    for r in range(reps): out[r]=rel[RNG.choice(idx,size,replace=False)].mean()
    return out
nullM=rand_null(nM); nullD=rand_null(nD)
zM=(mM-nullM.mean())/nullM.std(); zD=(mD-nullD.mean())/nullD.std()
pM=(nullM>=mM).mean(); pD=(nullD>=mD).mean()
print(f"  modulatory vs random(size {nM}): z={zM:+.2f}  p={pM:.4f}")
print(f"  identity   vs random(size {nD}): z={zD:+.2f}  p={pD:.4f}")

# modulatory vs identity head-to-head: label-permutation on pooled M∪D
pool=np.r_[relM,relD]; lab=np.r_[np.ones(nM),np.zeros(nD)]; obs=mM-mD
perm=np.array([ (lambda p: p[:nM].mean()-p[nM:].mean())(RNG.permutation(pool)) for _ in range(20000)])
pMD=(perm>=obs).mean()
print(f"  modulatory - identity = {obs:+.4e}  | perm p(mod>ident)={pMD:.4f}  (perm mean {perm.mean():+.2e}±{perm.std():.1e})")

print("\n"+"="*80); print("VERDICT"); print("="*80)
mod_beats_rand = pM<0.05; id_beats_rand = pD<0.05; mod_beats_id = pMD<0.05; id_beats_mod=(1-pMD)<0.05
if mod_beats_rand and mod_beats_id:
    v="MODULATORY — FP residual is neuromodulatory (mod>identity, mod>random). Connects to north-star modulatory latch."
elif id_beats_rand and id_beats_mod:
    v="IDENTITY — FP is type-based self-avoidance re-derived; modulatory thread stays node-level."
elif (not mod_beats_rand) and (not id_beats_rand):
    v="UNRESOLVABLE — neither beats random; PCs too entangled / signal too small to attribute. Bridge real but unattributable at this resolution."
elif abs(zM-zD)<1 and not mod_beats_id and not id_beats_mod:
    v="UNRESOLVABLE — modulatory ≈ identity; loading test lacks resolution to separate them."
else:
    v=f"MIXED — mod_vs_rand p={pM:.3f}, id_vs_rand p={pD:.3f}, mod_vs_id p={pMD:.3f}. Read explicitly, do not over-call."
print(" ",v)
# top modulatory + identity genes on the FP PCs, for transparency
gi=np.argsort(-rel)
print("\n  top-15 FP-relevant genes overall:")
for g in gi[:15]:
    tag="[MOD]" if mod_mask[g] else ("[ID]" if id_mask[g] else "")
    print(f"    {genes[g]:12s} rel={rel[g]:.3e} {tag}")
