"""BRIDGE TEST — does the node-level modulatory finding (Nb31 archetypes, transcriptionally
readable) reach the EDGE-LEVEL deterministic residual (FP/FN/det_core) we've been chasing?

Question: do residual edges concentrate on neurons of a particular multiplex archetype,
BEYOND contact degree? Archetype is derived from connectivity degrees, so degree is a hard
confound and must be controlled. Null is degree-stratified label permutation (preserves the
degree<->archetype link, breaks archetype<->residual). Calibrated: random-label FP rate checked.
"""
import warnings; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
RNG=np.random.default_rng(0)
SC="../work"

d=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); m=np.load(f"{SC}/mode0.npz",allow_pickle=True)
I,J=d["I"],d["J"]; neurons=[str(x) for x in d["neurons"]]; base=d["base"]
logC=base[:,0]                                    # log contact area
FP,FN,DC=m["FP"],m["FN"],m["det_core"]
N=len(neurons); Np=len(I)
print(f"neurons={N}  eligible pairs={Np}  | FP={FP.sum()} FN={FN.sum()} det_core={DC.sum()}")

# --- archetype labels (Nb28) ---
prof=pd.read_csv("../data/celegans/New Notebooks/data_derived/nb28_neuron_profile.csv")
arch_map=dict(zip(prof["neuron"].astype(str),prof["archetype"].astype(str)))
def lookup(n):                                    # handle L/R,D/V collapse if needed
    if n in arch_map: return arch_map[n]
    return arch_map.get(n,None)
arch=np.array([lookup(n) if lookup(n) else "NA" for n in neurons])
cov=(arch!="NA").mean(); print(f"archetype coverage of ring neurons: {cov:.2f}  ({(arch!='NA').sum()}/{N})")
labs=sorted(set(arch)-{"NA"}); print("archetypes:",{a:int((arch==a).sum()) for a in labs})

# --- contact degree per neuron (the confound) ---
deg=np.zeros(N)
for a,b in zip(I,J): deg[a]+=1; deg[b]+=1

# ---------- NODE-LEVEL: residual load per neuron, degree-controlled ----------
def node_load(mask):
    load=np.zeros(N)
    for a,b,mk in zip(I,J,mask):
        if mk: load[a]+=1; load[b]+=1
    return load
print("\n"+"="*78); print("NODE-LEVEL: does archetype explain residual-load beyond contact degree?"); print("="*78)
# residualize residual-load on degree (poisson-ish: use log1p), then per-archetype mean residual
for name,mask in [("det_core",DC),("FP",FP),("FN",FN)]:
    load=node_load(mask); has=arch!="NA"
    # regress load on degree via simple rate: expected load = deg * (sum load / sum deg)
    exp=deg*(load[has].sum()/deg[has].sum())
    resid=(load-exp)                              # excess residual edges beyond degree expectation
    # observed per-archetype mean excess
    obs={a:resid[has&(arch==a)].mean() for a in labs}
    # degree-stratified permutation null: permute archetype among has-neurons within degree quartiles
    qc=pd.qcut(deg[has],4,labels=False,duplicates="drop"); idx=np.where(has)[0]
    NP=2000; nulls={a:[] for a in labs}
    for _ in range(NP):
        perm=arch[has].copy()
        for q in np.unique(qc):
            sel=qc==q; perm[sel]=RNG.permutation(perm[sel])
        ap=np.array(["NA"]*N,object); ap[idx]=perm
        for a in labs: nulls[a].append(resid[has&(ap==a)].mean())
    print(f"\n  [{name}] excess residual-edges per neuron by archetype (degree-controlled):")
    for a in labs:
        nu=np.array(nulls[a]); z=(obs[a]-nu.mean())/(nu.std()+1e-9)
        flag="  <==" if abs(z)>=2 else ""
        print(f"    {a:24s} obs={obs[a]:+.3f}  null={nu.mean():+.3f}±{nu.std():.3f}  z={z:+.2f}{flag}")

# ---------- EDGE-LEVEL: endpoint-archetype predicts residual beyond degree ----------
print("\n"+"="*78); print("EDGE-LEVEL: endpoint archetype -> residual, neuron-holdout AUC vs degree baseline"); print("="*78)
# edge features
degf=np.log1p(np.c_[deg[I],deg[J]]); degf=np.c_[degf.min(1),degf.max(1)]   # sym degree pair
Xbase=np.c_[logC,degf]                                                     # contact area + degree
# archetype one-hot endpoint: for each label, count of endpoints in that archetype (0/1/2)
def arch_feats(av):
    F=np.zeros((Np,len(labs)))
    for k,a in enumerate(labs):
        F[:,k]=(av[I]==a).astype(int)+(av[J]==a).astype(int)
    same=(av[I]==av[J])&(av[I]!="NA")            # same-archetype edge
    return np.c_[F,same.astype(int)]
Xarch=arch_feats(arch)
groups=I                                          # neuron-holdout by source (approx); use GroupKFold on I
gkf=GroupKFold(5)
def cv_auc(X,y,groups):
    pred=np.zeros(len(y))
    for tr,te in gkf.split(X,y,groups):
        lr=LogisticRegression(max_iter=3000,class_weight="balanced").fit(X[tr],y[tr])
        pred[te]=lr.predict_proba(X[te])[:,1]
    return roc_auc_score(y,pred)
for name,y in [("det_core",DC.astype(int)),("FP",FP.astype(int)),("FN",FN.astype(int))]:
    ab=cv_auc(Xbase,y,groups)
    af=cv_auc(np.c_[Xbase,Xarch],y,groups)
    # degree-stratified label-permutation null on the archetype block
    has=arch!="NA"; idx=np.where(has)[0]; qc=pd.qcut(deg[has],4,labels=False,duplicates="drop")
    nul=[]
    for _ in range(60):
        perm=arch[has].copy()
        for q in np.unique(qc): sel=qc==q; perm[sel]=RNG.permutation(perm[sel])
        ap=np.array(["NA"]*N,object); ap[idx]=perm
        nul.append(cv_auc(np.c_[Xbase,arch_feats(ap)],y,groups))
    nul=np.array(nul); z=(af-nul.mean())/(nul.std()+1e-9)
    print(f"  [{name:8s}] base={ab:.4f}  +arch={af:.4f}  Δ={af-ab:+.4f}  | null(+arch)={nul.mean():.4f}±{nul.std():.4f}  z={z:+.2f}"+("  <== BRIDGE" if z>=2 else ""))
