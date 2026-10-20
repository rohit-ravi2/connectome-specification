"""BRIDGE TEST — HARDENED degree control. The positive was driven by Wired_broadcaster =
high-degree hubs. Coarse quartile / linear-degree control can leak if residual-load scales
super-linearly with degree. Here: (1) node-level uses a flexible nonlinear degree fit
(isotonic on degree) so 'excess' = beyond the true degree curve; permutation null at DECILE
resolution. (2) edge-level baseline adds nonlinear degree terms (deg, deg^2, sqrt) so the
archetype block must beat a curve, not a line. Same degree-stratified label-perm null.
Calibration: random-partition FP rate on the edge test.
"""
import warnings; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
RNG=np.random.default_rng(0)
SC="../work"
d=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); m=np.load(f"{SC}/mode0.npz",allow_pickle=True)
I,J=d["I"],d["J"]; neurons=[str(x) for x in d["neurons"]]; logC=d["base"][:,0]
FP,FN,DC=m["FP"],m["FN"],m["det_core"]; N=len(neurons); Np=len(I)
prof=pd.read_csv("../data/celegans/New Notebooks/data_derived/nb28_neuron_profile.csv")
am=dict(zip(prof["neuron"].astype(str),prof["archetype"].astype(str)))
arch=np.array([am.get(n,"NA") for n in neurons])
labs=sorted(set(arch)-{"NA"})
deg=np.zeros(N)
for a,b in zip(I,J): deg[a]+=1; deg[b]+=1

print("="*78); print("NODE-LEVEL, ISOTONIC degree control (excess beyond the true degree curve)"); print("="*78)
def node_load(mask):
    load=np.zeros(N)
    for a,b,mk in zip(I,J,mask):
        if mk: load[a]+=1; load[b]+=1
    return load
has=arch!="NA"; idx=np.where(has)[0]
qc=pd.qcut(deg[has],10,labels=False,duplicates="drop")     # DECILE strata
for name,mask in [("det_core",DC),("FP",FP),("FN",FN)]:
    load=node_load(mask)
    iso=IsotonicRegression(out_of_bounds="clip").fit(deg[has],load[has])
    resid=load-iso.predict(deg)                            # excess beyond monotone degree curve
    obs={a:resid[has&(arch==a)].mean() for a in labs}
    NP=2000; nulls={a:[] for a in labs}
    for _ in range(NP):
        perm=arch[has].copy()
        for q in np.unique(qc): sel=qc==q; perm[sel]=RNG.permutation(perm[sel])
        ap=np.array(["NA"]*N,object); ap[idx]=perm
        for a in labs: nulls[a].append(resid[has&(ap==a)].mean())
    print(f"\n  [{name}] isotonic-excess residual load by archetype (decile-perm null):")
    for a in labs:
        if (arch==a).sum()<5: continue
        nu=np.array(nulls[a]); z=(obs[a]-nu.mean())/(nu.std()+1e-9)
        print(f"    {a:24s} n={int((arch==a).sum()):2d} obs={obs[a]:+.3f} null={nu.mean():+.3f}±{nu.std():.3f} z={z:+.2f}"+("  <==" if abs(z)>=2 else ""))

print("\n"+"="*78); print("EDGE-LEVEL, NONLINEAR degree baseline (archetype must beat a curve)"); print("="*78)
dI,dJ=deg[I],deg[J]; lo=np.minimum(dI,dJ); hi=np.maximum(dI,dJ)
Xbase=np.c_[logC, np.log1p(lo),np.log1p(hi), lo,hi, lo**2,hi**2, np.sqrt(lo),np.sqrt(hi), lo*hi]
def arch_feats(av):
    F=np.zeros((Np,len(labs)))
    for k,a in enumerate(labs): F[:,k]=(av[I]==a).astype(int)+(av[J]==a).astype(int)
    same=((av[I]==av[J])&(av[I]!="NA")).astype(int)
    return np.c_[F,same]
gkf=GroupKFold(5)
def cv_auc(X,y,groups):
    pred=np.zeros(len(y))
    for tr,te in gkf.split(X,y,groups):
        pred[te]=LogisticRegression(max_iter=3000,class_weight="balanced").fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
    return roc_auc_score(y,pred)
Xarch=arch_feats(arch); qc2=pd.qcut(deg[has],10,labels=False,duplicates="drop")
for name,y in [("det_core",DC.astype(int)),("FP",FP.astype(int)),("FN",FN.astype(int))]:
    ab=cv_auc(Xbase,y,I); af=cv_auc(np.c_[Xbase,Xarch],y,I)
    nul=[]
    for _ in range(50):
        perm=arch[has].copy()
        for q in np.unique(qc2): sel=qc2==q; perm[sel]=RNG.permutation(perm[sel])
        ap=np.array(["NA"]*N,object); ap[idx]=perm
        nul.append(cv_auc(np.c_[Xbase,arch_feats(ap)],y,I))
    nul=np.array(nul); z=(af-nul.mean())/(nul.std()+1e-9)
    print(f"  [{name:8s}] nl-base={ab:.4f} +arch={af:.4f} Δ={af-ab:+.4f} | null={nul.mean():.4f}±{nul.std():.4f} z={z:+.2f}"+("  <== BRIDGE" if z>=2 else "  (dead)"))

# --- CALIBRATION: random fake-archetype partitions, same sizes, FP rate on det_core ---
print("\n"+"="*78); print("CALIBRATION: random fake-archetype labels (matched sizes), det_core edge test"); print("="*78)
sizes=[int((arch==a).sum()) for a in labs]; y=DC.astype(int); ab=cv_auc(Xbase,y,I)
fp=0; TT=20
for t in range(TT):
    fake=np.array(["NA"]*N,object); pool=RNG.permutation(idx)
    c=0
    for k,s in enumerate(sizes): fake[pool[c:c+s]]=f"F{k}"; c+=s
    af=cv_auc(np.c_[Xbase,arch_feats(fake)],y,I)
    # null for this fake set
    nul=[]
    for _ in range(12):
        perm=fake[has].copy()
        for q in np.unique(qc2): sel=qc2==q; perm[sel]=RNG.permutation(perm[sel])
        ap=np.array(["NA"]*N,object); ap[idx]=perm
        nul.append(cv_auc(np.c_[Xbase,arch_feats(ap)],y,I))
    nul=np.array(nul); z=(af-nul.mean())/(nul.std()+1e-9)
    if z>=2: fp+=1
print(f"  random fake-archetype FP rate (z>=2): {fp}/{TT} = {fp/TT:.2%}  (should be ~5% if calibrated)")
