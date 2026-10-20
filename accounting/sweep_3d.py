import json, csv, re, math, numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
D="../data/celegans/data/connectome"
SK="../seed/data/external/mechanistic_frontier/witvliet2021_developmental/data/skeletons"
# skeletons (adult Dataset7), subsample points per neuron
sk=json.load(open(f"{SK}/Dataset7_skeletons.json"))
pts={}
for nm,v in sk.items():
    co=list(v['coords'].values())
    if len(co)<3: continue
    arr=np.array(co,float); step=max(1,len(arr)//70); pts[nm]=arr[::step][:80]
print(f"neurons with skeleton: {len(pts)}")
R=list(csv.DictReader(open("data/processed/dev_stabilization_program/master_directed_chemical_pairs.csv")))
R=[r for r in R if r['src'] in pts and r['tgt'] in pts]
def c(n,f=float): return np.array([f(r[n]) for r in R])
a7=c('a7',int); a8=c('a8',int); disagree=(a7!=a8).astype(int)
THR=1000.0  # nm apposition threshold
f3d=[]
for r in R:
    P,Q=pts[r['src']],pts[r['tgt']]; Dm=cdist(P,Q)
    mind=Dm.min(); near=(Dm<THR)
    cofasc=int(near.sum())                       # apposition extent (point-pair count)
    cov_s=float(near.any(1).mean())              # frac of src points apposed
    cov_t=float(near.any(0).mean())              # frac of tgt points apposed
    f3d.append([math.log1p(mind),math.log1p(cofasc),cov_s,cov_t])
F3=np.array(f3d)
base=['log_area','shared','cos','cdeg_s','cdeg_t','ntsame','src_share','tgt_share','src_rank','tgt_rank']
Xb=np.column_stack([c(k) for k in base]); Xe=np.hstack([Xb,F3])
cv=StratifiedKFold(5,shuffle=True,random_state=0)
def disp(X):
    y2=np.concatenate([a7,a8]); X2=np.vstack([X,X])
    p=np.clip(cross_val_predict(GradientBoostingClassifier(random_state=0),X2,y2,cv=cv,method='predict_proba')[:,1][:len(R)],1e-4,1-1e-4)
    return disagree.mean()/(2*p*(1-p)).mean()
yrep=((a7==1)&(a8==1)).astype(int)
nf={n:i%5 for i,n in enumerate(sorted(set([r['src'] for r in R]+[r['tgt'] for r in R])))}
def nho(X):
    pred=np.full(len(R),np.nan)
    for f in range(5):
        te=[k for k,r in enumerate(R) if nf[r['src']]==f and nf[r['tgt']]==f]; tr=[k for k,r in enumerate(R) if nf[r['src']]!=f and nf[r['tgt']]!=f]
        if len(te)<20: continue
        pred[np.array(te)]=GradientBoostingClassifier(random_state=0).fit(X[tr],yrep[tr]).predict_proba(X[te])[:,1]
    m=~np.isnan(pred); return roc_auc_score(yrep[m],pred[m])
print(f"pairs used (both have skeleton): {len(R)}")
print("CANDIDATE: finer 3D geometry (co-fasciculation/min-dist/apposition-coverage) beyond scalar contact area")
print(f"  baseline (scalar contact+topology): dispersion_ratio={disp(Xb):.2f}  AUC(nho)={nho(Xb):.3f}")
print(f"  + 3D morphology features:           dispersion_ratio={disp(Xe):.2f}  AUC(nho)={nho(Xe):.3f}")
print(f"  + 3D ALONE (replace contact-area):  ", end="")
Xa=np.column_stack([c(k) for k in ['shared','cos','cdeg_s','cdeg_t','ntsame']]); Xa3=np.hstack([Xa,F3])
print(f"dispersion_ratio={disp(Xa3):.2f}  AUC(nho)={nho(Xa3):.3f}")
print("  (baseline ratio ~0.48; ->1 = finer geometry captures the sharp propensity)")
