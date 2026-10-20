import sys, json, numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); z2=np.load(f"{SC}/resid2.npz")
I,J,y7,y8,base,katz,neurons=z['I'],z['J'],z['y7'],z['y8'],z['base'],z['katz'],list(z['neurons'])
M=len(neurons); N=len(I); yrep=((y7==1)&(y8==1)).astype(int); idx={n:i for i,n in enumerate(neurons)}
X=np.hstack([base,katz]); MF=z2['MF']; r1=z2['r1']
cent={}
for dsn in (7,8):
    sk=json.load(open(f"../seed/data/external/mechanistic_frontier/witvliet2021_developmental/data/skeletons/Dataset{dsn}_skeletons.json"))
    for n,v in sk.items():
        if n in idx and v.get('coords'): cent.setdefault(n,[]).append(np.array(list(v['coords'].values()),float).mean(0))
POS=np.full((M,3),np.nan)
for n,v in cent.items(): POS[idx[n]]=np.mean(v,0)
mask=~np.isnan(POS).any(1); POS[~mask]=np.nanmean(POS[mask],0)
POS=(POS-POS.mean(0))/(POS.std(0)+1e-9)
# Sperry graded complementary matching features: per-axis source*target product + signed diff + |diff|
sperry=np.zeros((N,9))
for k in range(N):
    a,b=POS[I[k]],POS[J[k]]
    sperry[k]=[a[0]*b[0],a[1]*b[1],a[2]*b[2],a[0]-b[0],a[1]-b[1],a[2]-b[2],np.linalg.norm(a-b),a@b,np.linalg.norm(a)*np.linalg.norm(b)]
cv=StratifiedKFold(5,shuffle=True,random_state=0)
nf={n:i%5 for i,n in enumerate(sorted(set([neurons[i] for i in I]+[neurons[j] for j in J])))}
def nho(Xf):
    pred=np.full(N,np.nan)
    for f in range(5):
        te=[k for k in range(N) if nf[neurons[I[k]]]==f and nf[neurons[J[k]]]==f];tr=[k for k in range(N) if nf[neurons[I[k]]]!=f and nf[neurons[J[k]]]!=f]
        pred[np.array(te)]=GradientBoostingClassifier(random_state=0).fit(Xf[tr],yrep[tr]).predict_proba(Xf[te])[:,1]
    m=~np.isnan(pred);return roc_auc_score(yrep[m],pred[m])
a0=nho(np.hstack([X,MF])); a1=nho(np.hstack([X,MF,sperry]))
P(f"[SPERRY graded complementary matching] base+katz+mirror AUC={a0:.3f} -> +Sperry AUC={a1:.3f} (Δ={a1-a0:+.3f})")
# how much low-rank residual is recoverable? edge-holdout latent completion (descriptive upper bound)
from numpy.linalg import svd
R=np.zeros((M,M)); R[I,J]=r1
rng=np.random.default_rng(0); heldmask=rng.random(N)<0.2
Rt=R.copy(); Rt[I[heldmask],J[heldmask]]=0
Ut,st,Vt=svd(Rt)
for d in [4,8,16]:
    Rhat=(Ut[:,:d]*st[:d])@Vt[:d]
    pred=Rhat[I[heldmask],J[heldmask]]; true=r1[heldmask]
    P(f"  rank-{d:2d} latent completion of residual: held-out corr(pred,true)={np.corrcoef(pred,true)[0,1]:+.3f}")
