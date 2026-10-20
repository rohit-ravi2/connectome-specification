import sys,numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True)
I,J,neurons=z['I'].astype(int),z['J'].astype(int),list(z['neurons'])
base,katz=z['base'],z['katz']; y7,y8=z['y7'],z['y8']; N=len(I)
MF=np.load(f"{SC}/resid2.npz",allow_pickle=True)['MF']
r1=np.load(f"{SC}/resid2.npz",allow_pickle=True)['r1']
b=np.load(f"{SC}/bundle_feats.npz",allow_pickle=True)
Fa,Fb,Fc,Fd=b['Fa'],b['Fb'],b['Fc'],b['Fd']
yrep=((y7==1)&(y8==1)).astype(int)
BM=np.hstack([base,katz,MF])

endpts=sorted(set([neurons[k] for k in list(I)+list(J)]))
nf={n:i%5 for i,n in enumerate(endpts)}
NI=np.array([nf[neurons[int(I[k])]] for k in range(N)])
NJ=np.array([nf[neurons[int(J[k])]] for k in range(N)])

def holdout(X,seed=0):
    preds=np.full(N,np.nan); cov=np.zeros(N,bool)
    for f in range(5):
        test=(NI==f)&(NJ==f); train=(NI!=f)&(NJ!=f)
        if test.sum()==0: continue
        m=GradientBoostingClassifier(random_state=seed).fit(X[train],yrep[train])
        preds[test]=m.predict_proba(X[test])[:,1]; cov[test]=True
    return preds,cov,roc_auc_score(yrep[cov],preds[cov])

p0,cov,a0=holdout(BM); P(f"BASELINE BM AUC={a0:.4f} cov={cov.sum()}")
blocks={'a_modularity':Fa,'b_spectral':Fb,'c_kcore':Fc,'d_clique':Fd,
        'ALL':np.hstack([Fa,Fb,Fc,Fd])}
res={}
for name,F in blocks.items():
    p1,c1,a1=holdout(np.hstack([BM,F]))
    res[name]=(p1,a1)
    P(f"BM+{name:14s} AUC={a1:.4f}  dAUC={a1-a0:+.4f}")

# use ALL for hard/FP/FN
p1,a1=res['ALL']
hard=np.abs(r1)>0.3; FP=r1<-0.3; FN=r1>0.3
def improv(mask):
    m=mask&cov
    better=(np.abs(yrep-p1)[m]<np.abs(yrep-p0)[m]).mean()
    return better,m.sum()
for nm,mask in [('hard',hard),('FP',FP),('FN',FN)]:
    bb,n=improv(mask); P(f"{nm}: cov={n} fracImproved={bb:.3f}")
# discrimination AUCs on hard-968 (FP=neg,FN=pos): does model separate pruned vs kept?
mh=hard&cov
P(f"hard-968 AUC(yrep): BM={roc_auc_score(yrep[mh],p0[mh]):.4f}  BM+ALL={roc_auc_score(yrep[mh],p1[mh]):.4f}")
# FP mean prediction (lower=better, y=0). calibration check
mfp=FP&cov
P(f"FP mean pred: BM={p0[mfp].mean():.4f} BM+ALL={p1[mfp].mean():.4f} (target 0; lower=better)")
mfn=FN&cov
P(f"FN mean pred: BM={p0[mfn].mean():.4f} BM+ALL={p1[mfn].mean():.4f} (target 1; higher=better)")
# within-bundle (def a same-bundle) AUC
sb=(Fa[:,0]>0)&cov
P(f"within-modularity-bundle pairs={sb.sum()} AUC: BM={roc_auc_score(yrep[sb],p0[sb]):.4f} BM+ALL={roc_auc_score(yrep[sb],p1[sb]):.4f}")
np.savez(f"{SC}/bundle_holdout.npz",p0=p0,p1=p1,cov=cov,a0=a0,a1=a1)
