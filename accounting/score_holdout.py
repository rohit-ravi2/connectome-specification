import sys,numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True)
I,J,neurons=z['I'],z['J'],list(z['neurons']); base,katz=z['base'],z['katz']
y7,y8=z['y7'],z['y8']; N=len(I)
MF=np.load(f"{SC}/resid2.npz",allow_pickle=True)['MF']
DYN=np.load(f"{SC}/contactdyn.npz",allow_pickle=True)['DYN']
yrep=((y7==1)&(y8==1)).astype(int)
BM=np.hstack([base,katz,MF])
AUG=np.hstack([BM,DYN])

endpts=sorted(set([neurons[k] for k in list(I)+list(J)]))
nf={n:i%5 for i,n in enumerate(endpts)}
NI=np.array([nf[neurons[int(I[k])]] for k in range(N)])
NJ=np.array([nf[neurons[int(J[k])]] for k in range(N)])

def holdout(X):
    preds=np.full(N,np.nan); cov=np.zeros(N,bool)
    for f in range(5):
        test=(NI==f)&(NJ==f)
        train=(NI!=f)&(NJ!=f)
        if test.sum()==0: continue
        m=GradientBoostingClassifier(random_state=0).fit(X[train],yrep[train])
        preds[test]=m.predict_proba(X[test])[:,1]; cov[test]=True
    auc=roc_auc_score(yrep[cov],preds[cov])
    return preds,cov,auc

p0,c0,a0=holdout(BM); P(f"BASELINE neuron-holdout AUC={a0:.4f} cov={c0.sum()}")
p1,c1,a1=holdout(AUG); P(f"BM+DYN  neuron-holdout AUC={a1:.4f} cov={c1.sum()}")
P(f"delta AUC={a1-a0:+.4f}")
np.savez(f"{SC}/holdout_preds.npz",p0=p0,p1=p1,cov=(c0&c1),a0=a0,a1=a1)
