import sys,numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True)
I,J,neurons=z['I'],z['J'],list(z['neurons']); base,katz=z['base'],z['katz']
y7,y8=z['y7'],z['y8']; N=len(I); M=len(neurons)
MF=np.load(f"{SC}/resid2.npz",allow_pickle=True)['MF']
raw=np.load(f"{SC}/contactdyn.npz",allow_pickle=True)
DYN=raw['DYN']
yrep=((y7==1)&(y8==1)).astype(int)
BM=np.hstack([base,katz,MF])
# undirected pair -> DYN row (DYN is directed but symmetric build; map und key -> first occurrence)
und_row={}
for k in range(N):
    a,b=int(I[k]),int(J[k]); und_row.setdefault((min(a,b),max(a,b)),k)
endpts=sorted(set([neurons[k] for k in list(I)+list(J)]))
nf={n:i%5 for i,n in enumerate(endpts)}
NI=np.array([nf[neurons[int(I[k])]] for k in range(N)])
NJ=np.array([nf[neurons[int(J[k])]] for k in range(N)])
def holdout(X):
    preds=np.full(N,np.nan); cov=np.zeros(N,bool)
    for f in range(5):
        test=(NI==f)&(NJ==f); train=(NI!=f)&(NJ!=f)
        if test.sum()==0: continue
        m=GradientBoostingClassifier(random_state=0).fit(X[train],yrep[train])
        preds[test]=m.predict_proba(X[test])[:,1]; cov[test]=True
    return roc_auc_score(yrep[cov],preds[cov])
a0=holdout(BM); a1=holdout(np.hstack([BM,DYN]))
P(f"real baseline={a0:.4f} aug={a1:.4f} delta={a1-a0:+.4f}")
rng=np.random.default_rng(0); nulls=[]
K=20
for r in range(K):
    perm=rng.permutation(M)
    DYNs=np.zeros_like(DYN)
    for k in range(N):
        a,b=int(I[k]),int(J[k]); pa,pb=perm[a],perm[b]
        key=(min(pa,pb),max(pa,pb))
        src=und_row.get(key)
        DYNs[k]=DYN[src] if src is not None else DYN[k]
    ar=holdout(np.hstack([BM,DYNs])); nulls.append(ar-a0)
    P(f"  shuffle {r}: delta={ar-a0:+.4f}")
nulls=np.array(nulls)
P(f"NULL delta mean={nulls.mean():+.4f} std={nulls.std():.4f}  thr(mean+2std)={nulls.mean()+2*nulls.std():+.4f}")
P(f"REAL delta={a1-a0:+.4f}  PASS={ (a1-a0) > nulls.mean()+2*nulls.std() }")
np.savez(f"{SC}/perm_null.npz",nulls=nulls,real=a1-a0,a0=a0,a1=a1)
