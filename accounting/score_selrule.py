import sys,numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True)
I,J,neurons=z['I'].astype(int),z['J'].astype(int),list(z['neurons'])
base=z['base']; y7,y8=z['y7'],z['y8']; M=len(neurons); N=len(I)
r1=np.load(f"{SC}/resid2.npz",allow_pickle=True)['r1']
b=np.load(f"{SC}/bundle_feats.npz",allow_pickle=True); part_a=b['part_a']
yrep=((y7==1)&(y8==1)).astype(int)
A=np.zeros((M,M))
for k in range(N): A[I[k],J[k]]=np.expm1(base[k,0])
A=np.maximum(A,A.T); Ab=(A>0).astype(int)

# rank features (contact-only, no label leakage)
def ranks(members_of):
    """members_of[i] = boolean mask of allowed contact partners for ranking i's contacts."""
    Rk=np.zeros(N); Rkn=np.zeros(N); Frac=np.zeros(N)
    for k in range(N):
        i,j=int(I[k]),int(J[k])
        mask=members_of(i); mask=mask&(Ab[i]>0); mask[i]=False
        vals=A[i][mask]
        if len(vals)==0: Rk[k]=0;Rkn[k]=0.5;Frac[k]=1;continue
        r=(vals<A[i,j]).sum()  # how many weaker
        Rk[k]=r; Rkn[k]=r/max(len(vals)-1,1); Frac[k]=A[i,j]/(vals.max()+1e-9)
    return np.column_stack([Rk,Rkn,Frac])

allmask=lambda i: np.ones(M,bool)
bundmask=lambda i: (part_a==part_a[i])
Rglob=ranks(allmask)
Rbund=ranks(bundmask)
# also j-side (symmetric): rank among j's contacts
def ranks_j(members_of):
    Rk=np.zeros(N); Rkn=np.zeros(N); Frac=np.zeros(N)
    for k in range(N):
        i,j=int(I[k]),int(J[k])
        mask=members_of(j)&(Ab[j]>0); mask[j]=False
        vals=A[j][mask]
        if len(vals)==0: Rkn[k]=0.5;Frac[k]=1;continue
        Rk[k]=(vals<A[i,j]).sum(); Rkn[k]=Rk[k]/max(len(vals)-1,1); Frac[k]=A[i,j]/(vals.max()+1e-9)
    return np.column_stack([Rk,Rkn,Frac])
Rglob=np.hstack([Rglob,ranks_j(allmask)])
Rbund=np.hstack([Rbund,ranks_j(bundmask)])

_fm={n:i%5 for i,n in enumerate(sorted(set([neurons[q] for q in list(I)+list(J)])))}
NI=np.array([_fm[neurons[int(I[k])]] for k in range(N)]); NJ=np.array([_fm[neurons[int(J[k])]] for k in range(N)])
sb=(part_a[I]==part_a[J])  # within-modularity-bundle
raw=A[I,J].reshape(-1,1); rawlog=np.log1p(raw)

def holdout(X,restrict):
    preds=np.full(N,np.nan); cov=np.zeros(N,bool)
    for f in range(5):
        test=(NI==f)&(NJ==f)&restrict; train=(NI!=f)&(NJ!=f)&restrict
        if test.sum()==0 or train.sum()==0: continue
        m=GradientBoostingClassifier(random_state=0).fit(X[train],yrep[train])
        preds[test]=m.predict_proba(X[test])[:,1]; cov[test]=True
    return preds,cov,roc_auc_score(yrep[cov],preds[cov])

P(f"within-bundle pairs total={sb.sum()}  synapse rate={yrep[sb].mean():.3f}")
for name,X in [('rawArea',rawlog),('GLOBAL-rank',np.hstack([rawlog,Rglob])),('BUNDLE-rel-rank',np.hstack([rawlog,Rbund])),('BOTH',np.hstack([rawlog,Rglob,Rbund]))]:
    _,cov,auc=holdout(X,sb)
    P(f"[within-bundle] {name:16s} AUC={auc:.4f}  cov={cov.sum()}")
# FP-specific: among within-bundle, FP (pruned) vs the rest. Does bundle-rel rank rank FP low?
FP=r1<-0.3
mfp=sb&FP
P(f"within-bundle FP count={mfp.sum()}")
# AUC discriminating within-bundle synapse using bundle-rel vs global (already above). FP are the y=0 the baseline overpredicts.
# compare mean bundle-rel normalized rank of FP vs within-bundle synapsing pairs
print("FP mean bundle-rel-rank(i-side norm)=",Rbund[mfp,1].mean().round(3),
      " synapsing mean=",Rbund[sb&(yrep==1),1].mean().round(3),
      " |global:",Rglob[mfp,1].mean().round(3),Rglob[sb&(yrep==1),1].mean().round(3))
