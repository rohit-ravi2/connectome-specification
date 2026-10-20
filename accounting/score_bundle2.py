import sys,numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True)
I,J,neurons=z['I'].astype(int),z['J'].astype(int),list(z['neurons'])
base,katz=z['base'],z['katz']; y7,y8=z['y7'],z['y8']; M=len(neurons); N=len(I)
MF=np.load(f"{SC}/resid2.npz",allow_pickle=True)['MF']
r1=np.load(f"{SC}/resid2.npz",allow_pickle=True)['r1']
b=np.load(f"{SC}/bundle_feats.npz",allow_pickle=True)
part_a,part_b,core_arr=b['part_a'],b['part_b'],b['core_arr']
yrep=((y7==1)&(y8==1)).astype(int)
BM=np.hstack([base,katz,MF])
# matrices
A=np.zeros((M,M))
for k in range(N): A[I[k],J[k]]=np.expm1(base[k,0])
A=np.maximum(A,A.T); Ab=(A>0).astype(int)
Y=np.zeros((M,M),int)
for k in range(N): Y[I[k],J[k]]=yrep[k]

_fm={n:i%5 for i,n in enumerate(sorted(set([neurons[q] for q in list(I)+list(J)])))}
fnode=np.array([_fm.get(neurons[n],-1) for n in range(M)])
NI=fnode[I]; NJ=fnode[J]

# clique defs reused
import networkx as nx
G=nx.Graph(); G.add_nodes_from(range(M))
for k in range(N):
    if I[k]<J[k]: G.add_edge(int(I[k]),int(J[k]))
cliques=[c for c in nx.find_cliques(G) if len(c)>=4]
node_cliques=[[] for _ in range(M)]
for ci,c in enumerate(cliques):
    for n in c: node_cliques[n].append(ci)

def bundle_density(members_mask,exclude_fold):
    """directed within-bundle synapse density over pairs with BOTH endpoints NOT in exclude_fold (training-visible)."""
    mem=[n for n in range(M) if members_mask[n] and fnode[n]!=exclude_fold]
    if len(mem)<2: return np.nan,0
    idx=np.array(mem)
    sub=Ab[np.ix_(idx,idx)]; suby=Y[np.ix_(idx,idx)]
    np.fill_diagonal(sub,0)
    den=sub.sum();
    if den==0: return np.nan,0
    return suby[sub>0].mean() if den>0 else np.nan, den

# For efficiency: per fold, precompute density for each partition-bundle and each kcore-threshold and each clique
def fold_features(f):
    """return F (N x P) of NON-LEAKY bundle features, for TEST pairs of fold f (both endpoints==f).
       density computed from training neurons (fnode!=f) only."""
    P_=6
    F=np.zeros((N,P_))
    # partition a
    da={}
    for c in np.unique(part_a):
        mask=(part_a==c); d,_=bundle_density(mask,f); da[c]=d if not np.isnan(d) else 0.0
    db={}
    for c in np.unique(part_b):
        mask=(part_b==c); d,_=bundle_density(mask,f); db[c]=d if not np.isnan(d) else 0.0
    dk={}
    for kk in np.unique(core_arr):
        mask=(core_arr>=kk); d,_=bundle_density(mask,f); dk[kk]=d if not np.isnan(d) else 0.0
    dcl={}
    for ci,c in enumerate(cliques):
        mask=np.zeros(M,bool); mask[c]=True; d,_=bundle_density(mask,f); dcl[ci]=d if not np.isnan(d) else np.nan
    test=(NI==f)&(NJ==f)
    for k in np.where(test)[0]:
        i,j=int(I[k]),int(J[k])
        F[k,0]=da.get(part_a[i],0.0) if part_a[i]==part_a[j] else 0.0
        F[k,1]=int(part_a[i]==part_a[j])
        F[k,2]=db.get(part_b[i],0.0) if part_b[i]==part_b[j] else 0.0
        kk=min(core_arr[i],core_arr[j]); F[k,3]=dk.get(kk,0.0); F[k,4]=kk
        shared=set(node_cliques[i])&set(node_cliques[j])
        vals=[dcl[ci] for ci in shared if not np.isnan(dcl[ci])]
        F[k,5]=np.mean(vals) if vals else 0.0
    return F,test

def holdout(X_extra_fn):
    p0=np.full(N,np.nan); p1=np.full(N,np.nan); cov=np.zeros(N,bool)
    for f in range(5):
        test=(NI==f)&(NJ==f); train=(NI!=f)&(NJ!=f)
        if test.sum()==0: continue
        # baseline
        m0=GradientBoostingClassifier(random_state=0).fit(BM[train],yrep[train])
        p0[test]=m0.predict_proba(BM[test])[:,1]
        # extra features: must be computed fold-aware for BOTH train and test.
        Ftr,Fte=X_extra_fn(f,train,test)
        Xtr=np.hstack([BM[train],Ftr]); Xte=np.hstack([BM[test],Fte])
        m1=GradientBoostingClassifier(random_state=0).fit(Xtr,yrep[train])
        p1[test]=m1.predict_proba(Xte)[:,1]; cov[test]=True
    return p0,p1,cov

# For training rows, bundle density must ALSO exclude the row's fold? Training pairs are those with neither endpoint==f.
# To keep train/test consistent, compute density excluding fold f for BOTH (test uses fold-f exclusion; train pairs
# have endpoints in other folds, density from fnode!=f still includes their own labels -> mild train-side optimism,
# but TEST predictions are the scored quantity and are clean). Simpler & clean: compute one density map per fold excl f.
def make_fn():
    def fn(f,train,test):
        F,_=fold_features(f)  # fills only test rows non-leaky
        # for train rows we need features too; recompute densities excluding fold f (same map) applied to train pairs
        da={c:(bundle_density((part_a==c),f)[0] or 0.0) for c in np.unique(part_a)}
        db={c:(bundle_density((part_b==c),f)[0] or 0.0) for c in np.unique(part_b)}
        dk={kk:(bundle_density((core_arr>=kk),f)[0] or 0.0) for kk in np.unique(core_arr)}
        dcl={}
        for ci,c in enumerate(cliques):
            mask=np.zeros(M,bool); mask[c]=True; dcl[ci]=bundle_density(mask,f)[0]
        Ftr=np.zeros((train.sum(),6)); tr_idx=np.where(train)[0]
        for r,k in enumerate(tr_idx):
            i,j=int(I[k]),int(J[k])
            va=da.get(part_a[i],0.0); Ftr[r,0]=(va if (va==va) else 0.0) if part_a[i]==part_a[j] else 0.0
            Ftr[r,1]=int(part_a[i]==part_a[j])
            vb=db.get(part_b[i],0.0); Ftr[r,2]=(vb if vb==vb else 0.0) if part_b[i]==part_b[j] else 0.0
            kk=min(core_arr[i],core_arr[j]); vk=dk.get(kk,0.0); Ftr[r,3]=vk if vk==vk else 0.0; Ftr[r,4]=kk
            shared=set(node_cliques[i])&set(node_cliques[j])
            vals=[dcl[ci] for ci in shared if dcl[ci]==dcl[ci]]
            Ftr[r,5]=np.mean(vals) if vals else 0.0
        Fte=F[test]
        return Ftr,Fte
    return fn

p0,p1,cov=holdout(make_fn())
a0=roc_auc_score(yrep[cov],p0[cov]); a1=roc_auc_score(yrep[cov],p1[cov])
P(f"[NON-LEAKY] BASELINE BM AUC={a0:.4f}  BM+bundle-density AUC={a1:.4f}  dAUC={a1-a0:+.4f}  cov={cov.sum()}")
hard=np.abs(r1)>0.3; FP=r1<-0.3; FN=r1>0.3
for nm,mask in [('hard',hard),('FP',FP),('FN',FN)]:
    m=mask&cov; better=(np.abs(yrep-p1)[m]<np.abs(yrep-p0)[m]).mean()
    P(f"{nm}: cov={m.sum()} fracImproved={better:.3f}")
mh=hard&cov
P(f"hard AUC(yrep): BM={roc_auc_score(yrep[mh],p0[mh]):.4f} BM+bundle={roc_auc_score(yrep[mh],p1[mh]):.4f}")
mfp=FP&cov; mfn=FN&cov
P(f"FP mean pred: BM={p0[mfp].mean():.4f} BM+bundle={p1[mfp].mean():.4f}")
P(f"FN mean pred: BM={p0[mfn].mean():.4f} BM+bundle={p1[mfn].mean():.4f}")
np.savez(f"{SC}/bundle_holdout2.npz",p0=p0,p1=p1,cov=cov,a0=a0,a1=a1)
