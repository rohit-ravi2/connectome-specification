import sys, numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); z2=np.load(f"{SC}/resid2.npz"); zs=np.load(f"{SC}/subcellular.npz")
I,J,y7,y8,base,katz,neurons=z['I'],z['J'],z['y7'],z['y8'],z['base'],z['katz'],list(z['neurons'])
N=len(I); yrep=((y7==1)&(y8==1)).astype(int)
BM=np.hstack([base,katz,z2['MF']]); sub=zs['sub']; subloc=sub[:,[0,1,3,4,5,6]]
nf={n:i%5 for i,n in enumerate(sorted(set([neurons[i] for i in I]+[neurons[j] for j in J])))}
def nho(Xf):
    pred=np.full(N,np.nan)
    for f in range(5):
        te=[k for k in range(N) if nf[neurons[I[k]]]==f and nf[neurons[J[k]]]==f];tr=[k for k in range(N) if nf[neurons[I[k]]]!=f and nf[neurons[J[k]]]!=f]
        pred[np.array(te)]=GradientBoostingClassifier(random_state=0).fit(Xf[tr],yrep[tr]).predict_proba(Xf[te])[:,1]
    m=~np.isnan(pred);return roc_auc_score(yrep[m],pred[m])
a0=nho(BM); aall=nho(np.hstack([BM,sub])); aloc=nho(np.hstack([BM,subloc]))
P(f"base+katz+mirror {a0:.3f} | +sub-ALL {aall:.3f} (Δ{aall-a0:+.3f}) | +sub-loc {aloc:.3f} (Δ{aloc-a0:+.3f})")
rng=np.random.default_rng(0); nl=[]
for s in range(5):
    perm=rng.permutation(N); nl.append(nho(np.hstack([BM,subloc[perm]]))-a0)
nl=np.array(nl); P(f"[PERM control] real Δ={aloc-a0:+.3f} vs shuffled {nl.mean():+.3f}±{nl.std():.3f} -> {'REAL' if (aloc-a0)>nl.mean()+2*nl.std() else 'NOT beating null'}")
