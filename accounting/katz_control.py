import openpyxl, re, sys, numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
D="../data/celegans/data/connectome"
z=np.load("../work/deepfeats.npz",allow_pickle=True)
I,J,y7,y8,base,neurons=z['I'],z['J'],z['y7'],z['y8'],z['base'],list(z['neurons']); idx={n:i for i,n in enumerate(neurons)}; M=len(neurons)
N=len(I); yrep=((y7==1)&(y8==1)).astype(int)
def isneu(x): return bool(isinstance(x,str) and re.fullmatch(r"[A-Z]{2,4}[0-9]?[LR]?[0-9]?",str(x).strip()) and str(x).strip() in idx)
wc=openpyxl.load_workbook(f"{D}/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
cd=list(wc['hermaphrodite chemical'].iter_rows(values_only=True))
hr=max(range(6),key=lambda r:sum(isneu(x) for x in cd[r])); colmap={c:cd[hr][c].strip() for c in range(len(cd[hr])) if isneu(cd[hr][c])}
rhc=max(range(6),key=lambda c:sum(isneu(cd[r][c]) for r in range(hr+1,len(cd)) if c<len(cd[r])))
C=np.zeros((M,M))
for r in range(hr+1,len(cd)):
    if rhc<len(cd[r]) and isneu(cd[r][rhc]):
        pi=idx[cd[r][rhc].strip()]
        for c,post in colmap.items():
            if c<len(cd[r]) and cd[r][c] not in (None,''):
                try: C[pi,idx[post]]=1.0 if float(cd[r][c])>=1 else 0.0
                except: pass
rho=np.abs(np.linalg.eigvals(C)).max(); beta=0.5/rho
def katz_feats(Cm):
    K=np.linalg.inv(np.eye(M)-beta*Cm)-np.eye(M); C3=Cm@Cm@Cm; C4=C3@Cm
    return np.array([[K[i,j],K[j,i],np.log1p(C3[i,j]),np.log1p(C3[j,i]),np.log1p(C4[i,j]),np.log1p(C4[j,i])] for i,j in zip(I,J)])
nf={n:i%5 for i,n in enumerate(sorted(set([neurons[i] for i in I]+[neurons[j] for j in J])))}
def nho(X):
    pred=np.full(N,np.nan)
    for f in range(5):
        te=[k for k in range(N) if nf[neurons[I[k]]]==f and nf[neurons[J[k]]]==f];tr=[k for k in range(N) if nf[neurons[I[k]]]!=f and nf[neurons[J[k]]]!=f]
        pred[np.array(te)]=GradientBoostingClassifier(random_state=0).fit(X[tr],yrep[tr]).predict_proba(X[te])[:,1]
    m=~np.isnan(pred);return roc_auc_score(yrep[m],pred[m])
b=nho(base); real=nho(np.hstack([base,katz_feats(C)]))
P(f"baseline AUC={b:.3f} | +Katz(real) AUC={real:.3f}  gain={real-b:+.3f}")
rng=np.random.default_rng(0); perms=[]
for s in range(4):
    p=rng.permutation(M); Cp=C[np.ix_(p,p)]
    g=nho(np.hstack([base,katz_feats(Cp)]))-b; perms.append(g); P(f"  perm{s+1} node-shuffled-Katz gain={g:+.3f}")
perms=np.array(perms); P(f"REAL gain {real-b:+.3f} vs shuffled {perms.mean():+.3f}±{perms.std():.3f} -> {'REAL higher-order' if (real-b)>perms.mean()+2*perms.std() else 'NOT beating null'}")
