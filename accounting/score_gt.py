import sys, numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); g=np.load(f"{SC}/gnntensor.npz")
I,J,y7,y8,base,katz,neurons=z['I'],z['J'],z['y7'],z['y8'],z['base'],z['katz'],z['neurons']
gnn,tens=g['gnn'],g['tens']; N=len(I); disagree=(y7!=y8).astype(int); yrep=((y7==1)&(y8==1)).astype(int); cv=StratifiedKFold(5,shuffle=True,random_state=0)
bb=np.hstack([base,katz])
nf={n:i%5 for i,n in enumerate(sorted(set([neurons[i] for i in I]+[neurons[j] for j in J])))}
def nho(X,y=yrep):
    pred=np.full(N,np.nan)
    for f in range(5):
        te=[k for k in range(N) if nf[neurons[I[k]]]==f and nf[neurons[J[k]]]==f];tr=[k for k in range(N) if nf[neurons[I[k]]]!=f and nf[neurons[J[k]]]!=f]
        pred[np.array(te)]=GradientBoostingClassifier(random_state=0).fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
    m=~np.isnan(pred);return roc_auc_score(y[m],pred[m])
def disp(X):
    y2=np.concatenate([y7,y8]);X2=np.vstack([X,X]);p=np.clip(cross_val_predict(GradientBoostingClassifier(random_state=0),X2,y2,cv=cv,method='predict_proba')[:,1][:N],1e-4,1-1e-4)
    return disagree.mean()/(2*p*(1-p)).mean()
cfg={'bestbase':bb,'gnn':np.hstack([bb,gnn]),'tens':np.hstack([bb,tens]),'both':np.hstack([bb,gnn,tens])}
for name in sys.argv[1:]:
    X=cfg[name]; P(f"{name:9s} disp={disp(X):.3f}  AUC={nho(X):.3f}")
