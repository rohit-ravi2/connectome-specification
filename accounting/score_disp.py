import sys,numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict,StratifiedKFold
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True)
base,katz=z['base'],z['katz']; y7,y8=z['y7'],z['y8']; N=len(y7)
MF=np.load(f"{SC}/resid2.npz",allow_pickle=True)['MF']
DYN=np.load(f"{SC}/contactdyn.npz",allow_pickle=True)['DYN']
BM=np.hstack([base,katz,MF]); AUG=np.hstack([BM,DYN])
def disp(X):
    yy=np.concatenate([y7,y8])
    p=cross_val_predict(GradientBoostingClassifier(random_state=0),np.vstack([X,X]),yy,
        cv=StratifiedKFold(5,shuffle=True,random_state=0),method='predict_proba')[:,1][:N]
    p=np.clip(p,1e-4,1-1e-4)
    return np.mean(y7!=y8)/np.mean(2*p*(1-p))
P(f"dispersion BM={disp(BM):.4f}")
P(f"dispersion BM+DYN={disp(AUG):.4f}")
