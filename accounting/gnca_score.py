import sys,numpy as np
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
d=np.load(SC+"/deepfeats.npz",allow_pickle=True)
I,J=d['I'].astype(int),d['J'].astype(int); neurons=list(d['neurons'])
y7,y8=d['y7'],d['y8']; yrep=((y7==1)&(y8==1)).astype(int); N=len(I)
base,katz=d['base'],d['katz']
MF=np.load(SC+"/resid2.npz",allow_pickle=True)['MF']
r1=np.load(SC+"/resid2.npz",allow_pickle=True)['r1']
BM=np.hstack([base,katz,MF])
endpts=sorted(set([neurons[k] for k in list(I)+list(J)]))
nf={n:i%5 for i,n in enumerate(endpts)}
NI=np.array([nf[neurons[int(I[k])]] for k in range(N)]); NJ=np.array([nf[neurons[int(J[k])]] for k in range(N)])

# baseline neuron-holdout p0 (GBC on BM, same folds)
p0=np.full(N,np.nan); c0=np.zeros(N,bool)
for f in range(5):
    te=(NI==f)&(NJ==f); tr=(NI!=f)&(NJ!=f)
    m=GBC(random_state=0).fit(BM[tr],yrep[tr]); p0[te]=m.predict_proba(BM[te])[:,1]; c0[te]=True

hard=np.abs(r1)>0.3; FP=r1<-0.3; FN=r1>0.3
def disp(p,cov):
    pp=np.clip(p[cov],1e-4,1-1e-4)
    return np.mean((y7!=y8)[cov])/np.mean(2*pp*(1-pp))
def improv(pg,mask,cov):
    m=mask&cov&c0
    return (np.abs(yrep-pg)[m]<np.abs(yrep-p0)[m]).mean(), m.sum()

P(f"baseline BM neuron-holdout AUC={roc_auc_score(yrep[c0],p0[c0]):.4f} cov={c0.sum()}")
P(f"baseline disp(on holdout cov, comparable)={disp(p0,c0):.4f}")
P("")
import glob,os
for fn in sorted(glob.glob(SC+"/gnca_pred_*.npz")):
    name=os.path.basename(fn)[10:-4]
    z=np.load(fn); pg=z['preds']; cov=z['cov'].astype(bool)
    both=cov&c0
    auc=roc_auc_score(yrep[cov],pg[cov])
    dz=disp(pg,cov)
    hi,hn=improv(pg,hard,cov); fpi,fpn=improv(pg,FP,cov); fni,fnn=improv(pg,FN,cov)
    P(f"{name:14s} AUC={auc:.4f} disp={dz:.4f} | hard {hi*100:4.1f}%({hn}) FP {fpi*100:4.1f}%({fpn}) FN {fni*100:4.1f}%({fnn})")
