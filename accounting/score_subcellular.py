import sys, numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); z2=np.load(f"{SC}/resid2.npz"); zs=np.load(f"{SC}/subcellular.npz")
I,J,y7,y8,base,katz,neurons=z['I'],z['J'],z['y7'],z['y8'],z['base'],z['katz'],list(z['neurons'])
N=len(I); yrep=((y7==1)&(y8==1)).astype(int); disagree=(y7!=y8).astype(int)
BM=np.hstack([base,katz,z2['MF']]); r1=z2['r1']; sub=zs['sub']
subloc=sub[:,[0,1,3,4,5,6]]  # along_src,along_tgt,cfrac_src,cfrac_tgt,spread_src,spread_tgt (subcellular-specific)
cv=StratifiedKFold(5,shuffle=True,random_state=0)
nf={n:i%5 for i,n in enumerate(sorted(set([neurons[i] for i in I]+[neurons[j] for j in J])))}
def nho(Xf,ret=False):
    pred=np.full(N,np.nan)
    for f in range(5):
        te=[k for k in range(N) if nf[neurons[I[k]]]==f and nf[neurons[J[k]]]==f];tr=[k for k in range(N) if nf[neurons[I[k]]]!=f and nf[neurons[J[k]]]!=f]
        pred[np.array(te)]=GradientBoostingClassifier(random_state=0).fit(Xf[tr],yrep[tr]).predict_proba(Xf[te])[:,1]
    m=~np.isnan(pred); return (roc_auc_score(yrep[m],pred[m]),pred) if ret else roc_auc_score(yrep[m],pred[m])
def disp(Xf):
    y2=np.concatenate([y7,y8]);X2=np.vstack([Xf,Xf]);p=np.clip(cross_val_predict(GradientBoostingClassifier(random_state=0),X2,y2,cv=cv,method='predict_proba')[:,1][:N],1e-4,1-1e-4)
    return disagree.mean()/(2*p*(1-p)).mean()
a0,p0=nho(BM,ret=True); P(f"base+katz+mirror AUC={a0:.3f} disp={disp(BM):.3f}")
P(f"  subcellular-location ALONE AUC={nho(sub):.3f}")
for lab,F in [("+subcellular-specific(along/spread)",subloc),("+subcellular ALL(+dmin,pos)",sub)]:
    Xf=np.hstack([BM,F]); a1,p1=nho(Xf,ret=True)
    # improvement on the 968 confidently-wrong reproducible pairs
    hard=np.abs(r1)>0.3
    imp=(np.abs(yrep[hard]-p1[hard])<np.abs(yrep[hard]-p0[hard])).mean()
    P(f"  {lab:36s} AUC={a1:.3f} (Δ={a1-a0:+.3f}) disp={disp(Xf):.3f} | hard-968 improved {imp*100:.0f}%")
