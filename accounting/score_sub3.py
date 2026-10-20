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
a0,p0=nho(BM,ret=True); hard=np.abs(r1)>0.3
# individual sub-feature blocks: dmin(col2), along(0,1), spread(5,6), cfrac(3,4), pos(7,8,9)
blocks={'dmin(closest-approach)':[2],'along-arbor':[0,1],'cfrac-extent':[3,4],'spread-focal':[5,6],'3Dpos':[7,8,9],'ALL':list(range(10))}
for lab,cols in blocks.items():
    Xf=np.hstack([BM,sub[:,cols]]); a1,p1=nho(Xf,ret=True)
    imp=(np.abs(yrep[hard]-p1[hard])<np.abs(yrep[hard]-p0[hard])-1e-6).mean()
    d=disp(Xf) if lab in ('dmin(closest-approach)','ALL') else None
    P(f"  +{lab:22s} AUC={a1:.3f} (Δ{a1-a0:+.3f}) | hard-968 improved {imp*100:2.0f}%"+(f" | disp={d:.3f}" if d else ""))
P(f"\nbaseline disp=0.692; hard-968 = confidently-wrong reproducible pairs (|r1|>0.3)")
