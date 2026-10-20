import sys, csv, numpy as np
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
DE="../data/celegans/New Notebooks/data_derived"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); z2=np.load(f"{SC}/resid2.npz")
I,J,y7,y8,base,katz,neurons=z['I'],z['J'],z['y7'],z['y8'],z['base'],z['katz'],list(z['neurons'])
M=len(neurons); N=len(I); yrep=((y7==1)&(y8==1)).astype(int); disagree=(y7!=y8).astype(int); idx={n:i for i,n in enumerate(neurons)}
BM=np.hstack([base,katz,z2['MF']]); r1=z2['r1']
REL=np.zeros((M,M))
for a,b,r in list(csv.reader(open("../data/celegans/data/lineage/merged_lineage_table.csv")))[1:]:
    if a in idx and b in idx:
        try: REL[idx[a],idx[b]]=REL[idx[b],idx[a]]=float(r)
        except: pass
rel=np.array([REL[I[k],J[k]] for k in range(N)])
ez=np.load(f"{DE}/expression_tpm.npz",allow_pickle=True); exprN=[str(x) for x in ez['neurons']]; tpm=ez['tpm']; erow={n:k for k,n in enumerate(exprN)}
have=[n for n in neurons if n in erow]; Xe=np.log1p(tpm[[erow[n] for n in have]]); v=Xe.var(0)
Xe=np.nan_to_num((Xe[:,np.argsort(-v)[:3000]]-Xe[:,np.argsort(-v)[:3000]].mean(0))/(Xe[:,np.argsort(-v)[:3000]].std(0)+1e-9))
E=PCA(15,random_state=0).fit_transform(Xe); Emap={have[k]:E[k] for k in range(len(have))}; z0=np.zeros(15)
Ei=np.array([Emap.get(neurons[i],z0) for i in I]); Ej=np.array([Emap.get(neurons[j],z0) for j in J])
ecos=np.array([Ei[k]@Ej[k]/(np.linalg.norm(Ei[k])*np.linalg.norm(Ej[k])+1e-9) for k in range(N)])
BATT=np.column_stack([rel,ecos,Ei[:,:8],Ej[:,:8]])
cv=StratifiedKFold(5,shuffle=True,random_state=0)
nf={n:i%5 for i,n in enumerate(sorted(set([neurons[i] for i in I]+[neurons[j] for j in J])))}
def nho(Xf):
    pred=np.full(N,np.nan)
    for f in range(5):
        te=[k for k in range(N) if nf[neurons[I[k]]]==f and nf[neurons[J[k]]]==f];tr=[k for k in range(N) if nf[neurons[I[k]]]!=f and nf[neurons[J[k]]]!=f]
        pred[np.array(te)]=GradientBoostingClassifier(random_state=0).fit(Xf[tr],yrep[tr]).predict_proba(Xf[te])[:,1]
    m=~np.isnan(pred);return roc_auc_score(yrep[m],pred[m])
a0=nho(BM); aA=nho(np.hstack([BM,BATT]))
P(f"[JOINT KITCHEN-SINK] base+katz+mirror AUC={a0:.3f} -> +ALL(lineage+expr jointly) AUC={aA:.3f} (Δ={aA-a0:+.3f})")
# ---- SIZE OF THE REDUCIBLE TARGET (target-independent) ----
# dispersion gap: disp=0.69 means observed disagreement is 0.69x model-predicted sampling var.
# fully capturing determinism -> disp=1.0.  gap = reducible determinism remaining.
y2=np.concatenate([y7,y8]);X2=np.vstack([BM,BM]);p=np.clip(cross_val_predict(GradientBoostingClassifier(random_state=0),X2,y2,cv=cv,method='predict_proba')[:,1][:N],1e-4,1-1e-4)
dispv=disagree.mean()/(2*p*(1-p)).mean()
P(f"[TARGET SIZE] dispersion={dispv:.3f} (1.0=fully captured). Reducible determinism REMAINING = {(1-dispv)*100:.0f}% of sampling-scale variance -> NOT tiny, real structure uncaptured.")
P(f"  confidently-wrong reproducible pairs (|r1|>0.3): {(np.abs(r1)>0.3).sum()} ({(np.abs(r1)>0.3).mean()*100:.1f}%) = the concrete reducible target")
# is that low-rank structure REAL (something could reduce it) vs pattern artifact? (already z~8) -> reducible-in-principle
P(f"  mean|r1| overall={np.abs(r1).mean():.3f}")
