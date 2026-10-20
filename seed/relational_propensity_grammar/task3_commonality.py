import sys,numpy as np,pandas as pd,itertools
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/feats.npz",allow_pickle=True)
I,J=z['I'],z['J']; y7,y8,yrep=z['y7'],z['y8'],z['yrep']
CONTACT,HO,TIMING=z['CONTACT'],z['HO'],z['TIMING']; neurons=list(z['neurons']); N=len(I)
blocks={'C':CONTACT,'H':HO,'T':TIMING}
# neuron-disjoint folds
endpts=sorted(set([neurons[int(I[k])] for k in range(N)]+[neurons[int(J[k])] for k in range(N)]))
rng=np.random.RandomState(0); perm=rng.permutation(len(endpts))
nf={endpts[perm[i]]:i%5 for i in range(len(endpts))}
NI=np.array([nf[neurons[int(I[k])]] for k in range(N)]); NJ=np.array([nf[neurons[int(J[k])]] for k in range(N)])
covered=np.zeros(N,bool)
for f in range(5): covered |= ((NI==f)&(NJ==f))
P(f"N={N} covered(both-in-fold)={covered.sum()} yrep_rate_all={yrep.mean():.3f} yrep_rate_cov={yrep[covered].mean():.3f}")
def holdout_pred(X,y):
    pred=np.full(N,np.nan)
    for f in range(5):
        te=(NI==f)&(NJ==f); tr=(NI!=f)&(NJ!=f)
        if te.sum()==0: continue
        m=GradientBoostingClassifier(random_state=0).fit(X[tr],y[tr])
        pred[te]=m.predict_proba(X[te])[:,1]
    return pred
def bss_auc(X):
    p=holdout_pred(X,yrep); m=covered
    pc=np.clip(p[m],1e-4,1-1e-4); yc=yrep[m]
    base=yc.mean(); brier=np.mean((pc-yc)**2); brier_ref=np.mean((base-yc)**2)
    bss=1-brier/brier_ref
    auc=roc_auc_score(yc,pc)
    return bss,auc,brier
def disp_ratio(X):
    # propensity from doubled (y7,y8), neuron-disjoint; ratio on covered pairs
    yy=np.concatenate([y7,y8]); XX=np.vstack([X,X])
    NI2=np.concatenate([NI,NI]);NJ2=np.concatenate([NJ,NJ])
    pred=np.full(2*N,np.nan)
    for f in range(5):
        te=(NI2==f)&(NJ2==f); tr=(NI2!=f)&(NJ2!=f)
        if te.sum()==0: continue
        m=GradientBoostingClassifier(random_state=0).fit(XX[tr],yy[tr])
        pred[te]=m.predict_proba(XX[te])[:,1]
    p=np.clip(pred[:N],1e-4,1-1e-4); mm=covered
    return np.mean((y7[mm]!=y8[mm]))/np.mean(2*p[mm]*(1-p[mm]))
def X_of(sub):
    return np.hstack([blocks[b] for b in sub])
subsets=[]
for r in range(1,4):
    for c in itertools.combinations('CHT',r): subsets.append(''.join(c))
R={}; AUC={}; DISP={}; BRIER={}
for sub in subsets:
    b,a,br=bss_auc(X_of(sub)); d=disp_ratio(X_of(sub))
    R[sub]=b; AUC[sub]=a; DISP[sub]=d; BRIER[sub]=br
    P(f"  {sub:4s} BSS={b:+.4f} AUC={a:.3f} disp={d:.3f} brier={br:.4f}")
# commonality decomposition (R = BSS)
full='CHT'
U={'C':R['CHT']-R['HT'],'H':R['CHT']-R['CT'],'T':R['CHT']-R['CH']}
Cxy={'CH':R['CT']+R['HT']-R['T']-R['CHT'],
     'CT':R['CH']+R['HT']-R['H']-R['CHT'],
     'HT':R['CH']+R['CT']-R['C']-R['CHT']}
Cxyz=R['C']+R['H']+R['T']-R['CH']-R['CT']-R['HT']+R['CHT']
tot=R['CHT']
comp=[('unique_CONTACT',U['C']),('unique_HIGHERORDER',U['H']),('unique_TIMING',U['T']),
      ('shared_C&H',Cxy['CH']),('shared_C&T',Cxy['CT']),('shared_H&T',Cxy['HT']),
      ('shared_C&H&T',Cxyz)]
P("\n=== COMMONALITY DECOMPOSITION (metric=held-out BSS on yrep, neuron-disjoint) ===")
P(f"full-model BSS (CHT) = {tot:+.4f}  (sum of parts = {sum(v for _,v in comp):+.4f})")
for name,v in comp:
    P(f"  {name:20s} = {v:+.4f}  ({100*v/tot:+.1f}% of full)")
# save
rows=[]
for sub in subsets: rows.append(dict(subset=sub,BSS=R[sub],AUC=AUC[sub],disp_ratio=DISP[sub],brier=BRIER[sub]))
pd.DataFrame(rows).to_csv(f"{SC}/subset_scores.csv",index=False)
comp_rows=[dict(component=n,value=v,pct_of_full=100*v/tot) for n,v in comp]
comp_rows.append(dict(component='FULL_BSS',value=tot,pct_of_full=100.0))
pd.DataFrame(comp_rows).to_csv(f"{SC}/part1_commonality.csv",index=False)
P("saved part1_commonality.csv")
