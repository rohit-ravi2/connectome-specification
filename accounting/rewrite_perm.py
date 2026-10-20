import sys, csv, re, numpy as np
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); z2=np.load(f"{SC}/resid2.npz")
I,J,y7,y8,base,katz,neurons=z['I'],z['J'],z['y7'],z['y8'],z['base'],z['katz'],list(z['neurons'])
M=len(neurons); N=len(I); yrep=((y7==1)&(y8==1)).astype(int); idx={n:i for i,n in enumerate(neurons)}
BM=np.hstack([base,katz,z2['MF']]); disagree=(y7!=y8).astype(int)
stg=np.zeros((M,M))
with open("../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']!='chemical': continue
        ds=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
        if ds in (1,2,3,4) and r['pre'] in idx and r['post'] in idx: stg[idx[r['pre']],idx[r['post']]]=1
CA=np.zeros((M,M)); CA[I,J]=1
def rewrite(seed,tau=2,iters=25):
    G=seed.copy(); step=np.where(G>0,0,-1).astype(float)
    for t in range(1,iters+1):
        Cu=((G+G.T)>0).astype(float); score=Cu@Cu
        cand=(CA>0)&(G==0)&(score>=tau)&(np.eye(M)==0)
        if not cand.any(): break
        G[cand]=1; step[(step<0)&(G>0)]=t
    return step
def genfeat(seed):
    step=rewrite(seed); gen=np.array([step[I[k],J[k]] for k in range(N)])
    return np.column_stack([(gen>0).astype(float),np.where(gen<0,99,gen)])
cv=StratifiedKFold(5,shuffle=True,random_state=0)
nf={n:i%5 for i,n in enumerate(sorted(set([neurons[i] for i in I]+[neurons[j] for j in J])))}
def nho(Xf):
    pred=np.full(N,np.nan)
    for f in range(5):
        te=[k for k in range(N) if nf[neurons[I[k]]]==f and nf[neurons[J[k]]]==f];tr=[k for k in range(N) if nf[neurons[I[k]]]!=f and nf[neurons[J[k]]]!=f]
        pred[np.array(te)]=GradientBoostingClassifier(random_state=0).fit(Xf[tr],yrep[tr]).predict_proba(Xf[te])[:,1]
    m=~np.isnan(pred);return roc_auc_score(yrep[m],pred[m])
def disp(Xf):
    y2=np.concatenate([y7,y8]);X2=np.vstack([Xf,Xf]);p=np.clip(cross_val_predict(GradientBoostingClassifier(random_state=0),X2,y2,cv=cv,method='predict_proba')[:,1][:N],1e-4,1-1e-4)
    return disagree.mean()/(2*p*(1-p)).mean()
a0=nho(BM); areal=nho(np.hstack([BM,genfeat(stg)])); dreal=disp(np.hstack([BM,genfeat(stg)]))
P(f"real L1-seed rewrite: AUC {a0:.3f}->{areal:.3f} (Δ{areal-a0:+.3f}) disp {disp(BM):.3f}->{dreal:.3f}")
rng=np.random.default_rng(0); nl=[]
for s in range(5):
    p=rng.permutation(M); nl.append(nho(np.hstack([BM,genfeat(stg[np.ix_(p,p)])]))-a0)
nl=np.array(nl); P(f"[PERM: shuffle L1-graph node labels] real Δ={areal-a0:+.3f} vs shuffled {nl.mean():+.3f}±{nl.std():.3f} -> {'REAL' if (areal-a0)>nl.mean()+2*nl.std() else 'NOT beating null'}")
