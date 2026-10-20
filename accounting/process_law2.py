import sys, csv, re, numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); z2=np.load(f"{SC}/resid2.npz")
I,J,y7,y8,base,katz,neurons=z['I'],z['J'],z['y7'],z['y8'],z['base'],z['katz'],list(z['neurons']); idx={n:i for i,n in enumerate(neurons)}
M=len(neurons); N=len(I); disagree=(y7!=y8).astype(int)
GL1=np.zeros((M,M))
with open("../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']=='chemical':
            d=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
            if d in (1,2,3,4) and r['pre'] in idx and r['post'] in idx: GL1[idx[r['pre']],idx[r['post']]]=1
Cu=((GL1+GL1.T)>0).astype(float); Cff=GL1@GL1; l1deg=Cu.sum(1)
L1hist=np.column_stack([GL1[I,J],GL1[J,I],np.log1p(Cff[I,J]),l1deg[I],l1deg[J]])  # persistence + closure
BM=np.hstack([base,katz,z2['MF']])  # old feature model (contact+higher-order+symmetry), NO L1 synapse status
cv=StratifiedKFold(5,shuffle=True,random_state=0)
def disp(X):
    y2=np.concatenate([y7,y8]); X2=np.vstack([X,X])
    p=np.clip(cross_val_predict(GradientBoostingClassifier(random_state=0),X2,y2,cv=cv,method='predict_proba')[:,1][:N],1e-4,1-1e-4)
    return disagree.mean()/(2*p*(1-p)).mean(), p
d_bm,_=disp(BM)
d_l1,p_l1=disp(np.hstack([BM,L1hist]))
d_h,_=disp(L1hist)
P(f"[UNDER-DISPERSION — is it PERSISTENCE?] (per-animal estimator, apples-to-apples with the arc)")
P(f"  old feature model base+katz+mirror (NO L1-synapse status): dispersion={d_bm:.3f}")
P(f"  + L1 developmental history (persistence + closure):        dispersion={d_l1:.3f}")
P(f"  L1-history ALONE:                                          dispersion={d_h:.3f}")
P(f"  -> {'PERSISTENCE resolves under-dispersion' if d_l1>d_bm+0.1 else 'persistence does NOT collapse under-dispersion'}: gap-to-1.0 {1-d_bm:.2f} -> {1-d_l1:.2f}")
# generated dispersion (process ensemble) with correct per-animal p
rng=np.random.default_rng(0); dg=[(lambda a,b:(a!=b).mean()/(2*p_l1*(1-p_l1)).mean())((rng.random(N)<p_l1).astype(int),(rng.random(N)<p_l1).astype(int)) for _ in range(200)]
dg=np.array(dg); P(f"  process-generated ensemble dispersion (correct p) = {dg.mean():.3f}±{dg.std():.3f} (=1 by construction; real={d_l1:.3f} => residual under-dispersion {max(0,dg.mean()-d_l1):.3f} beyond persistence+closure)")
