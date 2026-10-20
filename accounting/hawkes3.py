import sys, csv, re, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); m0=np.load(f"{SC}/mode0.npz")
I,J,y7,y8,base,neurons=z['I'],z['J'],z['y7'],z['y8'],z['base'],list(z['neurons']); idx={n:i for i,n in enumerate(neurons)}
M=len(neurons); N=len(I); yrep=((y7==1)&(y8==1)).astype(int); L1c=m0['L1c']; area=base[:,0]
GL1=np.zeros((M,M))
with open("../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']=='chemical':
            d=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
            if d in (1,2,3,4) and r['pre'] in idx and r['post'] in idx: GL1[idx[r['pre']],idx[r['post']]]=1
L1e=(L1c>=2).astype(float); L1syn=GL1[I,J]
Cff0=GL1@GL1; lr=LogisticRegression(max_iter=1000).fit(np.column_stack([area,np.log1p(Cff0[I,J])])[L1e==0],yrep[L1e==0])
a,b=lr.coef_[0]; c0=lr.intercept_[0]
def sig(x): return 1/(1+np.exp(-np.clip(x,-30,30)))
cv=StratifiedKFold(5,shuffle=True,random_state=0)
def disp_estimator(ya,yb,feats):
    # IDENTICAL to arc: obs disagreement / 2 phat(1-phat), phat=CV feature-model on reproducible core
    yr=((ya==1)&(yb==1)).astype(int); dis=(ya!=yb).mean()
    y2=np.concatenate([ya,yb]); X2=np.vstack([feats,feats])
    p=np.clip(cross_val_predict(GradientBoostingClassifier(random_state=0),X2,y2,cv=cv,method='predict_proba')[:,1][:N],1e-4,1-1e-4)
    return dis/(2*p*(1-p)).mean()
# generate calibrated ensemble w/ modest pruning; take 2 synthetic animals
real_rate=(y7==1).mean(); lo,hi=-6,3
def gen1(cadj,rng):
    G=GL1.copy(); Cff=G@G; padd=sig(a*area+b*np.log1p(Cff[I,J])+c0+cadj); cur=G[I,J]>0
    add=(rng.random(N)<padd)&(~cur)
    for k in np.where(add)[0]: G[I[k],J[k]]=1
    pr=(rng.random(N)<0.2)&(GL1[I,J]>0); 
    for k in np.where(pr)[0]: G[I[k],J[k]]=0
    return G[I,J]>0
rng=np.random.default_rng(0)
for _ in range(16):
    mid=(lo+hi)/2; r=np.mean([gen1(mid,np.random.default_rng(q)).mean() for q in range(6)])
    if r>real_rate: hi=mid
    else: lo=mid
cadj=(lo+hi)/2
sa=gen1(cadj,np.random.default_rng(101)).astype(int); sb=gen1(cadj,np.random.default_rng(202)).astype(int)
# synthetic closure feature for the estimator
def cff_of(yy):
    G=np.zeros((M,M)); 
    for k in np.where(yy==1)[0]: G[I[k],J[k]]=1
    C=G@G; return np.log1p(C[I,J])
cffS=(cff_of(sa)+cff_of(sb))/2
featsA=np.column_stack([area,cffS])                 # contact+closure ONLY (like arc's adult-structure features)
featsB=np.column_stack([area,cffS,L1syn])           # + persistence
P(f"real edge rate {real_rate:.3f} | synthetic {sa.mean():.3f}/{sb.mean():.3f}")
P(f"[SYNTHETIC dispersion, identical estimator]")
P(f"  features contact+closure       = {disp_estimator(sa,sb,featsA):.2f}")
P(f"  features contact+closure+L1syn = {disp_estimator(sa,sb,featsB):.2f}")
P(f"[REAL dispersion, same estimator on the 2 adults]")
P(f"  features contact+closure       = {disp_estimator(y7,y8,np.column_stack([area,(cff_of(y7)+cff_of(y8))/2])):.2f}")
P(f"  features contact+closure+L1syn = {disp_estimator(y7,y8,np.column_stack([area,(cff_of(y7)+cff_of(y8))/2,L1syn])):.2f}")
