import sys, json, numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
DD="../data/celegans/data/connectome"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True)
I,J,y7,y8,base,katz,neurons=z['I'],z['J'],z['y7'],z['y8'],z['base'],z['katz'],list(z['neurons'])
M=len(neurons); N=len(I); yrep=((y7==1)&(y8==1)).astype(int); idx={n:i for i,n in enumerate(neurons)}; ns=set(neurons)
X=np.hstack([base,katz])
def mir(n):
    m=n[:-1]+('R' if n[-1]=='L' else 'L') if n[-1] in 'LR' else None
    return m if (m in ns) else None
Lrep={(I[k],J[k]):yrep[k] for k in range(N)}
mrep=np.full(N,-1.0); hs=np.zeros(N,bool)
for k in range(N):
    mi,mj=mir(neurons[I[k]]),mir(neurons[J[k]])
    if mi and mj and (idx[mi],idx[mj]) in Lrep: mrep[k]=Lrep[(idx[mi],idx[mj])]; hs[k]=True
MF=np.column_stack([hs.astype(float),np.where(hs,mrep,0.5)])
cv=StratifiedKFold(5,shuffle=True,random_state=0)
p1=cross_val_predict(GradientBoostingClassifier(random_state=0),np.hstack([X,MF]),yrep,cv=cv,method='predict_proba')[:,1]
r1=yrep-p1; R1=np.zeros((M,M)); R1[I,J]=r1
Ul,sv,Vr=np.linalg.svd(R1); Vr=Vr.T
P(f"residual-after-symmetry mean|r|={np.abs(r1).mean():.3f}; top singular vals {sv[:4].round(2)}")
# ---- node properties ----
t5=np.load("../data/tier5/connectome.npz",allow_pickle=True)
t5nt={str(n):str(v) for n,v in zip(t5["names"],t5["nt_primary"])}
def bcl(n): return (n[:-1] if n[-1] in 'LR' else n)
nt=[t5nt.get(neurons[i],t5nt.get(bcl(neurons[i]),"Unk")) for i in range(M)]
cent={}
for dsn in (7,8):
    sk=json.load(open(f"../seed/data/external/mechanistic_frontier/witvliet2021_developmental/data/skeletons/Dataset{dsn}_skeletons.json"))
    for n,v in sk.items():
        if n in idx and v.get('coords'): cent.setdefault(n,[]).append(np.array(list(v['coords'].values()),float).mean(0))
POS=np.full((M,3),np.nan)
for n,v in cent.items(): POS[idx[n]]=np.mean(v,0)
SENS={'ADF','ADL','AFD','ASE','ASG','ASH','ASI','ASJ','ASK','AWA','AWB','AWC','BAG','URX','IL1','IL2','OLL','OLQ','CEP','ADE','URY','FLP','URB','AVM','ALM','PLM','PVD','OLQ','IL','ASE'}
def isSens(n):
    c=bcl(n); c2=c.rstrip('DV') if c[-1:] in 'DV' else c
    return int(any(c.startswith(s) or c2.startswith(s) for s in SENS))
sens=np.array([isSens(n) for n in neurons]); cdeg=(base[:,3]);  # not per-node; recompute
deg=np.zeros(M)
for k in range(N): deg[I[k]]+=1
P(f"\nsensory neurons: {sens.sum()}/{M}")
def corr(a,b):
    m=~(np.isnan(a)|np.isnan(b)); 
    return np.corrcoef(a[m],b[m])[0,1] if m.sum()>10 else np.nan
P("\n[IDENTIFY hidden residual axis] |corr| of top-3 left/right singular vectors with node vars:")
for name,x in [("AP-position(x)",POS[:,0]),("DV(y)",POS[:,1]),("LR(z)",POS[:,2]),("degree",deg),
               ("sensory",sens.astype(float)),("NT=ACh",np.array([n=='Acetylcholine' or n=='ACh' for n in nt],float)),
               ("NT=Glu",np.array(['Glu' in n for n in nt],float)),("NT=GABA",np.array(['GABA' in n for n in nt],float))]:
    P(f"  {name:16s} L:[{corr(Ul[:,0],x):+.2f} {corr(Ul[:,1],x):+.2f} {corr(Ul[:,2],x):+.2f}]  R:[{corr(Vr[:,0],x):+.2f} {corr(Vr[:,1],x):+.2f} {corr(Vr[:,2],x):+.2f}]")
# top loading neurons on axis1
P("  axis-1 top+ neurons:",[neurons[i] for i in np.argsort(-Ul[:,0])[:6]]," top- :",[neurons[i] for i in np.argsort(Ul[:,0])[:6]])
# ---- PAIR-level residual correlates ----
P("\n[PAIR-level] does |residual| concentrate on relational categories? mean|r1| by group:")
sameNT=np.array([nt[I[k]]==nt[J[k]] for k in range(N)]); sameSens=np.array([sens[I[k]]==sens[J[k]]==1 for k in range(N)])
bothL=np.array([neurons[I[k]][-1]=='L' and neurons[J[k]][-1]=='L' for k in range(N)])
d3=np.array([np.linalg.norm(POS[I[k]]-POS[J[k]]) if not np.isnan(POS[I[k]]).any() and not np.isnan(POS[J[k]]).any() else np.nan for k in range(N)])
P(f"  same-NT {np.abs(r1[sameNT]).mean():.3f} vs diff {np.abs(r1[~sameNT]).mean():.3f} | both-sensory {np.abs(r1[sameSens]).mean():.3f} vs rest {np.abs(r1[~sameSens]).mean():.3f}")
P(f"  corr(|r1|, 3D-distance)={corr(np.abs(r1),d3):+.3f}")
np.savez(f"{SC}/resid2.npz",r1=r1,MF=MF,hs=hs,mrep=mrep)
