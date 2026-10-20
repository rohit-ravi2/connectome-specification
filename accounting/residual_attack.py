import sys, numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True)
I,J,y7,y8,base,katz,neurons=z['I'],z['J'],z['y7'],z['y8'],z['base'],z['katz'],list(z['neurons'])
M=len(neurons); N=len(I); yrep=((y7==1)&(y8==1)).astype(int); disagree=(y7!=y8).astype(int)
X=np.hstack([base,katz]); cv=StratifiedKFold(5,shuffle=True,random_state=0)
p=cross_val_predict(GradientBoostingClassifier(random_state=0),X,yrep,cv=cv,method='predict_proba')[:,1]
r=yrep-p  # residual
P(f"best-model AUC={roc_auc_score(yrep,p):.3f}  mean|resid|={np.abs(r).mean():.3f}")
# ================= (1) RESIDUAL LOW-RANK STRUCTURE =================
R=np.zeros((M,M)); R[I,J]=r
def topfrac(Rm,k=5):
    sv=np.linalg.svd(Rm,compute_uv=False); return (sv[:k]**2).sum()/ (sv**2).sum()
real=topfrac(R); rng=np.random.default_rng(0)
nulls=[]
for _ in range(30):
    rp=r[rng.permutation(N)]; Rn=np.zeros((M,M)); Rn[I,J]=rp; nulls.append(topfrac(Rn))
nulls=np.array(nulls)
P(f"\n[1] RESIDUAL LOW-RANK: top-5 SVD energy real={real:.3f} vs shuffle {nulls.mean():.3f}±{nulls.std():.3f}  z={(real-nulls.mean())/nulls.std():.1f}")
# ================= (2) PER-NEURON CONCENTRATION (hidden per-neuron variable) =================
resn=np.array([r[(I==n)|(J==n)].mean() if ((I==n)|(J==n)).sum()>3 else 0 for n in range(M)])
absn=np.array([np.abs(r[(I==n)|(J==n)]).mean() if ((I==n)|(J==n)).sum()>3 else np.nan for n in range(M)])
varreal=np.nanvar(resn); vnull=[]
for _ in range(30):
    rp=r[rng.permutation(N)]; vn=np.array([rp[(I==n)|(J==n)].mean() if ((I==n)|(J==n)).sum()>3 else 0 for n in range(M)]); vnull.append(np.nanvar(vn))
vnull=np.array(vnull); P(f"[2] PER-NEURON systematic resid var real={varreal:.4f} vs shuffle {vnull.mean():.4f}±{vnull.std():.4f}  z={(varreal-vnull.mean())/vnull.std():.1f}")
worst=np.argsort(-absn)[:8]; P("    most-mispredicted neurons:",[neurons[i] for i in worst])
# ================= (3) BILATERAL SYMMETRY / TEMPLATE =================
def mir(n):
    if n[-1]=='L': m=n[:-1]+'R'
    elif n[-1]=='R': m=n[:-1]+'L'
    else: return None
    return m if m in set(neurons) else None
idx={n:i for i,n in enumerate(neurons)}
yd={(I[k],J[k]):yrep[k] for k in range(N)}; pd={(I[k],J[k]):p[k] for k in range(N)}
my=np.full(N,-1); has=np.zeros(N,bool)
for k in range(N):
    ni,nj=neurons[I[k]],neurons[J[k]]; mi,mj=mir(ni),mir(nj)
    if mi and mj and (idx[mi],idx[mj]) in yd:
        my[k]=yd[(idx[mi],idx[mj])]; has[k]=True
P(f"\n[3] BILATERAL SYMMETRY: {has.sum()}/{N} pairs have a contralateral contacting mirror")
ag=(yrep[has]==my[has]).mean(); P(f"    edge<->mirror-edge AGREEMENT (repro-core) = {ag:.3f}")
h=has & (my>=0)
P(f"    P(edge | mirror=1)={yrep[h & (my==1)].mean():.3f}   P(edge | mirror=0)={yrep[h & (my==0)].mean():.3f}")
P(f"    mirror-state predicts edge: AUC={roc_auc_score(yrep[h],my[h]):.3f}  vs feature-model AUC={roc_auc_score(yrep[h],p[h]):.3f}")
# does mirror explain the FEATURE residual? correlation of mirror with r among uncertain feature preds
unc=h & (p>0.2)&(p<0.8)
if unc.sum()>30:
    P(f"    among feature-UNCERTAIN pairs (0.2<p<0.8, n={unc.sum()}): P(edge|mir=1)={yrep[unc&(my==1)].mean():.3f} P(edge|mir=0)={yrep[unc&(my==0)].mean():.3f}")
    P(f"      mirror AUC on uncertain={roc_auc_score(yrep[unc],my[unc]):.3f}  feature AUC on uncertain={roc_auc_score(yrep[unc],p[unc]):.3f}")
# corr mirror with signed residual
P(f"    corr(mirror-state, feature-residual r) = {np.corrcoef(my[h],r[h])[0,1]:.3f}")
np.savez(f"{SC}/resid.npz",p=p,r=r,my=my,has=has)
