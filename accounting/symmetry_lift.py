import sys, numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True)
I,J,y7,y8,base,katz,neurons=z['I'],z['J'],z['y7'],z['y8'],z['base'],z['katz'],list(z['neurons'])
M=len(neurons); N=len(I); yrep=((y7==1)&(y8==1)).astype(int); disagree=(y7!=y8).astype(int); idx={n:i for i,n in enumerate(neurons)}
X=np.hstack([base,katz]); ns=set(neurons)
def mir(n):
    if n[-1]=='L': m=n[:-1]+'R'
    elif n[-1]=='R': m=n[:-1]+'L'
    else: return None
    return m if m in ns else None
def mirror_of(lab):  # lab: dict (i,j)->val
    out=np.full(N,-1.0); hs=np.zeros(N,bool)
    for k in range(N):
        mi,mj=mir(neurons[I[k]]),mir(neurons[J[k]])
        if mi and mj and (idx[mi],idx[mj]) in lab: out[k]=lab[(idx[mi],idx[mj])]; hs[k]=True
    return out,hs
Lrep={(I[k],J[k]):yrep[k] for k in range(N)}; L7={(I[k],J[k]):y7[k] for k in range(N)}; L8={(I[k],J[k]):y8[k] for k in range(N)}
mrep,hs=mirror_of(Lrep); m7,_=mirror_of(L7); m8,_=mirror_of(L8)
MF=np.column_stack([hs.astype(float),np.where(hs,mrep,0.5)])  # [has_mirror, mirror-state]
# fold assignment: force each neuron and its mirror into DIFFERENT folds (leak-free symmetry test)
fold={}
for i,n in enumerate(neurons):
    b=n[:-1] if n[-1] in 'LR' else n; g=(hash(b)%5)
    fold[n]=g if (n[-1]!='R') else (g+1)%5
def nho(Xf):
    pred=np.full(N,np.nan)
    for f in range(5):
        te=[k for k in range(N) if fold[neurons[I[k]]]==f and fold[neurons[J[k]]]==f]
        tr=[k for k in range(N) if fold[neurons[I[k]]]!=f and fold[neurons[J[k]]]!=f]
        if len(te)<20: continue
        pred[np.array(te)]=GradientBoostingClassifier(random_state=0).fit(Xf[tr],yrep[tr]).predict_proba(Xf[te])[:,1]
    m=~np.isnan(pred);return roc_auc_score(yrep[m],pred[m]),m
a0,m0=nho(X); a1,m1=nho(np.hstack([X,MF]))
P(f"[LIFT, leak-free mirror-in-diff-fold] base+katz AUC={a0:.3f} -> +mirror-template AUC={a1:.3f}  (Δ={a1-a0:+.3f})")
# dispersion with PER-ANIMAL mirror (no cross-animal leak)
cv=StratifiedKFold(5,shuffle=True,random_state=0)
def disp_sym(withmir):
    if withmir:
        Xa=np.vstack([np.hstack([X,hs[:,None].astype(float),np.where(hs,m7,0.5)[:,None]]),
                      np.hstack([X,hs[:,None].astype(float),np.where(hs,m8,0.5)[:,None]])])
    else: Xa=np.vstack([X,X])
    y2=np.concatenate([y7,y8]); p=np.clip(cross_val_predict(GradientBoostingClassifier(random_state=0),Xa,y2,cv=cv,method='predict_proba')[:,1][:N],1e-4,1-1e-4)
    return disagree.mean()/(2*p*(1-p)).mean()
P(f"[DISPERSION, per-animal mirror] base+katz disp={disp_sym(False):.3f} -> +mirror disp={disp_sym(True):.3f}")
# does low-rank residual SURVIVE the mirror model? re-fit with mirror, re-SVD
p1=cross_val_predict(GradientBoostingClassifier(random_state=0),np.hstack([X,MF]),yrep,cv=cv,method='predict_proba')[:,1]
r1=yrep-p1; R1=np.zeros((M,M)); R1[I,J]=r1
sv=np.linalg.svd(R1,compute_uv=False); rng=np.random.default_rng(1); nl=[]
for _ in range(30):
    rp=r1[rng.permutation(N)]; Rn=np.zeros((M,M)); Rn[I,J]=rp; s2=np.linalg.svd(Rn,compute_uv=False); nl.append((s2[:5]**2).sum()/(s2**2).sum())
nl=np.array(nl); real=(sv[:5]**2).sum()/(sv**2).sum()
P(f"[RESIDUAL after mirror] top-5 SVD real={real:.3f} vs shuffle {nl.mean():.3f}±{nl.std():.3f} z={(real-nl.mean())/nl.std():.1f}  (mean|r|={np.abs(r1).mean():.3f})")
