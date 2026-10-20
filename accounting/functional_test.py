import sys, json, numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); z2=np.load(f"{SC}/resid2.npz")
I,J,y7,y8,base,katz,neurons=z['I'],z['J'],z['y7'],z['y8'],z['base'],z['katz'],list(z['neurons'])
M=len(neurons); N=len(I); yrep=((y7==1)&(y8==1)).astype(int); idx={n:i for i,n in enumerate(neurons)}
X=np.hstack([base,katz]); MF=z2['MF']; r1=z2['r1']
# Randi functional connectivity
F=np.zeros((M,M)); cov=0
for line in open("../data/celegans/data/randi2023/randi_funconn_wildty.tsv"):
    a,b,w=line.rstrip('\n').split('\t')
    if a in idx and b in idx:
        try: F[idx[a],idx[b]]=abs(float(w))
        except: pass
randi=np.array([[F[I[k],J[k]],F[J[k],I[k]],max(F[I[k],J[k]],F[J[k],I[k]])] for k in range(N)])
P(f"Randi coverage: {(randi.max(1)>0).sum()}/{N} pairs have nonzero funconn")
cv=StratifiedKFold(5,shuffle=True,random_state=0)
nf={n:i%5 for i,n in enumerate(sorted(set([neurons[i] for i in I]+[neurons[j] for j in J])))}
def nho(Xf):
    pred=np.full(N,np.nan)
    for f in range(5):
        te=[k for k in range(N) if nf[neurons[I[k]]]==f and nf[neurons[J[k]]]==f];tr=[k for k in range(N) if nf[neurons[I[k]]]!=f and nf[neurons[J[k]]]!=f]
        pred[np.array(te)]=GradientBoostingClassifier(random_state=0).fit(Xf[tr],yrep[tr]).predict_proba(Xf[te])[:,1]
    m=~np.isnan(pred);return roc_auc_score(yrep[m],pred[m])
a_bm=nho(np.hstack([X,MF])); a_bmr=nho(np.hstack([X,MF,randi]))
P(f"[FUNCTIONAL] base+katz+mirror AUC={a_bm:.3f} -> +Randi-funconn AUC={a_bmr:.3f}  (Δ={a_bmr-a_bm:+.3f})")
# does funconn correlate with residual (on covered pairs)?
cvd=randi.max(1)>0
P(f"  corr(|residual|, |funconn|) on covered pairs = {np.corrcoef(np.abs(r1[cvd]),randi[cvd].max(1))[0,1]:+.3f}")
P(f"  mean|resid| covered {np.abs(r1[cvd]).mean():.3f} vs uncovered {np.abs(r1[~cvd]).mean():.3f}")
# ---- is the low-rank residual DRIVEN by sensory block? ----
t5=np.load("../data/tier5/connectome.npz",allow_pickle=True)
def bcl(n): return (n[:-1] if n[-1] in 'LR' else n)
SENS={'ADF','ADL','AFD','ASE','ASG','ASH','ASI','ASJ','ASK','AWA','AWB','AWC','BAG','URX','IL1','IL2','OLL','OLQ','CEP','ADE','URY','FLP','URB','AVM','ALM','PLM','PVD'}
def isS(n):
    c=bcl(n); c2=c.rstrip('DV')
    return any(c.startswith(s) or c2.startswith(s) for s in SENS)
sens=np.array([isS(n) for n in neurons])
nonsens_pair=np.array([not(sens[I[k]] or sens[J[k]]) for k in range(N)])
def topfrac_sub(mask):
    R=np.zeros((M,M)); R[I[mask],J[mask]]=r1[mask]; sv=np.linalg.svd(R,compute_uv=False); return (sv[:5]**2).sum()/(sv**2).sum(),mask.sum()
rng=np.random.default_rng(2)
def nulltop(mask):
    out=[]
    rr=r1[mask]; ii,jj=I[mask],J[mask]
    for _ in range(30):
        R=np.zeros((M,M)); R[ii,jj]=rr[rng.permutation(len(rr))]; sv=np.linalg.svd(R,compute_uv=False); out.append((sv[:5]**2).sum()/(sv**2).sum())
    return np.array(out)
for lab,mask in [("ALL pairs",np.ones(N,bool)),("NON-sensory pairs only",nonsens_pair),("SENSORY-touching pairs",~nonsens_pair)]:
    real,n=topfrac_sub(mask); nl=nulltop(mask); P(f"  low-rank {lab:24s} real={real:.3f} shuffle={nl.mean():.3f}±{nl.std():.3f} z={(real-nl.mean())/nl.std():.1f} (n={n})")
