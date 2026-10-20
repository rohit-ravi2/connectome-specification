"""Robustness of the 'no pairwise address-code' finding:
 (Control 4) THRESHOLD: repeat ceiling + low-rank on strong cores (>=1/>=2/>=4 synapses).
             The variable halo is weak edges; a cleaner strong-core RAISES the noise ceiling
             and is the fair test of whether an address-code signal hides under edge noise.
 (Model 3)   BIOLOGICAL CLASS-PAIR: does NT source-target compatibility add over geometry?
 (Control 2) LABEL PERMUTATION: NT class-pair lift must beat permuted NT labels.
"""
import openpyxl, csv, re, math, numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy import sparse

CONTACT="../data/celegans/data/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx"
EDGES="../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv"
T5="../data/tier5/connectome.npz"
RNG=np.random.default_rng(0)

wb=openpyxl.load_workbook(CONTACT,read_only=True,data_only=True); ws=wb['adult nerve ring neighbors']
data=list(ws.iter_rows(values_only=True)); neurons=[h for h in data[0][1:]]; idx={n:i for i,n in enumerate(neurons)}; M=len(neurons)
A=np.zeros((M,M))
for row in data[1:]:
    rn=row[0]
    if rn is None or rn not in idx: continue
    for j,val in enumerate(row[1:]):
        if val in (None,''): continue
        try: A[idx[rn],j]=float(val)
        except: pass
A=np.maximum(A,A.T); deg_area=A.sum(1); deg_num=(A>0).sum(1)

# per-adult synapse counts per undirected pair
syn=defaultdict(lambda: defaultdict(int))
with open(EDGES) as f:
    for r in csv.DictReader(f):
        ds=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
        if ds not in (7,8): continue
        pre,post=r['pre'],r['post']; syn[(min(pre,post),max(pre,post))][ds]+=int(r['synapses'])

# geometry features + NT labels
import numpy as _np
t5=np.load(T5,allow_pickle=True); t5nt={str(n):str(v) for n,v in zip(t5["names"],t5["nt_primary"])}
def base_class(n):
    b=str(n)
    for s in("L","R"):
        if b.endswith(s): b=b[:-1]
    return b
nt_of=[t5nt.get(neurons[i], t5nt.get(base_class(neurons[i]),"Unknown")) for i in range(M)]

I,J,geom,strengthmax,in7,in8=[],[],[],[],[],[]
for i in range(M):
    for j in range(i+1,M):
        if A[i,j]<=0: continue
        pr=(min(neurons[i],neurons[j]),max(neurons[i],neurons[j])); s=syn.get(pr,{})
        I.append(i);J.append(j)
        in7.append(s.get(7,0)); in8.append(s.get(8,0))
        strengthmax.append(max(s.get(7,0),s.get(8,0)))
        vi,vj=A[i],A[j]; cos=float(vi@vj/(np.linalg.norm(vi)*np.linalg.norm(vj)+1e-9)); shared=int(((A[i]>0)&(A[j]>0)).sum())
        geom.append([math.log1p(A[i,j]),cos,shared,math.log1p(min(deg_area[i],deg_area[j])),
                     math.log1p(max(deg_area[i],deg_area[j])),min(deg_num[i],deg_num[j]),max(deg_num[i],deg_num[j])])
I=np.array(I);J=np.array(J);G=np.array(geom); G=(G-G.mean(0))/(G.std(0)+1e-9)
in7=np.array(in7);in8=np.array(in8);strengthmax=np.array(strengthmax); N=len(I)
def bits(yy,p): p=np.clip(p,1e-6,1-1e-6); return -(yy*np.log2(p)+(1-yy)*np.log2(1-p)).sum()
cv=StratifiedKFold(5,shuffle=True,random_state=0)

print("="*88); print("CONTROL 4 — THRESHOLD ROBUSTNESS (strong core = present in both adults, min-synapse >= thr)"); print("="*88)
print(f"  {'thr':>4} | {'core N':>6} | {'rate':>5} | {'ceiling AUC (7<->8)':>19} | {'geometry AUC':>12} | {'geom==ceiling?':>13}")
for thr in [1,2,4]:
    y=((in7>=thr)&(in8>=thr)).astype(int)
    if y.sum()<30: print(f"  {thr:>4} | too few positives"); continue
    # ceiling: does 'adult7 >=thr' predict 'adult8 >=thr'
    ya=(in7>=thr).astype(int); yb=(in8>=thr).astype(int)
    p1=yb[ya==1].mean() if (ya==1).any() else 0; p0=yb[ya==0].mean() if (ya==0).any() else 0
    ceil=roc_auc_score(yb,np.where(ya==1,p1,p0))
    ceil2=roc_auc_score(ya,np.where(yb==1,yb[yb==1].mean() if False else (ya[yb==1].mean()),ya[yb==0].mean()))
    pg=cross_val_predict(GradientBoostingClassifier(random_state=0),G,y,cv=cv,method='predict_proba')[:,1]
    gauc=roc_auc_score(y,pg)
    print(f"  {thr:>4} | {int(y.sum()):>6} | {y.mean():.3f} | {ceil:>19.3f} | {gauc:>12.3f} | {'YES' if gauc>=ceil-0.02 else 'gap '+f'{ceil-gauc:+.3f}':>13}")

print("\n"+"="*88); print("MODEL 3 — BIOLOGICAL NT CLASS-PAIR compatibility over geometry (both-adults core)"); print("="*88)
y=((in7>=1)&(in8>=1)).astype(int)
cats=sorted(set(nt_of)); cix={c:k for k,c in enumerate(cats)}; K=len(cats)
def ntpair_feats(labels):
    # unordered NT-pair one-hot
    P=sparse.lil_matrix((N,K*K))
    for k in range(N):
        a,b=cix[labels[I[k]]],cix[labels[J[k]]]; lo,hi=min(a,b),max(a,b); P[k,lo*K+hi]=1
    return P.tocsr()
def evalX(X,label):
    p=cross_val_predict(LogisticRegression(max_iter=3000,C=1.0),X,y,cv=cv,method='predict_proba')[:,1]
    print(f"  {label:38s} bits={bits(y,p):7.0f}  AUC={roc_auc_score(y,p):.4f}"); return roc_auc_score(y,p)
Gs=sparse.csr_matrix(G)
a_g=evalX(Gs,"geometry")
a_gn=evalX(sparse.hstack([Gs,ntpair_feats(nt_of)]).tocsr(),f"geometry + NT class-pair ({K} NT classes)")
# Control 2: permuted NT labels
perm_aucs=[]
for s in range(10):
    pl=list(np.array(nt_of)[RNG.permutation(M)])
    perm_aucs.append(evalX(sparse.hstack([Gs,ntpair_feats(pl)]).tocsr(),f"  [perm {s+1}] geometry + shuffled-NT") if s<0 else
                     roc_auc_score(y,cross_val_predict(LogisticRegression(max_iter=2000),sparse.hstack([Gs,ntpair_feats(pl)]).tocsr(),y,cv=cv,method='predict_proba')[:,1]))
perm_aucs=np.array(perm_aucs)
print(f"\n  NT class-pair lift over geometry: ΔAUC={a_gn-a_g:+.4f}")
print(f"  permuted-NT control: mean ΔAUC={perm_aucs.mean()-a_g:+.4f} ± {perm_aucs.std():.4f}  "
      f"-> real NT {'BEATS' if (a_gn-perm_aucs.mean())>2*perm_aucs.std()+1e-9 else 'does NOT beat'} shuffled")
print("="*88)
