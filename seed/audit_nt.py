"""Round-2 AUDIT of the NT class-pair lift. (A) reconcile LR vs GB baseline + does NT
survive GB + calibrated log-loss + fold-level paired deltas. (B) neuron-holdout gate."""
import openpyxl, csv, re, math, numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy import sparse
RNG=np.random.default_rng(0)
CONTACT="../data/celegans/data/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx"
EDGES="../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv"
T5="../data/tier5/connectome.npz"
import os; print("T5 npz exists:", os.path.exists(T5))

wb=openpyxl.load_workbook(CONTACT,read_only=True,data_only=True); ws=wb['adult nerve ring neighbors']
data=list(ws.iter_rows(values_only=True)); neurons=[h for h in data[0][1:]]; idx={n:i for i,n in enumerate(neurons) if isinstance(n,str)}; M=len(neurons)
A=np.zeros((M,M))
for row in data[1:]:
    rn=row[0]
    if rn not in idx: continue
    for j,val in enumerate(row[1:]):
        if val in (None,''): continue
        try: A[idx[rn],j]=float(val)
        except: pass
A=np.maximum(A,A.T); deg_area=A.sum(1); deg_num=(A>0).sum(1)
present=defaultdict(set)
with open(EDGES) as f:
    for r in csv.DictReader(f):
        ds=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
        if ds in (7,8): present[(min(r['pre'],r['post']),max(r['pre'],r['post']))].add(ds)
t5=np.load(T5,allow_pickle=True); t5nt={str(n):str(v) for n,v in zip(t5["names"],t5["nt_primary"])}
def base_class(n):
    b=str(n)
    for s in("L","R"):
        if b.endswith(s): b=b[:-1]
    return b
nt_of=[t5nt.get(neurons[i], t5nt.get(base_class(neurons[i]),"Unknown")) if isinstance(neurons[i],str) else "Unknown" for i in range(M)]
from collections import Counter
print("NT class counts:", dict(Counter(nt_of)))

I,J,y,geom=[],[],[],[]
for i in range(M):
    for j in range(i+1,M):
        if A[i,j]<=0 or not isinstance(neurons[i],str) or not isinstance(neurons[j],str): continue
        s=present.get((min(neurons[i],neurons[j]),max(neurons[i],neurons[j])),set())
        I.append(i);J.append(j); y.append(1 if s=={7,8} else 0)
        vi,vj=A[i],A[j]; cos=float(vi@vj/(np.linalg.norm(vi)*np.linalg.norm(vj)+1e-9)); shared=int(((A[i]>0)&(A[j]>0)).sum())
        geom.append([math.log1p(A[i,j]),cos,shared,math.log1p(min(deg_area[i],deg_area[j])),math.log1p(max(deg_area[i],deg_area[j])),min(deg_num[i],deg_num[j]),max(deg_num[i],deg_num[j])])
I=np.array(I);J=np.array(J);y=np.array(y,float);G=np.array(geom); Gs=(G-G.mean(0))/(G.std(0)+1e-9); N=len(y)
cats=sorted(set(nt_of)); cix={c:k for k,c in enumerate(cats)}; K=len(cats)
def ntpair(labels):
    P=sparse.lil_matrix((N,K*K))
    for k in range(N):
        a,b=cix[labels[I[k]]],cix[labels[J[k]]]; lo,hi=min(a,b),max(a,b); P[k,lo*K+hi]=1
    return P.tocsr()
NT=ntpair(nt_of)
def logloss(yy,p): p=np.clip(p,1e-6,1-1e-6); return -(yy*np.log2(p)+(1-yy)*np.log2(1-p)).sum()
def platt(p,yy):  # CV-fit temperature+bias on held-out probs (same for both models -> fair compare)
    z=np.log(np.clip(p,1e-6,1-1e-6)/np.clip(1-p,1e-6,1-1e-6)).reshape(-1,1)
    lr=LogisticRegression().fit(z,yy); return lr.predict_proba(z)[:,1]
cv=StratifiedKFold(5,shuffle=True,random_state=0); folds=list(cv.split(np.zeros(N),y))
print(f"\nN={N} pairs, reproducible {int(y.sum())} ({100*y.mean():.1f}%), NT classes K={K}\n")
print("="*92); print("PART A — baseline reconciliation + does NT survive the STRONG (GB) baseline"); print("="*92)
def run(model_fn, X, name):
    p=cross_val_predict(model_fn(), X, y, cv=folds, method='predict_proba')[:,1]
    pc=platt(p,y)
    print(f"  {name:32s} AUC={roc_auc_score(y,p):.4f}  bits={logloss(y,p):6.0f}  "
          f"calib-bits={logloss(y,pc):6.0f}  Brier={brier_score_loss(y,p):.4f}")
    return p
pgLR = run(lambda:LogisticRegression(max_iter=3000,C=1.0), Gs, "geometry [LR]")
pgnLR= run(lambda:LogisticRegression(max_iter=3000,C=1.0), sparse.hstack([sparse.csr_matrix(Gs),NT]).tocsr(), "geometry+NT [LR]")
pgGB = run(lambda:GradientBoostingClassifier(random_state=0), Gs, "geometry [GB]")
pgnGB= run(lambda:GradientBoostingClassifier(random_state=0), np.hstack([Gs, NT.toarray()]), "geometry+NT [GB]")
print(f"\n  ΔAUC NT over geometry:  LR {roc_auc_score(y,pgnLR)-roc_auc_score(y,pgLR):+.4f}   "
      f"GB {roc_auc_score(y,pgnGB)-roc_auc_score(y,pgGB):+.4f}")
print(f"  Δbits NT over geometry: LR {logloss(y,pgnLR)-logloss(y,pgLR):+.0f}   GB {logloss(y,pgnGB)-logloss(y,pgGB):+.0f}")
# fold-level paired ΔAUC (LR)
dA=[]
for tr,te in folds:
    for X,st in [(sparse.csr_matrix(Gs),'g'),(sparse.hstack([sparse.csr_matrix(Gs),NT]).tocsr(),'gn')]:
        pass
    lg=LogisticRegression(max_iter=3000).fit(sparse.csr_matrix(Gs)[tr],y[tr]).predict_proba(sparse.csr_matrix(Gs)[te])[:,1]
    lgn=LogisticRegression(max_iter=3000).fit(sparse.hstack([sparse.csr_matrix(Gs),NT]).tocsr()[tr],y[tr]).predict_proba(sparse.hstack([sparse.csr_matrix(Gs),NT]).tocsr()[te])[:,1]
    dA.append(roc_auc_score(y[te],lgn)-roc_auc_score(y[te],lg))
dA=np.array(dA); print(f"  fold-level paired ΔAUC (LR): {dA.mean():+.4f} ± {dA.std():.4f}  per-fold {np.round(dA,4)}")

print("\n"+"="*92); print("PART B — NEURON-HOLDOUT GATE (strict: test pairs have BOTH endpoints unseen in training)"); print("="*92)
nfold=RNG.integers(0,5,size=M)
def neuron_holdout(X):
    pred=np.full(N,np.nan)
    for f in range(5):
        H=set(np.where(nfold==f)[0])
        te=np.array([k for k in range(N) if I[k] in H and J[k] in H])
        tr=np.array([k for k in range(N) if I[k] not in H and J[k] not in H])
        if len(te)<10: continue
        Xtr,Xte=X[tr],X[te]
        lr=LogisticRegression(max_iter=3000).fit(Xtr,y[tr]); pred[te]=lr.predict_proba(Xte)[:,1]
    m=~np.isnan(pred); return pred,m
Xg=sparse.csr_matrix(Gs); Xgn=sparse.hstack([sparse.csr_matrix(Gs),NT]).tocsr()
pg,m=neuron_holdout(Xg); pgn,m2=neuron_holdout(Xgn); mm=m&m2
print(f"  tested pairs (both endpoints held out): {mm.sum()}  positives {int(y[mm].sum())}")
print(f"  geometry [LR]     : AUC={roc_auc_score(y[mm],pg[mm]):.4f}  bits={logloss(y[mm],pg[mm]):.0f}")
print(f"  geometry+NT [LR]  : AUC={roc_auc_score(y[mm],pgn[mm]):.4f}  bits={logloss(y[mm],pgn[mm]):.0f}")
print(f"  ΔAUC under neuron-holdout: {roc_auc_score(y[mm],pgn[mm])-roc_auc_score(y[mm],pg[mm]):+.4f}  "
      f"Δbits: {logloss(y[mm],pgn[mm])-logloss(y[mm],pg[mm]):+.0f}")
print("="*92)
