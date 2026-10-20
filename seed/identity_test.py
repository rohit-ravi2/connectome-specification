"""Rigorous make-or-break: does the residual survive geometry + per-neuron IDENTITY?
If per-neuron labels (one-hot i,j) + geometry explain wiring -> labels suffice, CLAIM DIES.
If large residual remains -> wiring is irreducibly PAIRWISE (partner-specific), claim robust."""
import openpyxl, csv, re, math, numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy import sparse

CONTACT = "../data/celegans/data/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx"
EDGES = "../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv"

wb = openpyxl.load_workbook(CONTACT, read_only=True, data_only=True)
ws = wb['adult nerve ring neighbors']; data=list(ws.iter_rows(values_only=True))
neurons=[h for h in data[0][1:]]; idx={n:i for i,n in enumerate(neurons)}; M=len(neurons)
A=np.zeros((M,M))
for row in data[1:]:
    rn=row[0]
    if rn is None or rn not in idx: continue
    for j,val in enumerate(row[1:]):
        if val in (None,''): continue
        try: A[idx[rn],j]=float(val)
        except (TypeError,ValueError): pass
A=np.maximum(A,A.T); deg_area=A.sum(1); deg_num=(A>0).sum(1)

chem=defaultdict(set); elec=defaultdict(set)
with open(EDGES) as f:
    for r in csv.DictReader(f):
        ds=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
        if ds not in (7,8): continue
        pre,post,typ=r['pre'],r['post'],r['type']
        (elec if typ=='electrical' else chem)[((min(pre,post),max(pre,post)) if typ=='electrical' else (pre,post))].add(ds)
repro=set()
for e,dss in chem.items():
    if len(dss)==2: repro.add((min(e),max(e)))
for e,dss in elec.items():
    if len(dss)==2: repro.add(e)

geom=[]; ids=[]; ys=[]
for i in range(M):
    for j in range(i+1,M):
        if A[i,j]<=0: continue
        vi,vj=A[i],A[j]
        cos=float(vi@vj/(np.linalg.norm(vi)*np.linalg.norm(vj)+1e-9))
        shared=int(((A[i]>0)&(A[j]>0)).sum())
        geom.append([math.log1p(A[i,j]),cos,shared,
                     math.log1p(min(deg_area[i],deg_area[j])),math.log1p(max(deg_area[i],deg_area[j])),
                     min(deg_num[i],deg_num[j]),max(deg_num[i],deg_num[j])])
        ids.append((i,j)); ys.append(1 if (neurons[i],neurons[j]) in repro or (neurons[j],neurons[i]) in repro else 0)
geom=np.array(geom); y=np.array(ys); N=len(y); pbar=y.mean()

# one-hot identity (symmetric: +1 for both endpoints)
onehot=sparse.lil_matrix((N,M))
for k,(i,j) in enumerate(ids): onehot[k,i]=1; onehot[k,j]=1
onehot=onehot.tocsr()
geom_std=(geom-geom.mean(0))/(geom.std(0)+1e-9)

def bits(model, X):
    cv=StratifiedKFold(5,shuffle=True,random_state=0)
    p=cross_val_predict(model,X,y,cv=cv,method='predict_proba')[:,1]
    p=np.clip(p,1e-6,1-1e-6)
    return -(y*np.log2(p)+(1-y)*np.log2(1-p)).sum(), roc_auc_score(y,p)

base=N*(-(pbar*math.log2(pbar)+(1-pbar)*math.log2(1-pbar)))
b_geom,a_geom=bits(GradientBoostingClassifier(random_state=0),geom)
b_id,a_id=bits(LogisticRegression(max_iter=2000,C=1.0),onehot)
Xcomb=sparse.hstack([sparse.csr_matrix(geom_std),onehot]).tocsr()
b_comb,a_comb=bits(LogisticRegression(max_iter=3000,C=1.0),Xcomb)

ID=math.log2(118)
for name,b,a in [("base (rate only)",base,0.5),("geometry (GB)",b_geom,a_geom),
                 ("per-neuron identity (one-hot)",b_id,a_id),("geometry + identity",b_comb,a_comb)]:
    print(f"  H(conn | {name:30s}) = {b:7.0f} bits  ({b/M:5.2f} bits/neuron)  AUC {a:.3f}")
print(f"\n  per-neuron identity budget = {ID:.2f} bits/neuron")
r=b_comb/M
print(f"\n  RESIDUAL after geometry+identity = {r:.2f} bits/neuron")
if r>ID:
    print(f"  -> SURVIVES: even geometry+per-neuron-identity leaves {r:.1f} bits/neuron (> {ID:.1f}).")
    print(f"     Wiring is IRREDUCIBLY PAIRWISE — not explained by labels+geometry.")
else:
    print(f"  -> COLLAPSES: geometry+identity explains the wiring; per-neuron labels suffice. NULL.")
