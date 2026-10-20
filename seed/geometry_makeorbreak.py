"""MAKE-OR-BREAK: does wiring specificity survive the BEST geometric model?
Give geometry every advantage (area + neighborhood similarity + shared neighbors +
degree), cross-validated strong classifier -> honest held-out residual entropy H(conn|geometry).
If residual >> per-neuron identity budget (~7 bits/neuron): per-neuron labels provably
insufficient (robust). If it collapses below: geometry suffices = Brittin, null."""
import openpyxl, csv, re, math, numpy as np
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold

CONTACT = "../data/celegans/data/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx"
EDGES = "../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv"

# contact area matrix
wb = openpyxl.load_workbook(CONTACT, read_only=True, data_only=True)
ws = wb['adult nerve ring neighbors']; data = list(ws.iter_rows(values_only=True))
neurons = [h for h in data[0][1:]]; idx = {n:i for i,n in enumerate(neurons)}; M=len(neurons)
A = np.zeros((M,M))
for row in data[1:]:
    rn = row[0]
    if rn is None or rn not in idx: continue
    for j,val in enumerate(row[1:]):
        if val in (None,''): continue
        try: A[idx[rn], j] = float(val)
        except (TypeError,ValueError): pass
A = np.maximum(A, A.T)
deg_area = A.sum(1); deg_num = (A>0).sum(1)

# reproducible connectome
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

# contacting pairs -> features + label
X_full, X_area, ys = [], [], []
for i in range(M):
    for j in range(i+1,M):
        if A[i,j] <= 0: continue
        vi, vj = A[i], A[j]
        cos = float(vi@vj/(np.linalg.norm(vi)*np.linalg.norm(vj)+1e-9))
        shared = int(((A[i]>0)&(A[j]>0)).sum())
        area = A[i,j]
        feats = [math.log1p(area), cos, shared,
                 math.log1p(min(deg_area[i],deg_area[j])), math.log1p(max(deg_area[i],deg_area[j])),
                 min(deg_num[i],deg_num[j]), max(deg_num[i],deg_num[j])]
        X_full.append(feats); X_area.append([math.log1p(area)])
        ys.append(1 if (neurons[i],neurons[j]) in repro or (neurons[j],neurons[i]) in repro else 0)
X_full=np.array(X_full); X_area=np.array(X_area); y=np.array(ys); N=len(y)
pbar=y.mean()

def bits(model, X):
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    p = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:,1]
    p = np.clip(p, 1e-6, 1-1e-6)
    ll = -(y*np.log2(p)+(1-y)*np.log2(1-p)).sum()
    from sklearn.metrics import roc_auc_score
    return ll, roc_auc_score(y, p)

base = N*(-(pbar*math.log2(pbar)+(1-pbar)*math.log2(1-pbar)))
area_bits, area_auc = bits(LogisticRegression(max_iter=1000), X_area)
full_bits, full_auc = bits(GradientBoostingClassifier(random_state=0), X_full)

ID_BUDGET = math.log2(118)   # per-neuron identity-code capacity (~118 classes)
print(f"contacting pairs N={N}, reproducible connections={int(y.sum())} ({100*pbar:.1f}%)")
print(f"\n  H(connectome)                    = {base:7.0f} bits   ({base/M:5.2f} bits/neuron)")
print(f"  H(connectome | contact AREA)     = {area_bits:7.0f} bits   ({area_bits/M:5.2f} bits/neuron)  AUC {area_auc:.3f}")
print(f"  H(connectome | FULL GEOMETRY)    = {full_bits:7.0f} bits   ({full_bits/M:5.2f} bits/neuron)  AUC {full_auc:.3f}")
print(f"     (features: area, neighborhood-cosine, shared-neighbors, degree x2 -- cross-validated)")
print(f"\n  per-neuron identity budget       = {ID_BUDGET:5.2f} bits/neuron  (log2 ~118 classes)")
resid_per_neuron = full_bits/M
print(f"\n  VERDICT: residual after best geometry = {resid_per_neuron:.2f} bits/neuron vs identity budget {ID_BUDGET:.2f}")
if resid_per_neuron > ID_BUDGET:
    print(f"  -> SURVIVES: residual ({resid_per_neuron:.1f}) EXCEEDS identity budget ({ID_BUDGET:.1f}) by "
          f"{resid_per_neuron/ID_BUDGET:.1f}x. Per-neuron labels provably insufficient; wiring needs pairwise info.")
else:
    print(f"  -> COLLAPSES: geometry drives residual BELOW identity budget. Confirms Brittin; null.")
