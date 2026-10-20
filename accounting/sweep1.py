import openpyxl, csv, math, numpy as np
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
D="../data/celegans/data/connectome"
# contact matrix
wb=openpyxl.load_workbook(f"{D}/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx",read_only=True,data_only=True)
data=list(wb['adult nerve ring neighbors'].iter_rows(values_only=True)); neurons=[h for h in data[0][1:] if isinstance(h,str)]; idx={n:i for i,n in enumerate(neurons)}; M=len(neurons)
A=np.zeros((M,M))
for row in data[1:]:
    if row[0] in idx:
        for j,v in enumerate(row[1:]):
            if v not in (None,''):
                try: A[idx[row[0]],j]=float(v)
                except: pass
A=np.maximum(A,A.T)
R=list(csv.DictReader(open("data/processed/dev_stabilization_program/master_directed_chemical_pairs.csv")))
def c(n,f=float): return np.array([f(r[n]) for r in R])
a7=c('a7',int); a8=c('a8',int)
base_keys=['log_area','shared','cos','cdeg_s','cdeg_t','ntsame','src_share','tgt_share','src_rank','tgt_rank']
Xb=np.column_stack([c(k) for k in base_keys])
cv=StratifiedKFold(5,shuffle=True,random_state=0)

print("="*74); print("CANDIDATE 1 — DISTRIBUTIONAL: is the halo = Bernoulli sampling of feature propensity?"); print("="*74)
# fit p(edge in one animal | features), pooled over both adults
y2=np.concatenate([a7,a8]); X2=np.vstack([Xb,Xb])
p=cross_val_predict(GradientBoostingClassifier(random_state=0),X2,y2,cv=cv,method='predict_proba')[:,1][:len(R)]
p=np.clip(p,1e-4,1-1e-4)
disagree=(a7!=a8).astype(int)
# predicted disagreement under independent Bernoulli(p): 2p(1-p)
pred_dis=2*p*(1-p)
print(f"  observed halo/disagreement rate:        {disagree.mean():.4f}")
print(f"  predicted (Bernoulli sampling 2p(1-p)): {pred_dis.mean():.4f}")
ratio=disagree.mean()/pred_dis.mean()
print(f"  ratio obs/pred = {ratio:.2f}  -> {'UNDER-dispersed (features miss real propensity = residual structure)' if ratio<0.85 else 'OVER-dispersed (extra variance beyond sampling)' if ratio>1.15 else 'MATCHES sampling (halo = specified stochasticity; gap explained as distributional)'}")
# calibration by p-bin
print("  calibration (p-bin: observed single-animal rate vs p, and disagreement vs 2p(1-p)):")
bins=np.quantile(p,[.2,.4,.6,.8])
for lo,hi in [(-1,bins[0]),(bins[0],bins[1]),(bins[1],bins[2]),(bins[2],bins[3]),(bins[3],2)]:
    m=(p>lo)&(p<=hi)
    if m.sum(): print(f"    p in ({max(lo,0):.2f},{min(hi,1):.2f}]: mean_p={p[m].mean():.2f} obs_edge_rate={((a7[m].mean()+a8[m].mean())/2):.2f} obs_disagree={disagree[m].mean():.2f} pred_disagree={pred_dis[m].mean():.2f}")

print("\n"+"="*74); print("CANDIDATE 3+4 — PHYSICAL/HIGHER-ORDER: does mesoscale contact structure add?"); print("="*74)
# spectral embedding of contact graph
deg=A.sum(1); Dm=np.diag(deg); L=Dm-A
w,V=np.linalg.eigh(L); emb=V[:,1:11]  # 10 nontrivial eigenvectors
km=KMeans(8,n_init=5,random_state=0).fit(emb).labels_
# higher-order contact features per pair
def hof(r):
    i,j=idx.get(r['src']),idx.get(r['tgt'])
    if i is None or j is None: return [0,0,0,0]
    tri=float(np.minimum(A[i],A[j]).sum())            # shared strong-contact context
    embd=float(np.linalg.norm(emb[i]-emb[j]))         # spectral distance
    samecom=1.0 if km[i]==km[j] else 0.0              # contact-community co-membership
    embdot=float(emb[i]@emb[j])
    return [math.log1p(tri),embd,samecom,embdot]
Xho=np.array([hof(r) for r in R]); Xe=np.hstack([Xb,Xho])
yrep=((a7==1)&(a8==1)).astype(int)
nf={n:i%5 for i,n in enumerate(sorted(set([r['src'] for r in R]+[r['tgt'] for r in R])))}
def nho(X):
    pred=np.full(len(R),np.nan)
    for f in range(5):
        te=[k for k,r in enumerate(R) if nf[r['src']]==f and nf[r['tgt']]==f]; tr=[k for k,r in enumerate(R) if nf[r['src']]!=f and nf[r['tgt']]!=f]
        if len(te)<20: continue
        pred[np.array(te)]=GradientBoostingClassifier(random_state=0).fit(X[tr],yrep[tr]).predict_proba(X[te])[:,1]
    m=~np.isnan(pred); return roc_auc_score(yrep[m],pred[m])
ab,ae=nho(Xb),nho(Xe)
print(f"  predict reproducible edge, NEURON-HOLDOUT:")
print(f"    pairwise contact/topology baseline: AUC={ab:.3f}")
print(f"    + mesoscale contact (spectral/community/triangle): AUC={ae:.3f}  (Δ={ae-ab:+.3f})")
print(f"  -> {'higher-order/physical structure ADDS (candidate alive)' if ae-ab>0.01 else 'no add on first operationalization (needs richer spatial features before ruling out)'}")
