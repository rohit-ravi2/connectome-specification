import csv, re, math, numpy as np
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
D="../data/celegans/data/connectome"
R=list(csv.DictReader(open("data/processed/dev_stabilization_program/master_directed_chemical_pairs.csv")))
# developmental gap-junction coupling (non-circular activity substrate)
gj=defaultdict(set)
with open(f"{D}/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']=='electrical':
            ds=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1)); gj[(min(r['pre'],r['post']),max(r['pre'],r['post']))].add(ds)
def gjdev(r):
    v=gj.get((min(r['src'],r['tgt']),max(r['src'],r['tgt'])),set()); return len(v&{1,2,3,4,5,6})
# Randi adult functional connectivity (directed; CIRCULAR caveat)
randi=defaultdict(float)
with open("../data/celegans/data/randi2023/randi_funconn_wildty.tsv") as f:
    for line in f:
        p=line.rstrip('\n').split('\t')
        if len(p)>=3:
            try: randi[(p[0],p[1])]=float(p[2])
            except: pass
def c(n,f=float): return np.array([f(r[n]) for r in R])
a7=c('a7',int); a8=c('a8',int); disagree=(a7!=a8).astype(int)
base=['log_area','shared','cos','cdeg_s','cdeg_t','ntsame','src_share','tgt_share','src_rank','tgt_rank']
Xb=np.column_stack([c(k) for k in base])
GJ=np.array([[math.log1p(gjdev(r)), 1.0 if gjdev(r)>0 else 0.0] for r in R])
RA=np.array([[math.log1p(abs(randi.get((r['src'],r['tgt']),0.0)))] for r in R])
yrep=((a7==1)&(a8==1)).astype(int)
cv=StratifiedKFold(5,shuffle=True,random_state=0)
nf={n:i%5 for i,n in enumerate(sorted(set([r['src'] for r in R]+[r['tgt'] for r in R])))}
def nho(X):
    pred=np.full(len(R),np.nan)
    for f in range(5):
        te=[k for k,r in enumerate(R) if nf[r['src']]==f and nf[r['tgt']]==f]; tr=[k for k,r in enumerate(R) if nf[r['src']]!=f and nf[r['tgt']]!=f]
        if len(te)<20: continue
        pred[np.array(te)]=GradientBoostingClassifier(random_state=0).fit(X[tr],yrep[tr]).predict_proba(X[te])[:,1]
    m=~np.isnan(pred); return roc_auc_score(yrep[m],pred[m])
def disp(X):
    y2=np.concatenate([a7,a8]); X2=np.vstack([X,X])
    p=np.clip(cross_val_predict(GradientBoostingClassifier(random_state=0),X2,y2,cv=cv,method='predict_proba')[:,1][:len(R)],1e-4,1-1e-4)
    return disagree.mean()/(2*p*(1-p)).mean()
print("CANDIDATE 2 — activity-dependent completion. Metric: AUC(neuron-holdout) + dispersion ratio (->1 = aligns)")
print(f"  gap-junction dev-coupled chemical pairs: {int((GJ[:,1]>0).sum())} / {len(R)}")
for nm,X in [("baseline (contact/topology/NT)",Xb),
             ("+ developmental GAP-JUNCTION (clean)",np.hstack([Xb,GJ])),
             ("+ Randi funct-conn (CIRCULAR caveat)",np.hstack([Xb,RA])),
             ("+ both",np.hstack([Xb,GJ,RA]))]:
    print(f"  {nm:38s} AUC={nho(X):.3f}  dispersion_ratio={disp(X):.2f}")
print("  (baseline dispersion ratio was 0.48; ->1 means the feature captured the sharp propensity)")
