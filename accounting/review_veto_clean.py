import csv, math, re, numpy as np
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
D="../data/celegans/data/connectome"
R=[r for r in csv.DictReader(open("data/processed/specification_accounting/master_edge_table.csv")) if r['contact']=='1']
presC=defaultdict(set)
with open(f"{D}/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']=='chemical':
            ds=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1)); presC[(min(r['pre'],r['post']),max(r['pre'],r['post']))].add(ds)
for r in R:
    pr=(min(r['src'],r['tgt']),max(r['src'],r['tgt'])); s=presC.get(pr,set())
    r['a8c']=int(8 in s); r['a7c']=int(7 in s)
def col(n): return np.array([int(r[n]) for r in R])
def colf(n): return np.array([float(r[n]) for r in R])
def h(p): return 0.0 if p<=0 or p>=1 else -p*math.log2(p)-(1-p)*math.log2(1-p)
y=col('a8c'); a7=col('a7c')
# chemical single-animal floor
fl=sum((a7==v).mean()*h(y[a7==v].mean()) for v in (0,1) if (a7==v).sum())
Xs=np.column_stack([colf('log_area'),colf('shared_nb'),colf('cos_sim'),colf('cdeg_min'),colf('cdeg_max'),
                    np.array([1.0 if r['src_nt']==r['tgt_nt'] else 0.0 for r in R])])
Xe=np.column_stack([Xs,colf('same_class'),colf('lr_homolog')])  # NON-leaky scalars only
neur=sorted(set([r['src'] for r in R]+[r['tgt'] for r in R])); nf={n:i%5 for i,n in enumerate(neur)}
def nho(X):
    pred=np.full(len(R),np.nan)
    for f in range(5):
        te=[k for k,r in enumerate(R) if nf[r['src']]==f and nf[r['tgt']]==f]; tr=[k for k,r in enumerate(R) if nf[r['src']]!=f and nf[r['tgt']]!=f]
        if len(te)<20: continue
        pred[np.array(te)]=GradientBoostingClassifier(random_state=0).fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
    m=~np.isnan(pred); p=np.clip(pred[m],1e-6,1-1e-6)
    return roc_auc_score(y[m],pred[m]), -(y[m]*np.log2(p)+(1-y[m])*np.log2(1-p)).mean(), int(m.sum())
as_,rs,n=nho(Xs); ae,re_,_=nho(Xe)
print("=== CLEAN VETO: single-animal chemical ad8, neuron-holdout, non-leaky features ===")
print(f"  chemical single-animal floor H(a8c|a7c) = {fl:.4f} bits/pair")
print(f"  structure:            AUC={as_:.3f}  residual={rs:.4f} bits/pair")
print(f"  +same_class/lr(scalar): AUC={ae:.3f}  residual={re_:.4f}  (ΔAUC={ae-as_:+.3f})")
print(f"  residual vs floor: {rs:.3f} vs {fl:.3f} -> gap {rs-fl:+.3f} bits/pair")
print(f"  VERDICT: coarse class/LR adds {'STRUCTURE' if ae-as_>0.02 else '~NOTHING (residual noise-like beyond geometry)'} under neuron-holdout")
