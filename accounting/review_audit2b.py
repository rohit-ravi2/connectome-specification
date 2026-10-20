import csv, math, re, numpy as np
from collections import defaultdict, Counter
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
D="../data/celegans/data/connectome"
R=[r for r in csv.DictReader(open("data/processed/specification_accounting/master_edge_table.csv")) if r['contact']=='1']
presC=defaultdict(set)
with open(f"{D}/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']=='chemical':
            ds=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1)); presC[(min(r['pre'],r['post']),max(r['pre'],r['post']))].add(ds)
for r in R:
    pr=(min(r['src'],r['tgt']),max(r['src'],r['tgt'])); r['repC']=int(7 in presC.get(pr,set()) and 8 in presC.get(pr,set()))
def col(n,f=float): return np.array([f(r[n]) for r in R])
y=col('repC',int); N=len(R)
# class-pair reproducibility rate as ONE target-encoded scalar (full-data = optimistic for structure)
cprate=defaultdict(list)
for r in R: cprate[frozenset((r['src_class'],r['tgt_class']))].append(r['repC'])
cpr=np.array([np.mean(cprate[frozenset((r['src_class'],r['tgt_class']))]) for r in R])
Xs=np.column_stack([col('log_area'),col('shared_nb'),col('cos_sim'),col('cdeg_min'),col('cdeg_max'),
                    np.array([1.0 if r['src_nt']==r['tgt_nt'] else 0.0 for r in R])])
Xe=np.column_stack([Xs,col('same_class'),col('lr_homolog'),cpr])
cv=StratifiedKFold(5,shuffle=True,random_state=0)
def h(p): return 0.0 if p<=0 or p>=1 else -p*math.log2(p)-(1-p)*math.log2(1-p)
p=cross_val_predict(GradientBoostingClassifier(random_state=0),Xs,y,cv=cv,method='predict_proba')[:,1]
resid=-(y*np.log2(np.clip(p,1e-6,1-1e-6))+(1-y)*np.log2(np.clip(1-p,1e-6,1-1e-6))).mean()
print("=== Phase 5 VETO (chemical-only contacting) ===")
print(f"  structural residual entropy = {resid:.4f} bits/pair  vs floor 0.617 [0.591,0.640]  -> "
      f"{'ABOVE CI (real gap)' if resid>0.640 else 'within/near CI'}")
def ecv(X): return roc_auc_score(y,cross_val_predict(GradientBoostingClassifier(random_state=0),X,y,cv=cv,method='predict_proba')[:,1])
neur=sorted(set([r['src'] for r in R]+[r['tgt'] for r in R])); nf={n:i%5 for i,n in enumerate(neur)}
def nho(X):
    pred=np.full(N,np.nan)
    for f in range(5):
        te=np.array([k for k,r in enumerate(R) if nf[r['src']]==f and nf[r['tgt']]==f])
        tr=np.array([k for k,r in enumerate(R) if nf[r['src']]!=f and nf[r['tgt']]!=f])
        if len(te)<20 or len(set(y[tr]))<2: continue
        pred[te]=GradientBoostingClassifier(random_state=0).fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
    m=~np.isnan(pred); return roc_auc_score(y[m],pred[m]),int(m.sum())
a_s,a_se=ecv(Xs),ecv(Xe); (ns,_),(nse,nc)=nho(Xs),nho(Xe)
print(f"  edge-CV:        structure {a_s:.3f} -> +class/LR/cprate {a_se:.3f} (Δ={a_se-a_s:+.3f})  [cprate is OPTIMISTIC full-data encode]")
print(f"  NEURON-HOLDOUT: structure {ns:.3f} -> +class/LR/cprate {nse:.3f} (Δ={nse-ns:+.3f}) [n={nc}]")
print(f"  VERDICT: residual {'STRUCTURED' if nse-ns>0.02 else 'NOISE-like under neuron-holdout'}")

print("\n=== Phase 6: PATH-ENTROPY CIRCULARITY ===")
def t3(r): return (1 if int(r['L1n'])>0 else 0,int(r['L2']),int(r['L3']))
def t5(r): return t3(r)+(int(r['ad7']),int(r['ad8']))
core=[r for r in R if r['repC']==1]; halo=[r for r in R if r['halo']=='1']
def st(rows,tf,nb):
    c=Counter(tf(r) for r in rows); n=sum(c.values()); return -sum((v/n)*math.log2(v/n) for v in c.values()), np.mean([sum(tf(r))/nb for r in rows])
for lab,tf,nb in [("5-bit incl adult (Round2, circular)",t5,5),("3-bit L1-L3 only (non-circular)",t3,3)]:
    ce,cp=st(core,tf,nb); he,hp=st(halo,tf,nb)
    print(f"  {lab:38s} core ent={ce:.2f}/per={cp:.2f} | halo ent={he:.2f}/per={hp:.2f} | sep={he-ce:+.2f}")
