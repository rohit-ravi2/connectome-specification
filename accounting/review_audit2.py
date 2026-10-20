import csv, math, re, numpy as np
from collections import defaultdict, Counter
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
D="../data/celegans/data/connectome"
R=[r for r in csv.DictReader(open("data/processed/specification_accounting/master_edge_table.csv")) if r['contact']=='1']
# chemical-only reproducible target (fix Flaw 1)
presC=defaultdict(set)
with open(f"{D}/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']=='chemical':
            ds=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1)); presC[(min(r['pre'],r['post']),max(r['pre'],r['post']))].add(ds)
for r in R:
    pr=(min(r['src'],r['tgt']),max(r['src'],r['tgt'])); r['repC']=int(7 in presC.get(pr,set()) and 8 in presC.get(pr,set()))
def col(n,f=float): return np.array([f(r[n]) for r in R])
Xs=np.column_stack([col('log_area'),col('shared_nb'),col('cos_sim'),col('cdeg_min'),col('cdeg_max'),
                    np.array([1.0 if r['src_nt']==r['tgt_nt'] else 0.0 for r in R])])
y=col('repC',int); cv=StratifiedKFold(5,shuffle=True,random_state=0)
def h(p): return 0.0 if p<=0 or p>=1 else -p*math.log2(p)-(1-p)*math.log2(1-p)

print("=== Phase 5 VETO: is the contacting-pair residual STRUCTURED or NOISE? (chemical-only) ===")
# structural residual entropy vs floor
p=cross_val_predict(GradientBoostingClassifier(random_state=0),Xs,y,cv=cv,method='predict_proba')[:,1]
resid=-(y*np.log2(np.clip(p,1e-6,1-1e-6))+(1-y)*np.log2(np.clip(1-p,1e-6,1-1e-6))).mean()
print(f"  structural model residual entropy (chemical repro) = {resid:.4f} bits/pair; edge-CV AUC={roc_auc_score(y,p):.3f}")
# class + L/R features
cats=sorted(set([r['src_class'] for r in R]+[r['tgt_class'] for r in R])); cix={c:i for i,c in enumerate(cats)}; K=len(cats)
CP=np.zeros((len(R),K*K))
for k,r in enumerate(R): a,b=cix[r['src_class']],cix[r['tgt_class']]; lo,hi=min(a,b),max(a,b); CP[k,lo*K+hi]=1
extra=np.column_stack([CP, col('lr_homolog'), col('same_class')])
# EDGE-CV vs NEURON-HOLDOUT
def edgecv(X): return roc_auc_score(y,cross_val_predict(GradientBoostingClassifier(random_state=0),X,y,cv=cv,method='predict_proba')[:,1])
neur=sorted(set([r['src'] for r in R]+[r['tgt'] for r in R])); nf={n:i%5 for i,n in enumerate(neur)}
def neuronholdout(X):
    pred=np.full(len(R),np.nan)
    for f in range(5):
        te=np.array([k for k,r in enumerate(R) if nf[r['src']]==f and nf[r['tgt']]==f])
        tr=np.array([k for k,r in enumerate(R) if nf[r['src']]!=f and nf[r['tgt']]!=f])
        if len(te)<20 or len(set(y[tr]))<2: continue
        m=GradientBoostingClassifier(random_state=0).fit(X[tr],y[tr]); pred[te]=m.predict_proba(X[te])[:,1]
    mm=~np.isnan(pred); return roc_auc_score(y[mm],pred[mm]),mm.sum()
a_s=edgecv(Xs); a_se=edgecv(np.hstack([Xs,extra]))
ns,_=neuronholdout(Xs); nse,ncnt=neuronholdout(np.hstack([Xs,extra]))
print(f"  edge-CV:        structure {a_s:.3f} -> +class/LR {a_se:.3f}  (Δ={a_se-a_s:+.3f})")
print(f"  NEURON-HOLDOUT: structure {ns:.3f} -> +class/LR {nse:.3f}  (Δ={nse-ns:+.3f})  [n={ncnt}]")
print(f"  VERDICT: residual {'STRUCTURED (survives neuron-holdout)' if nse-ns>0.02 else 'NOISE-like (extra features add ~0 under neuron-holdout)'}")

print("\n=== Phase 6: PATH-ENTROPY CIRCULARITY (exclude adult bits from trajectory) ===")
def traj3(r): return (1 if int(r['L1n'])>0 else 0, int(r['L2']), int(r['L3']))  # NO adult bits
def traj5(r): return (1 if int(r['L1n'])>0 else 0, int(r['L2']), int(r['L3']), int(r['ad7']), int(r['ad8']))
core=[r for r in R if r['repC']==1]; halo=[r for r in R if r['halo']=='1']
def stats(rows,tf,nb):
    c=Counter(tf(r) for r in rows); n=sum(c.values()); ent=-sum((v/n)*math.log2(v/n) for v in c.values())
    per=np.mean([sum(tf(r))/nb for r in rows]); return ent,per
for lab,tf,nb in [("5-bit (incl adult = Round2, CIRCULAR)",traj5,5),("3-bit (L1-L3 only, non-circular)",traj3,3)]:
    ce,cp=stats(core,tf,nb); he,hp=stats(halo,tf,nb)
    print(f"  {lab:42s} core: ent={ce:.2f} persist={cp:.2f} | halo: ent={he:.2f} persist={hp:.2f} | sep {he-ce:+.2f}")
print("  -> if the 3-bit (developmental-only) separation shrinks vs 5-bit, the split was partly circular")

print(f"\n=== Contacting-pair residual vs floor CI ===")
print(f"  structural residual {resid:.3f} vs contacting floor 0.617 [0.591,0.640]  -> "
      f"{'ABOVE floor CI = gap real' if resid>0.640 else 'within/near floor CI'}")
