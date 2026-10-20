"""Part 1 (P1): maximum-entropy / predictive-entropy ladder. Predict a single adult's
edges (ad8) from cumulative constraint layers; residual bits vs animal-to-animal floor."""
import csv, math, numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
R=list(csv.DictReader(open("data/processed/specification_accounting/master_edge_table.csv")))
def col(name,rows,f=float): return np.array([f(r[name]) for r in rows])
def h(p): return 0.0 if p<=0 or p>=1 else -p*math.log2(p)-(1-p)*math.log2(1-p)
def ladder(rows,label):
    y=col('ad8',rows,int); N=len(y); base=N*h(y.mean())
    feats={'degree':['cdeg_min','cdeg_max'],'contact':['contact'],'+area':['log_area'],
           '+topology':['shared_nb','cos_sim'],'+NT_same':['__ntsame'],'+dev_persist':['L1n','L2','L3']}
    # precompute same-NT
    for r in rows: r['__ntsame']=1.0 if r['src_nt']==r['tgt_nt'] else 0.0
    cum=[]; out=[]; cv=StratifiedKFold(5,shuffle=True,random_state=0)
    prev=base
    # layer 0 = base rate
    out.append(('base_rate',base/N,0.0,0.5)); 
    for name,fs in feats.items():
        cum+=fs; X=np.column_stack([col(f,rows) for f in cum])
        p=cross_val_predict(GradientBoostingClassifier(random_state=0),X,y,cv=cv,method='predict_proba')[:,1]
        p=np.clip(p,1e-6,1-1e-6); bits=-(y*np.log2(p)+(1-y)*np.log2(1-p)).sum()
        auc=roc_auc_score(y,p) if len(set(y))>1 else .5
        out.append((name,bits/N,(prev-bits)/N,auc)); prev=bits
    print(f"\n=== {label} (N={N}, edge rate={y.mean():.3f}) ===")
    print(f"  {'layer':14s} {'resid bits/pair':>15} {'bits removed':>13} {'AUC':>6}")
    for nm,bp,rm,au in out: print(f"  {nm:14s} {bp:15.4f} {rm:13.4f} {au:6.3f}")
    return out,base/N,prev/N
# floor over each universe
def floor(rows):
    a7=col('ad7',rows,int); a8=col('ad8',rows,int); fl=0.0
    for v in(0,1):
        m=a7==v
        if m.sum(): fl+=m.mean()*h(a8[m].mean())
    return fl
allp=R; cont=[r for r in R if r['contact']=='1']
print("MORPHOSPACE FUNNEL — residual entropy per constraint layer, vs replicate noise floor")
oA,bA,fA=ladder(allp,"ALL pairs (full morphospace)")
print(f"  -> ALL-pairs noise floor H(ad8|ad7)={floor(allp):.4f} ; final constrained residual={fA:.4f}")
oC,bC,fC=ladder(cont,"CONTACTING pairs (where the entropy is)")
print(f"  -> CONTACTING noise floor H(ad8|ad7)={floor(cont):.4f} ; final constrained residual={fC:.4f}")
with open("data/processed/specification_accounting/P1_entropy_ladder.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["universe","layer","resid_bits_per_pair","bits_removed","auc"])
    for u,o in [("all",oA),("contact",oC)]:
        for nm,bp,rm,au in o: w.writerow([u,nm,round(bp,5),round(rm,5),round(au,4)])
    w.writerow(["all","NOISE_FLOOR",round(floor(allp),5),"",""]); w.writerow(["contact","NOISE_FLOOR",round(floor(cont),5),"",""])
print("\nartifact: P1_entropy_ladder.csv")
