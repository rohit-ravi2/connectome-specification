import csv, math, numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
R=[r for r in csv.DictReader(open("data/processed/specification_accounting/master_edge_table.csv")) if r['contact']=='1']
def col(n,f=float): return np.array([f(r[n]) for r in R])
def h(p): return 0.0 if p<=0 or p>=1 else -p*math.log2(p)-(1-p)*math.log2(1-p)
def ll(y,p): p=np.clip(p,1e-6,1-1e-6); return -(y*np.log2(p)+(1-y)*np.log2(1-p)).mean()
Xstruct=np.column_stack([col('log_area'),col('shared_nb'),col('cos_sim'),col('cdeg_min'),col('cdeg_max'),
                         np.array([1.0 if r['src_nt']==r['tgt_nt'] else 0.0 for r in R])])
y8=col('ad8',int); yrep=col('reproducible',int); yhalo=col('halo',int); floor=0.617
cv=StratifiedKFold(5,shuffle=True,random_state=0)

print("="*70); print("PART 3 — RATE-DISTORTION: minimal model complexity vs distortion (predict ad8)")
print("="*70); print(f"  {'rate (leaves)':>14} {'distortion(bits)':>16} {'AUC':>6}   floor={floor}")
rows3=[]
for leaves in [2,4,8,16,32,64]:
    p=cross_val_predict(DecisionTreeClassifier(max_leaf_nodes=leaves,random_state=0),Xstruct,y8,cv=cv,method='predict_proba')[:,1]
    d=ll(y8,p); a=roc_auc_score(y8,p); rate=math.log2(leaves)  # ~bits to specify which leaf
    rows3.append((leaves,rate,d,a)); print(f"  {leaves:>14} {d:16.4f} {a:6.3f}   (rate~{rate:.0f} bits/decision)")
reach=[r for r in rows3 if r[2]<=floor+0.01]
print(f"  -> minimal model reaching noise floor: {reach[0][0] if reach else '>64'} leaves "
      f"(~{reach[0][1] if reach else '>6'} bits) at distortion {reach[0][2] if reach else 'n/a':.4f}" if reach else
      f"  -> even 64-leaf model (distortion {rows3[-1][2]:.4f}) does NOT reach floor {floor}")

print("\n"+"="*70); print("PART 4 — RESIDUAL: is the reproducible core predictable and the halo noise?")
print("="*70)
prep=cross_val_predict(GradientBoostingClassifier(random_state=0),Xstruct,yrep,cv=cv,method='predict_proba')[:,1]
# halo only among pairs present in >=1 adult (the variable set)
var_mask=(col('ad7',int)|col('ad8',int)).astype(bool)
Xv=Xstruct[var_mask]; yh=yhalo[var_mask]
ph=cross_val_predict(GradientBoostingClassifier(random_state=0),Xv,yh,cv=StratifiedKFold(5,shuffle=True,random_state=0),method='predict_proba')[:,1]
print(f"  predict REPRODUCIBLE (both adults) from structure: AUC={roc_auc_score(yrep,prep):.3f}  (signal)")
print(f"  predict HALO (which animal) from structure:        AUC={roc_auc_score(yh,ph):.3f}  (~0.5 = noise)")
# residual compressibility: does class add beyond structure for reproducible?
from scipy import sparse
cats=sorted(set([r['src_class'] for r in R]+[r['tgt_class'] for r in R])); cix={c:i for i,c in enumerate(cats)}; K=len(cats)
CP=sparse.lil_matrix((len(R),K*K))
for k,r in enumerate(R):
    a,b=cix[r['src_class']],cix[r['tgt_class']]; lo,hi=min(a,b),max(a,b); CP[k,lo*K+hi]=1
CP=CP.tocsr()
p_s=cross_val_predict(LogisticRegression(max_iter=2000),sparse.csr_matrix(Xstruct),yrep,cv=cv,method='predict_proba')[:,1]
p_sc=cross_val_predict(LogisticRegression(max_iter=2000),sparse.hstack([sparse.csr_matrix(Xstruct),CP]).tocsr(),yrep,cv=cv,method='predict_proba')[:,1]
print(f"  residual compressibility: structure AUC={roc_auc_score(yrep,p_s):.3f} -> +class-pair AUC={roc_auc_score(yrep,p_sc):.3f}  (Δ={roc_auc_score(yrep,p_sc)-roc_auc_score(yrep,p_s):+.3f})")

print("\n"+"="*70); print("PART 2 — PATH ENTROPY: developmental trajectory of strong-core vs weak-halo")
print("="*70)
def traj(r): return (1 if int(r['L1n'])>0 else 0, int(r['L2']), int(r['L3']), int(r['ad7']), int(r['ad8']))
from collections import Counter
def path_ent(rows):
    c=Counter(traj(r) for r in rows); n=sum(c.values())
    ent=-sum((v/n)*math.log2(v/n) for v in c.values())
    persist=np.mean([sum(traj(r))/5 for r in rows])
    return ent, persist, n
groups={'strong-core(repro)':[r for r in R if r['reproducible']=='1'],
        'weak-halo':[r for r in R if r['halo']=='1'],
        'contact-nonedge':[r for r in R if r['ad7']=='0' and r['ad8']=='0']}
print(f"  {'group':22s} {'path-entropy(bits)':>18} {'mean persistence':>16} {'n':>6}")
rows2=[]
for g,rr in groups.items():
    e,per,n=path_ent(rr); rows2.append((g,e,per,n)); print(f"  {g:22s} {e:18.3f} {per:16.3f} {n:6d}")
# save
with open("data/processed/specification_accounting/D_rate_distortion_curve.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["leaves","rate_bits","distortion_bits","auc"]); w.writerows(rows3)
with open("data/processed/specification_accounting/A_path_entropy.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["group","path_entropy_bits","mean_persistence","n"]); w.writerows(rows2)
print("\nartifacts: D_rate_distortion_curve.csv, A_path_entropy.csv")
