import openpyxl, csv, re, math, numpy as np
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
D="../data/celegans/data/connectome"; T5="../data/tier5/connectome.npz"
wb=openpyxl.load_workbook(f"{D}/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx",read_only=True,data_only=True)
data=list(wb['adult nerve ring neighbors'].iter_rows(values_only=True)); neurons=[h for h in data[0][1:] if isinstance(h,str)]; idx={n:i for i,n in enumerate(neurons)}; M=len(neurons)
A=np.zeros((M,M))
for row in data[1:]:
    if row[0] in idx:
        for j,v in enumerate(row[1:]):
            if v not in (None,''):
                try: A[idx[row[0]],j]=float(v)
                except: pass
A=np.maximum(A,A.T); rowsum=A.sum(1); cdeg=(A>0).sum(1)
# ranks/shares of each contact
def rank(i,j):  # fractional rank of A[i,j] among i's contacts (1=largest)
    cs=A[i][A[i]>0]; return float((cs<A[i,j]).mean())  # 0..1, higher=bigger relative
t5=np.load(T5,allow_pickle=True); t5nt={str(n):str(v) for n,v in zip(t5["names"],t5["nt_primary"])}
def bc(n):
    b=n
    for s in("L","R"):
        if b.endswith(s):b=b[:-1]
    return b
NT=[t5nt.get(neurons[i],t5nt.get(bc(neurons[i]),"Unknown")) for i in range(M)]
# chemical directed presence per animal
dchem=defaultdict(set); dev=defaultdict(set)
with open(f"{D}/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']!='chemical': continue
        ds=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1)); dchem[(r['pre'],r['post'])].add(ds)
        if ds<=6: dev[(r['pre'],r['post'])].add(ds)
# directed contacting pairs
rows=[]
for i in range(M):
    for j in range(M):
        if i==j or A[i,j]<=0: continue
        s=dchem.get((neurons[i],neurons[j]),set()); d=dev.get((neurons[i],neurons[j]),set())
        rows.append(dict(i=i,j=j,a7=int(7 in s),a8=int(8 in s),
            log_area=math.log1p(A[i,j]),shared=int(((A[i]>0)&(A[j]>0)).sum()),
            cos=float(A[i]@A[j]/(np.linalg.norm(A[i])*np.linalg.norm(A[j])+1e-9)),
            cdeg_s=int(cdeg[i]),cdeg_t=int(cdeg[j]),
            src_share=A[i,j]/(rowsum[i]+1e-9),tgt_share=A[i,j]/(rowsum[j]+1e-9),
            src_rank=rank(i,j),tgt_rank=rank(j,i),
            ntsame=1.0 if NT[i]==NT[j] else 0.0,
            devpre=int(len(d)),src=neurons[i],tgt=neurons[j]))
N=len(rows); y=np.array([r['a8'] for r in rows]); a7=np.array([r['a7'] for r in rows])
def h(p): return 0.0 if p<=0 or p>=1 else -p*math.log2(p)-(1-p)*math.log2(1-p)
floor=sum((a7==v).mean()*h(y[a7==v].mean()) for v in(0,1) if (a7==v).sum())
print(f"directed contacting chemical pairs N={N}  ad8 rate={y.mean():.3f}")
print(f"NOISE FLOOR H(ad8|ad7) directed chemical = {floor:.4f} bits/pair")
def X(keys): return np.column_stack([[r[k] for r in rows] for k in keys])
cv=StratifiedKFold(5,shuffle=True,random_state=0)
def resid(keys):
    p=cross_val_predict(GradientBoostingClassifier(random_state=0),X(keys),y,cv=cv,method='predict_proba')[:,1]
    pp=np.clip(p,1e-6,1-1e-6); return -(y*np.log2(pp)+(1-y)*np.log2(1-pp)).mean(), roc_auc_score(y,p)
print("\n=== ENTROPY LADDER (directed chemical contacting) ===")
ladder=[("degree",['cdeg_s','cdeg_t']),("+area",['log_area']),("+topology",['shared','cos']),
        ("+COMPETITION",['src_share','tgt_share','src_rank','tgt_rank']),("+dev_persist",['devpre']),("+NT",['ntsame'])]
cum=[]; print(f"  {'layer':14s} {'resid bits':>11} {'AUC':>6}  (floor {floor:.3f})")
for nm,fs in ladder:
    cum+=fs; b,au=resid(cum); print(f"  {nm:14s} {b:11.4f} {au:6.3f}  gap={b-floor:+.3f}")
# NEURON-HOLDOUT: does COMPETITION add beyond absolute geometry?
nf={n:i%5 for i,n in enumerate(sorted(set([r['src'] for r in rows]+[r['tgt'] for r in rows])))}
def nho(keys):
    XX=X(keys); pred=np.full(N,np.nan)
    for f in range(5):
        te=[k for k,r in enumerate(rows) if nf[r['src']]==f and nf[r['tgt']]==f]; tr=[k for k,r in enumerate(rows) if nf[r['src']]!=f and nf[r['tgt']]!=f]
        if len(te)<20: continue
        pred[np.array(te)]=GradientBoostingClassifier(random_state=0).fit(XX[tr],y[tr]).predict_proba(XX[te])[:,1]
    m=~np.isnan(pred); return roc_auc_score(y[m],pred[m])
base=['log_area','shared','cos','cdeg_s','cdeg_t','ntsame']
print("\n=== PHASE 4 VETO: does COMPETITION / TIMING compress residual under NEURON-HOLDOUT? ===")
print(f"  structure(abs geom+NT):          AUC={nho(base):.3f}")
print(f"  + contact COMPETITION:           AUC={nho(base+['src_share','tgt_share','src_rank','tgt_rank']):.3f}")
print(f"  + dev_persist (timing):          AUC={nho(base+['devpre']):.3f}")
print(f"  + competition + timing:          AUC={nho(base+['src_share','tgt_share','src_rank','tgt_rank','devpre']):.3f}")
import csv as _c
with open("data/processed/contacting_residual_final/master_directed_chemical_pairs.csv","w",newline="") as f:
    w=_c.DictWriter(f,list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("\nartifact: master_directed_chemical_pairs.csv")
