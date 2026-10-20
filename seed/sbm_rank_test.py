"""Decisive test: is the reproducible connectome a low-rank CLASS code or high-rank PAIRWISE?
Logistic latent-factor model: logit(i,j) = b + a_i + a_j + <U_i,U_j>, U in R^d.
CV over pairs, sweep rank d. Residual collapsing at small d => ~d-class address code.
Residual staying high => irreducibly pairwise (partner-specific)."""
import openpyxl, csv, re, math, numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

CONTACT="../data/celegans/data/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx"
EDGES="../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv"

wb=openpyxl.load_workbook(CONTACT,read_only=True,data_only=True); ws=wb['adult nerve ring neighbors']
data=list(ws.iter_rows(values_only=True)); neurons=[h for h in data[0][1:]]; idx={n:i for i,n in enumerate(neurons)}; M=len(neurons)
A=np.zeros((M,M))
for row in data[1:]:
    rn=row[0]
    if rn is None or rn not in idx: continue
    for j,val in enumerate(row[1:]):
        if val in (None,''): continue
        try: A[idx[rn],j]=float(val)
        except (TypeError,ValueError): pass
A=np.maximum(A,A.T)
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

I,J,y=[],[],[]
for i in range(M):
    for j in range(i+1,M):
        if A[i,j]<=0: continue
        I.append(i); J.append(j)
        y.append(1 if (neurons[i],neurons[j]) in repro or (neurons[j],neurons[i]) in repro else 0)
I=np.array(I); J=np.array(J); y=np.array(y,float); N=len(y)
sig=lambda z: 1/(1+np.exp(-z))

def fit_predict(tr, te, d, steps=3000, lr=0.05, lam=1e-3):
    Itr,Jtr,ytr=I[tr],J[tr],y[tr]
    b=math.log(ytr.mean()/(1-ytr.mean())); a=np.zeros(M); U=np.random.default_rng(0).normal(0,0.3,(M,d)) if d>0 else None
    ma=np.zeros(M); va=np.zeros(M); mb=vb=0.0
    if d>0: mU=np.zeros((M,d)); vU=np.zeros((M,d))
    for t in range(1,steps+1):
        logit=b+a[Itr]+a[Jtr]+((U[Itr]*U[Jtr]).sum(1) if d>0 else 0)
        g=sig(logit)-ytr
        gb=g.mean()
        ga=np.zeros(M); np.add.at(ga,Itr,g); np.add.at(ga,Jtr,g); ga=ga/len(g)+lam*a
        b-=lr*gb  # simple for b
        # Adam for a
        ma=0.9*ma+0.1*ga; va=0.999*va+0.001*ga*ga
        a-=lr*(ma/(1-0.9**t))/(np.sqrt(va/(1-0.999**t))+1e-8)
        if d>0:
            gU=np.zeros((M,d)); np.add.at(gU,Itr,g[:,None]*U[Jtr]); np.add.at(gU,Jtr,g[:,None]*U[Itr]); gU=gU/len(g)+lam*U
            mU=0.9*mU+0.1*gU; vU=0.999*vU+0.001*gU*gU
            U-=lr*(mU/(1-0.9**t))/(np.sqrt(vU/(1-0.999**t))+1e-8)
    Ite,Jte=I[te],J[te]
    logit=b+a[Ite]+a[Jte]+((U[Ite]*U[Jte]).sum(1) if d>0 else 0)
    return sig(logit)

pbar=y.mean(); base=N*(-(pbar*math.log2(pbar)+(1-pbar)*math.log2(1-pbar)))
print(f"N={N} contacting pairs, {int(y.sum())} reproducible ({100*pbar:.1f}%)")
print(f"  base H(connectome) = {base:.0f} bits ({base/M:.2f}/neuron)")
print(f"  [ref] geometry model residual = 2907 bits (16.0/neuron)\n")
print(f"  {'rank d':>7} | {'residual bits':>13} | {'bits/neuron':>11} | {'AUC':>5}")
cv=StratifiedKFold(5,shuffle=True,random_state=0)
for d in [0,2,4,8,16,32]:
    preds=np.zeros(N)
    for tr,te in cv.split(I,y): preds[te]=fit_predict(tr,te,d)
    p=np.clip(preds,1e-6,1-1e-6)
    resid=-(y*np.log2(p)+(1-y)*np.log2(1-p)).sum()
    print(f"  {d:>7} | {resid:>13.0f} | {resid/M:>11.2f} | {roc_auc_score(y,p):.3f}")
