"""Phase 7 — does LINEAGE add to reproducible-synapse prediction BEYOND the connectome-seed
baseline (geometry + developmental-arrival + source-NT)? Decisive branch test.
Directed reproducible chem synapse (both adults) on directed contacting pairs among lineage-
covered neurons. Incremental baseline stack + lineage block + lineage-permutation null.
"""
import warnings, re; warnings.simplefilter("ignore")
import numpy as np, pandas as pd, openpyxl
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from scipy import sparse
RNG=np.random.default_rng(0); D="../data/celegans/data"

lf=pd.read_csv(D+"/processed/developmental_rules/lineage_features.csv")
paths={r.neuron:r.lineage_path.split("/") for r in lf.itertuples()}
founder={r.neuron:r.founder for r in lf.itertuples()}
bt=pd.read_csv(D+"/processed/developmental_rules/birthtime_earlylate.csv")
born={r.neuron:(1 if r.born=="early" else 0) for r in bt.itertuples()}
t5=np.load("../data/tier5/connectome.npz",allow_pickle=True)
t5nt={str(n):str(v) for n,v in zip(t5["names"],t5["nt_primary"])}
def bclass(n):
    for s in "LRDV":
        if n.endswith(s): return n[:-1]
    return n
def nnt(n):
    x=str(t5nt.get(n,t5nt.get(bclass(n),"Unknown")))
    if x.startswith("ACh") or x.startswith("Acetyl"): return "ACh"
    if x.startswith("Glu"): return "Glu"
    for k,l in [("GABA","GABA"),("Dopamine","DA"),("Serot","5HT"),("Octopamine","OA"),("Tyramine","TA")]:
        if k in x: return l
    return "Unknown"
def td(a,b):
    pa,pb=paths[a],paths[b]; k=0
    while k<len(pa) and k<len(pb) and pa[k]==pb[k]: k+=1
    return (len(pa)-k)+(len(pb)-k)

# contact
wb=openpyxl.load_workbook(D+"/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx",read_only=True,data_only=True)
ws=wb['adult nerve ring neighbors']; data=list(ws.iter_rows(values_only=True)); hdr=[h for h in data[0][1:]]
cov=lambda n: n in paths and n in born
A=defaultdict(float)
for row in data[1:]:
    rn=row[0]
    if not cov(rn): continue
    for j,val in enumerate(row[1:]):
        cn=hdr[j]
        if cov(cn) and val not in (None,'') and rn!=cn:
            try: A[(rn,cn)]=float(val)
            except: pass
Aud=defaultdict(float)
for (a,b),v in A.items(): Aud[(min(a,b),max(a,b))]=max(Aud[(min(a,b),max(a,b))],v)
cdeg=defaultdict(int); ring=sorted(set([a for a,b in A]+[b for a,b in A]))
for (a,b),v in Aud.items():
    if v>0: cdeg[a]+=1; cdeg[b]+=1
# neighborhood vectors for cos/shared
neigh={n:set() for n in ring}
for (a,b),v in Aud.items():
    if v>0: neigh[a].add(b); neigh[b].add(a)

# reproducible directed chem + dev-arrival
pres=defaultdict(set); chemdir=defaultdict(set)
for r in pd.read_csv(D+"/connectome/witvliet2020/witvliet_combined_edges.csv").itertuples():
    ds=int(re.search(r'witvliet_2020_(\d)',r.source_file).group(1))
    if not(cov(r.pre) and cov(r.post)): continue
    pres[(min(r.pre,r.post),max(r.pre,r.post))].add(ds)
    if r.type=="chemical": chemdir[(r.pre,r.post)].add(ds)

# directed contacting candidate pairs
I,J,geom,dev,y,td_,bp,sf,snt=[],[],[],[],[],[],[],[],[]
for (a,b),v in Aud.items():
    if v<=0: continue
    for (s,t) in [(a,b),(b,a)]:
        area=Aud[(min(s,t),max(s,t))]
        ni,nj=neigh[s],neigh[t]; cos=len(ni&nj)/((len(ni)*len(nj))**0.5+1e-9); sh=len(ni&nj)
        geom.append([np.log1p(area),cos,sh,np.log1p(cdeg[s]),np.log1p(cdeg[t])])
        dev.append(sum(1 for d in (1,2,3,4,5,6) if d in pres.get((min(s,t),max(s,t)),set())))
        sc=chemdir.get((s,t),set()); y.append(1 if (7 in sc and 8 in sc) else 0)
        td_.append(td(s,t)); bp.append([born[s],born[t],1 if born[s]==born[t]==1 else 0])
        sf.append(1.0 if founder[s]==founder[t] else 0.0); snt.append(nnt(s)); I.append(s); J.append(t)
y=np.array(y); G=np.array(geom); dev=np.array(dev,float).reshape(-1,1); td_=np.array(td_,float).reshape(-1,1)
bp=np.array(bp,float); sf=np.array(sf).reshape(-1,1)
ntc=sorted(set(snt)); ntx={c:k for k,c in enumerate(ntc)}
SNT=np.zeros((len(y),len(ntc)));
for k,s in enumerate(snt): SNT[k,ntx[s]]=1
N=len(y); print(f"directed contacting pairs (lineage-covered) {N} | reproducible chem {int(y.sum())} ({y.mean():.3f})")
skf=StratifiedKFold(5,shuffle=True,random_state=0)
def auc(X): return roc_auc_score(y,cross_val_predict(LogisticRegression(max_iter=2500,class_weight="balanced"),X,y,cv=skf,method="predict_proba")[:,1])
def bits(X):
    p=np.clip(cross_val_predict(LogisticRegression(max_iter=2500,class_weight="balanced"),X,y,cv=skf,method="predict_proba")[:,1],1e-6,1-1e-6)
    return float(-(y*np.log2(p)+(1-y)*np.log2(1-p)).sum())

print("\n"+"="*84); print("PHASE 7 — reproducible-synapse baseline stack + LINEAGE"); print("="*84)
Bgeo=G; Bgd=np.column_stack([G,dev]); Bgdn=np.column_stack([G,dev,SNT])
LIN=np.column_stack([td_,bp,sf])
for lab,X in [("geometry",Bgeo),("geometry+dev-arrival",Bgd),("geometry+dev+srcNT (connectome-seed baseline)",Bgdn),
              ("  +LINEAGE (td,born,founder)",np.column_stack([Bgdn,LIN]))]:
    print(f"  {lab:46s} AUC={auc(X):.4f} bits={bits(X):.0f}")
base_auc=auc(Bgdn); full_auc=auc(np.column_stack([Bgdn,LIN])); dA=full_auc-base_auc
# lineage permutation null (shuffle lineage+born labels across neurons)
alln=sorted(set(I)|set(J))
perm=[]
for _ in range(15):
    pmap={n:alln[k] for n,k in zip(alln,RNG.permutation(len(alln)))}
    tdp=np.array([td(pmap[I[k]],pmap[J[k]]) for k in range(N)],float).reshape(-1,1)
    bpp=np.array([[born[pmap[I[k]]],born[pmap[J[k]]],1 if born[pmap[I[k]]]==born[pmap[J[k]]]==1 else 0] for k in range(N)],float)
    sfp=np.array([1.0 if founder[pmap[I[k]]]==founder[pmap[J[k]]] else 0.0 for k in range(N)]).reshape(-1,1)
    perm.append(auc(np.column_stack([Bgdn,tdp,bpp,sfp]))-base_auc)
perm=np.array(perm)
print(f"\n  ΔAUC(lineage | geometry+dev+NT) = {dA:+.4f}")
print(f"  lineage-permutation null ΔAUC = {perm.mean():+.4f} ± {perm.std():.4f}  z={(dA-perm.mean())/(perm.std()+1e-9):+.1f}")
print("\n"+"="*84)
alive = dA>0.01 and (dA-perm.mean())/(perm.std()+1e-9)>3
print("PHASE 7 VERDICT:", "LINEAGE ADDS beyond geometry+dev+NT -> developmental rule beyond contact"
      if alive else "LINEAGE ADDS ~NOTHING beyond geometry+dev+NT -> no developmental wiring grammar beyond contact+timing")
print("="*84)
