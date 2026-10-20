"""LOOSE END: is the directional pairwise NT pattern (Glu->ACh, ACh->GABA enriched) DEVELOPMENTAL?
Gate 6 showed developmental arrival (present in L1-L3 datasets 1-6) absorbed ~half the source-NT
main effect. Now test the PAIRWISE part specifically:
 A. src×tgt NT residual matrix under geometry vs geometry+dev-arrival baseline (does it shrink?)
 B. pairwise-over-additive AUC lift under geometry vs geometry+dev baseline
 C. mean developmental-arrival by src×tgt NT cell (are enriched cells developmentally EARLY?)
Directed chem reproducible (both adults) target.
"""
import openpyxl, csv, re, math, numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy import sparse
CONTACT="../data/celegans/data/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx"
EDGES="../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv"
T5="../data/tier5/connectome.npz"

wb=openpyxl.load_workbook(CONTACT,read_only=True,data_only=True); ws=wb['adult nerve ring neighbors']
data=list(ws.iter_rows(values_only=True)); neurons=[h for h in data[0][1:]]; idx={n:i for i,n in enumerate(neurons)}; M=len(neurons)
A=np.zeros((M,M))
for row in data[1:]:
    rn=row[0]
    if rn is None or rn not in idx: continue
    for j,val in enumerate(row[1:]):
        if val in (None,''): continue
        try: A[idx[rn],j]=float(val)
        except: pass
A=np.maximum(A,A.T); deg_area=A.sum(1); deg_num=(A>0).sum(1)
chem_ds=defaultdict(set)   # directed chem edge -> datasets present
with open(EDGES) as f:
    for r in csv.DictReader(f):
        ds=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
        if r['type']=='chemical': chem_ds[(r['pre'],r['post'])].add(ds)
t5=np.load(T5,allow_pickle=True); t5nt={str(n):str(v) for n,v in zip(t5["names"],t5["nt_primary"])}
def bclass(n):
    b=str(n)
    for s in("L","R"):
        if b.endswith(s): b=b[:-1]
    return b
def normnt(x):
    x=str(x)
    if x.startswith("ACh") or x.startswith("Acetyl"): return "ACh"
    if x.startswith("Glu"): return "Glu"
    for key,lab in [("GABA","GABA"),("Dopamine","DA"),("Serot","5HT"),("Octopamine","OA"),("Tyramine","TA")]:
        if key in x: return lab
    return "Unknown"
nt_of=np.array([normnt(t5nt.get(str(neurons[i]),t5nt.get(bclass(neurons[i]),"Unknown"))) for i in range(M)],dtype=object)

DI,DJ,dg,dy,dev=[],[],[],[],[]
for i in range(M):
    for j in range(M):
        if i==j or A[i,j]<=0: continue
        DI.append(i);DJ.append(j)
        vi,vj=A[i],A[j]; cos=float(vi@vj/(np.linalg.norm(vi)*np.linalg.norm(vj)+1e-9)); sh=int(((A[i]>0)&(A[j]>0)).sum())
        dg.append([math.log1p(A[i,j]),cos,sh,math.log1p(deg_area[i]),math.log1p(deg_area[j]),deg_num[i],deg_num[j]])
        s=chem_ds.get((str(neurons[i]),str(neurons[j])),set())
        dy.append(1 if (7 in s and 8 in s) else 0)
        dev.append(sum(1 for d in (1,2,3,4,5,6) if d in s))   # directed developmental arrival (L1-L3), 0..6
DI=np.array(DI);DJ=np.array(DJ); DG=np.array(dg); DG=(DG-DG.mean(0))/(DG.std(0)+1e-9); dy=np.array(dy); dev=np.array(dev,float); Nd=len(DI)
DEV=((dev-dev.mean())/(dev.std()+1e-9)).reshape(-1,1)
snt=nt_of[DI]; tnt=nt_of[DJ]
cv=StratifiedKFold(5,shuffle=True,random_state=0)
Gs=sparse.csr_matrix(DG); GD=sparse.hstack([Gs,sparse.csr_matrix(DEV)]).tocsr()
p_g =cross_val_predict(LogisticRegression(max_iter=3000),Gs,dy,cv=cv,method='predict_proba')[:,1]
p_gd=cross_val_predict(LogisticRegression(max_iter=3000),GD,dy,cv=cv,method='predict_proba')[:,1]
r_g=dy-p_g; r_gd=dy-p_gd
big=["ACh","Glu","GABA"]

print("="*88); print("A. src×tgt NT residual matrix (×100): geometry baseline  vs  geometry+dev baseline"); print("="*88)
for tag,r in [("geometry",r_g),("geometry+dev",r_gd)]:
    print(f"\n  [{tag}]  src\\tgt |"+"".join(f"{c:>11}" for c in big))
    for sc in big:
        line=f"          {sc:>7} |"
        for tc in big:
            m=(snt==sc)&(tnt==tc)
            line+=(f"{r[m].mean()*100:>7.2f}({m.sum():>3})" if m.sum()>=10 else f"{'--':>11}")
        print(line)

print("\n"+"="*88); print("B. pairwise-over-additive AUC lift: geometry vs geometry+dev baseline"); print("="*88)
ntc=sorted(set(nt_of)); ntx={c:k for k,c in enumerate(ntc)}
def src_oh():
    P=sparse.lil_matrix((Nd,len(ntc)));
    for k in range(Nd): P[k,ntx[nt_of[DI[k]]]]=1
    return P.tocsr()
def tgt_oh():
    P=sparse.lil_matrix((Nd,len(ntc)));
    for k in range(Nd): P[k,ntx[nt_of[DJ[k]]]]=1
    return P.tocsr()
def pair_oh():
    P=sparse.lil_matrix((Nd,len(ntc)**2));
    for k in range(Nd): P[k,ntx[nt_of[DI[k]]]*len(ntc)+ntx[nt_of[DJ[k]]]]=1
    return P.tocsr()
def aucX(X): return roc_auc_score(dy,cross_val_predict(LogisticRegression(max_iter=3000),X.tocsr(),dy,cv=cv,method='predict_proba')[:,1])
S,T,P=src_oh(),tgt_oh(),pair_oh()
for base_tag,B in [("geometry",Gs),("geometry+dev",GD)]:
    a_add=aucX(sparse.hstack([B,S,T])); a_pair=aucX(sparse.hstack([B,P]))
    print(f"  [{base_tag}] additive src+tgt NT AUC={a_add:.4f} | pairwise AUC={a_pair:.4f} | pairwise-over-additive = {a_pair-a_add:+.4f}")

print("\n"+"="*88); print("C. mean developmental-arrival (0-6, L1-L3) by src×tgt NT cell — are enriched cells EARLY?"); print("="*88)
print(f"  overall mean dev-arrival among directed contacting pairs = {dev.mean():.3f}; among reproducible = {dev[dy==1].mean():.3f}")
print(f"  src\\tgt |"+"".join(f"{c:>11}" for c in big))
for sc in big:
    line=f"  {sc:>7} |"
    for tc in big:
        m=(snt==sc)&(tnt==tc)
        line+=(f"{dev[m].mean():>7.2f}({m.sum():>3})" if m.sum()>=10 else f"{'--':>11}")
    print(line)
print("  (enriched cells Glu->ACh / ACh->GABA: if their dev-arrival >> overall, the pairwise pattern is developmental)")
