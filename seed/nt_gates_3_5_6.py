"""GATE 3 (directed) + GATE 5 (threshold) + GATE 6 (topology/dev baseline) for the NT signal.
- G6: is NT still additive after the STRONGEST non-molecular baseline (geometry + developmental
  arrival from L1-L3 datasets + contact topology)? The main kill-shot.
- G3: directed chem target -> is the signal truly source×target PAIRWISE, or just source-NT main effect?
- G5: does the NT lift hold across strong-core thresholds (>=1/>=2/>=4 synapses)?
NT labels normalized (merge duplicate ACh labels).
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
RNG=np.random.default_rng(0)

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

# per-dataset presence (undirected & directed chem), synapse counts adults
undir_ds=defaultdict(set); chemdir_ds=defaultdict(set); syn_ad=defaultdict(lambda: defaultdict(int))
with open(EDGES) as f:
    for r in csv.DictReader(f):
        ds=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
        pre,post,typ=r['pre'],r['post'],r['type']
        undir_ds[(min(pre,post),max(pre,post))].add(ds)
        if typ=='chemical':
            chemdir_ds[(pre,post)].add(ds)
            if ds in(7,8): syn_ad[(min(pre,post),max(pre,post))][ds]+=int(r['synapses'])

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
    if "GABA" in x: return "GABA"
    if "Dopamine" in x: return "DA"
    if "Serot" in x: return "5HT"
    if "Octopamine" in x: return "OA"
    if "Tyramine" in x: return "TA"
    return "Unknown"
nt_of=np.array([normnt(t5nt.get(str(neurons[i]),t5nt.get(bclass(neurons[i]),"Unknown"))) for i in range(M)],dtype=object)
cats=sorted(set(nt_of)); cix={c:k for k,c in enumerate(cats)}; K=len(cats)
print(f"normalized NT classes {K}: {[(c,int((nt_of==c).sum())) for c in cats]}")

def bits(yy,p): p=np.clip(p,1e-6,1-1e-6); return float(-(yy*np.log2(p)+(1-yy)*np.log2(1-p)).sum())
cv=StratifiedKFold(5,shuffle=True,random_state=0)
def oof(X,y): return cross_val_predict(LogisticRegression(max_iter=3000,C=1.0),X,y,cv=cv,method='predict_proba')[:,1]

# ---- build undirected candidate table (geometry + dev-arrival) ----
UI,UJ,geom,dev=[],[],[],[]
syn7,syn8=[],[]
for i in range(M):
    for j in range(i+1,M):
        if A[i,j]<=0: continue
        UI.append(i);UJ.append(j)
        vi,vj=A[i],A[j]; cos=float(vi@vj/(np.linalg.norm(vi)*np.linalg.norm(vj)+1e-9)); sh=int(((A[i]>0)&(A[j]>0)).sum())
        geom.append([math.log1p(A[i,j]),cos,sh,math.log1p(min(deg_area[i],deg_area[j])),
                     math.log1p(max(deg_area[i],deg_area[j])),min(deg_num[i],deg_num[j]),max(deg_num[i],deg_num[j])])
        pr=(min(str(neurons[i]),str(neurons[j])),max(str(neurons[i]),str(neurons[j])))
        dss=undir_ds.get(pr,set()); dev.append(sum(1 for d in (1,2,3,4,5,6) if d in dss))  # L1-L3 arrival count 0..6
        syn7.append(syn_ad.get(pr,{}).get(7,0)); syn8.append(syn_ad.get(pr,{}).get(8,0))
UI=np.array(UI);UJ=np.array(UJ); G=np.array(geom); G=(G-G.mean(0))/(G.std(0)+1e-9)
dev=np.array(dev,float); DEV=((dev-dev.mean())/(dev.std()+1e-9)).reshape(-1,1)
syn7=np.array(syn7);syn8=np.array(syn8); N=len(UI)
def ntpair_und(labels):
    P=sparse.lil_matrix((N,K*K))
    for k in range(N):
        a,b=cix[labels[UI[k]]],cix[labels[UJ[k]]]; lo,hi=min(a,b),max(a,b); P[k,lo*K+hi]=1
    return P.tocsr()
Gs=sparse.csr_matrix(G); GD=sparse.hstack([Gs,sparse.csr_matrix(DEV)]).tocsr()

print("\n"+"="*88); print("GATE 6 — does NT survive the strongest non-molecular baseline (geometry + dev-arrival)?"); print("="*88)
y=((syn7>=1)&(syn8>=1)).astype(int)
for lab,X in [("geometry",Gs),("geometry+dev-arrival",GD)]:
    p=oof(X,y); print(f"  {lab:24s} AUC={roc_auc_score(y,p):.4f} bits={bits(y,p):.0f}")
pgd=oof(GD,y); pgdn=oof(sparse.hstack([GD,ntpair_und(nt_of)]).tocsr(),y)
perm=np.array([roc_auc_score(y,oof(sparse.hstack([GD,ntpair_und(nt_of[RNG.permutation(M)])]).tocsr(),y))-roc_auc_score(y,pgd) for _ in range(15)])
dA=roc_auc_score(y,pgdn)-roc_auc_score(y,pgd)
print(f"  geometry+dev + NT-pair   AUC={roc_auc_score(y,pgdn):.4f} bits={bits(y,pgdn):.0f}")
print(f"  ΔAUC(NT | geom+dev) = {dA:+.4f}  Δbits={bits(y,pgd)-bits(y,pgdn):+.0f}  perm {perm.mean():+.4f}±{perm.std():.4f} z={(dA-perm.mean())/(perm.std()+1e-9):+.1f}")

print("\n"+"="*88); print("GATE 5 — NT lift across strong-core thresholds (geometry baseline)"); print("="*88)
for thr in [1,2,4]:
    yt=((syn7>=thr)&(syn8>=thr)).astype(int)
    if yt.sum()<40: print(f"  thr>={thr}: too few"); continue
    pg=oof(Gs,yt); pgn=oof(sparse.hstack([Gs,ntpair_und(nt_of)]).tocsr(),yt)
    pm=np.array([roc_auc_score(yt,oof(sparse.hstack([Gs,ntpair_und(nt_of[RNG.permutation(M)])]).tocsr(),yt))-roc_auc_score(yt,pg) for _ in range(8)])
    dd=roc_auc_score(yt,pgn)-roc_auc_score(yt,pg)
    print(f"  thr>={thr} (pos {int(yt.sum())}): geomAUC {roc_auc_score(yt,pg):.4f} -> +NT {roc_auc_score(yt,pgn):.4f}  ΔAUC {dd:+.4f}  perm z={(dd-pm.mean())/(pm.std()+1e-9):+.1f}")

print("\n"+"="*88); print("GATE 3 — DIRECTED chem: source-NT vs target-NT vs source×target PAIRWISE"); print("="*88)
# directed chemical reproducible target on directed contacting pairs
DI,DJ,dgeom,dy=[],[],[],[]
for i in range(M):
    for j in range(M):
        if i==j or A[i,j]<=0: continue
        DI.append(i);DJ.append(j)
        vi,vj=A[i],A[j]; cos=float(vi@vj/(np.linalg.norm(vi)*np.linalg.norm(vj)+1e-9)); sh=int(((A[i]>0)&(A[j]>0)).sum())
        dgeom.append([math.log1p(A[i,j]),cos,sh,math.log1p(deg_area[i]),math.log1p(deg_area[j]),deg_num[i],deg_num[j]])
        s=chemdir_ds.get((str(neurons[i]),str(neurons[j])),set()); dy.append(1 if (7 in s and 8 in s) else 0)
DI=np.array(DI);DJ=np.array(DJ); DG=np.array(dgeom); DG=(DG-DG.mean(0))/(DG.std(0)+1e-9); dy=np.array(dy); Nd=len(DI)
DGs=sparse.csr_matrix(DG)
def onehot(vals,ncat):
    P=sparse.lil_matrix((Nd,ncat));
    for k,v in enumerate(vals): P[k,v]=1
    return P.tocsr()
srcNT=onehot([cix[nt_of[i]] for i in DI],K); tgtNT=onehot([cix[nt_of[j]] for j in DJ],K)
pairNT=sparse.lil_matrix((Nd,K*K))
for k in range(Nd): pairNT[k,cix[nt_of[DI[k]]]*K+cix[nt_of[DJ[k]]]]=1
pairNT=pairNT.tocsr()
print(f"  directed contacting pairs {Nd} | directed reproducible chem {int(dy.sum())} ({dy.mean():.3f})")
for lab,X in [("geometry",DGs),("geometry+srcNT",sparse.hstack([DGs,srcNT])),
              ("geometry+tgtNT",sparse.hstack([DGs,tgtNT])),
              ("geometry+src+tgtNT (additive)",sparse.hstack([DGs,srcNT,tgtNT])),
              ("geometry+src×tgt NT (pairwise)",sparse.hstack([DGs,pairNT]))]:
    p=oof(X.tocsr(),dy); print(f"  {lab:32s} AUC={roc_auc_score(dy,p):.4f} bits={bits(dy,p):.0f}")
print("  (pairwise > additive => genuinely source×target; else mostly main-effect)")
