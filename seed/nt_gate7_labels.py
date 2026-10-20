"""GATE 7 — is source-NT UNIQUELY informative, or a proxy for source cell-type/modality?
Gate 3 showed the signal is a SOURCE effect. So compete source-NT against source functional
type (sensory / command-inter / motor / other, from nb28_neuron_profile). Directed chem target.
Key test: does source-NT add beyond source-modality (and vice versa)?
"""
import openpyxl, csv, re, math, numpy as np, pandas as pd
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy import sparse

CONTACT="../data/celegans/data/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx"
EDGES="../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv"
T5="../data/tier5/connectome.npz"
PROF="../data/celegans/New Notebooks/data_derived/nb28_neuron_profile.csv"
RNG=np.random.default_rng(0)
MOTOR={'VA','VB','VD','DA','DB','DD','AS','VC','RMD','RME','RIM','RMF','RMH','RIV','SMD','SMB','URA'}

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
chemdir=defaultdict(set)
with open(EDGES) as f:
    for r in csv.DictReader(f):
        ds=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
        if r['type']=='chemical': chemdir[(r['pre'],r['post'])].add(ds)

t5=np.load(T5,allow_pickle=True); t5nt={str(n):str(v) for n,v in zip(t5["names"],t5["nt_primary"])}
def bclass(n):
    b=str(n)
    for s in("L","R"):
        if b.endswith(s): b=b[:-1]
    return b
def normnt(x):
    x=str(x)
    for pre,lab in [("ACh","ACh"),("Acetyl","ACh"),("Glu","Glu")]:
        if x.startswith(pre): return lab
    for key,lab in [("GABA","GABA"),("Dopamine","DA"),("Serot","5HT"),("Octopamine","OA"),("Tyramine","TA")]:
        if key in x: return lab
    return "Unknown"
nt_of=np.array([normnt(t5nt.get(str(neurons[i]),t5nt.get(bclass(neurons[i]),"Unknown"))) for i in range(M)],dtype=object)

prof=pd.read_csv(PROF).set_index("neuron")
def modality(i):
    n=str(neurons[i])
    if n in prof.index:
        if int(prof.loc[n,"is_sensory"])==1: return "sensory"
        if int(prof.loc[n,"is_command"])==1: return "command"
    if bclass(n) in MOTOR or n in MOTOR: return "motor"
    return "inter"
mod_of=np.array([modality(i) for i in range(M)],dtype=object)
ntc=sorted(set(nt_of)); ntx={c:k for k,c in enumerate(ntc)}; Knt=len(ntc)
mdc=sorted(set(mod_of)); mdx={c:k for k,c in enumerate(mdc)}; Kmd=len(mdc)
print(f"NT classes {Knt}: {[(c,int((nt_of==c).sum())) for c in ntc]}")
print(f"modality {Kmd}: {[(c,int((mod_of==c).sum())) for c in mdc]}")

DI,DJ,dgeom,dy=[],[],[],[]
for i in range(M):
    for j in range(M):
        if i==j or A[i,j]<=0: continue
        DI.append(i);DJ.append(j)
        vi,vj=A[i],A[j]; cos=float(vi@vj/(np.linalg.norm(vi)*np.linalg.norm(vj)+1e-9)); sh=int(((A[i]>0)&(A[j]>0)).sum())
        dgeom.append([math.log1p(A[i,j]),cos,sh,math.log1p(deg_area[i]),math.log1p(deg_area[j]),deg_num[i],deg_num[j]])
        s=chemdir.get((str(neurons[i]),str(neurons[j])),set()); dy.append(1 if (7 in s and 8 in s) else 0)
DI=np.array(DI);DJ=np.array(DJ); DG=np.array(dgeom); DG=(DG-DG.mean(0))/(DG.std(0)+1e-9); dy=np.array(dy); Nd=len(DI)
DGs=sparse.csr_matrix(DG)
def oh(vals,K):
    P=sparse.lil_matrix((Nd,K))
    for k,v in enumerate(vals): P[k,v]=1
    return P.tocsr()
srcNT=oh([ntx[nt_of[i]] for i in DI],Knt); srcMOD=oh([mdx[mod_of[i]] for i in DI],Kmd)
cv=StratifiedKFold(5,shuffle=True,random_state=0)
def bits(yy,p): p=np.clip(p,1e-6,1-1e-6); return float(-(yy*np.log2(p)+(1-yy)*np.log2(1-p)).sum())
def A_(X): p=cross_val_predict(LogisticRegression(max_iter=3000,C=1.0),X.tocsr(),dy,cv=cv,method='predict_proba')[:,1]; return roc_auc_score(dy,p),bits(dy,p)
print(f"\ndirected reproducible chem {int(dy.sum())}/{Nd} ({dy.mean():.3f})")
base=A_(DGs)
tests=[("geometry",DGs),
       ("geometry + source-MODALITY",sparse.hstack([DGs,srcMOD])),
       ("geometry + source-NT",sparse.hstack([DGs,srcNT])),
       ("geometry + srcMOD + srcNT",sparse.hstack([DGs,srcMOD,srcNT]))]
res={}
for lab,X in tests:
    a,b=A_(X); res[lab]=a; print(f"  {lab:32s} AUC={a:.4f} bits={b:.0f}")
print("\n  KEY DECOMPOSITION:")
print(f"    source-MODALITY over geometry : {res['geometry + source-MODALITY']-res['geometry']:+.4f}")
print(f"    source-NT over geometry       : {res['geometry + source-NT']-res['geometry']:+.4f}")
print(f"    source-NT over geometry+MOD   : {res['geometry + srcMOD + srcNT']-res['geometry + source-MODALITY']:+.4f}  <-- is NT unique beyond modality?")
print(f"    source-MOD over geometry+NT   : {res['geometry + srcMOD + srcNT']-res['geometry + source-NT']:+.4f}")
