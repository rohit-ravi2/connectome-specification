"""Did I under-explore the feature space? Rich LEAKAGE-SAFE features + nonlinear model + interactions
vs plain contact area, multiple targets, neuron-disjoint CV. If nothing beats area -> ceiling has teeth.
Features are ALL from the CONTACT graph + lineage + NT (never the synapse graph)."""
import openpyxl, csv, re, math, json, numpy as np
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
DATA="../data/celegans/data"
C=f"{DATA}/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx"
E=f"{DATA}/connectome/witvliet2020/witvliet_combined_edges.csv"
tree=json.load(open("../lineage/jtauber_lineage/cells.json"));dep={}
def w(n,d):
    dep[n["name"]]=d
    for c in n.get("children",[]): w(c,d+1)
w(tree,0)
wb=openpyxl.load_workbook(C,read_only=True,data_only=True);ws=wb['adult nerve ring neighbors']
rows=list(ws.iter_rows(values_only=True));alln=[h for h in rows[0][1:]];idx={n:i for i,n in enumerate(alln)};M=len(alln)
A=np.zeros((M,M))
for r in rows[1:]:
    rn=r[0]
    if rn in idx:
        for j,v in enumerate(r[1:]):
            if v not in (None,''):
                try:A[idx[rn],j]=float(v)
                except:pass
A=np.maximum(A,A.T);B=(A>0).astype(float)
deg_a=A.sum(1);deg_n=B.sum(1)
# higher-order contact topology: common neighbors, jaccard, triangle count via contact graph, 2-hop
CN=B@B  # common contact-neighbors count
tri=np.zeros(M)  # triangles each node participates in (contact graph)
for i in range(M): tri[i]=0.5*(B[i]@B@B[i])
kcore=B.sum(1)  # proxy (degree); fine
# NT + strong-core targets
import numpy as np
d=np.load("../data/tier5/connectome.npz",allow_pickle=True)
nm=[str(x) for x in d["names"]];ntp=[str(x) for x in d["nt_primary"]]
def ntn(x):
    for k,v in[("Acetyl","ACh"),("Glu","Glu"),("GABA","GABA"),("Dopamine","DA"),("Serotonin","5HT"),("Octopamine","OA"),("Tyramine","TA")]:
        if k in x:return v
    return "Unk"
NT={a:ntn(b) for a,b in zip(nm,ntp)}
pres=defaultdict(set)
for r in csv.DictReader(open(E)):
    ds=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
    if ds in(7,8) and r['type']!='electrical':
        k=(min(r['pre'],r['post']),max(r['pre'],r['post']));pres[k].add(ds)
# also max synapse for thresholds
syn=defaultdict(lambda:defaultdict(int))
for r in csv.DictReader(open(E)):
    ds=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
    if ds in(7,8) and r['type']!='electrical':
        k=(min(r['pre'],r['post']),max(r['pre'],r['post']));syn[k][ds]=max(syn[k][ds],int(float(r.get('synapses',1) or 1)))
pairs=[];AREA=[];RICH=[];NToh=[];meta=[]
for i in range(M):
    for j in range(i+1,M):
        if A[i,j]<=0:continue
        a,b=alln[i],alln[j]
        if a not in dep or b not in dep or a not in NT or b not in NT:continue
        vi,vj=A[i],A[j];cos=float(vi@vj/(np.linalg.norm(vi)*np.linalg.norm(vj)+1e-9))
        sh=int(((B[i]>0)&(B[j]>0)).sum());jac=sh/((deg_n[i]+deg_n[j]-sh)+1e-9)
        la=math.log1p(A[i,j])
        rich=[la,cos,sh,jac,math.log1p(min(deg_a[i],deg_a[j])),math.log1p(max(deg_a[i],deg_a[j])),
              min(deg_n[i],deg_n[j]),max(deg_n[i],deg_n[j]),math.log1p(tri[i]+1),math.log1p(tri[j]+1),
              CN[i,j],abs(dep[a]-dep[b]),min(dep[a],dep[b]),
              la*cos,la*jac,cos*jac]  # interactions
        AREA.append([la]);RICH.append(rich);pairs.append((a,b))
        NTk=tuple(sorted((NT[a],NT[b])));NToh.append(NTk)
        k=(min(a,b),max(a,b));meta.append((len(pres.get(k,())),max(syn[k].values()) if k in syn else 0))
AREA=np.array(AREA);RICH=np.array(RICH);meta=np.array(meta)
ntk=sorted(set(NToh));nti={x:i for i,x in enumerate(ntk)};NTM=np.zeros((len(pairs),len(ntk)))
for r,x in enumerate(NToh):NTM[r,nti[x]]=1
def ndauc(X,y,seed=0):
    neu=sorted(set(a for a,b in pairs)|set(b for a,b in pairs))
    rng=np.random.default_rng(seed);f=np.array_split(rng.permutation(neu),5);nf={n:i for i,ff in enumerate(f) for n in ff}
    pr=np.full(len(y),np.nan)
    for k in range(5):
        te=np.array([nf[a]==k and nf[b]==k for a,b in pairs]);tr=np.array([nf[a]!=k and nf[b]!=k for a,b in pairs])
        if te.sum()<10 or y[tr].sum()<5 or y[te].sum()<3 or len(set(y[te]))<2:continue
        c=make_pipeline(StandardScaler(with_mean=False),GradientBoostingClassifier(random_state=seed));c.fit(X[tr],y[tr]);pr[te]=c.predict_proba(X[te])[:,1]
    m=~np.isnan(pr);return roc_auc_score(y[m],pr[m])
def m5(X,y):return np.mean([ndauc(X,y,s) for s in range(5)])
print("neuron-disjoint AUC (GradientBoosting). Does RICH leakage-safe topology+interactions+NT beat plain contact AREA?")
print(f"{'target':22s} {'N':>6s} {'pos%':>5s} {'AREA':>7s} {'RICH':>7s} {'RICH+NT':>8s} {'Δ(rich-area)':>12s}")
for name,y in [("strongcore>=1",(meta[:,0]==2).astype(int)),
               ("strong>=2syn",((meta[:,0]==2)&(meta[:,1]>=2)).astype(int)),
               ("strong>=4syn",((meta[:,0]==2)&(meta[:,1]>=4)).astype(int)),
               ("weak-halo(1anim)",(meta[:,0]==1).astype(int))]:
    if y.sum()<15:continue
    a=m5(AREA,y);r=m5(RICH,y);rn=m5(np.hstack([RICH,NTM]),y)
    print(f"{name:22s} {len(y):>6d} {100*y.mean():>5.1f} {a:>7.3f} {r:>7.3f} {rn:>8.3f} {r-a:>+12.3f}")
