"""HIGHER-ORDER SPATIAL GEOMETRY — does 3D configurational geometry (skeleton-derived, NON-graph) predict the
reproducible gap, and subsume the topological baseline? Non-circularity is automatic: spatial features are not
graph features, so if they predict connectivity it's spatial-config->wiring (a specifier), not A~A.
SPATIAL (non-graph): arbor length/branch-points/extent/reach, co-fasciculation length, contact-location-on-arbor,
approach geometry. TOPO (graph, possibly-circular): contact-degree, common-contact-neighbors(triangles).
Report ALONGSIDE contact-area (NOT net-of). Held-out: neuron-disjoint CV + Cook. Capacity-matched null.
"""
import warnings,sys,json,csv,re; warnings.simplefilter("ignore")
import numpy as np, openpyxl, pandas as pd
from collections import defaultdict
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
RNG=np.random.default_rng(0); D="../data/celegans/data"
SKD="../seed/data/external/mechanistic_frontier/witvliet2021_developmental/data/skeletons"
sk=json.load(open(f"{SKD}/Dataset7_skeletons.json"))
cg=pd.read_csv(D+"/expression/cengen/thresholded/021821_liberal_threshold1.csv"); classes=list(cg.columns[3:]); cls_idx={c:k for k,c in enumerate(classes)}
def n2c(n):
    if n in cls_idx: return cls_idx[n]
    b=n
    for s in "LR":
        if b.endswith(s): b=b[:-1]; break
    if b in cls_idx: return cls_idx[b]
    for s in "DV":
        if b.endswith(s) and b[:-1] in cls_idx: return cls_idx[b[:-1]]
    return cls_idx.get(b,None)
# contact
wb0=openpyxl.load_workbook(f"{D}/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx",read_only=True,data_only=True)
ws=wb0['adult nerve ring neighbors']; dd=list(ws.iter_rows(values_only=True)); hdr=[h for h in dd[0][1:]]
ca=defaultdict(float)
for r in dd[1:]:
    rn=r[0]
    if rn is None: continue
    for jj,v in enumerate(r[1:]):
        cn=hdr[jj]
        if cn and v not in (None,'') and rn!=cn:
            try: ca[(min(rn,cn),max(rn,cn))]=max(ca[(min(rn,cn),max(rn,cn))],float(v))
            except: pass
# neurons with skeleton + contact + class
ring=[n for n in hdr if n in sk and sk[n].get('coords') and n2c(n) is not None]
# ---- per-neuron SPATIAL features ----
def build(n):
    v=sk[n]; co=v['coords']; ids=list(co.keys()); pts=np.array([co[i] for i in ids],float)
    dtr=np.array([v['dist_to_root'].get(i,0) for i in ids],float)
    if len(pts)>90:
        s=RNG.choice(len(pts),90,replace=False); pts=pts[s]; dtr=dtr[s]
    # branch points from arbor tree (parent dict): node with >=2 children
    ch=defaultdict(int)
    for node,par in v['arbor'].items(): ch[par]+=1
    nbr=sum(1 for k2,c in ch.items() if c>=2)
    return pts,(dtr/(dtr.max()+1e-9)),float(v['length']),nbr,pts.std(0).mean(),float(np.max(dtr)),len(v.get('ends',[]))
PT={};DTR={};ALEN={};NBR={};EXT={};MDTR={};NEND={}
for n in ring:
    PT[n],DTR[n],ALEN[n],NBR[n],EXT[n],MDTR[n],NEND[n]=build(n)
trees={n:cKDTree(PT[n]) for n in ring}
M=len(ring); idx={n:i for i,n in enumerate(ring)}
pairs=[(ring[i],ring[j]) for i in range(M) for j in range(i+1,M)]; Np=len(pairs)
THR=800.0  # nm, co-fasciculation proximity
def pairfeat(a,b):
    d,_=trees[b].query(PT[a]); close=d<THR
    cof=close.sum(); mind=d.min(); meanc=d[close].mean() if close.any() else d.min()
    loc_a=DTR[a][close].mean() if close.any() else 0.5
    d2,_=trees[a].query(PT[b]); close2=d2<THR; loc_b=DTR[b][close2].mean() if close2.any() else 0.5
    spread=PT[a][close].std(0).mean() if close.sum()>1 else 0.0
    return [cof,mind,meanc,loc_a,loc_b,spread,np.log1p(ALEN[a]),np.log1p(ALEN[b]),NBR[a],NBR[b],EXT[a],EXT[b],MDTR[a],MDTR[b]]
P(f"neurons w/ skeleton+contact+class {M} | pairs {Np} | building spatial features...")
SPAT=np.array([pairfeat(a,b) for a,b in pairs])
area=np.array([ca.get((min(a,b),max(a,b)),0.0) for a,b in pairs]); CA=np.column_stack([np.log1p(area)])
# ---- TOPO (graph) features ----
Cadj=np.zeros((M,M))
for k,(a,b) in enumerate(pairs):
    if area[k]>0: Cadj[idx[a],idx[b]]=1; Cadj[idx[b],idx[a]]=1
cdeg=Cadj.sum(1); comm=Cadj@Cadj
TOPO=np.column_stack([cdeg[[idx[a] for a,b in pairs]],cdeg[[idx[b] for a,b in pairs]],np.array([comm[idx[a],idx[b]] for a,b in pairs])])
# ---- targets ----
syn=defaultdict(int)
with open(f"{D}/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']!='chemical': continue
        m=re.search(r'witvliet_2020_(\d)',r['source_file'])
        if m and int(m.group(1)) in (7,8): syn[(min(r['pre'],r['post']),max(r['pre'],r['post']))]+=1
yW=np.array([1 if syn.get((min(a,b),max(a,b)),0)>=2 else 0 for a,b in pairs])
wb=openpyxl.load_workbook(f"{D}/connectome/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
A=list(wb['hermaphrodite chemical'].iter_rows(values_only=True)); nset=set(ring)
def nm(seq): return sum(1 for x in seq if isinstance(x,str) and x.strip() in nset)
hrow=int(np.argmax([nm(r) for r in A])); cols=[(str(x).strip() if x is not None else None) for x in A[hrow]]
hcol=int(np.argmax([nm([A[r][c] for r in range(len(A))]) for c in range(min(6,len(A[0])))]))
GC=np.zeros((M,M))
for r in range(hrow+1,len(A)):
    rn=A[r][hcol]; rn=rn.strip() if isinstance(rn,str) else None
    if rn not in idx: continue
    for c in range(hcol+1,len(A[r])):
        cn=cols[c]; v=A[r][c]
        if cn in idx and v not in (None,''):
            try:
                if float(v)>0: GC[idx[rn],idx[cn]]=1
            except: pass
yC=np.array([1 if (GC[idx[a],idx[b]] or GC[idx[b],idx[a]]) else 0 for a,b in pairs])
elig=area>0
# ---- neuron-disjoint CV ----
fold=np.array([RNG.integers(5) for _ in ring])
pf=np.array([max(fold[idx[a]],fold[idx[b]]) if fold[idx[a]]==fold[idx[b]] else -1 for a,b in pairs])
def cvauc(F,y):
    pred=np.full(Np,np.nan)
    for f in range(5):
        te=(pf==f)&elig; tr=(pf!=f)&(pf>=0)&elig
        if te.sum()<10 or tr.sum()<50: continue
        clf=HistGradientBoostingClassifier(max_iter=200,random_state=0).fit(F[tr],yW[tr])
        pred[te]=clf.predict_proba(F[te])[:,1]
    ok=~np.isnan(pred)
    return roc_auc_score(y[ok],pred[ok]), pred, ok
P("\n"+"="*82); P("DECISIVE TEST — spatial-config vs topology predicting the gap (neuron-disjoint CV)"); P("="*82)
sets={'contact-area (bulk)':CA,'CA + SPATIAL(non-graph)':np.hstack([CA,SPAT]),
      'CA + TOPO(graph)':np.hstack([CA,TOPO]),'CA + SPATIAL + TOPO':np.hstack([CA,SPAT,TOPO])}
res={}
for nmk,F in sets.items():
    aW,pred,ok=cvauc(F,yW); aC=roc_auc_score(yC[ok],pred[ok]); res[nmk]=aW
    P(f"  {nmk:26s} AUC(Witvliet held-out)={aW:.3f}  AUC(Cook out-of-sample)={aC:.3f}")
# capacity null: CA + random features matched to SPATIAL dim
rn_=[]
for s in range(3):
    Rf=np.random.default_rng(s).standard_normal((Np,SPAT.shape[1]))
    rn_.append(cvauc(np.hstack([CA,Rf]),yW)[0])
P(f"  CA + RANDOM (capacity null, matched dim)  AUC={np.mean(rn_):.3f}±{np.std(rn_):.3f}")
P("\n"+"="*82)
spatial_adds=res['CA + SPATIAL(non-graph)']-res['contact-area (bulk)']
topo_beyond_spatial=res['CA + SPATIAL + TOPO']-res['CA + SPATIAL(non-graph)']
spatial_vs_capacity=res['CA + SPATIAL(non-graph)']-np.mean(rn_)
P(f"  SPATIAL adds over contact-area: {spatial_adds:+.3f} | vs capacity-null: {spatial_vs_capacity:+.3f}")
P(f"  TOPO adds BEYOND spatial:       {topo_beyond_spatial:+.3f}  (small => spatial SUBSUMES topology)")
if spatial_vs_capacity>0.02 and topo_beyond_spatial<0.015:
    P("  => CAPSTONE 1: higher-order SPATIAL geometry predicts the gap & SUBSUMES topology. The topological 0.75")
    P("     was underfit spatial configuration all along — NON-CIRCULAR, characterized. Connectome = two-scale")
    P("     spatial specification (contact-area bulk + higher-order config residual). Mystery dissolved into structure.")
elif spatial_vs_capacity>0.02 and topo_beyond_spatial>=0.015:
    P("  => CAPSTONE 2 (bounded): spatial explains PART; topology retains independent signal -> split is real-spatial")
    P("     + still-open circular-topological piece needing its own non-circularity test.")
else:
    P("  => CAPSTONE 3 (deflationary): spatial config does NOT predict the gap beyond capacity; the topological")
    P("     signal is graph-predicting-graph (circular). No specification past contact-area. Door shuts on evidence.")
P("="*82)
