"""Steelman topology->geometry: rank/scale-free topology features so the Witvliet->Cook cross-map
gets its fairest shot. Also test whether a RICHER topology feature set (add katz/common-neighbor
spectra) beats geometry at predicting geometry's own residual. Capacity null included.
"""
import warnings,sys,json,csv,re; warnings.simplefilter("ignore")
import numpy as np, openpyxl, pandas as pd
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, rankdata
def P(*a): print(*a); sys.stdout.flush()
D="../data/celegans/data"
pj=json.load(open("../data/network_positions.json"))
pos={c["cell"]:np.array([c["x"],c["y"],c["z"]],float) for c in pj["cells"] if None not in (c.get("x"),c.get("y"),c.get("z"))}
wb0=openpyxl.load_workbook(f"{D}/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx",read_only=True,data_only=True)
data=list(wb0['adult nerve ring neighbors'].iter_rows(values_only=True)); hdr=[h for h in data[0][1:]]
neurons=[h for h in data[0][1:] if isinstance(h,str) and h in pos]; idx={n:i for i,n in enumerate(neurons)}; M=len(neurons)
A=np.zeros((M,M))
for row in data[1:]:
    if row[0] in idx:
        for j,v in enumerate(row[1:]):
            cn=hdr[j]
            if cn in idx and v not in (None,''):
                try: A[idx[row[0]],idx[cn]]=float(v)
                except: pass
A=np.maximum(A,A.T)
W7=np.zeros((M,M)); W8=np.zeros((M,M))
with open(f"{D}/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']!='chemical': continue
        m=re.search(r'witvliet_2020_(\d)',r['source_file'])
        if m and r['pre'] in idx and r['post'] in idx:
            if int(m.group(1))==7: W7[idx[r['pre']],idx[r['post']]]+=1
            elif int(m.group(1))==8: W8[idx[r['pre']],idx[r['post']]]+=1
Wrep=((W7>0)&(W8>0)).astype(float)
wc=openpyxl.load_workbook(f"{D}/connectome/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
cd=list(wc['hermaphrodite chemical'].iter_rows(values_only=True))
def isneu(x): return bool(isinstance(x,str) and str(x).strip() in idx)
hr=max(range(6),key=lambda r:sum(isneu(x) for x in cd[r])); colmap={c:cd[hr][c].strip() for c in range(len(cd[hr])) if isneu(cd[hr][c])}
rhc=max(range(6),key=lambda c:sum(isneu(cd[r][c]) for r in range(hr+1,len(cd)) if c<len(cd[r])))
C=np.zeros((M,M))
for r in range(hr+1,len(cd)):
    if rhc<len(cd[r]) and isneu(cd[r][rhc]):
        pi=idx[cd[r][rhc].strip()]
        for c,post in colmap.items():
            if c<len(cd[r]) and cd[r][c] not in (None,''):
                try:
                    if float(cd[r][c])>=1: C[pi,idx[post]]=1.0
                except: pass
OP=[(i,j) for i in range(M) for j in range(M) if i!=j and A[i,j]>0]
oi=np.array([p[0] for p in OP]); oj=np.array([p[1] for p in OP]); nO=len(OP)
logarea=np.log1p(A[oi,oj])
def rk(x): return rankdata(x)/len(x)              # scale-free rank transform
def feats_scalefree(G):
    kout=G.sum(1); kin=G.sum(0); Gu=((G+G.T)>0).astype(float); comm=Gu@Gu; ff=G@G
    # rank-normalize degree-like features per-dataset -> scale invariant
    return np.column_stack([rk(kout[oi]),rk(kin[oi]),rk(kout[oj]),rk(kin[oj]),
                            G[oj,oi],rk(comm[oi,oj]),rk(ff[oi,oj])])
Ft=feats_scalefree(Wrep); FtC=feats_scalefree(C)
# within-Witvliet neuron-disjoint T->G with scale-free feats
def ndcv(Feat,y,seed=0,k=5):
    rng=np.random.default_rng(seed); fold=rng.integers(0,k,M); pred=np.full(len(y),np.nan)
    for f in range(k):
        te_n=set(np.where(fold==f)[0])
        te=np.array([(oi[p] in te_n and oj[p] in te_n) for p in range(len(y))])
        tr=np.array([(oi[p] not in te_n and oj[p] not in te_n) for p in range(len(y))])
        if te.sum()==0 or tr.sum()==0: continue
        pred[te]=HGBR(max_iter=250,random_state=0).fit(Feat[tr],y[tr]).predict(Feat[te])
    return pred
p=ndcv(Ft,logarea); m=~np.isnan(p)
P(f"scale-free T->G (topo->area) within-Witvliet neuron-disjoint R2={r2_score(logarea[m],p[m]):.3f} spearman={spearmanr(logarea[m],p[m]).statistic:.3f}")
mdl=HGBR(max_iter=250,random_state=0).fit(Ft,logarea)
P(f"scale-free COOK cross-map (fit Witvliet-topo -> apply Cook-topo) R2={r2_score(logarea,mdl.predict(FtC)):.3f} spearman={spearmanr(logarea,mdl.predict(FtC)).statistic:.3f}")
# capacity null
Rf=np.random.default_rng(3).standard_normal((nO,7)); pr=ndcv(Rf,logarea); mr=~np.isnan(pr)
P(f"  random-7 capacity null R2={r2_score(logarea[mr],pr[mr]):.3f}")
