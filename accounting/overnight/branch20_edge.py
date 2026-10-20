"""BRANCH 20 — the EDGE side. Geometry codes WHICH pairs connect; tonight dissected the ARROW. Does identity add
anything to the undirected EDGE (which contacting pairs form a chemical synapse) beyond contact area? B13b hinted
synapses sit nearer-in-rank than random contacts. Fair test: universe = contacting pairs (nerve-ring), target =
chemical synapse present (undirected), baseline = contact area (+ degree), then add identity features (|rank-gap|,
identity cosine-similarity). Does identity improve edge prediction, degree-controlled? Expect (per prior rounds)
contact dominates; quantify any residual identity contribution to the edge.
"""
import warnings,sys,openpyxl; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
def P(*a): print(*a); sys.stdout.flush()
D="../data/celegans/data"
cg=pd.read_csv(f"{D}/expression/cengen/thresholded/021821_liberal_threshold1.csv")
classes=list(cg.columns[3:]); Xf=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values.T>0
br=Xf.mean(0); Xc=Xf[:,(br>0.05)&(br<0.95)].astype(float); cls_idx={c:k for k,c in enumerate(classes)}
def n2c(n):
    if not isinstance(n,str): return None
    if n in cls_idx: return n
    b=n
    for s in "LR":
        if b.endswith(s): b=b[:-1]; break
    if b in cls_idx: return b
    for s in "DV":
        if b.endswith(s) and b[:-1] in cls_idx: return b[:-1]
    return None
# contact area (adult nerve ring) at class level (max over member pairs)
wb0=openpyxl.load_workbook(f"{D}/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx",read_only=True,data_only=True)
ws=wb0['adult nerve ring neighbors']; dd=list(ws.iter_rows(values_only=True)); hdr=[h for h in dd[0][1:]]
ca=defaultdict(float)
for r in dd[1:]:
    rn=r[0]; ca_=n2c(str(rn)) if rn is not None else None
    if ca_ is None: continue
    for jj,v in enumerate(r[1:]):
        cn=hdr[jj]; cb=n2c(str(cn)) if cn else None
        if cb and cb!=ca_ and v not in (None,''):
            try: ca[(min(ca_,cb),max(ca_,cb))]=max(ca[(min(ca_,cb),max(ca_,cb))],float(v))
            except: pass
# Cook chemical (undirected presence) + learn rank
wb=openpyxl.load_workbook(f"{D}/connectome/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
A=list(wb['hermaphrodite chemical'].iter_rows(values_only=True))
def looks(n): return isinstance(n,str) and len(n)>=2 and n[0].isalpha()
hr=max(range(len(A)),key=lambda r:sum(looks(x) for x in A[r])); cols=[(str(x).strip() if x is not None else None) for x in A[hr]]
hc=max(range(min(6,len(A[0]))),key=lambda c:sum(looks(A[r][c]) for r in range(len(A))))
chemD=[]
for r in range(hr+1,len(A)):
    cpre=n2c(A[r][hc])
    if not cpre: continue
    for c in range(hc+1,len(A[r])):
        cpost=n2c(cols[c]); v=A[r][c]
        if cpost and cpost!=cpre and v not in (None,''):
            try:
                if float(v)>0: chemD.append((cpre,cpost))
            except: pass
chem=set(chemD); chem_u=set((min(a,b),max(a,b)) for a,b in chemD)
asym=[(a,b) for (a,b) in chem if (b,a) not in chem]
Fr=[];yr=[]
for a,b in asym:
    d=Xc[cls_idx[a]]-Xc[cls_idx[b]]; Fr+=[d,-d]; yr+=[1,0]
w=LogisticRegression(max_iter=1000,C=0.5).fit(np.array(Fr),np.array(yr)).coef_[0]
s={c:float(Xc[cls_idx[c]]@w) for c in classes if c in cls_idx}
# degree (chemical undirected)
deg=defaultdict(int)
for a,b in chem_u: deg[a]+=1; deg[b]+=1
# universe = contacting class-pairs, mapped
pairs=[(a,b) for (a,b) in ca if a in s and b in s and ca[(a,b)]>0]
y=np.array([1 if (a,b) in chem_u else 0 for a,b in pairs])
carea=np.array([np.log1p(ca[(a,b)]) for a,b in pairs])
rgap=np.array([abs(s[a]-s[b]) for a,b in pairs])
sim=np.array([float(Xc[cls_idx[a]]@Xc[cls_idx[b]]) for a,b in pairs])
degf=np.array([deg[a]+deg[b] for a,b in pairs])
P(f"BRANCH 20 — {len(pairs)} contacting class-pairs, {y.sum()} chemical synapses ({y.mean():.2f} base rate)")
kf=KFold(5,shuffle=True,random_state=0)
def auc(Fm): return roc_auc_score(y,cross_val_predict(HistGradientBoostingClassifier(max_iter=200,random_state=0),Fm,y,cv=kf,method='predict_proba')[:,1])
aC=auc(carea[:,None])
aCD=auc(np.c_[carea,degf])
aCDI=auc(np.c_[carea,degf,rgap,sim])
aCI=auc(np.c_[carea,rgap,sim])
P(f"\n  synapse-presence | contact-area only         AUC {aC:.3f}")
P(f"                    | contact + degree           AUC {aCD:.3f}")
P(f"                    | contact + identity(gap,sim) AUC {aCI:.3f}  (Δ vs contact {aCI-aC:+.3f})")
P(f"                    | contact + degree + identity AUC {aCDI:.3f} (Δ vs contact+degree {aCDI-aCD:+.3f})")
# permutation null: shuffle identity features across pairs, re-fit contact+degree+shuffled-identity
np.random.seed(0); nd=[]
for _ in range(20):
    pr=np.random.permutation(len(pairs))
    nd.append(auc(np.c_[carea,degf,rgap[pr],sim[pr]])-aCD)
z=((aCDI-aCD)-np.mean(nd))/(np.std(nd)+1e-9)
P(f"\n  identity ΔAUC over contact+degree = {aCDI-aCD:+.3f} vs shuffled-identity {np.mean(nd):+.3f}±{np.std(nd):.3f} -> z={z:+.1f}")
P(f"\n  VERDICT: {'identity adds REAL residual to the EDGE beyond contact+degree (z>2) — which contacting pairs synapse is partly identity-shaped (nearer-in-rank pairs favored), not purely geometric' if z>2 and aCDI-aCD>0.01 else 'the EDGE is ~purely geometric — identity adds little beyond contact+degree (Δ%.3f, z=%.1f); geometry decides WHO connects, identity only decides WHICH WAY (the arrow)'%(aCDI-aCD,z)}.")
