"""BRANCH 13b — is the rank-layering BEYOND geometry? B13's null rewired among all neurons, so the layering could
be entirely the contact constraint (edges must be between touching neurons; touching neurons are near in rank).
Clean test: null = random ordered CONTACTING pairs (same count). If real edges have smaller rank-gap than random
CONTACTING pairs, the identity-rank adds locality beyond geometry (the arrow is laid out within the contact map);
if real ~ random-contacting, the layering IS the geometry.
"""
import warnings,sys,openpyxl; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
def P(*a): print(*a); sys.stdout.flush()
RNG=np.random.default_rng(1); D="../data/celegans/data"
cg=pd.read_csv(f"{D}/expression/cengen/thresholded/021821_liberal_threshold1.csv")
classes=list(cg.columns[3:]); Xf=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values.T>0
br=Xf.mean(0); Xc=Xf[:,(br>0.05)&(br<0.95)].astype(float); cls_idx={c:k for k,c in enumerate(classes)}
def n2c(n):
    if not isinstance(n,str): return None
    if n in cls_idx: return n
    b=n
    for su in "LR":
        if b.endswith(su): b=b[:-1]; break
    if b in cls_idx: return b
    for su in "DV":
        if b.endswith(su) and b[:-1] in cls_idx: return b[:-1]
    return None
wb=openpyxl.load_workbook(f"{D}/connectome/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
A=list(wb['hermaphrodite chemical'].iter_rows(values_only=True))
def looks(n): return isinstance(n,str) and len(n)>=2 and n[0].isalpha()
hr=max(range(len(A)),key=lambda r:sum(looks(x) for x in A[r])); cols=[(str(x).strip() if x is not None else None) for x in A[hr]]
hc=max(range(min(6,len(A[0]))),key=lambda c:sum(looks(A[r][c]) for r in range(len(A))))
Ed=[]
for r in range(hr+1,len(A)):
    cpre=n2c(A[r][hc])
    if not cpre: continue
    for c in range(hc+1,len(A[r])):
        cpost=n2c(cols[c]); v=A[r][c]
        if cpost and cpost!=cpre and v not in (None,''):
            try:
                if float(v)>0: Ed.append((cpre,cpost))
            except: pass
chem=set(Ed); asym=[(a,b) for (a,b) in chem if (b,a) not in chem]
F=[];y=[]
for a,b in asym:
    d=Xc[cls_idx[a]]-Xc[cls_idx[b]]; F+=[d,-d]; y+=[1,0]
w=LogisticRegression(max_iter=1000,C=0.5).fit(np.array(F),np.array(y)).coef_[0]
S={c:float(Xc[cls_idx[c]]@w) for c in classes if c in cls_idx}
from scipy.stats import rankdata
cl=[c for c in classes]; rv=rankdata([S[c] for c in cl]); pr={cl[i]:rv[i]/len(cl) for i in range(len(cl))}
# contact map at CLASS level
wb0=openpyxl.load_workbook(f"{D}/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx",read_only=True,data_only=True)
ws=wb0['adult nerve ring neighbors']; dd=list(ws.iter_rows(values_only=True)); hdr=[h for h in dd[0][1:]]
contact=set()
for r in dd[1:]:
    rn=r[0]
    if rn is None: continue
    ca=n2c(str(rn))
    if ca is None: continue
    for jj,v in enumerate(r[1:]):
        cn=hdr[jj]; cb=n2c(str(cn)) if cn else None
        if cb and cb!=ca and v not in (None,''):
            try:
                if float(v)>0: contact.add((ca,cb)); contact.add((cb,ca))
            except: pass
# real edges that are class-contacting and rank-mapped
real=[(a,b) for a,b in Ed if (a,b) in contact and a in pr and b in pr]
realgap=np.array([abs(pr[a]-pr[b]) for a,b in real])
cpairs=[(a,b) for (a,b) in contact if a in pr and b in pr]
P(f"BRANCH 13b — {len(real)} class-contacting directed edges | {len(cpairs)} ordered contacting pairs")
P(f"  real edge mean |rank-gap| = {realgap.mean():.3f}")
nm=[]
for _ in range(1000):
    samp=[cpairs[i] for i in RNG.integers(0,len(cpairs),len(real))]
    nm.append(np.mean([abs(pr[a]-pr[b]) for a,b in samp]))
z=(realgap.mean()-np.mean(nm))/np.std(nm)
P(f"  random CONTACTING-pair null: {np.mean(nm):.3f}±{np.std(nm):.3f} -> z={z:+.1f}")
P(f"\n  VERDICT: {'the identity-rank adds LOCALITY BEYOND geometry — real synapses connect nearer-in-rank than random contacts (z<-2): the feedforward cascade is layered WITHIN the contact map, not just by it' if z<-2 else ('the layering IS the geometry — within contacting pairs, real edges are NOT nearer-in-rank than random contacts (z=%.1f): contact adjacency already carries the rank-locality'%z)}.")
