"""BRANCH 16 — where do E vs I neurons sit on the identity-rank / hierarchy? B8 showed inhibitory EDGES are weakly
more feedback. Neuron-level question: do GABAergic (inhibitory) neuron classes occupy a characteristic position on
the identity-rank s (and trophic h)? C. elegans GABA is largely motor-side (body GABAergic motor neurons DD/VD,
RME, RIS, AVL, DVB). Prediction: GABA -> low presynaptic-source rank / high trophic (sink/motor end); glutamate
(sensory-rich) -> high rank; ACh -> broad. Null: shuffle NT labels. Ties the E/I identity to the functional axis.
"""
import warnings,sys; warnings.simplefilter("ignore")
import numpy as np, pandas as pd, openpyxl
from sklearn.linear_model import LogisticRegression
from scipy.stats import kruskal
def P(*a): print(*a); sys.stdout.flush()
RNG=np.random.default_rng(0); D="../data/celegans/data"
cg=pd.read_csv(f"{D}/expression/cengen/thresholded/021821_liberal_threshold1.csv")
classes=list(cg.columns[3:]); Xf=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values.T>0
gname=cg["gene_name"].astype(str).str.lower().values
br=Xf.mean(0); Xc=Xf[:,(br>0.05)&(br<0.95)].astype(float); cls_idx={c:k for k,c in enumerate(classes)}
def ex(g,ci):
    wq=np.where(gname==g)[0]; return bool(Xf[ci,wq[0]]) if len(wq) else False
def nt(ci):
    gaba=ex("unc-25",ci) or ex("unc-47",ci); glu=ex("eat-4",ci); ach=ex("unc-17",ci) or ex("cha-1",ci)
    if gaba: return "GABA"
    if ach and not glu: return "ACh"
    if glu and not ach: return "Glu"
    if glu and ach: return "ACh+Glu"
    return "none"
NT={c:nt(k) for k,c in enumerate(classes)}
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
                if float(v)>0: Ed.append((cpre,cpost,float(v)))
            except: pass
chem=set((a,b) for a,b,w in Ed); asym=[(a,b) for (a,b) in chem if (b,a) not in chem]
F=[];y=[]
for a,b in asym:
    d=Xc[cls_idx[a]]-Xc[cls_idx[b]]; F+=[d,-d]; y+=[1,0]
w=LogisticRegression(max_iter=1000,C=0.5).fit(np.array(F),np.array(y)).coef_[0]
neur=sorted({a for a,b,w_ in Ed}|{b for a,b,w_ in Ed}); idx={n:i for i,n in enumerate(neur)}; M=len(neur)
s={n:float(Xc[cls_idx[n]]@w) for n in neur}
W=np.zeros((M,M))
for a,b,w_ in Ed: W[idx[a],idx[b]]+=w_
indeg=W.sum(0); outdeg=W.sum(1)
h=np.linalg.lstsq(np.diag(indeg+outdeg)-(W+W.T)+np.ones((M,M))/M, indeg-outdeg, rcond=None)[0]; h=h-h.min()
hh={neur[i]:h[i] for i in range(M)}
P(f"BRANCH 16 — identity-rank s and trophic h by neurotransmitter class:")
groups={}
for lab in ["Glu","ACh","ACh+Glu","GABA"]:
    ns=[n for n in neur if NT.get(n)==lab]
    if len(ns)<4: continue
    ss=[s[n] for n in ns]; hs=[hh[n] for n in ns]; groups[lab]=ss
    P(f"    {lab:8s} (n={len(ns):2d}): identity-rank s={np.mean(ss):+.3f}  trophic h={np.mean(hs):.3f}")
if "GABA" in groups and "Glu" in groups:
    ks,pk=kruskal(*groups.values())
    P(f"\n  Kruskal-Wallis on identity-rank across NT: H={ks:.1f} p={pk:.1e}")
    # is GABA the lowest-rank (motor-end)?
    order=sorted(groups,key=lambda g:np.mean(groups[g]))
    P(f"    rank order (low presynaptic-source -> high): {' < '.join(order)}")
    # permutation null on GABA mean-rank
    gaba_ns=[n for n in neur if NT.get(n)=="GABA"]; gm=np.mean([s[n] for n in gaba_ns])
    alls=np.array([s[n] for n in neur]); null=[np.mean(RNG.choice(alls,len(gaba_ns),replace=False)) for _ in range(2000)]
    z=(gm-np.mean(null))/np.std(null)
    P(f"    GABA mean rank {gm:+.3f} vs shuffle {np.mean(null):+.3f}±{np.std(null):.3f} -> z={z:+.1f} ({'GABA sits motor-side/low-rank' if z<-2 else 'GABA position not extreme'})")
P(f"\n  VERDICT: E/I identity maps onto the functional axis if GABA occupies the motor-end (low presynaptic-source rank);\n  read the ordering + z above — a coherent NT->position map means inhibition is placed at a specific hierarchy level.")
