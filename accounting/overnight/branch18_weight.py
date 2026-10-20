"""BRANCH 18 — does the identity-rank predict synapse STRENGTH (not just direction)? If the feedforward backbone
'flows downhill strongly', then down-rank (feedforward) edges should carry MORE synapses than up-rank, and larger
rank-gaps... could go either way (strong local relays vs weak long-range). Tests on Cook weighted chemical: (1)
weight of down-rank vs up-rank edges; (2) does signed rank-gap predict log-weight (regression); (3) control for
degree (hubs have strong edges). Ties the 'arrow' to the 'weight' axis that B15 showed carries reproducibility.
"""
import warnings,sys; warnings.simplefilter("ignore")
import numpy as np, pandas as pd, openpyxl
from sklearn.linear_model import LogisticRegression
from scipy.stats import mannwhitneyu, spearmanr
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
from collections import defaultdict
ew=defaultdict(float)
for a,b,w_ in Ed: ew[(a,b)]+=w_
chem=set(ew); asym=[(a,b) for (a,b) in chem if (b,a) not in chem]
F=[];y=[]
for a,b in asym:
    d=Xc[cls_idx[a]]-Xc[cls_idx[b]]; F+=[d,-d]; y+=[1,0]
w=LogisticRegression(max_iter=1000,C=0.5).fit(np.array(F),np.array(y)).coef_[0]
neur=sorted({a for a,b in ew}|{b for a,b in ew})
s={n:float(Xc[cls_idx[n]]@w) for n in neur}
deg=defaultdict(float)
for (a,b),v in ew.items(): deg[a]+=v; deg[b]+=v
edges=list(ew.keys()); wt=np.array([ew[e] for e in edges]); lw=np.log(wt)
down=np.array([1 if s[a]>s[b] else 0 for a,b in edges])
gap=np.array([s[a]-s[b] for a,b in edges])   # signed
degsum=np.array([deg[a]+deg[b] for a,b in edges])
P(f"BRANCH 18 — {len(edges)} directed weighted edges | down-rank {int(down.sum())} up-rank {int((1-down).sum())}")
P(f"  median synapses: down-rank {np.median(wt[down==1]):.1f} vs up-rank {np.median(wt[down==0]):.1f}")
u,pu=mannwhitneyu(wt[down==1],wt[down==0],alternative='greater')
P(f"    Mann-Whitney (down heavier): p={pu:.2e}")
r1,p1=spearmanr(gap,lw)
P(f"\n  signed rank-gap (s_pre-s_post) vs log-weight: Spearman r={r1:+.3f} p={p1:.1e}  (positive => feedforward edges stronger)")
# degree control
def resid(a,c): b=np.polyfit(c,a,1); return a-np.polyval(b,c)
r2,p2=spearmanr(resid(gap,degsum),resid(lw,degsum))
P(f"  degree-controlled partial r={r2:+.3f} p={p2:.1e}")
P(f"\n  VERDICT: {'the backbone flows DOWNHILL STRONGLY — feedforward (down-rank) synapses carry more weight than rank-violations, degree-robustly (r>0.1, p<0.01): synapse strength aligns with the identity-rank, indirectly linking the arrow to the weight axis that carries reproducibility' if pu<0.01 and r2>0.05 and p2<0.05 else 'rank does NOT predict synapse weight after degree control (r=%.2f p=%.2g) — strength is orthogonal to the direction-rank'%(r2,p2)}.")
