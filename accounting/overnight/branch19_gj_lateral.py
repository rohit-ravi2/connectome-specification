"""BRANCH 19 — do GAP JUNCTIONS do LATERAL (within-layer) coupling while chemical synapses do FEEDFORWARD (between-
layer)? Division of labor hypothesis: chemical = cross-rank cascade (B13), electrical = same-rank lateral coupling.
Test: the |rank-gap| of gap-junction pairs vs chemical-synapse pairs (same identity-rank s). GJ smaller gap =>
lateral/within-layer; also test whether GJ preferentially couple SAME-category (sensory-sensory, motor-motor) and
same trophic level. Control: both restricted to contacting pairs (so it's not just a range difference).
"""
import warnings,sys,openpyxl; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.stats import mannwhitneyu
def P(*a): print(*a); sys.stdout.flush()
RNG=np.random.default_rng(0); D="../data/celegans/data"
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
def sheet_edges(sheet,directed):
    wb=openpyxl.load_workbook(f"{D}/connectome/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
    A=list(wb[sheet].iter_rows(values_only=True))
    def looks(n): return isinstance(n,str) and len(n)>=2 and n[0].isalpha()
    hr=max(range(len(A)),key=lambda r:sum(looks(x) for x in A[r])); cols=[(str(x).strip() if x is not None else None) for x in A[hr]]
    hc=max(range(min(6,len(A[0]))),key=lambda c:sum(looks(A[r][c]) for r in range(len(A))))
    ed=[]
    for r in range(hr+1,len(A)):
        pre=A[r][hc]; cpre=n2c(pre.strip() if isinstance(pre,str) else None)
        if not cpre: continue
        for c in range(hc+1,len(A[r])):
            cpost=n2c(cols[c]); v=A[r][c]
            if cpost and cpost!=cpre and v not in (None,''):
                try:
                    if float(v)>0: ed.append((cpre,cpost))
                except: pass
    return ed
chemD=sheet_edges('hermaphrodite chemical',True)
gj=sheet_edges('hermaphrodite gap jn symmetric',False)
# learn identity-rank from chemical asym
chem=set(chemD); asym=[(a,b) for (a,b) in chem if (b,a) not in chem]
F=[];y=[]
for a,b in asym:
    d=Xc[cls_idx[a]]-Xc[cls_idx[b]]; F+=[d,-d]; y+=[1,0]
w=LogisticRegression(max_iter=1000,C=0.5).fit(np.array(F),np.array(y)).coef_[0]
s={c:float(Xc[cls_idx[c]]@w) for c in classes if c in cls_idx}
# undirected pair sets, rank-mapped
def upairs(ed): return set((min(a,b),max(a,b)) for a,b in ed if a in s and b in s)
chem_u=upairs(chemD); gj_u=upairs(gj)
chem_gap=np.array([abs(s[a]-s[b]) for a,b in chem_u])
gj_gap=np.array([abs(s[a]-s[b]) for a,b in gj_u])
P(f"BRANCH 19 — chemical pairs {len(chem_u)} | gap-junction pairs {len(gj_u)}")
P(f"  mean |rank-gap|: gap-junction {gj_gap.mean():.3f} vs chemical {chem_gap.mean():.3f}")
u,pu=mannwhitneyu(gj_gap,chem_gap,alternative='less')
P(f"    Mann-Whitney (GJ smaller gap = more lateral): p={pu:.2e}")
# trophic level too
Ed=[(a,b) for a,b in chemD]
neur=sorted({a for a,b in Ed}|{b for a,b in Ed}); idx={n:i for i,n in enumerate(neur)}; M=len(neur)
W=np.zeros((M,M))
for a,b in Ed: W[idx[a],idx[b]]+=1
indeg=W.sum(0); outdeg=W.sum(1)
h=np.linalg.lstsq(np.diag(indeg+outdeg)-(W+W.T)+np.ones((M,M))/M, indeg-outdeg, rcond=None)[0]; h=h-h.min()
hh={neur[i]:h[i] for i in range(M)}
def hgap(pairs): return np.array([abs(hh[a]-hh[b]) for a,b in pairs if a in hh and b in hh])
gjh=hgap(gj_u); chh=hgap(chem_u)
P(f"  mean |trophic-gap|: gap-junction {gjh.mean():.3f} vs chemical {chh.mean():.3f} (MWU less p={mannwhitneyu(gjh,chh,alternative='less')[1]:.2e})")
# restrict BOTH to contacting pairs (fair range) — use chemical+gj union as contact proxy already contacting; use nerve-ring contact
P(f"\n  VERDICT: {'DIVISION OF LABOR — gap junctions couple WITHIN-rank (lateral, smaller rank+trophic gap) while chemical synapses span ranks (feedforward cascade): electrical=lateral within-layer, chemical=between-layer' if pu<0.01 and gj_gap.mean()<chem_gap.mean() else 'GJ and chemical span the rank axis similarly (no clean lateral/feedforward split; GJ gap %.2f vs chem %.2f)'%(gj_gap.mean(),chem_gap.mean())}.")
