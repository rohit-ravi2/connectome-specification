"""BRANCH 15 — is the identity-coded FEEDFORWARD (down-rank) the REPRODUCIBLE developmental backbone? The project's
north star is cross-animal reproducibility. We now have an identity-rank that sets the arrow. Test: across the 8
Witvliet animals, are DOWN-rank (feedforward, identity-consistent) chemical edges present in MORE animals than
UP-rank (rank-violating) edges? If yes — controlling for synapse weight/degree (reproducibility tracks weight) —
then the identity-rank IS the conserved backbone and the deviations are the variable part. This links the arrow
synthesis to developmental reproducibility.
"""
import warnings,sys,csv; warnings.simplefilter("ignore")
import numpy as np, pandas as pd, openpyxl
from collections import defaultdict, Counter
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
# Cook rank
wb=openpyxl.load_workbook(f"{D}/connectome/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
A=list(wb['hermaphrodite chemical'].iter_rows(values_only=True))
def looks(n): return isinstance(n,str) and len(n)>=2 and n[0].isalpha()
hr=max(range(len(A)),key=lambda r:sum(looks(x) for x in A[r])); cols=[(str(x).strip() if x is not None else None) for x in A[hr]]
hc=max(range(min(6,len(A[0]))),key=lambda c:sum(looks(A[r][c]) for r in range(len(A))))
cookEd=set()
for r in range(hr+1,len(A)):
    cpre=n2c(A[r][hc])
    if not cpre: continue
    for c in range(hc+1,len(A[r])):
        cpost=n2c(cols[c]); v=A[r][c]
        if cpost and cpost!=cpre and v not in (None,''):
            try:
                if float(v)>0: cookEd.add((cpre,cpost))
            except: pass
asym=[(a,b) for (a,b) in cookEd if (b,a) not in cookEd]
F=[];y=[]
for a,b in asym:
    d=Xc[cls_idx[a]]-Xc[cls_idx[b]]; F+=[d,-d]; y+=[1,0]
w=LogisticRegression(max_iter=1000,C=0.5).fit(np.array(F),np.array(y)).coef_[0]
s={c:float(Xc[cls_idx[c]]@w) for c in classes if c in cls_idx}
# Witvliet: per-animal presence + total synapses
present=defaultdict(set); syn=Counter()
with open(f"{D}/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']=='chemical':
            a=n2c(r['pre']); b=n2c(r['post'])
            if a and b and a!=b and a in s and b in s:
                present[(a,b)].add(r['source_file']); syn[(a,b)]+=int(r['synapses'])
edges=list(present.keys())
nrep=np.array([len(present[e]) for e in edges])          # in how many of 8 animals
wt=np.array([syn[e] for e in edges])                     # total synapses
downrank=np.array([1 if s[e[0]]>s[e[1]] else 0 for e in edges])  # feedforward
P(f"BRANCH 15 — {len(edges)} Witvliet chem class-edges | down-rank(ff) {int(downrank.sum())} up-rank {int((1-downrank).sum())}")
P(f"  mean #animals present: down-rank {nrep[downrank==1].mean():.2f} vs up-rank {nrep[downrank==0].mean():.2f}")
u,pu=mannwhitneyu(nrep[downrank==1],nrep[downrank==0],alternative='greater')
P(f"    Mann-Whitney (down>up): p={pu:.2e}")
# control for weight: reproducibility tracks synapse count. within weight strata, does down-rank still win?
P(f"\n  weight-stratified (reproducibility tracks synapse count):")
import numpy as np
qs=np.quantile(wt,[0,.25,.5,.75,1.0])
for k in range(4):
    m=(wt>=qs[k])&(wt<=qs[k+1])
    d1=nrep[m&(downrank==1)]; d0=nrep[m&(downrank==0)]
    if len(d0)>3 and len(d1)>3:
        P(f"    weight [{qs[k]:.0f}-{qs[k+1]:.0f}]: down {d1.mean():.2f} (n{len(d1)}) vs up {d0.mean():.2f} (n{len(d0)})")
# partial: regress out log-weight, correlate residual reproducibility with down-rank
lw=np.log1p(wt); b=np.polyfit(lw,nrep,1); res=nrep-np.polyval(b,lw)
rho,pp=spearmanr(res,downrank)
P(f"\n  weight-residualized reproducibility vs down-rank: Spearman r={rho:+.3f} p={pp:.1e}")
P(f"\n  VERDICT: {'the identity-coded FEEDFORWARD (down-rank) edges are the REPRODUCIBLE backbone — present in more animals than rank-violations even at matched synapse weight: the identity-rank is the conserved developmental skeleton, deviations are the variable part' if pu<0.01 and rho>0 else 'down-rank not more reproducible after weight control (r=%.2f) — reproducibility is carried by weight, not rank-consistency'%rho}.")
