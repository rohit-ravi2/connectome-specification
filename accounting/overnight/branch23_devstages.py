"""BRANCH 23 — is the identity-rank's control of synapse DIRECTION present from the EARLIEST developmental stage, or
does it sharpen with maturation? Witvliet 2020 has 8 staged connectomes (datasets 1-2 = L1 birth, ..., 7-8 = adult).
Apply the (Cook-derived, adult-expression) identity-rank s to each stage's chemical synapses: what fraction flow
DOWN the rank, and what is the direction-AUC, at each stage? If the arrow is birth-configured, direction should
already obey the rank at L1; if experience/activity refines it, down-rank fraction should rise L1->adult. Distinguishes
a genetically-preset direction axis from an activity-refined one.
"""
import warnings,sys,csv,re; warnings.simplefilter("ignore")
import numpy as np, pandas as pd, openpyxl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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
chem=set()
for r in range(hr+1,len(A)):
    cpre=n2c(A[r][hc])
    if not cpre: continue
    for c in range(hc+1,len(A[r])):
        cpost=n2c(cols[c]); v=A[r][c]
        if cpost and cpost!=cpre and v not in (None,''):
            try:
                if float(v)>0: chem.add((cpre,cpost))
            except: pass
asym=[(a,b) for (a,b) in chem if (b,a) not in chem]
F=[];y=[]
for a,b in asym:
    d=Xc[cls_idx[a]]-Xc[cls_idx[b]]; F+=[d,-d]; y+=[1,0]
w=LogisticRegression(max_iter=1000,C=0.5).fit(np.array(F),np.array(y)).coef_[0]
s={c:float(Xc[cls_idx[c]]@w) for c in classes if c in cls_idx}
# Witvliet per-stage chemical edges
from collections import defaultdict
stage=defaultdict(lambda: defaultdict(int))
with open(f"{D}/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']=='chemical':
            m=re.search(r'witvliet_2020_(\d)',r['source_file'])
            if not m: continue
            a=n2c(r['pre']); b=n2c(r['post'])
            if a and b and a!=b and a in s and b in s:
                stage[int(m.group(1))][(a,b)]+=int(r['synapses'])
P("BRANCH 23 — identity-rank -> direction across Witvliet developmental stages:")
P("  (datasets 1-2 ~ L1 birth ... 7-8 ~ adult)")
labels={1:'L1(0h)',2:'L1(5h)',3:'L1(8h)',4:'L1(16h)',5:'L2(23h)',6:'L3',7:'adult',8:'adult'}
rows=[]
for st in sorted(stage):
    ed=stage[st]; both=[(a,b) for (a,b) in ed if (b,a) not in ed]  # asym within stage
    if len(both)<20:
        P(f"    stage {st} {labels.get(st,''):9s}: only {len(both)} asym pairs (skip)"); continue
    down=np.mean([1 if s[a]>s[b] else 0 for a,b in both])
    yv=[];sc=[]
    for a,b in both: yv+=[1,0]; sc+=[s[a]-s[b], s[b]-s[a]]
    auc=roc_auc_score(yv,sc)
    rows.append((st,len(both),down,auc))
    P(f"    stage {st} {labels.get(st,''):9s}: {len(both):4d} asym pairs | down-rank {down:.3f} | direction-AUC {auc:.3f}")
if len(rows)>=3:
    early=np.mean([r[3] for r in rows[:2]]); late=np.mean([r[3] for r in rows[-2:]])
    from scipy.stats import spearmanr
    rho,pp=spearmanr([r[0] for r in rows],[r[3] for r in rows])
    P(f"\n  early-stage AUC {early:.3f} vs late-stage AUC {late:.3f}; trend vs stage Spearman r={rho:+.2f} p={pp:.2f}")
    P(f"\n  VERDICT: {'the direction axis is BIRTH-CONFIGURED — the identity-rank already governs synapse direction at the earliest L1 stage (~as well as adult); the arrow is genetically preset, not activity-refined' if early>0.75 and abs(late-early)<0.06 else ('the arrow SHARPENS with maturation (early %.2f -> late %.2f) — direction is refined developmentally' % (early,late) if late>early+0.06 else 'stage trend weak/mixed (early %.2f late %.2f, r=%.2f)'%(early,late,rho))}.")
