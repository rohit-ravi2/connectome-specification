"""BRANCH 25 — does the anatomical identity-rank predict the direction of MEASURED FUNCTIONAL signal flow? Randi 2023
whole-brain functional connectivity: stimulate i, measure response in j -> directed influence fc[i,j]. Test: for
pairs with asymmetric functional influence (fc[i,j] != fc[j,i]), does the identity-rank agree (stronger influence
FROM the higher-rank/source neuron)? If yes, the identity arrow predicts real signal-flow direction (closes identity
->anatomy->function). If null, functional influence is emergent/polysynaptic (as prior structure->dynamics tests
found) and direction is not read off the anatomical rank. Compare against the anatomical connectome's own predictivity.
"""
import warnings,sys,openpyxl; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
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
# identity rank from Cook chemical
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
# Randi functional connectivity (directed, SIGNED: +activating / -suppressing). Direction of influence is i->j
# regardless of sign; STRENGTH is |value|. Keep the max-magnitude value per class-pair.
fc={}
with open(f"{D}/randi2023/randi_funconn_wildty.tsv") as f:
    for line in f:
        p=line.split('\t')
        if len(p)>=3:
            i=n2c(p[0].strip()); j=n2c(p[1].strip())
            try: v=float(p[2])
            except: continue
            if i and j and i!=j:
                if abs(v)>abs(fc.get((i,j),0)): fc[(i,j)]=v
# asymmetric functional pairs by INFLUENCE MAGNITUDE
seen=set(); pairs=[]
for (i,j) in list(fc):
    if (j,i) in seen or (i,j) in seen: continue
    seen.add((i,j))
    a=abs(fc.get((i,j),0)); b=abs(fc.get((j,i),0))
    if a!=b and i in s and j in s:
        # direction of stronger functional influence (by magnitude)
        if a>b: pairs.append((i,j))   # i influences j more
        else: pairs.append((j,i))
P(f"BRANCH 25 — {len(fc)} directed functional edges | {len(pairs)} asymmetric functional pairs (rank-mapped)")
# does identity-rank predict functional-flow direction?
down=np.mean([1 if s[a]>s[b] else 0 for a,b in pairs])
yv=[];sc=[]
for a,b in pairs: yv+=[1,0]; sc+=[s[a]-s[b], s[b]-s[a]]
au=roc_auc_score(yv,sc)
P(f"  identity-rank -> functional-flow direction: down-rank {down:.3f} | AUC {au:.3f}")
# reference: does the ANATOMICAL arrow (chemical asym) predict functional direction? (upper bound-ish)
anat_agree=[]
for a,b in pairs:
    if (a,b) in chem and (b,a) not in chem: anat_agree.append(1)
    elif (b,a) in chem and (a,b) not in chem: anat_agree.append(0)
if anat_agree: P(f"  (of pairs also anatomically-asymmetric, {len(anat_agree)}: functional agrees with anatomy {np.mean(anat_agree):.3f})")
# null: shuffle s labels
import numpy as np
alls=np.array([s[n] for n in s]); RNG=np.random.default_rng(0)
null=[np.mean([1 if RNG.choice(alls)>RNG.choice(alls) else 0 for _ in range(len(pairs))]) for _ in range(200)]
P(f"\n  VERDICT: {'the identity-rank predicts FUNCTIONAL signal-flow direction (AUC %.2f>0.6) — identity->anatomy->function closes: the molecular arrow anticipates measured brain dynamics'%au if au>0.6 else 'the identity-rank does NOT predict functional-flow direction (AUC %.2f ~chance) — measured functional influence is emergent/polysynaptic, not read off the anatomical arrow (consistent with prior structure->dynamics walls)'%au}.")
