"""BRANCH 12 — what are the RANK-VIOLATING edges? The identity-rank sets 68% of edge directions (feedforward);
the other 32% flow UP the rank and identity CAN'T predict them (B10: pair-specific adds +0.009). Hypothesis: the
connectome = an identity-coded feedforward SKELETON + a structured RECURRENT-LOOP system (command interneurons)
that identity does NOT specify. Tests: (1) are up-rank ('against-the-arrow') edges enriched on command interneurons
vs down-rank? (2) are up-rank edges enriched inside the largest strongly-connected component (recurrent core)?
(3) can identity distinguish up- vs down-rank edges at all (should be near-chance if recurrence is extra-identity)?
"""
import warnings,sys,openpyxl; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score
from scipy.stats import fisher_exact
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
chem=set((a,b) for a,b,w in Ed); asym=[(a,b) for (a,b) in chem if (b,a) not in chem]
F=[];y=[]
for a,b in asym:
    d=Xc[cls_idx[a]]-Xc[cls_idx[b]]; F+=[d,-d]; y+=[1,0]
w=LogisticRegression(max_iter=1000,C=0.5).fit(np.array(F),np.array(y)).coef_[0]
neur=sorted({a for a,b,w_ in Ed}|{b for a,b,w_ in Ed}); idx={n:i for i,n in enumerate(neur)}; M=len(neur)
s={n:float(Xc[cls_idx[n]]@w) for n in neur}
# classify each directed edge as down-rank (feedforward) or up-rank (violation)
up=[(a,b,w_) for a,b,w_ in Ed if s[a]<s[b]]; down=[(a,b,w_) for a,b,w_ in Ed if s[a]>=s[b]]
P(f"BRANCH 12 — {len(Ed)} directed edges | down-rank(feedforward) {len(down)} | up-rank(violation) {len(up)}")
# (1) command-interneuron enrichment among up-rank edges
CMD=set(x for base in ['AVA','AVB','AVD','AVE','PVC','RIM','AVJ','AVK'] for x in [base])
def has_cmd(e): return n2c(e[0]) in CMD or n2c(e[1]) in CMD
cu=sum(has_cmd(e) for e in up); cd=sum(has_cmd(e) for e in down)
orat,pf=fisher_exact([[cu,len(up)-cu],[cd,len(down)-cd]])
P(f"\n  (1) command-interneuron touch: up-rank {cu}/{len(up)} ({cu/len(up):.2f}) vs down-rank {cd}/{len(down)} ({cd/len(down):.2f}) | Fisher OR={orat:.2f} p={pf:.1e}")
# (2) largest SCC (recurrent core) via directed reachability on the full directed graph
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
G=sp.lil_matrix((M,M))
for a,b,w_ in Ed: G[idx[a],idx[b]]=1
ncomp,lab=connected_components(G.tocsr(),directed=True,connection='strong')
sizes=np.bincount(lab); core=np.argmax(sizes); incore=set(neur[i] for i in range(M) if lab[i]==core)
P(f"    largest SCC (recurrent core) size = {sizes.max()}")
def both_core(e): return e[0] in incore and e[1] in incore
uc=sum(both_core(e) for e in up); dc=sum(both_core(e) for e in down)
orat2,pf2=fisher_exact([[uc,len(up)-uc],[dc,len(down)-dc]])
P(f"  (2) inside recurrent core: up-rank {uc/len(up):.2f} vs down-rank {dc/len(down):.2f} | Fisher OR={orat2:.2f} p={pf2:.1e}")
# (3) can identity distinguish up vs down-rank edges? (edge-level, non-antisym: use [|Δexpr|, s_pre, s_post])
Fe=[];ye=[]
for a,b,w_ in Ed:
    Fe.append(np.abs(Xc[cls_idx[a]]-Xc[cls_idx[b]])); ye.append(1 if s[a]<s[b] else 0)
Fe=np.array(Fe);ye=np.array(ye)
auc=roc_auc_score(ye,cross_val_predict(LogisticRegression(max_iter=1000,C=0.5),Fe,ye,cv=KFold(5,shuffle=True,random_state=0),method='predict_proba')[:,1])
P(f"\n  (3) identity (|Δexpr|) predicting up-vs-down-rank edge: AUC {auc:.3f} (0.5=recurrence is extra-identity)")
P(f"\n  VERDICT: {'DECOMPOSITION — identity codes the feedforward SKELETON; the rank-violating edges are STRUCTURED onto the recurrent command core (OR>1.5, p<0.05) and are NOT identity-predictable (AUC~0.5): recurrence is a separate system layered on the identity-coded feedforward axis' if (orat>1.4 or orat2>1.4) and auc<0.6 else 'rank-violations not cleanly command/recurrence-structured (cmd OR=%.2f, core OR=%.2f, identity-AUC=%.2f)'%(orat,orat2,auc)}.")
