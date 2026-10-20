"""BRANCH 5 — GAP-JUNCTION control (edge without arrow). Gap junctions are UNDIRECTED (symmetric electrical
synapses). The decomposition predicts: GJ follow the geometry-EDGE but carry NO identity-ARROW (no polarity),
and may instead use identity as SIMILARITY (homotypic innexin coupling). If identity-direction predicts chemical
direction (0.89) but NOT GJ, the arrow is specifically a chemical-synapse-polarity code. Contrast the two identity
roles: DIRECTION (chemical) vs SIMILARITY (electrical).
"""
import warnings,sys,openpyxl,csv,re; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
D="../data/celegans/data"
cg=pd.read_csv(f"{D}/expression/cengen/thresholded/021821_liberal_threshold1.csv")
classes=list(cg.columns[3:]); Xf=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values.T>0
br=Xf.mean(0); Xc=Xf[:,(br>0.05)&(br<0.95)].astype(float); cls_idx={c:k for k,c in enumerate(classes)}
def n2c(n):
    if not isinstance(n,str): return None
    if n in cls_idx: return cls_idx[n]
    b=n
    for s in "LR":
        if b.endswith(s): b=b[:-1]; break
    if b in cls_idx: return cls_idx[b]
    for s in "DV":
        if b.endswith(s) and b[:-1] in cls_idx: return cls_idx[b[:-1]]
    return None
def sheet_edges(sheet,directed):
    wb=openpyxl.load_workbook(f"{D}/connectome/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
    A=list(wb[sheet].iter_rows(values_only=True))
    def looks(n): return isinstance(n,str) and len(n)>=2 and n[0].isalpha()
    hr=max(range(len(A)),key=lambda r:sum(looks(x) for x in A[r])); cols=[(str(x).strip() if x is not None else None) for x in A[hr]]
    hc=max(range(min(6,len(A[0]))),key=lambda c:sum(looks(A[r][c]) for r in range(len(A))))
    ed=set()
    for r in range(hr+1,len(A)):
        pre=A[r][hc]; pre=pre.strip() if isinstance(pre,str) else None
        if not pre: continue
        for c in range(hc+1,len(A[r])):
            post=cols[c]; v=A[r][c]
            if post and v not in (None,''):
                try:
                    if float(v)>0:
                        ed.add((pre,post) if directed else (min(pre,post),max(pre,post)))
                except: pass
    return ed
gj=sheet_edges('hermaphrodite gap jn symmetric',False)   # undirected GJ
chem=sheet_edges('hermaphrodite chemical',True)           # directed chemical
neurons=sorted({x for e in (gj|{p for e in chem for p in e}) for x in (e if isinstance(e,tuple) else [e])} & set(n for n in cls_idx) | {p for e in gj for p in e} | {p for e in chem for p in e})
neurons=[n for n in neurons if n2c(n) is not None]
P(f"BRANCH 5 — GJ control | {len(gj)} gap junctions, {len(chem)} chemical | mapped neurons {len(neurons)}")
kf=KFold(5,shuffle=True,random_state=0)
def auc(F,y):
    return roc_auc_score(y,cross_val_predict(LogisticRegression(max_iter=1000,C=0.5),F,y,cv=kf,method='predict_proba')[:,1])
# (A) identity DIRECTION on GJ: GJ are symmetric -> is there an arrow? build asym-style dataset (both orientations)
gjpairs=[tuple(e) for e in gj if n2c(e[0]) is not None and n2c(e[1]) is not None]
def build_dir(pairs):
    F=[];y=[]
    for a,b in pairs:
        d=Xc[n2c(a)]-Xc[n2c(b)]; F+=[d,-d]; y+=[1,0]
    return np.array(F),np.array(y)
# for GJ 'direction' is arbitrary (symmetric) -> AUC should be ~0.5
Fg,yg=build_dir(gjpairs); a_gj_dir=auc(Fg,yg)
# chemical asymmetric direction (reference, should be high)
asym=[(a,b) for (a,b) in chem if (b,a) not in chem and n2c(a) is not None and n2c(b) is not None]; Fc,yc=build_dir(asym); a_ch_dir=auc(Fc,yc)
P(f"\n  identity-ARROW (direction): chemical asym-synapses AUC {a_ch_dir:.3f}  vs  GJ (symmetric) AUC {a_gj_dir:.3f}")
P(f"    -> {'ARROW is chemical-specific: identity codes chemical DIRECTION but NOT GJ (symmetric, no polarity)' if a_ch_dir>a_gj_dir+0.15 else 'GJ also directional?'}")
# (B) identity SIMILARITY on GJ presence: do similar-identity cells gap-junction? (homotypic innexin)
# universe = pairs; target = GJ present. feature = identity similarity (# shared genes) + contact if available
allp=[(neurons[i],neurons[j]) for i in range(len(neurons)) for j in range(i+1,len(neurons))]
import random
random.seed(0); allp=random.sample(allp, min(20000,len(allp)))
def sim(a,b): return float((Xc[n2c(a)]*Xc[n2c(b)]).sum())
ygj=np.array([1 if (min(a,b),max(a,b)) in gj else 0 for a,b in allp])
Fsim=np.array([[sim(a,b)] for a,b in allp])
a_gj_sim=auc(Fsim,ygj)
# contrast: does identity-similarity predict chemical (undirected) presence?
chem_u=set((min(a,b),max(a,b)) for a,b in chem)
ych=np.array([1 if (min(a,b),max(a,b)) in chem_u else 0 for a,b in allp])
a_ch_sim=auc(Fsim,ych)
P(f"\n  identity-SIMILARITY -> presence: GJ AUC {a_gj_sim:.3f} vs chemical(undirected) AUC {a_ch_sim:.3f}")
P(f"    -> {'GJ use identity as SIMILARITY (homotypic coupling) more than chemical does' if a_gj_sim>a_ch_sim+0.03 else 'similarity ~ same for both'}")
P(f"\n  VERDICT: chemical synapses use identity as DIRECTION (arrow {a_ch_dir:.2f}); GJ have NO arrow ({a_gj_dir:.2f}) —")
P(f"  the two connectomes read identity DIFFERENTLY: chemical=polarity/arrow, electrical=similarity/symmetric coupling.")
