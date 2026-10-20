"""BRANCH 5b — GJ direction control, LEAKAGE-FIXED (pair-grouped CV: both orientations of a pair in the same
fold, so the model can't memorize a pair). Clean test: chemical asym-synapses should retain a real arrow; GJ
(symmetric) should collapse to ~0.5 (no arrow). Confirms the arrow is a chemical-synapse-polarity code.
"""
import warnings,sys,openpyxl; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, GroupKFold
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
                    if float(v)>0: ed.add((pre,post) if directed else (min(pre,post),max(pre,post)))
                except: pass
    return ed
gj=[tuple(e) for e in sheet_edges('hermaphrodite gap jn symmetric',False) if n2c(e[0]) is not None and n2c(e[1]) is not None]
chem=sheet_edges('hermaphrodite chemical',True)
asym=[(a,b) for (a,b) in chem if (b,a) not in chem and n2c(a) is not None and n2c(b) is not None]
P(f"BRANCH 5b (leak-fixed) | GJ pairs {len(gj)} | chemical asym {len(asym)}")
def dir_auc(pairs):
    F=[];y=[];g=[]
    for k,(a,b) in enumerate(pairs):
        d=Xc[n2c(a)]-Xc[n2c(b)]; F+=[d,-d]; y+=[1,0]; g+=[k,k]   # SAME group -> both orientations same fold
    F=np.array(F);y=np.array(y);g=np.array(g)
    pred=cross_val_predict(LogisticRegression(max_iter=1000,C=0.5),F,y,cv=GroupKFold(5),groups=g,method='predict_proba')[:,1]
    return roc_auc_score(y,pred)
a_ch=dir_auc(asym); a_gj=dir_auc(gj)
P(f"\n  identity-ARROW, PAIR-GROUPED CV (no leakage):")
P(f"    chemical asym-synapses (real polarity) AUC = {a_ch:.3f}")
P(f"    gap junctions (symmetric, no polarity)  AUC = {a_gj:.3f}")
P(f"\n  VERDICT: {'CONFIRMED — chemical synapses carry an identity-ARROW ('+f'{a_ch:.2f}'+'), gap junctions do NOT ('+f'{a_gj:.2f}'+', ~chance): the arrow is a chemical-synapse polarity code, absent from symmetric electrical coupling.' if a_ch>a_gj+0.12 else f'inconclusive (chem {a_ch:.2f}, gj {a_gj:.2f})'}")
