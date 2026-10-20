"""FRONTIER 1 (P1) — what organizes the dense/decentralized NEUROPEPTIDE network? The chemical connectome's arrow is
an identity-rank; the NPP net (Ripoll-Sanchez, real) is 14-20x denser and rank-decoupled. Hypothesis: it has NO
specific address structure — connectivity is MASS-ACTION expression breadth (who makes many NPPs broadcasts to who
expresses many GPCRs), a promiscuous degree-product field, not a wiring code. Tests: (1) does a degree/marginal
(gravity) model W_ij ~ out_i*in_j explain the network? (2) does anything specific (identity similarity / chemical
rank) add beyond degrees? (3) is NPP out-broadcasting / in-receiving aligned with the chemical sensory->motor rank?
(4) is out-strength just NPP-gene repertoire size, in-strength just receptor repertoire?
"""
import warnings,sys,openpyxl; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
def P(*a): print(*a); sys.stdout.flush()
D="../data/celegans/data"
cg=pd.read_csv(f"{D}/expression/cengen/thresholded/021821_liberal_threshold1.csv")
classes=list(cg.columns[3:]); gname=cg["gene_name"].astype(str).str.lower().values
Xall=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values.T>0
br=Xall.mean(0); Xc=Xall[:,(br>0.05)&(br<0.95)].astype(float); cls_idx={c:k for k,c in enumerate(classes)}
def n2c(n):
    if not isinstance(n,str): return None
    n=n.strip()
    if n in cls_idx: return n
    b=n
    for s in "LR":
        if b.endswith(s): b=b[:-1]; break
    if b in cls_idx: return b
    for s in "DV":
        if b.endswith(s) and b[:-1] in cls_idx: return b[:-1]
    return None
# chemical identity rank
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
F=[];yy=[]
for a,b in asym:
    d=Xc[cls_idx[a]]-Xc[cls_idx[b]]; F+=[d,-d]; yy+=[1,0]
w=LogisticRegression(max_iter=1000,C=0.5).fit(np.array(F),np.array(yy)).coef_[0]
srank={c:float(Xc[cls_idx[c]]@w) for c in classes if c in cls_idx}
# NPP repertoire per class: # flp/nlp/ins genes expressed
npp=np.array([g.startswith(("flp-","nlp-","ins-")) for g in gname])
npp_count={c:int(Xall[cls_idx[c],npp].sum()) for c in classes if c in cls_idx}
# NPP long-range adjacency
df=pd.read_csv(f"{D}/ripollsanchez2023/01022024_neuropeptide_connectome_long_range_model.csv",index_col=0)
rows=list(df.index); colns=list(df.columns); Wm=df.values.astype(float)
ri=[n2c(r) for r in rows]; ci=[n2c(c) for c in colns]
keepr=[i for i,x in enumerate(ri) if x in srank]; keepc=[j for j,x in enumerate(ci) if x in srank]
Rn=[ri[i] for i in keepr]; Cn=[ci[j] for j in keepc]; W=Wm[np.ix_(keepr,keepc)]
Wb=(W>0).astype(int); n=len(Rn)
out_i=Wb.sum(1); in_j=Wb.sum(0)
P(f"FRONTIER 1 (P1) — NPP long-range net {n}x{len(Cn)} | edges {int(Wb.sum())} density {Wb.mean():.2f}")
# (1) degree/gravity model: predict edge presence from log(out_i)+log(in_j)
pairs=[(i,j) for i in range(n) for j in range(len(Cn)) if Rn[i]!=Cn[j]]
y=np.array([Wb[i,j] for i,j in pairs])
Xdeg=np.array([[np.log1p(out_i[i]),np.log1p(in_j[j])] for i,j in pairs])
kf=KFold(5,shuffle=True,random_state=0)
auc_deg=roc_auc_score(y,cross_val_predict(LogisticRegression(max_iter=1000),Xdeg,y,cv=kf,method='predict_proba')[:,1])
P(f"\n  (1) degree/gravity model (out_i,in_j) -> edge: AUC {auc_deg:.3f} (high => connectivity is a degree-product field)")
# (2) does specific identity add beyond degrees? identity cosine sim + chemical rank gap
cidx={c:cls_idx[c] for c in set(Rn)|set(Cn)}
def sim(a,b): return float(Xc[cidx[a]]@Xc[cidx[b]])
Xspec=np.array([[np.log1p(out_i[i]),np.log1p(in_j[j]),sim(Rn[i],Cn[j]),abs(srank[Rn[i]]-srank[Cn[j]])] for i,j in pairs])
auc_spec=roc_auc_score(y,cross_val_predict(LogisticRegression(max_iter=1000),Xspec,y,cv=kf,method='predict_proba')[:,1])
P(f"  (2) +identity-sim +chemical-rank-gap: AUC {auc_spec:.3f} (Δ over degrees {auc_spec-auc_deg:+.3f}) — specific structure beyond mass-action?")
# (3) NPP broadcasting/receiving vs chemical sensory->motor rank
so=spearmanr([srank[c] for c in Rn], out_i)[0]; si=spearmanr([srank[c] for c in Cn], in_j)[0]
P(f"\n  (3) chemical rank s vs NPP OUT-degree (broadcast): r={so:+.3f} | vs IN-degree (receive): r={si:+.3f}")
P(f"      (s>0 = sensory-source; positive out-corr => sensory neurons broadcast peptides)")
# (4) is out/in-degree just repertoire size?
ro=spearmanr([npp_count[c] for c in Rn], out_i)[0]
P(f"  (4) NPP out-degree vs #NPP genes expressed (repertoire size): r={ro:+.3f}")
P(f"\n  VERDICT: {'the NPP network is a MASS-ACTION expression field — edge presence is a degree/repertoire product (AUC %.2f from degrees alone), specific identity adds little (Δ%.3f), organized by broadcast/receive REPERTOIRE not a wiring address code'%(auc_deg,auc_spec-auc_deg) if auc_deg>0.75 and auc_spec-auc_deg<0.03 else 'NPP net has structure beyond degree-product (Δ%.3f) — specific pairing/identity matters'%(auc_spec-auc_deg)}.")
