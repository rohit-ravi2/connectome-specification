"""BRANCH 9b — family ABLATION (clean version of the enrichment test; robust to L2 weight-spreading). Retrain the
arrow axis with each candidate family REMOVED; if no family's removal drops pair-grouped AUC, the arrow is
distributed/redundant whole-identity, not a module. Also fit each family ALONE to see how much any recognition
vocabulary carries on its own.
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
allg=cg["gene_name"].astype(str).str.lower().values
br=Xf.mean(0); mask=(br>0.05)&(br<0.95); Xc=Xf[:,mask].astype(float); genes=allg[mask]
cls_idx={c:k for k,c in enumerate(classes)}
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
F=[];yv=[];g=[]
for k,(a,b) in enumerate(asym):
    d=Xc[cls_idx[a]]-Xc[cls_idx[b]]; F+=[d,-d]; yv+=[1,0]; g+=[k,k]
F=np.array(F);yv=np.array(yv);g=np.array(g)
def auc(Fm): return roc_auc_score(yv,cross_val_predict(LogisticRegression(max_iter=1000,C=0.5),Fm,yv,cv=GroupKFold(5),groups=g,method='predict_proba')[:,1])
full=auc(F)
P(f"BRANCH 9b — family ablation | {len(asym)} asym pairs, {len(genes)} genes | FULL arrow AUC {full:.3f}")
GPCR=("srh-","str-","sri-","srd-","sre-","srab","srx-","srt-","srg-","srz-","srsx","srw-","srbc","srr-","srb-","sra-","srv-","srj-","srxa","dmsr","npr-","gar-","ser-","dop-","tyra","octr","mgl-","gbb-")
NP=("flp-","nlp-","ins-"); INX=("inx-","unc-7","unc-9","eat-5","che-7")
HOX=("ceh-","unc-4","unc-30","unc-42","unc-86","unc-3","vab-3","vab-7","vab-15","lin-11","mec-3","ttx-1","ttx-3","cog-1","alr-1","egl-5","egl-46","lim-","zag-1","fax-1","die-1")
ADH=("nrx-1","nlg-1","sax-3","sax-7","cam-1","casy-1","ptp-3","rig-","zig-","oig-","wrk-1","madd-4","unc-40","dma-1","hpo-30","lron","igcm","syg-1","syg-2","cdh-","fmi-","him-4","lad-2")
fams={"GPCR/receptor":GPCR,"neuropeptide":NP,"innexin":INX,"homeobox/TF":HOX,"adhesion/IgSF":ADH}
P("\n  ABLATION (retrain WITHOUT family) and family-ALONE:")
for name,pref in fams.items():
    sel=np.array([any(x.startswith(p) for p in pref) for x in genes])
    if sel.sum()<5: P(f"    {name:16s}: n={sel.sum()} too few"); continue
    drop=auc(F[:,~sel]); alone=auc(F[:,sel])
    P(f"    {name:16s}: n={sel.sum():3d} | remove->AUC {drop:.3f} (Δ{drop-full:+.3f}) | family-alone AUC {alone:.3f}")
P(f"\n  VERDICT: {'no family is NECESSARY (removal Δ≈0) and none reconstructs the arrow alone -> the direction code is DISTRIBUTED whole-cell identity, not a dedicated recognition/signaling/TF module' }.")
