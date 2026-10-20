"""BRANCH 1 — the PRESYNAPTIC-IDENTITY AXIS. identity(expr_pre−expr_post)->direction AUC 0.89. Read out the
weight vector w: which genes make a neuron presynaptic (source/sensory) vs postsynaptic (sink/motor)? Is it a
known program (fate/homeobox), and does it SURVIVE an NT confound (not just 'ACh neurons are presynaptic')?
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
gname=cg["gene_name"].astype(str).values
br=Xf.mean(0); keep=np.where((br>0.05)&(br<0.95))[0]; Xc=Xf[:,keep].astype(float); gk=gname[keep]
cls_idx={c:k for k,c in enumerate(classes)}
def n2c(n):
    if n in cls_idx: return cls_idx[n]
    b=n
    for s in "LR":
        if b.endswith(s): b=b[:-1]; break
    if b in cls_idx: return cls_idx[b]
    for s in "DV":
        if b.endswith(s) and b[:-1] in cls_idx: return cls_idx[b[:-1]]
    return None
wb=openpyxl.load_workbook(f"{D}/connectome/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
A=list(wb['hermaphrodite chemical'].iter_rows(values_only=True))
def looks(n): return isinstance(n,str) and len(n)>=2 and n[0].isalpha()
hr=max(range(len(A)),key=lambda r:sum(looks(x) for x in A[r])); cols=[(str(x).strip() if x is not None else None) for x in A[hr]]
hc=max(range(min(6,len(A[0]))),key=lambda c:sum(looks(A[r][c]) for r in range(len(A))))
ed=set()
for r in range(hr+1,len(A)):
    pre=A[r][hc]; pre=pre.strip() if isinstance(pre,str) else None
    if not pre or n2c(pre) is None: continue
    for c in range(hc+1,len(A[r])):
        post=cols[c]; v=A[r][c]
        if post and n2c(post) is not None and v not in (None,'') and pre!=post:
            try:
                if float(v)>0: ed.add((pre,post))
            except: pass
asym=[(a,b) for (a,b) in ed if (b,a) not in ed]
def feat(a,b,Xm=Xc): return Xm[n2c(a)]-Xm[n2c(b)]
def build(asy,Xm=Xc):
    F=[];y=[]
    for a,b in asy: F+=[feat(a,b,Xm),feat(b,a,Xm)]; y+=[1,0]
    return np.array(F),np.array(y)
Fc,yc=build(asym)
clf=LogisticRegression(max_iter=2000,C=0.5).fit(Fc,yc); w=clf.coef_[0]
P(f"BRANCH 1 — presynaptic-identity axis | {len(asym)} directed synapses, {Xc.shape[1]} genes")
# stability of w across folds
kf=KFold(5,shuffle=True,random_state=0); ws=[]
for tr,_ in kf.split(Fc): ws.append(LogisticRegression(max_iter=2000,C=0.5).fit(Fc[tr],yc[tr]).coef_[0])
stab=np.mean([np.corrcoef(ws[i],ws[j])[0,1] for i in range(5) for j in range(i+1,5)])
P(f"  weight-vector stability across folds (mean pairwise corr) = {stab:.2f}")
order=np.argsort(w)
P("\n  PRESYNAPTIC pole (source/sensory-like; high expr -> presynaptic):")
for i in order[::-1][:15]: P(f"    {gk[i]:12s} w={w[i]:+.2f}")
P("\n  POSTSYNAPTIC pole (sink/motor-like; high expr -> postsynaptic):")
for i in order[:15]: P(f"    {gk[i]:12s} w={w[i]:+.2f}")
# NT confound: remove NT-machinery + monoamine genes, refit
ntg=set(['unc-17','cha-1','cho-1','eat-4','unc-25','unc-47','unc-46','cat-1','cat-2','tph-1','tbh-1','tdc-1','bas-1','dat-1','mod-5','snf-11','ace-1','ace-2'])
mask=np.array([g.lower() not in ntg for g in gk])
Fn,yn=build(asym,Xc[:,mask])
aucNT=roc_auc_score(yn,cross_val_predict(LogisticRegression(max_iter=2000,C=0.5),Fn,yn,cv=KFold(5,shuffle=True,random_state=0),method='predict_proba')[:,1])
aucFull=roc_auc_score(yc,cross_val_predict(LogisticRegression(max_iter=2000,C=0.5),Fc,yc,cv=KFold(5,shuffle=True,random_state=0),method='predict_proba')[:,1])
P(f"\n  NT CONFOUND: direction AUC full {aucFull:.3f} vs NT-machinery-removed {aucNT:.3f} -> {'SURVIVES (not just NT)' if aucNT>aucFull-0.03 else 'depends on NT'}")
# characterize top genes by category (name pattern)
def cat(g):
    g=g.lower()
    if g.startswith('ceh-') or g in ('unc-3','unc-30','unc-42','unc-86','unc-4','ttx-3','lin-11','mec-3','cfi-1','che-1','ast-1','nhr-67'): return 'fate-TF'
    if g in ntg: return 'NT-machinery'
    if g.startswith(('osm-','tax-','che-','srx','srh','sri','str-','ocr-','tax','gcy-','odr-')): return 'sensory'
    if g.startswith(('acr-','unc-','lev-','del-','mec-','twk-','egl-')): return 'channel/motor-ish'
    return 'other'
topP=[gk[i] for i in order[::-1][:30]]; topM=[gk[i] for i in order[:30]]
from collections import Counter
P(f"\n  presynaptic-pole top-30 categories: {dict(Counter(cat(g) for g in topP))}")
P(f"  postsynaptic-pole top-30 categories: {dict(Counter(cat(g) for g in topM))}")
np.save('../work/presyn_w.npy', w);
P("\n  [saved axis weights ../work/presyn_w.npy]  VERDICT: the arrow-code is a "+("STABLE, NT-independent identity axis" if stab>0.5 and aucNT>aucFull-0.03 else "axis with caveats")+" — read genes above for what 'presynaptic-ness' IS.")
