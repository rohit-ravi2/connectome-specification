"""SPEC 2 — is the developmental decision boundary geometrically located? Full contact-area distribution per
reproducibility bin (absent / variable / repro-present); OVERLAP coefficient between variable and each extreme;
threshold locations (present|variable and variable|absent); whether thresholds vary by circuit type (trophic level)
and developmental stage (L1 vs LATE). Report distributions + overlaps + thresholds; no pass/fail.
"""
import warnings,sys,csv,openpyxl; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
D="../data/celegans/data"
# ---- contact area (adult nerve ring) ----
def load_contact(sheet):
    df=pd.read_excel(f"{D}/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx",sheet,header=None)
    names=[str(x).strip() for x in df.iloc[0,1:].tolist()]
    Mv=df.iloc[1:,1:].apply(pd.to_numeric,errors="coerce").values
    rns=[str(x).strip() for x in df.iloc[1:,0].tolist()]
    area={}
    for r,rnm in enumerate(rns):
        for c,cn in enumerate(names):
            v=Mv[r,c]
            if pd.notna(v) and v>0:
                area[(rnm,cn)]=max(area.get((rnm,cn),0),v)  # directed key kept both ways below
    return area
areaU=load_contact('adult nerve ring neighbors')  # undirected-ish (both keys present in matrix)
# ---- witvliet per-dataset presence ----
syn=defaultdict(dict)
with open(f"{D}/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']!='chemical': continue
        ds=int(r['source_file'].split('_')[2].split('.')[0])
        syn[(r['pre'],r['post'])][ds]=int(r['synapses'])
# ---- identity-rank s (for circuit-type stratification) ----
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
def looks(x): return isinstance(x,str) and len(x)>=2 and x[0].isalpha()
hr=max(range(len(A)),key=lambda r:sum(looks(x) for x in A[r])); cols=[(str(x).strip() if x is not None else None) for x in A[hr]]
hc=max(range(min(6,len(A[0]))),key=lambda c:sum(looks(A[r][c]) for r in range(len(A))))
chem=set()
for r in range(hr+1,len(A)):
    cp=n2c(A[r][hc])
    if not cp: continue
    for c in range(hc+1,len(A[r])):
        q=n2c(cols[c]); v=A[r][c]
        if q and q!=cp and v not in (None,''):
            try:
                if float(v)>0: chem.add((cp,q))
            except: pass
asym=[(a,b) for (a,b) in chem if (b,a) not in chem]
F=[];y=[]
for a,b in asym:
    d=Xc[cls_idx[a]]-Xc[cls_idx[b]]; F+=[d,-d]; y+=[1,0]
w=LogisticRegression(max_iter=1000,C=0.5).fit(np.array(F),np.array(y)).coef_[0]
srank={c:float(Xc[cls_idx[c]]@w) for c in classes if c in cls_idx}
# ---- build pair table with area + per-set k ----
def build(dslist):
    nds=len(dslist); rows=[]
    for (pre,post),ar in areaU.items():
        if pre==post: continue
        sc=syn.get((pre,post),{})
        k=sum(1 for d in dslist if sc.get(d,0)>0)
        b='absent' if k==0 else ('repro_present' if k==nds else 'variable')
        sp=srank.get(n2c(pre)); rows.append((pre,post,ar,k,b,sp))
    return pd.DataFrame(rows,columns=['pre','post','area','k','bin','s_pre'])
def overlap_coef(x1,x2,bins):
    h1,_=np.histogram(x1,bins=bins,density=True); h2,_=np.histogram(x2,bins=bins,density=True)
    wd=np.diff(bins); return float(np.sum(np.minimum(h1,h2)*wd))
def best_threshold(lo_area,hi_area):
    # area cutoff maximizing balanced separation (Youden) hi>lo
    yv=np.r_[np.zeros(len(lo_area)),np.ones(len(hi_area))]; xv=np.r_[lo_area,hi_area]
    auc=roc_auc_score(yv,xv)
    cuts=np.unique(np.quantile(xv,np.linspace(.02,.98,60)))
    best=None
    for t in cuts:
        tpr=(hi_area>t).mean(); fpr=(lo_area>t).mean(); j=tpr-fpr
        if best is None or j>best[1]: best=(t,j,tpr,1-fpr)
    return auc,best
def report(df,tag):
    P("\n"+"="*66); P(f"[{tag}]  bins: "+str(dict(df['bin'].value_counts())))
    la=np.log10(df['area'].clip(lower=1))
    bins=np.linspace(la.min(),la.max(),25)
    for b in ['absent','variable','repro_present']:
        x=la[df['bin']==b]
        hist,_=np.histogram(x,bins=8)
        P(f"  {b:<14} n={len(x):5d} log10area: med={np.median(x):.2f} q25={np.percentile(x,25):.2f} q75={np.percentile(x,75):.2f}  hist(8bin)={list(hist)}")
    va=la[df['bin']=='variable']; ab=la[df['bin']=='absent']; pr=la[df['bin']=='repro_present']
    P(f"  OVERLAP coef  variable∩absent = {overlap_coef(va,ab,bins):.3f} | variable∩present = {overlap_coef(va,pr,bins):.3f} | absent∩present = {overlap_coef(ab,pr,bins):.3f}")
    auc1,b1=best_threshold(df[df['bin']=='variable']['area'].values, df[df['bin']=='repro_present']['area'].values)
    auc2,b2=best_threshold(df[df['bin']=='absent']['area'].values, df[df['bin']=='variable']['area'].values)
    P(f"  THRESHOLD present|variable : area={b1[0]:.0f} (AUC {auc1:.3f}, sep TPR={b1[2]:.2f}/TNR={b1[3]:.2f})")
    P(f"  THRESHOLD variable|absent  : area={b2[0]:.0f} (AUC {auc2:.3f}, sep TPR={b2[2]:.2f}/TNR={b2[3]:.2f})")
    return b1[0],b2[0]
P("SPEC2 — contact-area zone hardening")
dfL1=build([1,2,3,4]); dfLATE=build([5,6,7,8])
t1_L1=report(dfL1,"L1 replicates 1-4"); t1_LATE=report(dfLATE,"LATE 5-8")
# ---- circuit-type stratification (trophic/identity side) on ALL 1-8 ----
dfALL=build([1,2,3,4,5,6,7,8]); dfALL=dfALL.dropna(subset=['s_pre'])
med=dfALL['s_pre'].median()
for side,mask in [("SENSORY-side (high s_pre)",dfALL.s_pre>med),("MOTOR-side (low s_pre)",dfALL.s_pre<=med)]:
    report(dfALL[mask],side)
P("\nSUMMARY thresholds: present|variable  L1 vs LATE = %.0f vs %.0f ; variable|absent L1 vs LATE = %.0f vs %.0f"%(t1_L1[0],t1_LATE[0],t1_L1[1],t1_LATE[1]))
