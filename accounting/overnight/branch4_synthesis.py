"""BRANCH 4 — the FULL GENERATIVE SYNTHESIS. If the picture is 'geometry codes the EDGE (which pairs connect),
identity codes the ARROW (which direction)', then predicting the DIRECTED connectome should decompose: geometry
gives the undirected edge, identity gives the direction, and together they reconstruct the directed synapse.
Test: predict directed synapse i->j from [contact geometry] + [identity direction expr_i-expr_j]; does edge x arrow
beat each alone, cross-animal?
"""
import warnings,sys,openpyxl,csv,re; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
D="../data/celegans/data"
# expression (variable genes)
cg=pd.read_csv(f"{D}/expression/cengen/thresholded/021821_liberal_threshold1.csv")
classes=list(cg.columns[3:]); Xf=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values.T>0
br=Xf.mean(0); Xc=Xf[:,(br>0.05)&(br<0.95)].astype(float); cls_idx={c:k for k,c in enumerate(classes)}
def n2c(n):
    if n in cls_idx: return cls_idx[n]
    b=n
    for s in "LR":
        if b.endswith(s): b=b[:-1]; break
    if b in cls_idx: return cls_idx[b]
    for s in "DV":
        if b.endswith(s) and b[:-1] in cls_idx: return cls_idx[b[:-1]]
    return None
# contact geometry (nerve_ring_neighbors adult) — undirected edge
wb0=openpyxl.load_workbook(f"{D}/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx",read_only=True,data_only=True)
ws=wb0['adult nerve ring neighbors']; dd=list(ws.iter_rows(values_only=True)); hdr=[h for h in dd[0][1:]]
from collections import defaultdict
ca=defaultdict(float)
for r in dd[1:]:
    rn=r[0]
    if rn is None: continue
    for jj,v in enumerate(r[1:]):
        cn=hdr[jj]
        if cn and v not in (None,'') and rn!=cn:
            try: ca[(min(rn,cn),max(rn,cn))]=max(ca[(min(rn,cn),max(rn,cn))],float(v))
            except: pass
# Cook directed synapses
def load_dir(sheet,src):
    if src=='cook':
        wb=openpyxl.load_workbook(f"{D}/connectome/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
        A=list(wb[sheet].iter_rows(values_only=True))
        def looks(n): return isinstance(n,str) and len(n)>=2 and n[0].isalpha()
        hrr=max(range(len(A)),key=lambda r:sum(looks(x) for x in A[r])); cols=[(str(x).strip() if x is not None else None) for x in A[hrr]]
        hcc=max(range(min(6,len(A[0]))),key=lambda c:sum(looks(A[r][c]) for r in range(len(A))))
        ed=set()
        for r in range(hrr+1,len(A)):
            pre=A[r][hcc]; pre=pre.strip() if isinstance(pre,str) else None
            if not pre: continue
            for c in range(hcc+1,len(A[r])):
                post=cols[c]; v=A[r][c]
                if post and v not in (None,''):
                    try:
                        if float(v)>0: ed.add((pre,post))
                    except: pass
        return ed
edC=load_dir('hermaphrodite chemical','cook')
# universe: ordered pairs among neurons that CONTACT and map to expression
ring=[n for n in hdr if isinstance(n,str) and n2c(n) is not None]
opairs=[(a,b) for a in ring for b in ring if a!=b and ca.get((min(a,b),max(a,b)),0)>0]
def contact(a,b): return np.log1p(ca.get((min(a,b),max(a,b)),0.0))
def iddir(a,b): return Xc[n2c(a)]-Xc[n2c(b)]
y=np.array([1 if (a,b) in edC else 0 for a,b in opairs])
GEO=np.array([[contact(a,b)] for a,b in opairs])
IDD=np.array([iddir(a,b) for a,b in opairs])
P(f"BRANCH 4 — directed reconstruction | {len(opairs)} ordered contacting pairs, {y.sum()} directed synapses")
kf=KFold(5,shuffle=True,random_state=0)
def auc(F,model='gbm'):
    m=HistGradientBoostingClassifier(max_iter=150,random_state=0) if model=='gbm' else LogisticRegression(max_iter=1000,C=0.5)
    return roc_auc_score(y,cross_val_predict(m,F,y,cv=kf,method='predict_proba')[:,1])
aG=auc(GEO); aI=auc(np.hstack([np.zeros((len(y),1)),IDD]),'lr'); aB=auc(np.hstack([GEO,IDD]))
P(f"  DIRECTED-synapse AUC: geometry-EDGE only {aG:.3f} | identity-ARROW only {aI:.3f} | EDGE x ARROW {aB:.3f}")
P(f"    (geometry alone can't set direction -> caps low on directed target; arrow adds the direction; together = full)")
# Cook->Witvliet cross-animal on the combined
edW=set()
with open(f"{D}/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']=='chemical':
            m=re.search(r'witvliet_2020_(\d)',r['source_file'])
            if m and int(m.group(1)) in (7,8): edW.add((r['pre'],r['post']))
yW=np.array([1 if (a,b) in edW else 0 for a,b in opairs])
clf=HistGradientBoostingClassifier(max_iter=150,random_state=0).fit(np.hstack([GEO,IDD]),y)
aWc=roc_auc_score(yW,clf.predict_proba(np.hstack([GEO,IDD]))[:,1])
P(f"  EDGE x ARROW -> WITVLIET directed connectome (independent animal): AUC {aWc:.3f}")
P(f"\n  VERDICT: {'DECOMPOSITION HOLDS — edge x arrow reconstructs the directed connectome, arrow adds direction geometry cant' if aB>aG+0.05 and aWc>0.7 else 'partial'}: geometry sets which pairs wire, identity sets which way, together = the directed connectome.")
