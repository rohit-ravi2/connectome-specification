"""SPEC 1 — does functional CO-ACTIVITY predict NPP coupling BEYOND degree? Partial-correlation ladder:
raw -> control NPP out/in-degree -> control NPP + CHEMICAL degree. Also the chemical-layer relationship (A3's actual
residual) and the reverse direction. Report partial Spearman at each stage; name which control kills it.
"""
import warnings,sys,openpyxl; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
def P(*a): print(*a); sys.stdout.flush()
D="../data/celegans/data"
cg=pd.read_csv(f"{D}/expression/cengen/thresholded/021821_liberal_threshold1.csv")
classes=list(cg.columns[3:]); cls_idx={c:k for k,c in enumerate(classes)}
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
# ---- Randi matrix M[i,j] = response of j when i stimulated ----
rand_rows=[]
with open(f"{D}/randi2023/randi_funconn_wildty.tsv") as f:
    for line in f:
        p=line.split('\t')
        if len(p)>=3:
            try: rand_rows.append((p[0].strip(),p[1].strip(),float(p[2])))
            except: pass
rn=sorted({a for a,b,v in rand_rows}|{b for a,b,v in rand_rows}); ri={n:i for i,n in enumerate(rn)}; NR=len(rn)
M=np.zeros((NR,NR))
for a,b,v in rand_rows: M[ri[a],ri[b]]=v
def coact_matrix(Mat,mode):
    X=Mat.copy() if mode=='resp' else Mat.T.copy(); np.fill_diagonal(X,0.0)
    return np.nan_to_num(np.corrcoef(X.T))
Ccoact=0.5*(coact_matrix(M,'resp')+coact_matrix(M,'infl'))
# ---- NPP long-range ----
npp=pd.read_csv(f"{D}/ripollsanchez2023/01022024_neuropeptide_connectome_long_range_model.csv",index_col=0)
nrows=list(npp.index); ncols=list(npp.columns); NW=npp.values.astype(float)
# ---- Cook chemical degree ----
wb=openpyxl.load_workbook(f"{D}/connectome/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
A=list(wb['hermaphrodite chemical'].iter_rows(values_only=True))
def looks(x): return isinstance(x,str) and len(x)>=2 and x[0].isalpha()
hr=max(range(len(A)),key=lambda r:sum(looks(x) for x in A[r])); hc=max(range(min(6,len(A[0]))),key=lambda c:sum(looks(A[r][c]) for r in range(len(A))))
cols=[(str(x).strip() if x is not None else None) for x in A[hr]]
chemdeg=pd.Series(dtype=float)
cd={}
for r in range(hr+1,len(A)):
    pre=A[r][hc]; pre=pre.strip() if isinstance(pre,str) else None
    if not pre: continue
    for c in range(hc+1,len(A[r])):
        post=cols[c]; v=A[r][c]
        if post and v not in (None,''):
            try:
                if float(v)>0: cd[pre]=cd.get(pre,0)+1; cd[post]=cd.get(post,0)+1
            except: pass
# ---- build ordered-pair table over Randi neurons that map to NPP + have chem degree ----
def npp_idx(name):
    if name in nrows: return nrows.index(name)
    return None
rows=[]
nrows_set={r:i for i,r in enumerate(nrows)}; ncols_set={c:i for i,c in enumerate(ncols)}
npp_out={r: (NW[i]>0).sum() for i,r in enumerate(nrows)}
npp_in ={c: (NW[:,j]>0).sum() for j,c in enumerate(ncols)}
for i,a in enumerate(rn):
    ia=nrows_set.get(a)
    for j,b in enumerate(rn):
        if i==j: continue
        jb=ncols_set.get(b)
        if ia is None or jb is None: continue
        w=NW[ia,jb]
        rows.append((a,b,Ccoact[i,j],w,npp_out.get(a,0),npp_in.get(b,0),cd.get(a,0),cd.get(b,0)))
df=pd.DataFrame(rows,columns=['a','b','coact','npp_w','npp_out','npp_in','chem_a','chem_b'])
df['npp_edge']=(df.npp_w>0).astype(int)
P(f"SPEC1 — {len(df)} ordered pairs over {len(set(df.a))} Randi∩NPP neurons; NPP-edge base rate {df.npp_edge.mean():.3f}")
def partial_spear(y,x,Z):
    # partial Spearman: rank all, regress out Z (with intercept), correlate residuals
    yr=rankdata(y); xr=rankdata(x)
    if Z is None or Z.shape[1]==0:
        return spearmanr(y,x)[0]
    Zr=np.column_stack([rankdata(Z[:,k]) for k in range(Z.shape[1])])
    Zr=np.column_stack([np.ones(len(yr)),Zr])
    ry=yr-Zr@np.linalg.lstsq(Zr,yr,rcond=None)[0]
    rx=xr-Zr@np.linalg.lstsq(Zr,xr,rcond=None)[0]
    return np.corrcoef(ry,rx)[0,1]
nppZ=df[['npp_out','npp_in']].values
allZ=df[['npp_out','npp_in','chem_a','chem_b']].values
P("\n=== co-activity vs NPP WEIGHT — partial-correlation ladder ===")
P(f"  raw spearman                         : {partial_spear(df.coact.values,df.npp_w.values,None):+.4f}")
P(f"  | NPP out+in degree                  : {partial_spear(df.coact.values,df.npp_w.values,nppZ):+.4f}")
P(f"  | NPP + CHEMICAL degree              : {partial_spear(df.coact.values,df.npp_w.values,allZ):+.4f}")
P("\n=== co-activity vs NPP EDGE (binary) — ladder ===")
P(f"  raw                                  : {partial_spear(df.coact.values,df.npp_edge.values,None):+.4f}")
P(f"  | NPP out+in degree                  : {partial_spear(df.coact.values,df.npp_edge.values,nppZ):+.4f}")
P(f"  | NPP + CHEMICAL degree              : {partial_spear(df.coact.values,df.npp_edge.values,allZ):+.4f}")
# reverse: symmetric for partial corr; report explicitly + a degree-product reference
P(f"\n  degree-product (npp_out*npp_in) vs npp_w spearman = {spearmanr(df.npp_out*df.npp_in, df.npp_w)[0]:+.3f}  (the confound magnitude)")
P(f"  coact vs degree-product spearman = {spearmanr(df.coact, df.npp_out*df.npp_in)[0]:+.3f}  (how much coact IS degree)")
# ---- chemical layer (A3's actual residual) ----
# chem-edge presence per pair
chempairs=set()
for r in range(hr+1,len(A)):
    pre=A[r][hc]; pre=pre.strip() if isinstance(pre,str) else None
    if not pre: continue
    for c in range(hc+1,len(A[r])):
        post=cols[c]; v=A[r][c]
        if post and v not in (None,''):
            try:
                if float(v)>0: chempairs.add((pre,post))
            except: pass
df['chem_edge']=[1 if (a,b) in chempairs else 0 for a,b in zip(df.a,df.b)]
P("\n=== co-activity vs CHEMICAL EDGE (A3's actual positive residual) — ladder ===")
P(f"  chem-edge base rate {df.chem_edge.mean():.3f}")
P(f"  raw                                  : {partial_spear(df.coact.values,df.chem_edge.values,None):+.4f}")
P(f"  | chemical degree (a,b)              : {partial_spear(df.coact.values,df.chem_edge.values,df[['chem_a','chem_b']].values):+.4f}")
P(f"  | chem + NPP degree                  : {partial_spear(df.coact.values,df.chem_edge.values,allZ):+.4f}")
P("\nNOTE: partial Spearman is symmetric, so 'coact->NPP beyond degree' and 'NPP->coact beyond degree' are the SAME coefficient (bidirectional by construction). Read which control (if any) drives each coefficient to ~0.")
