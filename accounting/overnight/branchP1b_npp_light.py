"""FRONTIER 1 (P1b) — lightweight: degree-product structure of the NPP net + specific-structure delta (subsampled)
+ cheap correlations. Answers: is the neuropeptide network a MASS-ACTION degree/repertoire-product field, or does
specific identity add? Is NPP broadcast/receive aligned with the chemical sensory->motor rank?
"""
import warnings,sys,openpyxl; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
def P(*a): print(*a); sys.stdout.flush()
RNG=np.random.default_rng(0); D="../data/celegans/data"
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
Fc=[];yy=[]
for a,b in asym:
    d=Xc[cls_idx[a]]-Xc[cls_idx[b]]; Fc+=[d,-d]; yy+=[1,0]
w=LogisticRegression(max_iter=1000,C=0.5).fit(np.array(Fc),np.array(yy)).coef_[0]
srank={c:float(Xc[cls_idx[c]]@w) for c in classes if c in cls_idx}
npp=np.array([g.startswith(("flp-","nlp-","ins-")) for g in gname])
npp_count={c:int(Xall[cls_idx[c],npp].sum()) for c in classes if c in cls_idx}
df=pd.read_csv(f"{D}/ripollsanchez2023/01022024_neuropeptide_connectome_long_range_model.csv",index_col=0)
rows=list(df.index); colns=list(df.columns); Wm=df.values.astype(float)
ri=[n2c(r) for r in rows]; ci=[n2c(c) for c in colns]
keepr=[i for i,x in enumerate(ri) if x in srank]; keepc=[j for j,x in enumerate(ci) if x in srank]
Rn=[ri[i] for i in keepr]; Cn=[ci[j] for j in keepc]; W=Wm[np.ix_(keepr,keepc)]
Wb=(W>0).astype(int); n=len(Rn); m=len(Cn)
out_i=Wb.sum(1); in_j=Wb.sum(0)
P(f"FRONTIER 1 (P1b) — NPP long-range {n}x{m} | edges {int(Wb.sum())} density {Wb.mean():.2f}")
# feature matrices, VECTORIZED, subsample pairs
allpairs=np.array([(i,j) for i in range(n) for j in range(m) if Rn[i]!=Cn[j]])
sub=allpairs[RNG.choice(len(allpairs),min(9000,len(allpairs)),replace=False)]
Xe=np.array([Xc[cls_idx[Rn[i]]] for i in range(n)]); Xf2=np.array([Xc[cls_idx[Cn[j]]] for j in range(m)])
srk=np.array([srank[c] for c in Rn]); sck=np.array([srank[c] for c in Cn])
y=np.array([Wb[i,j] for i,j in sub])
loi=np.log1p(out_i); lij=np.log1p(in_j)
Xdeg=np.array([[loi[i],lij[j]] for i,j in sub])
simv=np.array([float(Xe[i]@Xf2[j]) for i,j in sub])
gapv=np.array([abs(srk[i]-sck[j]) for i,j in sub])
Xspec=np.c_[Xdeg,simv,gapv]
kf=KFold(5,shuffle=True,random_state=0)
def au(Xm): return roc_auc_score(y,cross_val_predict(LogisticRegression(max_iter=500),Xm,y,cv=kf,method='predict_proba')[:,1])
adeg=au(Xdeg); aspec=au(Xspec)
P(f"\n  (1) degree/gravity (out_i,in_j)->edge AUC {adeg:.3f}  (subsampled {len(sub)} pairs)")
P(f"  (2) +identity-sim+chem-rank-gap AUC {aspec:.3f} (Δ {aspec-adeg:+.3f})")
so=spearmanr(srk,out_i)[0]; si=spearmanr(sck,in_j)[0]
P(f"\n  (3) chemical rank s vs NPP OUT-degree r={so:+.3f} | vs IN-degree r={si:+.3f}  (s>0=sensory-source)")
ro=spearmanr([npp_count[c] for c in Rn],out_i)[0]
P(f"  (4) NPP out-degree vs #NPP genes (flp/nlp/ins repertoire) r={ro:+.3f}")
P(f"\n  VERDICT: {'the NPP net is a MASS-ACTION degree/repertoire-product field (edge AUC %.2f from degrees, specific identity adds Δ%.3f) — organized by broadcast/receive REPERTOIRE not a wiring address code'%(adeg,aspec-adeg) if adeg>0.8 and aspec-adeg<0.03 else 'NPP net has specific structure beyond degree-product (Δ%.3f)'%(aspec-adeg)}.")
