"""FRONTIER 2 (P2) — is the identity-RANK downstream of the HOMEOBOX terminal-selector code? Hobert: neuron identity
= a combinatorial homeobox code. The arrow is a whole-transcriptome identity scalar s. Question: can homeobox-TF
expression alone reconstruct the rank s — and does it do so BETTER than random same-size gene panels (i.e. is the rank
specifically a homeobox-read identity, or just generic transcriptome)? Tests: (1) CV R2 predicting s from homeobox
expression; (2) vs random same-size non-homeobox panels (the GRN-branch killer control); (3) how few homeobox genes
suffice (sparse); (4) which homeobox genes top-load on the rank.
"""
import warnings,sys,openpyxl; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeCV, Lasso
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score
def P(*a): print(*a); sys.stdout.flush()
RNG=np.random.default_rng(0); D="../data/celegans/data"
cg=pd.read_csv(f"{D}/expression/cengen/thresholded/021821_liberal_threshold1.csv")
classes=list(cg.columns[3:]); gname=cg["gene_name"].astype(str).str.lower().values
Xall=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values.T>0
br=Xall.mean(0); mask=(br>0.05)&(br<0.95); Xc=Xall[:,mask].astype(float); gmask=gname[mask]; cls_idx={c:k for k,c in enumerate(classes)}
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
# identity rank from chemical
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
neur=sorted({a for a,b in chem}|{b for a,b in chem}); neur=[n for n in neur if n in cls_idx]
s=np.array([float(Xc[cls_idx[n]]@w) for n in neur]); Xn=np.array([Xc[cls_idx[n]] for n in neur])
# homeobox set (Reilly 2020: ceh-* + curated non-ceh)
NONCEH=["unc-4","unc-30","unc-39","unc-42","unc-62","unc-86","lin-11","lin-39","lim-4","lim-6","lim-7",
        "mec-3","mab-5","egl-5","php-3","nob-1","vab-7","vab-15","ttx-1","ttx-3","cog-1","alr-1","tab-1",
        "dsc-1","pha-2","dve-1","pros-1","zfh-2","zag-1","hmbx-1","psa-3","eyg-1"]
hb=np.array([g.startswith("ceh-") or g in NONCEH for g in gmask])
Xhb=Xn[:,hb]; hbgenes=gmask[hb]
# keep variably-expressed homeobox among these neurons
vv=(Xhb.mean(0)>0.03)&(Xhb.mean(0)<0.97); Xhb=Xhb[:,vv]; hbgenes=hbgenes[vv]
P(f"FRONTIER 2 (P2) — {len(neur)} neurons | homeobox genes used: {Xhb.shape[1]}")
kf=KFold(5,shuffle=True,random_state=0)
def cvr2(Xm):
    pred=cross_val_predict(RidgeCV(alphas=np.logspace(-2,3,20)),Xm,s,cv=kf)
    return r2_score(s,pred)
r2_hb=cvr2(Xhb)
# full transcriptome ceiling
r2_full=cvr2(Xn)
P(f"\n  (1) predict identity-rank s from HOMEOBOX expression: CV R2 = {r2_hb:.3f}")
P(f"      (full-transcriptome ceiling CV R2 = {r2_full:.3f})")
# (2) random same-size non-homeobox panels
nonhb_idx=np.where(~hb)[0]; k=Xhb.shape[1]; rand=[]
for _ in range(200):
    cols_=RNG.choice(nonhb_idx,k,replace=False); rand.append(cvr2(Xn[:,cols_]))
rand=np.array(rand); z=(r2_hb-rand.mean())/rand.std()
P(f"  (2) random same-size ({k}) non-homeobox panels: R2 {rand.mean():.3f}±{rand.std():.3f} -> homeobox z={z:+.1f} ({'homeobox-SPECIFIC' if z>2 else 'NOT better than random panel = generic identity'})")
# (3) sparse: how few homeobox genes (Lasso path)
from sklearn.linear_model import LassoCV
las=LassoCV(cv=5,random_state=0,max_iter=5000).fit(Xhb,s)
nnz=int((np.abs(las.coef_)>1e-6).sum())
P(f"  (3) sparse Lasso: {nnz} homeobox genes retained, CV-ish R2 {r2_score(s,las.predict(Xhb)):.3f} (in-sample)")
# (4) top homeobox predictors (ridge coef on standardized)
from sklearn.preprocessing import StandardScaler
Xs=StandardScaler().fit_transform(Xhb); coef=RidgeCV(alphas=np.logspace(-2,3,20)).fit(Xs,s).coef_
order=np.argsort(-np.abs(coef))[:12]
P(f"  (4) top homeobox loadings on the rank: "+", ".join(f"{hbgenes[i]}({coef[i]:+.2f})" for i in order))
P(f"\n  VERDICT: {'the identity-rank is substantially a HOMEOBOX-read code (R2=%.2f, z=%.1f>2 vs random panels) — the terminal-selector homeobox combination that specifies neuron identity also sets its position on the wiring-direction axis'%(r2_hb,z) if z>2 and r2_hb>0.15 else ('homeobox predicts the rank (R2=%.2f) but NOT better than a random same-size gene panel (z=%.1f) — the rank is GENERIC whole-cell identity, not specifically homeobox-encoded'%(r2_hb,z))}.")
