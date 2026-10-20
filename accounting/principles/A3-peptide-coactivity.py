"""A3 — Is the NEUROPEPTIDE network organized by functional CO-ACTIVITY (Randi)?
Co-activity = similarity of response profiles in the Randi functional connectome.
Compare co-activity vs identity-rank vs degree as predictors of NPP edge weight;
check chemical-synapse prediction; find peptide-coupled/chemically-distant pairs;
test NPP-expression-similarity -> co-activity self-reinforcement.
"""
import warnings,sys,openpyxl; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
D="../data/celegans/data"
OUT="../accounting/principles"

# ---------- CeNGEN expression + n2c ----------
cg=pd.read_csv(f"{D}/expression/cengen/thresholded/021821_liberal_threshold1.csv")
classes=list(cg.columns[3:]); gname=cg["gene_name"].astype(str).str.lower().values
Xall=(cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values.T>0)
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

# ---------- Randi functional matrix: M[i,j]= response of j when i stimulated ----------
r=pd.read_csv(f"{D}/randi2023/randi_funconn_wildty.tsv",sep='\t',header=None,names=['i','j','v'])
rn=sorted(set(r['i'])|set(r['j'])); ridx={c:k for k,c in enumerate(rn)}; NR=len(rn)
M=np.zeros((NR,NR))
for a,b,v in zip(r['i'],r['j'],r['v']): M[ridx[a],ridx[b]]=v
P(f"Randi matrix {NR}x{NR}, nonzero {np.mean(M!=0):.3f}, signed range [{M.min():.0f},{M.max():.0f}]")

# ---------- CO-ACTIVITY: correlate RESPONSE profiles (columns) across third-neuron stimulations ----------
# response profile of neuron a = column a = how a responds to each stimulation. exclude self-rows.
def coact_matrix(Mat, mode='resp'):
    if mode=='resp': X=Mat.copy()            # columns = responders; corr columns
    else: X=Mat.T.copy()                     # rows = influence; corr rows-as-cols
    # zero out diagonal (self stim/self resp) to avoid trivial
    np.fill_diagonal(X,0.0)
    C=np.corrcoef(X.T)  # correlate columns
    C=np.nan_to_num(C)
    return C
Cresp=coact_matrix(M,'resp')   # co-response (similar responders)
Cinfl=coact_matrix(M,'infl')   # co-influence (similar outgoing effect)
# primary co-activity = average of both (functional co-embedding)
Ccoact=0.5*(Cresp+Cinfl)
P(f"co-activity(resp) mean off-diag {Cresp[np.triu_indices(NR,1)].mean():+.3f}, "
  f"co-influence mean {Cinfl[np.triu_indices(NR,1)].mean():+.3f}")

# ---------- identity rank s (chemical asymmetric axis) ----------
wb=openpyxl.load_workbook(f"{D}/connectome/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
A=list(wb['hermaphrodite chemical'].iter_rows(values_only=True))
def looks(n): return isinstance(n,str) and len(n)>=2 and n[0].isalpha()
hr=max(range(len(A)),key=lambda r_:sum(looks(x) for x in A[r_])); cols=[(str(x).strip() if x is not None else None) for x in A[hr]]
hc=max(range(min(6,len(A[0]))),key=lambda c:sum(looks(A[rr][c]) for rr in range(len(A))))
chem=set()
for rr in range(hr+1,len(A)):
    cpre=n2c(A[rr][hc])
    if not cpre: continue
    for c in range(hc+1,len(A[rr])):
        cpost=n2c(cols[c]); v=A[rr][c]
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

# ---------- NPP connectome (mid-range primary) ----------
def load_npp(model):
    df=pd.read_csv(f"{D}/ripollsanchez2023/01022024_neuropeptide_connectome_{model}_range_model.csv",index_col=0)
    return df
npp_df=load_npp('mid')
npp_names=list(npp_df.index)  # cell-level names
Wnpp_full=npp_df.values.astype(float)
npp_idx={nm:k for k,nm in enumerate(npp_names)}

# ---------- align Randi cells <-> NPP cells (both cell-level) ----------
common=[c for c in rn if c in npp_idx]
P(f"common Randi&NPP cells: {len(common)}")
ci_r=[ridx[c] for c in common]; ci_n=[npp_idx[c] for c in common]
CO=Ccoact[np.ix_(ci_r,ci_r)]         # co-activity among common cells
Wn=Wnpp_full[np.ix_(ci_n,ci_n)]      # NPP directed weight among common cells
Wn_sym=np.maximum(Wn,Wn.T)           # symmetric NPP coupling
n=len(common)

# NPP degrees (for confound)
out_i=(Wn>0).sum(1); in_j=(Wn>0).sum(0); deg=(Wn_sym>0).sum(1)

# ---------- build pair table (unordered i<j) ----------
iu,ju=np.triu_indices(n,1)
co=CO[iu,ju]
wnpp=Wn_sym[iu,ju]
wnpp_bin=(wnpp>0).astype(int)
degprod=np.log1p(deg[iu])+np.log1p(deg[ju])
# identity distance (class-level rank gap) & expression similarity
def cls_of(c):
    cc=n2c(c); return cc if cc in cls_idx else None
scls=[cls_of(c) for c in common]
rankgap=np.array([abs(srank.get(scls[i],np.nan)-srank.get(scls[j],np.nan)) for i,j in zip(iu,ju)])
valid_rank=~np.isnan(rankgap)

P("\n=== (2) predict NPP coupling: co-activity vs identity-rank vs degree ===")
P(f"pairs {len(co)}, NPP-edge base rate {wnpp_bin.mean():.3f}")
def sp(x,y,mask=None):
    if mask is not None: x,y=x[mask],y[mask]
    ok=np.isfinite(x)&np.isfinite(y)
    return spearmanr(x[ok],y[ok])[0]
P(f"  co-activity vs NPP-weight      spearman r={sp(co,wnpp):+.3f}")
P(f"  co-activity vs NPP-edge(bin)   spearman r={sp(co,wnpp_bin):+.3f}")
P(f"  identity-rankgap vs NPP-weight spearman r={sp(rankgap,wnpp):+.3f}  (neg=closer identity->more coupled)")
P(f"  degree-product vs NPP-weight   spearman r={sp(degprod,wnpp):+.3f}  (DEGREE CONFOUND)")
P(f"  co-activity vs degree-product  spearman r={sp(co,degprod):+.3f}  (how much co-activity IS degree)")

# partial: does co-activity add over degree for NPP edge? logistic CV AUC
kf=KFold(5,shuffle=True,random_state=0)
def auc(cols_):
    Xf=np.column_stack(cols_); ok=np.all(np.isfinite(Xf),1)
    return roc_auc_score(wnpp_bin[ok], cross_val_predict(LogisticRegression(max_iter=1000),Xf[ok],wnpp_bin[ok],cv=kf,method='predict_proba')[:,1])
P(f"\n  AUC NPP-edge from degree-prod only     : {auc([degprod]):.3f}")
P(f"  AUC from co-activity only              : {auc([co]):.3f}")
P(f"  AUC from identity-rankgap only         : {auc([np.nan_to_num(rankgap,nan=np.nanmedian(rankgap))]):.3f}")
P(f"  AUC degree+coactivity                  : {auc([degprod,co]):.3f}  (Δ over degree = coactivity add)")
P(f"  AUC degree+coactivity+rankgap          : {auc([degprod,co,np.nan_to_num(rankgap,nan=np.nanmedian(rankgap))]):.3f}")

# ---------- (3) co-activity predict CHEMICAL synapse presence ----------
# chemical edge among common cells (class-level presence -> map back)
chem_pair=np.zeros(len(co))
for k,(i,j) in enumerate(zip(iu,ju)):
    a,b=scls[i],scls[j]
    if a and b and ((a,b) in chem or (b,a) in chem): chem_pair[k]=1
P("\n=== (3) co-activity predict CHEMICAL synapse presence (expected weak-ish; but functional<-chem) ===")
P(f"  chem-edge base rate {chem_pair.mean():.3f}")
P(f"  co-activity vs chemical-edge   spearman r={sp(co,chem_pair):+.3f}")
P(f"  co-activity vs NPP-edge        spearman r={sp(co,wnpp_bin):+.3f}  (compare)")
P(f"  AUC chem-edge from co-activity : {auc.__self__ if False else roc_auc_score(chem_pair, cross_val_predict(LogisticRegression(max_iter=1000),co.reshape(-1,1),chem_pair,cv=kf,method='predict_proba')[:,1]):.3f}")

# ---------- (4) peptide-coupled but chemically/hierarchically DISTANT ----------
P("\n=== (4) STRONG NPP coupling + NO chemical synapse + LARGE identity gap ===")
score=wnpp.copy().astype(float)
cand=[]
for k,(i,j) in enumerate(zip(iu,ju)):
    if wnpp[k]>0 and chem_pair[k]==0 and np.isfinite(rankgap[k]):
        cand.append((wnpp[k], rankgap[k], co[k], common[i], common[j]))
cand.sort(key=lambda t:(-t[0],-t[1]))
P(f"  {len(cand)} pairs NPP-coupled & no-chem-synapse (of {int((wnpp>0).sum())} NPP edges)")
P(f"  top by NPP weight then rank-gap (weight, rankgap, coact, cellA, cellB):")
for t in cand[:15]: P(f"    w={t[0]:.0f} gap={t[1]:.2f} coact={t[2]:+.2f}  {t[3]}-{t[4]}")
# among NPP-coupled no-chem pairs, are they MORE or LESS co-active than NPP-coupled+chem?
mask_nc=(wnpp_bin==1)&(chem_pair==0); mask_c=(wnpp_bin==1)&(chem_pair==1)
P(f"  co-activity: NPP-coupled+chem {co[mask_c].mean():+.3f} (n={mask_c.sum()}) vs NPP-coupled+NOchem {co[mask_nc].mean():+.3f} (n={mask_nc.sum()})")

# ---------- (5) NPP EXPRESSION similarity -> co-activity? (self-reinforcing) ----------
# NPP expression profile per class: flp/nlp/ins ligands + npp GPCR receptors
lig=np.array([g.startswith(("flp-","nlp-","ins-")) for g in gname])
# receptor genes: npr-, gpcr candidates -> use names starting npr- or containing 'receptor'? use npr-/ *-r? keep npr- and frpr, dmsr, etc common NPP GPCR prefixes
recpref=("npr-","frpr-","dmsr-","nmur-","trhr-","seb-","aex-","egl-6","dop-","ser-","tyra-","octr-","gnrr-","ckr-","pdfr-")
rec=np.array([g.startswith(recpref) for g in gname])
nppgene=lig|rec
def npp_profile(c):
    cc=cls_of(c)
    if cc is None: return None
    return Xall[cls_idx[cc],nppgene].astype(float)
P("\n=== (5) NPP-expression similarity -> co-activity? (self-reinforcing) ===")
prof={c:npp_profile(c) for c in common}
valid=[c for c in common if prof[c] is not None and prof[c].sum()>0]
cpos={c:common.index(c) for c in valid}
def cos(a,b):
    da,db=prof[a],prof[b]; d=np.linalg.norm(da)*np.linalg.norm(db)
    return float(da@db/d) if d>0 else 0.0
nppsim=[];coa=[];degp=[];wnp=[]
for x in range(len(valid)):
    for y in range(x+1,len(valid)):
        a,b=valid[x],valid[y]; ia,ib=cpos[a],cpos[b]
        nppsim.append(cos(a,b)); coa.append(CO[ia,ib])
        degp.append(np.log1p(deg[ia])+np.log1p(deg[ib])); wnp.append(Wn_sym[ia,ib])
nppsim=np.array(nppsim);coa=np.array(coa);degp=np.array(degp);wnp=np.array(wnp)
P(f"  valid cells with NPP-expr {len(valid)}, pairs {len(nppsim)}")
P(f"  NPP-expr-sim vs co-activity   spearman r={sp(nppsim,coa):+.3f}")
P(f"  NPP-expr-sim vs NPP-coupling  spearman r={sp(nppsim,wnp):+.3f}  (expr drives NPP net by construction)")
P(f"  NPP-expr-sim vs degree-prod   spearman r={sp(nppsim,degp):+.3f}")
# high vs low NPP-expr-sim co-activity contrast
hi=nppsim>np.quantile(nppsim,0.8); lo=nppsim<np.quantile(nppsim,0.2)
P(f"  co-activity: top-20% NPP-expr-sim {coa[hi].mean():+.3f} vs bottom-20% {coa[lo].mean():+.3f}")

# ---------- robustness: co-activity vs NPP across the 3 range models ----------
P("\n=== robustness across NPP range models (co-activity vs NPP-weight spearman) ===")
for model in ['short','mid','long']:
    Wf=load_npp(model).values.astype(float)
    Wnm=Wf[np.ix_(ci_n,ci_n)]; Wsm=np.maximum(Wnm,Wnm.T)
    wv=Wsm[iu,ju]
    P(f"  {model:5s}: coact-vs-NPPweight r={sp(co,wv):+.3f} | degprod-vs-NPPweight r={sp((np.log1p((Wsm>0).sum(1))[iu]+np.log1p((Wsm>0).sum(1))[ju]),wv):+.3f} | edges {int((Wsm>0).sum()/2)}")
P("\nDONE")
