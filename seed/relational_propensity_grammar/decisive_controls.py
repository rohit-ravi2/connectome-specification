#!/usr/bin/env python3
"""Settle the verdict: is the relational dispersion move (0.539->0.599) a GENUINE grammar or an artifact?
Controls: (1) CALIBRATION reliability base vs relational; (2) HALO discrimination (theta=1 core vs 0.5 vs 0);
(3) COOK cross-pipeline context (kills W7~=W8 conservation re-description); (4) INTERPRETABLE rule directions
(does role-equivalence SUPPRESS, complementarity PROMOTE?) via leakage-safe cross-animal logistic."""
import csv, re, os, math
import numpy as np, openpyxl
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve

C="../data/celegans/data"
OUT=os.path.expanduser("../seed/data/processed/relational_propensity_grammar")
EDGES=f"{C}/connectome/witvliet2020/witvliet_combined_edges.csv"
CONTACT=f"{C}/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx"
COOK=os.path.expanduser("../seed/data/external/mechanistic_frontier/cook2019_adult/SI 2 Synapse adjacency matrices.xlsx")
NPZ="../data/tier5/connectome.npz"

animal_edges=defaultdict(set)
for r in csv.DictReader(open(EDGES)):
    if r['type']=='electrical': continue
    a=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1)); animal_edges[a].add((r['pre'],r['post']))
W7,W8=animal_edges[7],animal_edges[8]
wb=openpyxl.load_workbook(CONTACT,read_only=True,data_only=True); ws=wb['adult nerve ring neighbors']
rows=list(ws.iter_rows(values_only=True)); alln=[h for h in rows[0][1:]]; ix={n:i for i,n in enumerate(alln) if n}
M=len(alln); A=np.zeros((M,M))
for row in rows[1:]:
    rn=row[0]
    if rn in ix:
        for j,v in enumerate(row[1:]):
            if v not in (None,''):
                try:A[ix[rn],j]=float(v)
                except:pass
A=np.maximum(A,A.T); Bnb=(A>0).astype(float); deg=A.sum(1); degn=Bnb.sum(1)
d=np.load(NPZ,allow_pickle=True)
def ntn(x):
    for k,v in [("Acetyl","ACh"),("Glu","Glu"),("GABA","GABA"),("Dopamine","mono"),("Serotonin","mono"),("Octopamine","mono"),("Tyramine","mono")]:
        if k in x: return v
    return "Unk"
NT={a:ntn(str(b)) for a,b in zip([str(x) for x in d["names"]],[str(x) for x in d["nt_primary"]])}
pairs=[(alln[i],alln[j]) for i in range(M) for j in range(M) if i!=j and alln[i] and alln[j] and A[i,j]>0]
ntk=sorted(set((NT.get(a,'Unk'),NT.get(b,'Unk')) for a,b in pairs)); ki={k:i for i,k in enumerate(ntk)}
def feats(a,b):
    i,j=ix[a],ix[b]; cos=float(A[i]@A[j]/(np.linalg.norm(A[i])*np.linalg.norm(A[j])+1e-9))
    cn=int(((Bnb[i]>0)&(Bnb[j]>0)).sum()); oh=[0]*len(ntk); oh[ki[(NT.get(a,'Unk'),NT.get(b,'Unk'))]]=1
    return [math.log1p(A[i,j]),cos,cn,math.log1p(deg[i]),math.log1p(deg[j]),degn[i],degn[j]]+oh
X=np.array([feats(a,b) for a,b in pairs])
y7=np.array([1 if (a,b) in W7 else 0 for a,b in pairs]); y8=np.array([1 if (a,b) in W8 else 0 for a,b in pairs])
y_any=((y7|y8)>0).astype(int)
neu=sorted(set(a for a,b in pairs)|set(b for a,b in pairs))
def folds(seed=0):
    rng=np.random.default_rng(seed); f=np.array_split(rng.permutation(neu),5); return {n:i for i,ff in enumerate(f) for n in ff}
def ndcv(Xf,y,seed=0,proba=True,model='gb'):
    nf=folds(seed); p=np.full(len(y),np.nan)
    for k in range(5):
        te=np.array([nf[a]==k and nf[b]==k for a,b in pairs]); tr=np.array([nf[a]!=k and nf[b]!=k for a,b in pairs])
        if te.sum()<10 or y[tr].sum()<5: continue
        clf=make_pipeline(StandardScaler(with_mean=False),GradientBoostingClassifier(random_state=seed) if model=='gb' else LogisticRegression(max_iter=2000))
        clf.fit(Xf[tr],y[tr]); p[te]=clf.predict_proba(Xf[te])[:,1]
    return p
def relfeats(ctx,pairs):
    outn=defaultdict(set); inn=defaultdict(set)
    for u,v in ctx: outn[u].add(v); inn[v].add(u)
    return np.array([[len(inn[a]&inn[b]),len(outn[a]&outn[b]),len(outn[a]&inn[b]),len(outn[b]&inn[a])] for a,b in pairs],float)
def dispersion(p,mask): return np.abs(y7[mask]-y8[mask]).sum()/(2*p[mask]*(1-p[mask])).sum()

# ---- Cook chemical edges (cross-pipeline context) ----
def load_cook():
    try:
        wb=openpyxl.load_workbook(COOK,read_only=True,data_only=True)
        for sh in wb.sheetnames:
            ws=wb[sh]; rr=list(ws.iter_rows(values_only=True))
            hdr=[str(x) if x is not None else '' for x in rr[0][1:]]
            if sum(1 for h in hdr if h in ix)<20: continue
            E=set()
            for row in rr[1:]:
                pre=str(row[0]) if row[0] else ''
                if pre not in ix: continue
                for j,v in enumerate(row[1:]):
                    if j<len(hdr) and hdr[j] in ix and v not in (None,'',0):
                        try:
                            if float(v)>0: E.add((pre,hdr[j]))
                        except:pass
            if len(E)>200: return E,sh
    except Exception as e: print("cook load err",e)
    return None,None
cook,csheet=load_cook()
print(f"Cook edges: {len(cook) if cook else 'FAILED'} (sheet {csheet})")

# ============ (1) CALIBRATION: is the dispersion move genuine or sharpening artifact? ============
p_base=np.nanmean(np.vstack([ndcv(X,y7),ndcv(X,y8)]),axis=0)
p_rel=np.nanmean(np.vstack([ndcv(np.hstack([X,relfeats(W8,pairs)]),y7),ndcv(np.hstack([X,relfeats(W7,pairs)]),y8)]),axis=0)
mb=~np.isnan(p_base); mr=~np.isnan(p_rel); m=mb&mr
# reliability: bin p, compare to actual any-adult rate. Genuine => calibrated; artifact => overconfident
def cal_err(p,y,mask):
    frac,mean=calibration_curve(y[mask],p[mask],n_bins=10,strategy='quantile'); return float(np.mean(np.abs(frac-mean)))
ece_base=cal_err(p_base,y_any,mb); ece_rel=cal_err(p_rel,y_any,mr)
print(f"\n=== (1) CALIBRATION (lower=better calibrated) ===")
print(f"  base ECE={ece_base:.3f}  relational ECE={ece_rel:.3f}  (relational more-calibrated iff <=base)")
print(f"  dispersion base={dispersion(p_base,mb):.3f} relational={dispersion(p_rel,mr):.3f}")
# genuine-agreement test: does relational raise p on CONCORDANT(theta=1) & keep INTERMEDIATE on DISCORDANT(theta=0.5)?
theta=(y7+y8)/2.0
for lab,sel in [("core theta=1",theta==1),("HALO theta=0.5",theta==0.5),("absent theta=0",theta==0)]:
    s=sel&m
    print(f"  mean p | {lab:16s} base={p_base[s].mean():.3f} relational={p_rel[s].mean():.3f}  (n={s.sum()})")

# ============ (2) HALO discrimination: can relational separate halo(0.5) from core(1)? ============
core_or_halo=m&(theta>=0.5)
yc=(theta[core_or_halo]==1).astype(int)  # 1=core, 0=halo among present-in->=1
auc_base_halo=roc_auc_score(yc,p_base[core_or_halo]); auc_rel_halo=roc_auc_score(yc,p_rel[core_or_halo])
print(f"\n=== (2) HALO vs CORE discrimination (AUC; can model tell theta=1 from theta=0.5?) ===")
print(f"  base AUC={auc_base_halo:.3f}  relational AUC={auc_rel_halo:.3f}  (d={auc_rel_halo-auc_base_halo:+.3f})")

# ============ (3) COOK cross-pipeline: does DIFFERENT-pipeline context still predict Witvliet? ============
if cook:
    p_cookctx=np.nanmean(np.vstack([ndcv(np.hstack([X,relfeats(cook,pairs)]),y7),ndcv(np.hstack([X,relfeats(cook,pairs)]),y8)]),axis=0)
    mc=~np.isnan(p_cookctx)
    print(f"\n=== (3) COOK cross-pipeline context (kills W7~=W8 re-description) ===")
    print(f"  AUC base={roc_auc_score(y_any[mb],p_base[mb]):.3f}  +Cook-context={roc_auc_score(y_any[mc],p_cookctx[mc]):.3f}")
    print(f"  dispersion base={dispersion(p_base,mb):.3f}  +Cook-context={dispersion(p_cookctx,mc):.3f}")

# ============ (4) INTERPRETABLE RULE DIRECTIONS (leakage-safe cross-animal logistic) ============
# standardized logistic coefficients: sign of equivalence(shared_in/out) vs complementarity(feedforward)
Rw7=relfeats(W7,pairs)  # context to predict y8
sc=StandardScaler().fit(Rw7); Rs=sc.transform(Rw7)
lr=LogisticRegression(max_iter=2000).fit(np.hstack([StandardScaler(with_mean=False).fit_transform(X),Rs]),y8)
rc=lr.coef_[0][-4:]
print(f"\n=== (4) RULE DIRECTIONS (standardized logit coef, W7-context -> W8 edge) ===")
for nm2,cf in zip(["shared_inputs(equiv)","shared_outputs(equiv)","feedforward a->k->b(compl)","feedback b->k->a"],rc):
    print(f"  {nm2:32s} beta={cf:+.3f}  ({'PROMOTES' if cf>0 else 'SUPPRESSES'} edge)")

with open(f"{OUT}/decisive_controls.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["metric","base","relational_or_cook"])
    w.writerow(["ECE_calibration",round(ece_base,4),round(ece_rel,4)])
    w.writerow(["dispersion_ratio",round(dispersion(p_base,mb),4),round(dispersion(p_rel,mr),4)])
    w.writerow(["halo_vs_core_AUC",round(auc_base_halo,4),round(auc_rel_halo,4)])
    if cook: w.writerow(["cook_crosspipe_dispersion",round(dispersion(p_base,mb),4),round(dispersion(p_cookctx,mc),4)])
    for nm2,cf in zip(["beta_shared_in","beta_shared_out","beta_feedforward","beta_feedback"],rc): w.writerow([nm2,"",round(float(cf),4)])
print("\ndecisive controls done ->",OUT)
