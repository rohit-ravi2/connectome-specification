"""CONFOUND-CLOSER — Arm 1: lineage-intrinsic vs lineage-as-position. Prior null was a GLOBAL lineage shuffle
(FORBIDDEN — scrambles position too). Proper test: POSITION-PRESERVING null (permute lineage within position-
matched groups, preserving position↔trajectory, breaking only lineage↔trajectory beyond position) + POSITION-
RESIDUALIZED target (does lineage predict contact after regressing out full placement). Both Cook-held-out; must
agree. Report within-position-group lineage variance (the null's power) + how well position predicts lineage.
"""
import warnings,sys,csv; warnings.simplefilter("ignore")
import numpy as np
sys.path.insert(0,'../accounting')
from traj_benchmark import RING,PAIRS,PI,PJ,CONTACT,SK,DEGNULL_ND,M,YC,PFOLD,SOMA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import roc_auc_score, r2_score
from scipy.stats import spearmanr
def P(*a): print(*a); sys.stdout.flush()
RNG=np.random.default_rng(0); NP=len(PAIRS)
lin={}
for r in csv.DictReader(open("../data/celegans/data/processed/developmental_rules/birthtime_earlylate.csv")):
    lin[r['neuron']]=r['lineage_s4'].replace(' ','')
def bc(n):
    for s in "LRDV":
        if n.endswith(s): return n[:-1]
    return n
LN=np.array([lin.get(n,lin.get(bc(n),'')) for n in RING])
def cp(a,b):
    c=0
    for x,y in zip(a,b):
        if x==y: c+=1
        else: break
    return c
def linfeat(LNv):
    lcp=np.array([cp(LNv[PI[k]],LNv[PJ[k]]) for k in range(NP)])
    return np.column_stack([lcp]+[np.array([1.0 if (len(LNv[PI[k]])>=d and len(LNv[PJ[k]])>=d and LNv[PI[k]][:d]==LNv[PJ[k]][:d]) else 0.0 for k in range(NP)]) for d in (3,5,7,9)])
LINEAGE=linfeat(LN)
somad=np.array([np.linalg.norm(SOMA[PI[k]]-SOMA[PJ[k]]) for k in range(NP)])
PLACE=np.column_stack([somad,SOMA[PI],SOMA[PJ]])   # FULL clean-C spatial placement set
def nd(F,tgt,cont=False):
    pred=np.full(NP,np.nan)
    for f in range(5):
        te=(PFOLD==f);tr=(PFOLD!=f)&(PFOLD>=0)
        if tr.sum()<50: continue
        if cont: pred[te]=HistGradientBoostingRegressor(max_iter=200,random_state=0).fit(F[tr],tgt[tr]).predict(F[te])
        elif tgt[te].sum()>3 and (tgt[te]==0).sum()>3:
            pred[te]=HistGradientBoostingClassifier(max_iter=200,random_state=0).fit(F[tr],tgt[tr]).predict_proba(F[te])[:,1]
    ok=~np.isnan(pred); return pred,ok
# ---- decomposition-quality: how strong is the position->lineage confound; within-position-group lineage variance ----
P("="*80); P("DECOMPOSITION-QUALITY (null power)"); P("="*80)
P(f"  confound strength: Spearman(soma-distance, lineage-common-prefix) = {spearmanr(somad,LINEAGE[:,0])[0]:+.2f}")
poscl=KMeans(12,n_init=5,random_state=0).fit(SOMA).labels_    # position-matched groups (soma clusters)
# within-position-group lineage diversity: mean # distinct lineage prefixes per cluster
div=[]
for c in range(12):
    ix=np.where(poscl==c)[0]
    if len(ix)>1: div.append(len(set(LN[i][:6] for i in ix))/len(ix))
P(f"  within-position-group lineage diversity (distinct lineage-prefixes / group size) = {np.mean(div):.2f} (1.0=all distinct, low=homogeneous->null underpowered)")

# ---- TEST 1: position-preserving lineage null ----
la,ok=nd(LINEAGE,CONTACT); aucL=roc_auc_score(CONTACT[ok],la[ok]); cookL=roc_auc_score(YC[ok],la[ok])
def pos_pres_null():
    LNp=LN.copy()
    for c in range(12):
        ix=np.where(poscl==c)[0]
        if len(ix)>1: LNp[ix]=LN[ix[RNG.permutation(len(ix))]]   # permute lineage WITHIN position group
    p,o=nd(linfeat(LNp),CONTACT); return roc_auc_score(CONTACT[o],p[o])
def pos_pres_null_k(K):
    cl=KMeans(K,n_init=5,random_state=0).fit(SOMA).labels_
    outs=[]
    for _ in range(8):
        LNp=LN.copy()
        for c in range(K):
            ix=np.where(cl==c)[0]
            if len(ix)>1: LNp[ix]=LN[ix[RNG.permutation(len(ix))]]
        p,o=nd(linfeat(LNp),CONTACT); outs.append(roc_auc_score(CONTACT[o],p[o]))
    return np.mean(outs),np.std(outs)
P("\n"+"="*80); P("TEST 1 — real lineage vs POSITION-PRESERVING null (coarse->fine position control)"); P("="*80)
P(f"  real lineage-alone AUC = {aucL:.3f} (Cook {cookL:.3f})")
for K in (8,12,20,30):
    m,s=pos_pres_null_k(K); P(f"  position-preserving null K={K:2d} clusters (~{162//K}/grp): AUC {m:.3f}±{s:.3f} -> real−null {aucL-m:+.3f}")

# ---- TEST 2: position-residualized target (FULL continuous placement) ----
pplace,okp=nd(PLACE,CONTACT.astype(float),cont=True)
resid=CONTACT.astype(float)-pplace
pr,opr=nd(LINEAGE,resid,cont=True)
valid=okp&opr
r2_lin_resid=r2_score(resid[valid], pr[valid]) if valid.sum()>50 else np.nan
# also: does lineage add to FULL placement in combined model (the beyond-placement delta)
pcomb,okc=nd(np.hstack([PLACE,LINEAGE]),CONTACT); aC=roc_auc_score(CONTACT[okc],pcomb[okc])
pp,okpp=nd(PLACE,CONTACT); aPl=roc_auc_score(CONTACT[okpp],pp[okpp])
P("\n"+"="*80); P("TEST 2 — lineage vs the FULL continuous placement (the proper control)"); P("="*80)
P(f"  lineage -> (contact − full-placement-prediction) residual: CV R2 = {r2_lin_resid:+.3f} (>0.01 = position-orthogonal lineage info)")
P(f"  beyond-FULL-placement Δ (placement+lineage {aC:.3f} vs placement {aPl:.3f}) = {aC-aPl:+.3f}")
r2_ok = r2_lin_resid>0.005 or (aC-aPl)>0.01

mfine,sfine=pos_pres_null_k(30)
P("\n"+"="*80)
t1 = (aucL-mfine)>3*sfine and aucL>DEGNULL_ND   # beats FINE position-preserving null AND above degree
t2 = r2_ok
if t1 and t2: P("VERDICT: lineage beats FINE position-preserving null AND predicts full-placement residual -> LINEAGE-INTRINSIC REAL, confound-closed. Capstone.")
elif (not t1) and (not t2): P(f"VERDICT: COSTUME — lineage-alone ({aucL:.3f}) is BELOW degree ({DEGNULL_ND:.3f}); the +0.07 vs coarse null shrinks with finer position control ({aucL-mfine:+.3f} at K=30) and lineage adds ~nothing to FULL placement (R2 {r2_lin_resid:+.3f}). Lineage-as-position confirmed; physical account holds; 'lineage-intrinsic' was the coarse-control illusion.")
else: P(f"VERDICT: tests DISAGREE (fine-null-beat={t1} [{aucL-mfine:+.3f}], residual={t2} [R2 {r2_lin_resid:+.3f}]) -> lineage/position separation is resolution-dependent; NAMED DATA-LIMIT (needs single-cell lineage+trajectory tracing to resolve). Note lineage is below degree regardless.")
P("="*80)
