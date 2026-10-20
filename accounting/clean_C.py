"""CLEAN-C QUANTIFICATION. Clean C = best PRE-PLACEMENT model (soma position from skeleton root, Sulston birth
time, AB-lineage relatedness) — NEVER adult arbor arrangement. Part 1: honest bulk fraction X (beyond degree,
Cook-held-out) + locate the misses. Every number here is CLEAN C. The pre-audit 0.868 was the ADULT-ARBOR leak.
"""
import warnings,sys,json,csv,numpy as np,pandas as pd; warnings.simplefilter("ignore")
sys.path.insert(0,'../accounting')
from bulk_benchmark import RING,PAIRS,PI,PJ,CONTACT,SK,neuron_disjoint_auc,DEGNULL_ND,M,AREA,YC,CADJ,CDEG,PFOLD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
NpP=len(PAIRS)
# ---- pre-placement features ----
def soma(n):
    st=SK[n].get('starts'); co=SK[n]['coords']
    key=str(st[0]) if st and str(st[0]) in co else list(co.keys())[0]
    return np.array(co[key],float)
SOMA=np.array([soma(n) for n in RING])
bt={};lin={}
for r in csv.DictReader(open("../seed/data/external/mechanistic_frontier/neuron_birth_times_sulston/neuron_birth_times.csv")):
    bt[r['neuron']]=float(r['birth_time_min'])
for r in csv.DictReader(open("../data/celegans/data/processed/developmental_rules/birthtime_earlylate.csv")):
    lin[r['neuron']]=r['lineage_s4'].replace(' ','')
def bc(n):
    for s in "LRDV":
        if n.endswith(s): return n[:-1]
    return n
def bof(n): return bt.get(n,bt.get(bc(n),np.nan))
def lof(n): return lin.get(n,lin.get(bc(n),''))
BT=np.array([bof(n) for n in RING]); LN=[lof(n) for n in RING]
def commonpref(a,b):
    c=0
    for x,y in zip(a,b):
        if x==y: c+=1
        else: break
    return c
somad=np.array([np.linalg.norm(SOMA[PI[k]]-SOMA[PJ[k]]) for k in range(NpP)])
bdiff=np.array([abs(BT[PI[k]]-BT[PJ[k]]) if not np.isnan(BT[PI[k]]) and not np.isnan(BT[PJ[k]]) else 0 for k in range(NpP)])
lcp=np.array([commonpref(LN[PI[k]],LN[PJ[k]]) for k in range(NpP)])
# clean-C feature matrix: soma 3D both + soma dist + birth diff/both + lineage common-prefix (ALL pre-placement)
F=np.column_stack([somad,SOMA[PI],SOMA[PJ],bdiff,BT[PI],BT[PJ],lcp])
P(f"clean-C features: soma-dist, soma-3D(both), birth-diff, birth(both), lineage-common-prefix ({F.shape[1]} dims, ALL pre-placement)")
P(f"  data coverage: soma {np.isfinite(SOMA[:,0]).mean():.2f} birth {np.isfinite(BT).mean():.2f} lineage {np.mean([len(x)>0 for x in LN]):.2f}")
# neuron-disjoint CV
def nd_auc(Fm):
    pred=np.full(NpP,np.nan)
    for f in range(5):
        te=(PFOLD==f); tr=(PFOLD!=f)&(PFOLD>=0)
        if CONTACT[te].sum()>3 and (CONTACT[te]==0).sum()>3 and tr.sum()>50:
            clf=HistGradientBoostingClassifier(max_iter=200,random_state=0).fit(Fm[tr],CONTACT[tr]); pred[te]=clf.predict_proba(Fm[te])[:,1]
    ok=~np.isnan(pred); return roc_auc_score(CONTACT[ok],pred[ok]), roc_auc_score(YC[ok],pred[ok]), pred, ok
aucCL,cookCL,predCL,ok=nd_auc(F)
# soma-only and lineage/birth-only for decomposition
aucSoma=nd_auc(np.column_stack([somad,SOMA[PI],SOMA[PJ]]))[0]
aucDev=nd_auc(np.column_stack([bdiff,BT[PI],BT[PJ],lcp]))[0]
CEIL=0.859  # adult-arbor (leaky) ceiling reference
P("\n"+"="*82); P("PART 1 — HONEST BULK FRACTION (clean pre-placement C)"); P("="*82)
P(f"  clean-C (soma+birth+lineage) contact AUC: neuron-disjoint {aucCL:.3f} | Cook {cookCL:.3f}")
P(f"    soma-geometry only:  {aucSoma:.3f} | developmental only (birth+lineage): {aucDev:.3f}")
P(f"  degree-preserving null: {DEGNULL_ND:.3f} | adult-arbor(leaky) ceiling: {CEIL:.3f}")
beyond=aucCL-DEGNULL_ND; denom=CEIL-DEGNULL_ND
X=beyond/denom if denom>0 else 0
P(f"\n  clean-C beyond degree = {beyond:+.3f} ; explainable-beyond-degree (ceiling−degree) = {denom:.3f}")
P(f"  ==> X = fraction of bulk-beyond-degree explained by clean physical placement = {X:+.2f}")
if aucCL<=DEGNULL_ND+0.01:
    P(f"  ==> clean placement does NOT beat the degree null -> X≈0: pre-placement info explains ~nothing of the bulk")
    P(f"      beyond degree. The bulk is the ADULT NEURITE MORPHOLOGY (grown trajectories in the ring), which is")
    P(f"      NOT derivable from soma+birth+lineage. (matches Phase-1: adult arrangement not computable from IC.)")
# ---- LOCATE THE MISSES ----
P("\n[LOCATE clean-C's bulk misses]")
err=np.abs(CONTACT[ok].astype(float)-predCL[ok])   # per-pair error
# is error random or structured? correlate with area, degree, soma-distance-region
from scipy.stats import spearmanr
P(f"  corr(|error|, contact-area) = {spearmanr(err,AREA[ok])[0]:+.2f} | corr(|error|, endpoint-degree-sum) = {spearmanr(err,(CDEG[PI]+CDEG[PJ])[ok])[0]:+.2f}")
# among real contacts clean-C misses (contact=1, pred low): what are they?
miss=(CONTACT[ok]==1)&(predCL[ok]<np.quantile(predCL[ok],0.5))
P(f"  real contacts clean-C MISSES (n={miss.sum()}): mean area {AREA[ok][miss].mean():.0f} vs hit-contacts area {AREA[ok][(CONTACT[ok]==1)&~miss].mean():.0f}")
P(f"  => misses are {'the STRONG large-area contacts' if AREA[ok][miss].mean()>AREA[ok][(CONTACT[ok]==1)&~miss].mean() else 'the WEAK small-area contacts'}")
# ---- PART 2/3 gating ----
P("\n"+"="*82)
if aucCL<=DEGNULL_ND+0.02:
    P("PART 2/3 STATUS: clean-C does not beat degree -> it has NO meaningful above-degree 'ground state' to derive a")
    P("  constrained/free degeneracy from. The pre-audit degeneracy-unification (Y) was ENTIRELY the adult-arbor leak")
    P("  and is VOID. Clean C cannot be tested for residual-degeneracy because it has no clean bulk signal to degenerate.")
    P(f"  Honest map: X≈{max(0,X):.2f} of bulk (clean placement ~ nothing beyond degree); Y = void (no clean degeneracy).")
    P("  REMAINDER = essentially the entire above-degree bulk = the adult neurite MORPHOLOGY (grown ring trajectories).")
    P("  Next question is SINGLE and located: what generates the neurite trajectories (morphogenesis) — the last open door.")
else:
    P(f"PART 2/3: clean-C beats degree (X={X:.2f}) -> build placement energy model, extract degeneracy, match noise-")
    P("  corrected reproducible/variable. (Proceed in follow-up.)")
P("="*82)
