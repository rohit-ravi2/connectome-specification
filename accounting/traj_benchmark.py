"""SHARED TRAJECTORY BENCHMARK (importable). Target = clean-C's BULK REMAINDER: the contact structure that
soma-PLACEMENT (soma position + birth + lineage) does NOT force. Each angle provides a per-pair feature; score()
reports whether it recovers the remainder BEYOND placement, BEYOND degree, neuron-disjoint held-out, + Cook-synapse
downstream. CRITICAL LEAK RULE (the C-audit lesson): contact = adult neurite adjacency, so any feature built from
the FOCAL pair's own adult trajectory predicts contact near-definitionally (=C's 0.868 leak). Each angle MUST
declare uses_focal_trajectory; if True, that number is leak-inflated like C and does NOT count — the clean test is
whether an UPSTREAM mechanism (gradient/pioneer-substrate/birth-order/tissue-env, not the focal adult path) adds
beyond placement. score() prints a LEAK-WARNING when flagged.
"""
import sys, numpy as np, csv;
sys.path.insert(0,'../accounting')
from bulk_benchmark import RING,PAIRS,PI,PJ,CONTACT,SK,DEGNULL_ND,M,AREA,YC,CDEG,PFOLD,IDX
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
NP=len(PAIRS)
# ---- placement baseline (soma root + Sulston birth + AB-lineage), STRICTLY pre-placement ----
def _soma(n):
    st=SK[n].get('starts'); co=SK[n]['coords']; k=str(st[0]) if st and str(st[0]) in co else list(co.keys())[0]
    return np.array(co[k],float)
SOMA=np.array([_soma(n) for n in RING])
_bt={};_lin={}
for r in csv.DictReader(open("../seed/data/external/mechanistic_frontier/neuron_birth_times_sulston/neuron_birth_times.csv")): _bt[r['neuron']]=float(r['birth_time_min'])
for r in csv.DictReader(open("../data/celegans/data/processed/developmental_rules/birthtime_earlylate.csv")): _lin[r['neuron']]=r['lineage_s4'].replace(' ','')
def _bc(n):
    for s in "LRDV":
        if n.endswith(s): return n[:-1]
    return n
BIRTH=np.array([_bt.get(n,_bt.get(_bc(n),0.0)) for n in RING])
LIN=[_lin.get(n,_lin.get(_bc(n),'')) for n in RING]
def _cp(a,b):
    c=0
    for x,y in zip(a,b):
        if x==y: c+=1
        else: break
    return c
_somad=np.array([np.linalg.norm(SOMA[PI[k]]-SOMA[PJ[k]]) for k in range(NP)])
_bdiff=np.abs(BIRTH[PI]-BIRTH[PJ])
_lcp=np.array([_cp(LIN[PI[k]],LIN[PJ[k]]) for k in range(NP)])
PLACEMENT=np.column_stack([_somad,SOMA[PI],SOMA[PJ],_bdiff,BIRTH[PI],BIRTH[PJ],_lcp])  # the clean-C baseline
def _nd_cv(F):
    pred=np.full(NP,np.nan)
    for f in range(5):
        te=(PFOLD==f); tr=(PFOLD!=f)&(PFOLD>=0)
        if CONTACT[te].sum()>3 and (CONTACT[te]==0).sum()>3 and tr.sum()>50:
            pred[te]=HistGradientBoostingClassifier(max_iter=200,random_state=0).fit(F[tr],CONTACT[tr]).predict_proba(F[te])[:,1]
    ok=~np.isnan(pred); return pred, ok
_pp,_ok=_nd_cv(PLACEMENT); PLACE_AUC=roc_auc_score(CONTACT[_ok],_pp[_ok]); P_PLACE=_pp  # clean-C remainder = residual of this
def score(feature, name, uses_focal_trajectory):
    """feature: (NP,) or (NP,k) array from your angle. uses_focal_trajectory: bool — declare honestly."""
    feature=np.asarray(feature,float);
    if feature.ndim==1: feature=feature[:,None]
    Fpp=np.hstack([PLACEMENT,feature])
    pp2,ok=_nd_cv(Fpp)
    aucP=roc_auc_score(CONTACT[ok],P_PLACE[ok]); aucPF=roc_auc_score(CONTACT[ok],pp2[ok])
    # beyond degree: does place+feature beat the degree null; does feature-ALONE beat degree
    fa,oka=_nd_cv(feature); auc_feat_alone=roc_auc_score(CONTACT[oka],fa[oka])
    cook=roc_auc_score(YC[ok],pp2[ok])
    d={'name':name,'uses_focal_trajectory':uses_focal_trajectory,'placement_only_auc':round(aucP,3),
       'placement_plus_feature_auc':round(aucPF,3),'BEYOND_PLACEMENT_delta':round(aucPF-aucP,3),
       'feature_alone_auc':round(auc_feat_alone,3),'degree_null':round(DEGNULL_ND,3),
       'beyond_degree':round(auc_feat_alone-DEGNULL_ND,3),'cook_synapse_auc':round(cook,3)}
    if uses_focal_trajectory: print("  *** LEAK WARNING: uses focal adult trajectory -> contact-from-trajectory is near-definitional (C's 0.868 leak). This number does NOT count as beyond-placement recovery. Provide an upstream-only version. ***")
    print("TRAJ-BENCH", d)
    # which contacts recovered: pairs place misses (contact=1, P_PLACE low) that place+feature gets (pp2 high)
    miss=(CONTACT==1)&(P_PLACE<np.nanmedian(P_PLACE[CONTACT==1]))
    rec=miss&(pp2>np.nanmedian(pp2[CONTACT==1]))
    print(f"  recovered {int(np.nansum(rec))} of {int(miss.sum())} placement-missed contacts")
    return d, rec
if __name__=="__main__":
    print(f"placement(clean-C) neuron-disjoint contact AUC = {PLACE_AUC:.3f} (below degree null {DEGNULL_ND:.3f}); target = the remainder above this.")
