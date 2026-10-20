"""ARM 1 — LINEAGE-INTRINSIC routing (the structural blind spot: nulls tested EXPRESSION, never cell-autonomous
LINEAGE). Gate outcome: pioneers reproducible-but-NOT-special (concentration rejected, Arm 6 fires) -> test
lineage on GENERAL routing (contact), not pioneer-special. Q: does Sulston lineage predict contact BEYOND
placement (soma) AND beyond degree, Cook-held-out, controlling lineage-as-position (sisters sit near each other)?
"""
import warnings,sys,csv; warnings.simplefilter("ignore")
import numpy as np
sys.path.insert(0,'../accounting')
from traj_benchmark import RING,PAIRS,PI,PJ,CONTACT,SK,DEGNULL_ND,M,YC,CDEG,PFOLD,SOMA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
RNG=np.random.default_rng(0); NP=len(PAIRS)
# lineage paths (Sulston AB lineage)
lin={}
for r in csv.DictReader(open("../data/celegans/data/processed/developmental_rules/birthtime_earlylate.csv")):
    lin[r['neuron']]=r['lineage_s4'].replace(' ','')
def bc(n):
    for s in "LRDV":
        if n.endswith(s): return n[:-1]
    return n
LN=[lin.get(n,lin.get(bc(n),'')) for n in RING]
P(f"lineage coverage: {np.mean([len(x)>0 for x in LN]):.2f} ({sum(len(x)>0 for x in LN)}/{M})")
def cp(a,b):
    c=0
    for x,y in zip(a,b):
        if x==y: c+=1
        else: break
    return c
# lineage features (cell-autonomous): common-prefix depth + shared-sublineage-at-depth-d indicators + same-moiety
lcp=np.array([cp(LN[PI[k]],LN[PJ[k]]) for k in range(NP)])
def share_at(d): return np.array([1.0 if (len(LN[PI[k]])>=d and len(LN[PJ[k]])>=d and LN[PI[k]][:d]==LN[PJ[k]][:d]) else 0.0 for k in range(NP)])
LINEAGE=np.column_stack([lcp]+[share_at(d) for d in (3,5,7,9)])
# placement = soma only (geometry, NO lineage)
somad=np.array([np.linalg.norm(SOMA[PI[k]]-SOMA[PJ[k]]) for k in range(NP)])
PLACE=np.column_stack([somad,SOMA[PI],SOMA[PJ]])
def nd(F):
    pred=np.full(NP,np.nan)
    for f in range(5):
        te=(PFOLD==f);tr=(PFOLD!=f)&(PFOLD>=0)
        if CONTACT[te].sum()>3 and (CONTACT[te]==0).sum()>3 and tr.sum()>50:
            pred[te]=HistGradientBoostingClassifier(max_iter=200,random_state=0).fit(F[tr],CONTACT[tr]).predict_proba(F[te])[:,1]
    ok=~np.isnan(pred); return roc_auc_score(CONTACT[ok],pred[ok]), roc_auc_score(YC[ok],pred[ok]), ok
aP,cP,_=nd(PLACE); aPL,cPL,_=nd(np.hstack([PLACE,LINEAGE])); aL,cL,_=nd(LINEAGE)
P("\n"+"="*80); P("ARM 1 — LINEAGE-INTRINSIC vs placement/degree (target=contact, neuron-disjoint, Cook)"); P("="*80)
P(f"  placement(soma) only:      AUC {aP:.3f} (Cook {cP:.3f})")
P(f"  placement + LINEAGE:       AUC {aPL:.3f} (Cook {cPL:.3f})  -> BEYOND-PLACEMENT Δ={aPL-aP:+.3f}")
P(f"  LINEAGE alone:             AUC {aL:.3f} (Cook {cL:.3f})  vs degree-null {DEGNULL_ND:.3f} -> beyond-degree {aL-DEGNULL_ND:+.3f}")
# lineage-shuffled null: shuffle lineage labels across neurons, recompute lineage-alone
def lin_shuffled():
    p=RNG.permutation(M); LNp=[LN[p[i]] for i in range(M)]
    lcp2=np.array([cp(LNp[PI[k]],LNp[PJ[k]]) for k in range(NP)])
    sh=np.column_stack([lcp2]+[np.array([1.0 if (len(LNp[PI[k]])>=d and LNp[PI[k]][:d]==LNp[PJ[k]][:d]) else 0.0 for k in range(NP)]) for d in (3,5,7,9)])
    return nd(sh)[0]
nulls=[lin_shuffled() for _ in range(6)]
P(f"  lineage-SHUFFLED null (alone): AUC {np.mean(nulls):.3f}±{np.std(nulls):.3f} -> real-vs-shuffle Δ={aL-np.mean(nulls):+.3f}")
# lineage-as-position control: does lineage add beyond soma? (already the beyond-placement Δ above)
P("\n"+"="*80)
wins = (aPL-aP>0.02) and (aL>DEGNULL_ND) and (aL>np.mean(nulls)+3*np.std(nulls)) and (cPL-cP>0.005)
P("VERDICT:", "LINEAGE-INTRINSIC ROUTING REAL — lineage predicts contact beyond placement AND degree, survives lineage-shuffle+Cook. The arc's answer: cell-autonomous lineage program (the untested blind spot)." if wins
  else f"LINEAGE does NOT clear — beyond-placement Δ={aPL-aP:+.3f}, lineage-alone {aL:.3f} vs degree {DEGNULL_ND:.3f} vs shuffle {np.mean(nulls):.3f}. Lineage adds little beyond soma-position (lineage-as-position costume) and doesn't beat degree. The cell-autonomous blind spot is checked and does NOT carry the general reproducible routing.")
P("="*80)
