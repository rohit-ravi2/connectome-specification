"""FATE-DETERMINANT CODE test (standalone — NOT 'implementing a confirmed lineage result': lineage FELL its
proper confound-closer last turn = costume/below-degree). Legitimate distinct Q: does the cell-autonomous fate
code (homeobox/terminal-selector TFs the neuron expresses) predict its ROUTING (which bundle it joins), beyond
POSITION and beyond RANDOM-IDENTITY (the homeobox-validating control)? Cell-autonomous per-neuron, NOT expression-
matching. Cook has no trajectories -> bundle prediction is within-Witvliet (flagged).
"""
import warnings,sys,csv; warnings.simplefilter("ignore")
import numpy as np, pandas as pd, networkx as nx
sys.path.insert(0,'../accounting')
from traj_benchmark import RING,PAIRS,PI,PJ,SK,M,SOMA
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
def P(*a): print(*a); sys.stdout.flush()
RNG=np.random.default_rng(0); NP=len(PAIRS)
# ---- bundles (co-routing clusters, connectivity-blind) ----
def cloud(n):
    pts=np.array(list(SK[n]['coords'].values()),float)
    return pts[RNG.choice(len(pts),min(80,len(pts)),replace=False)] if len(pts)>80 else pts
PT={n:cloud(n) for n in RING}; trees={n:cKDTree(PT[n]) for n in RING}
cf=np.array([int((trees[RING[PJ[k]]].query(PT[RING[PI[k]]])[0]<500).sum()) for k in range(NP)])
G=nx.Graph(); G.add_nodes_from(range(M)); thr=np.quantile(cf[cf>0],0.5)
for k in range(NP):
    if cf[k]>thr: G.add_edge(PI[k],PJ[k])
comms=nx.community.greedy_modularity_communities(G)
bund=np.full(M,-1)
for i,c in enumerate(comms):
    for n in c: bund[n]=i
P(f"bundles: {len(comms)} (sizes {sorted(len(c) for c in comms)})")
# ---- fate-determinant TF code (homeobox + terminal selectors), per-neuron expression ----
cg=pd.read_csv("../data/celegans/data/expression/cengen/thresholded/021821_liberal_threshold1.csv")
classes=list(cg.columns[3:]); X=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values
gname=cg["gene_name"].astype(str).str.lower().values; cls_idx={c:k for k,c in enumerate(classes)}
def n2c(n):
    if n in cls_idx: return cls_idx[n]
    b=n
    for s in "LR":
        if b.endswith(s): b=b[:-1]; break
    if b in cls_idx: return cls_idx[b]
    for s in "DV":
        if b.endswith(s) and b[:-1] in cls_idx: return cls_idx[b[:-1]]
    return cls_idx.get(b,None)
SEL=["unc-3","unc-30","unc-42","unc-86","unc-4","unc-39","unc-62","ttx-1","ttx-3","lin-11","lin-39","lim-4","lim-6","lim-7","mec-3","mab-5","egl-5","cfi-1","che-1","ast-1","ceh-14","ceh-10","cog-1","alr-1","nhr-67","pag-3","bnc-1"]
FATE=[g for g in gname if g.startswith("ceh-") or g in SEL]; FATE=list(dict.fromkeys(FATE))
CC=np.array([n2c(n) if n2c(n) is not None else 0 for n in RING])
Fmat=np.array([(X[np.where(gname==g)[0][0]]>0).astype(float) for g in FATE])  # nFate x classes
FATEv=Fmat[:,CC].T   # neuron x fate-TF (cell-autonomous fate identity vector)
P(f"fate-determinant TFs: {len(FATE)} (homeobox {sum(g.startswith('ceh-') for g in FATE)} + selectors {sum(g in SEL for g in FATE)})")
# ---- PHASE 0: fate<->lineage link (fate is lineage-set) ----
lin={}
for r in csv.DictReader(open("../data/celegans/data/processed/developmental_rules/birthtime_earlylate.csv")): lin[r['neuron']]=r['lineage_s4'].replace(' ','')
def bc(n):
    for s in "LRDV":
        if n.endswith(s): return n[:-1]
    return n
LN=[lin.get(n,lin.get(bc(n),'')) for n in RING]
def cp(a,b):
    c=0
    for x,y in zip(a,b):
        if x==y:c+=1
        else:break
    return c
lcp=np.array([cp(LN[PI[k]],LN[PJ[k]]) for k in range(NP)])
fate_sim=np.array([(FATEv[PI[k]]*FATEv[PJ[k]]).sum() for k in range(NP)])
P(f"\n[PHASE 0] fate<->lineage: Spearman(fate-similarity, lineage-common-prefix) = {spearmanr(fate_sim,lcp)[0]:+.2f} (should be +, fate is lineage-set)")
# ---- PHASE 1: predict bundle (per-neuron) from fate vs position vs random-identity ----
valid=bund>=0; y=bund[valid]
def loo_bacc(Fn):
    skf=StratifiedKFold(4,shuffle=True,random_state=0)
    pred=cross_val_predict(HistGradientBoostingClassifier(max_iter=150,random_state=0),Fn[valid],y,cv=skf)
    return balanced_accuracy_score(y,pred)
b_fate=loo_bacc(FATEv); b_pos=loo_bacc(SOMA); b_both=loo_bacc(np.hstack([SOMA,FATEv]))
# random-identity null: matched-size random gene sets
pool=[i for i in range(len(gname)) if 10<=(X[i]>0).sum()<=len(classes)-10]
rand=[loo_bacc(np.array([(X[i]>0).astype(float) for i in np.random.default_rng(s).choice(pool,len(FATE),replace=False)])[:,CC].T) for s in range(5)]
chance=1.0/len(comms)
P("\n"+"="*80); P("PHASE 1 — does fate code predict which BUNDLE a neuron joins?"); P("="*80)
P(f"  balanced-accuracy (chance={chance:.3f}, {len(comms)} bundles):")
P(f"    fate code:            {b_fate:.3f}")
P(f"    position (soma):      {b_pos:.3f}")
P(f"    position + fate:      {b_both:.3f}  -> beyond-position Δ={b_both-b_pos:+.3f}")
P(f"    random-identity null: {np.mean(rand):.3f} ± {np.std(rand):.3f}  -> fate vs random-identity Δ={b_fate-np.mean(rand):+.3f}")
P("\n"+"="*80)
beyond_pos = (b_both-b_pos)>0.03
beyond_rand = b_fate>np.mean(rand)+2*np.std(rand)
if beyond_pos and beyond_rand:
    P("VERDICT: fate code predicts bundle BEYOND position AND beats random-identity -> cell-autonomous fate")
    P("  determinant implements routing. The molecular reopening at the FATE level (Phase 2 to characterize).")
elif not beyond_rand:
    P(f"VERDICT: fate code TIES/loses to random-identity (Δ={b_fate-np.mean(rand):+.3f}) -> fate TFs don't SPECIFICALLY")
    P("  predict routing; it's generic cell-identity (~region), not a fate-specific routing code. The lineage-intrinsic")
    P("  determinant (itself weak/costume) is NOT cleanly transcriptional-fate either. Points to non-transcriptional")
    P("  (polarity/mechanics) OR simply that routing is not fate-coded. Consistent with identity-generic-for-wiring.")
else:
    P(f"VERDICT: fate beats random-identity but NOT position (beyond-pos Δ={b_both-b_pos:+.3f}) -> fate ≈ position (region);")
    P("  fate code is regional identity, does not add routing info beyond where the cell sits. Not a fate-routing code.")
P("  NOTE: bundles are Witvliet-only (Cook has no trajectories) -> no Cook-held-out on bundle target; flagged.")
P("="*80)
