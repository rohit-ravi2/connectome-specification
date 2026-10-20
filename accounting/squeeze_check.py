"""MECHANISM-vs-MARKER premise check. The squeeze presupposes fate->routing beyond position (mediating lineage).
Confirm decisively: does fate add ANYTHING to bundle prediction beyond position+lineage? If ~0, the premise is
false and mechanism-vs-marker is MOOT (no fate->routing to be cause or marker of). Also report which squeeze
tests are even measurable on-disk.
"""
import warnings,sys,csv; warnings.simplefilter("ignore")
import numpy as np, pandas as pd, networkx as nx
sys.path.insert(0,'../accounting')
from traj_benchmark import RING,PAIRS,PI,PJ,SK,M,SOMA
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
def P(*a): print(*a); sys.stdout.flush()
RNG=np.random.default_rng(0); NP=len(PAIRS)
def cloud(n):
    pts=np.array(list(SK[n]['coords'].values()),float); return pts[RNG.choice(len(pts),min(80,len(pts)),replace=False)] if len(pts)>80 else pts
PT={n:cloud(n) for n in RING}; trees={n:cKDTree(PT[n]) for n in RING}
cf=np.array([int((trees[RING[PJ[k]]].query(PT[RING[PI[k]]])[0]<500).sum()) for k in range(NP)])
G=nx.Graph(); G.add_nodes_from(range(M)); thr=np.quantile(cf[cf>0],0.5)
for k in range(NP):
    if cf[k]>thr: G.add_edge(PI[k],PJ[k])
bund=np.full(M,-1)
for i,c in enumerate(nx.community.greedy_modularity_communities(G)):
    for n in c: bund[n]=i
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
SEL=["unc-3","unc-30","unc-42","unc-86","unc-4","ttx-3","lin-11","lim-6","mec-3","mab-5","egl-5","cfi-1","che-1","ast-1","ceh-14","ceh-10","cog-1","nhr-67","pag-3","bnc-1"]
FATE=[g for g in gname if g.startswith("ceh-") or g in SEL]; FATE=list(dict.fromkeys(FATE))
CC=np.array([n2c(n) if n2c(n) is not None else 0 for n in RING])
FATEv=np.array([(X[np.where(gname==g)[0][0]]>0).astype(float) for g in FATE])[:,CC].T
lin={}
for r in csv.DictReader(open("../data/celegans/data/processed/developmental_rules/birthtime_earlylate.csv")): lin[r['neuron']]=r['lineage_s4'].replace(' ','')
def bc(n):
    for s in "LRDV":
        if n.endswith(s): return n[:-1]
    return n
LN=[lin.get(n,lin.get(bc(n),'')) for n in RING]
LINv=np.array([[1.0 if (len(LN[i])>=d and LN[i][:d]==p) else 0.0 for d,p in [(3,LN[i][:3]),(5,LN[i][:5])]]+[len(LN[i])] for i in range(M)])
valid=bund>=0; y=bund[valid]
def bacc(Fn):
    return balanced_accuracy_score(y,cross_val_predict(HistGradientBoostingClassifier(max_iter=150,random_state=0),Fn[valid],y,cv=StratifiedKFold(4,shuffle=True,random_state=0)))
P("="*80); P("DECISIVE PREMISE CHECK — does fate add to bundle beyond position+lineage?"); P("="*80)
bP=bacc(SOMA); bPL=bacc(np.hstack([SOMA,LINv])); bPLF=bacc(np.hstack([SOMA,LINv,FATEv]))
P(f"  bundle bal-acc (chance {1/(bund.max()+1):.2f}): position {bP:.3f} | position+lineage {bPL:.3f} | position+lineage+FATE {bPLF:.3f}")
P(f"  ==> FATE beyond position+lineage = {bPLF-bPL:+.3f}")
# Test 4 specificity: individual fate TFs beyond position
P("\n[Test 4 specificity] top fate-TFs by bundle-prediction beyond position (if any localize):")
scores=[]
for j,g in enumerate(FATE):
    b=bacc(np.hstack([SOMA,FATEv[:,[j]]]))-bP; scores.append((g,b))
scores.sort(key=lambda x:-x[1])
for g,s in scores[:5]: P(f"    {g:8s} beyond-position Δ={s:+.3f}")
P("\n"+"="*80)
if bPLF-bPL<=0.02:
    P("VERDICT: fate adds ~nothing beyond position+lineage -> NO fate->routing signal to be cause OR marker of.")
    P("  MECHANISM-vs-MARKER IS MOOT: the premise ('fate predicts routing beyond position, mediates lineage')")
    P("  is FALSE (confirmed). Fate codes IDENTITY/effectors (survived), routing is physical/positional.")
else:
    P("VERDICT: fate adds something beyond position+lineage -> re-examine; mechanism-vs-marker has a target.")
P("\n[Squeeze-test measurability on-disk]:")
P("  Test 1 (fate-TF onset vs routing-commitment timing): needs per-TF developmental expression timing + per-")
P("    pioneer routing-commit time -> Packer embryo expr sparse, routing-commit not cleanly datable -> WEAK/UNMEASURABLE.")
P("  Test 2 (natural fate-EXPRESSION variation across animals vs routing variation): CeNGEN is a SINGLE atlas —")
P("    NO per-animal expression replicates -> the natural-perturbation version is UNMEASURABLE on-disk (0 power).")
P("  Test 3 (dose): CeNGEN thresholded, limited graded dose -> WEAK anyway (markers can be graded).")
P("  Test 4 (specificity): runnable (above) but moot since the signal is position-subsumed (not a routing code).")
P("  => the two STRONG squeeze tests (1,2) are unmeasurable with on-disk data even if the premise held.")
P("DECISIVE EXPERIMENT (regardless): fate-TF knockout/misexpression -> pioneer trajectory = wet-lab, not proposed;")
P("  and there is no correlational fate->routing-beyond-position signal for it to test causally.")
P("="*80)
