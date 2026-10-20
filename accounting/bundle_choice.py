"""BUNDLE-CHOICE HEAD-TO-HEAD: pioneer-inheritance (physical) vs adhesion-sorting (molecular). Target = bundle
co-membership, defined CONNECTIVITY-BLIND (spatial co-routing clusters from coords, NO contact matrix). Phase-0
gate: is co-membership separable from contact? ARM A: pioneers placement-predictable (root) + followers inherit.
ARM B: adhesion-affinity predicts co-membership beyond a DEGREE-PRESERVING null (the Angle-3 costume at this level).
"""
import warnings,sys,csv; warnings.simplefilter("ignore")
import numpy as np, pandas as pd, networkx as nx
sys.path.insert(0,'../accounting')
from traj_benchmark import RING,PAIRS,PI,PJ,CONTACT,SK,DEGNULL_ND,M,AREA,YC,CDEG,PFOLD,IDX,SOMA,BIRTH
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
RNG=np.random.default_rng(0); NP=len(PAIRS)
# ---- Phase 0: connectivity-blind co-routing -> bundles ----
def cloud(n):
    pts=np.array(list(SK[n]['coords'].values()),float)
    return pts[RNG.choice(len(pts),min(90,len(pts)),replace=False)] if len(pts)>90 else pts
PT={n:cloud(n) for n in RING}; trees={n:cKDTree(PT[n]) for n in RING}
def cofasc(a,b): d,_=trees[b].query(PT[a]); return int((d<500).sum())
cf=np.array([cofasc(RING[PI[k]],RING[PJ[k]]) for k in range(NP)])   # coords only, no contact
# co-routing graph -> bundles (community detection)
G=nx.Graph(); G.add_nodes_from(range(M))
thr=np.quantile(cf[cf>0],0.5)
for k in range(NP):
    if cf[k]>thr: G.add_edge(PI[k],PJ[k],weight=cf[k])
comms=nx.community.greedy_modularity_communities(G,weight='weight')
bund=np.full(M,-1)
for i,c in enumerate(comms):
    for n in c: bund[n]=i
P(f"PHASE 0: co-routing graph -> {len(comms)} bundles (sizes {sorted(len(c) for c in comms)[-6:]}); unassigned {int((bund<0).sum())}")
comemb=np.array([1 if (bund[PI[k]]>=0 and bund[PI[k]]==bund[PJ[k]]) else 0 for k in range(NP)])
# universe = spatially co-located pairs (could plausibly bundle) = co-fasc>0
U=cf>0
P(f"  co-located universe (co-fasc>0): {U.sum()} pairs | same-bundle {comemb[U].sum()} | contact-rate in U {CONTACT[U].mean():.2f}")
# SEPARABILITY GATE: is co-membership just contact?
sep=roc_auc_score(CONTACT[U],comemb[U])
P(f"  [GATE] AUC(co-membership -> contact) among co-located = {sep:.3f}  (~1.0 => inextricable from contact)")
P(f"  Spearman(co-membership, contact) in U = {spearmanr(comemb[U],CONTACT[U])[0]:+.2f}")
GATE = sep<0.90
P(f"  => bundle target {'SEPARABLE enough from contact (proceed)' if GATE else 'INEXTRICABLE from contact — resolution limit; head-to-head cannot cleanly distinguish (report)'}")

# helper: neuron-disjoint CV AUC on the co-located universe for co-membership
foldn=np.array([RNG.integers(5) for _ in range(M)]); pf=np.array([max(foldn[PI[k]],foldn[PJ[k]]) if foldn[PI[k]]==foldn[PJ[k]] else -1 for k in range(NP)])
def ndU(F,tgt,mask):
    pred=np.full(NP,np.nan)
    for f in range(5):
        te=(pf==f)&mask; tr=(pf!=f)&(pf>=0)&mask
        if tgt[te].sum()>3 and (tgt[te]==0).sum()>3 and tr.sum()>40:
            pred[te]=HistGradientBoostingClassifier(max_iter=150,random_state=0).fit(F[tr],tgt[tr]).predict_proba(F[te])[:,1]
    ok=(~np.isnan(pred))&mask; return roc_auc_score(tgt[ok],pred[ok]) if ok.sum()>20 and tgt[ok].sum()>5 else np.nan
# degree null for co-membership
cl=(CDEG[PI]*CDEG[PJ]); degU=roc_auc_score(comemb[U],cl[U])
P(f"  degree-null (deg_i*deg_j -> co-membership) AUC = {degU:.3f}")

# ================= ARM A — PIONEER-INHERITANCE =================
P("\n"+"="*82); P("ARM A — PHYSICAL: pioneer-inheritance"); P("="*82)
early=BIRTH<=np.quantile(BIRTH,0.4); late=~early
somad=np.array([np.linalg.norm(SOMA[PI[k]]-SOMA[PJ[k]]) for k in range(NP)])
PLACE=np.column_stack([somad,SOMA[PI],SOMA[PJ]])
# A1 make-or-break: are EARLY (pioneer) pairs' co-membership placement-predictable?
Uearly=U&early[PI]&early[PJ]
a1=ndU(PLACE,comemb,Uearly)
P(f"  [ROOT] pioneer(early-early) co-membership from PLACEMENT (soma): AUC={a1:.3f} vs degree {degU:.3f} -> {'placement-predictable' if a1>degU else 'NOT placement-predictable => physical chain BROKEN at root'} (n={Uearly.sum()})")
# A2 followers inherit: does a late neurite share bundle via a shared earlier pioneer?
foll=np.zeros((M,M))
for i in range(M):
    for j in range(M):
        if i!=j and BIRTH[i]>BIRTH[j]:
            d,_=trees[RING[i]].query(PT[RING[j]])
            if (d<800).sum()>3: foll[i,j]=1
shp=np.array([(foll[PI[k]]*foll[PJ[k]]).sum() for k in range(NP)])
Ulate=U&(late[PI]|late[PJ])
a2=ndU(np.column_stack([shp,somad]),comemb,Ulate)
a2_bd=ndU(shp.reshape(-1,1),comemb,Ulate)
P(f"  [FOLLOWERS] late co-membership from pioneer-inheritance (shared-pioneer+placement): AUC={a2:.3f} | shared-pioneer alone {a2_bd:.3f} vs degree {degU:.3f}")
armA = (a1>degU) and (a2_bd>degU)
P(f"  ARM A verdict: {'physical chain holds (pioneers placement-predictable AND followers inherit)' if armA else 'chain does NOT close'}")

# ================= ARM B — ADHESION-SORTING =================
P("\n"+"="*82); P("ARM B — MOLECULAR: adhesion-sorting (degree-null is make-or-break)"); P("="*82)
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
ADH=[g for g in gname if g.startswith(('cdh-','oig-','igcm-','igdb-','lron-','zig-','rig-','casy-')) or g in ('sax-7','fmi-1','lad-2','rig-6','nlg-1','nrx-1','hmr-1','ptp-3')]
ADH=list(dict.fromkeys(ADH))
Am=np.array([(X[np.where(gname==g)[0][0]]>0).astype(float) for g in ADH])  # nAdh x classes
CC=np.array([n2c(n) if n2c(n) is not None else 0 for n in RING]); An=Am[:,CC]  # nAdh x neuron
# adhesion-affinity = # shared adhesion molecules (Steinberg matching)
adh_aff=np.array([(An[:,PI[k]]*An[:,PJ[k]]).sum() for k in range(NP)])
b_auc=ndU(adh_aff.reshape(-1,1),comemb,U)
P(f"  adhesion-affinity -> co-membership AUC={b_auc:.3f} vs degree-null {degU:.3f} -> beyond_degree={b_auc-degU:+.3f}")
# DEGREE-PRESERVING null: permute adhesion within degree-matched bins
bins=np.digitize(CDEG,np.quantile(CDEG,[.2,.4,.6,.8]))
def dpn():
    Ap=An.copy()
    for bb in range(5):
        ix=np.where(bins==bb)[0]
        if len(ix)>1: Ap[:,ix]=An[:,ix[RNG.permutation(len(ix))]]
    aff=np.array([(Ap[:,PI[k]]*Ap[:,PJ[k]]).sum() for k in range(NP)])
    return ndU(aff.reshape(-1,1),comemb,U)
nulls=[dpn() for _ in range(8)]; nulls=[x for x in nulls if not np.isnan(x)]
P(f"  DEGREE-PRESERVING null (adhesion permuted within degree bins): AUC={np.mean(nulls):.3f}±{np.std(nulls):.3f} | real Δ={b_auc-np.mean(nulls):+.3f}")
# decompose: adhesion residual orthogonal to degree
dp=(CDEG[PI]+CDEG[PJ]).reshape(-1,1); ares=adh_aff-LinearRegression().fit(dp[U],adh_aff[U]).predict(dp)
b_res=ndU(ares.reshape(-1,1),comemb,U)
P(f"  adhesion residual (orthogonal to degree) -> co-membership AUC={b_res:.3f} (chance 0.5)")
armB=(b_auc>degU) and (b_auc>np.mean(nulls)+3*np.std(nulls)) and (b_res>0.55)
P(f"  ARM B verdict: {'adhesion-sorting REAL beyond degree' if armB else 'FALLS to degree-costume (adhesion ≈ degree-preserving null / below degree)'}")

P("\n"+"="*82); P("HEAD-TO-HEAD"); P("="*82)
P(f"  target=bundle co-membership | gate {'PASS' if GATE else 'FAIL(inextricable)'} | degree-null {degU:.3f}")
P(f"  ARM A (physical): root placement-predictable={a1>degU} ({a1:.3f}), followers-inherit={a2_bd>degU} ({a2_bd:.3f}) -> {'HOLDS' if armA else 'broken'}")
P(f"  ARM B (molecular): adhesion beyond-degree={b_auc-degU:+.3f}, vs deg-pres-null Δ={b_auc-np.mean(nulls):+.3f} -> {'WINS' if armB else 'FALLS(costume)'}")
if not GATE: P("  OVERALL: bundle target inextricable from contact — cannot cleanly distinguish; resolution limit.")
elif armA and not armB: P("  OVERALL: PHYSICAL wins — bundle-choice=pioneer-inheritance; molecules generic at synapse/routing/sorting (airtight).")
elif armB and not armA: P("  OVERALL: MOLECULAR wins — adhesion-sorting is the one molecular level that acts. Located reopening.")
elif armA and armB: P("  OVERALL: BOTH — pioneers route, followers sort by affinity (dual).")
else: P("  OVERALL: NEITHER cleanly explains bundle co-membership — determinant one layer deeper.")
P("="*82)
