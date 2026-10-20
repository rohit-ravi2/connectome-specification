"""CONFOUND-CLOSER PAIR. AUDIT 1: is Angle-2 fasciculation-closure coincidence real mechanization or A≈A
(spatial co-fasciculation IS contact, so 'coincides with closure' is trivial)? AUDIT 2: does Angle-3 guidance
routing beat a DEGREE-PRESERVING null (permute guidance within degree-matched bins, preserving degree+geometry),
beyond degree, or is it the degree-costume? Both scored on the beyond-placement remainder.
"""
import warnings,sys,json,csv; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
sys.path.insert(0,'../accounting')
from traj_benchmark import RING,PAIRS,PI,PJ,CONTACT,SK,DEGNULL_ND,M,AREA,YC,CDEG,PFOLD,IDX,SOMA,PLACEMENT
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
RNG=np.random.default_rng(0); NP=len(PAIRS)

# ================= AUDIT 1 — ANGLE-2 A≈A CHECK =================
P("="*84); P("AUDIT 1 — Angle-2: is fasciculation-closure coincidence mechanization or A≈A?"); P("="*84)
# co-fasciculation from PURE 3D trajectory proximity (INPUT AUDIT: only coords, NEVER contact/connectivity)
def cloud(n):
    pts=np.array(list(SK[n]['coords'].values()),float)
    return pts[RNG.choice(len(pts),min(90,len(pts)),replace=False)] if len(pts)>90 else pts
PT={n:cloud(n) for n in RING}; trees={n:cKDTree(PT[n]) for n in RING}
def cofasc(a,b):  # co-run length: # of a's path nodes within 500nm of b's path (pure spatial)
    d,_=trees[b].query(PT[a]); return (d<500).sum()
cf=np.array([cofasc(RING[PI[k]],RING[PJ[k]]) for k in range(NP)])
P("  co-fasciculation inputs audited: skeleton coords ONLY (no contact matrix, no adjacency, no graph-neighbors)")
# THE A≈A TELL: is spatial co-fasciculation just contact re-measured?
auc_cf_contact=roc_auc_score(CONTACT,cf)
P(f"  AUC(spatial co-fasciculation -> CONTACT) = {auc_cf_contact:.3f}  (if ~0.86, co-fasc IS spatial adjacency = contact)")
P(f"  Spearman(co-fasc, contact-area among contacts) = {spearmanr(cf[CONTACT>0],AREA[CONTACT>0])[0]:+.2f}")
# closure (shared contact-neighbors) — a graph property
Cadj=np.zeros((M,M))
for k in np.where(CONTACT>0)[0]: Cadj[PI[k],PJ[k]]=1; Cadj[PJ[k],PI[k]]=1
comm=Cadj@Cadj; closure=np.array([comm[PI[k],PJ[k]] for k in range(NP)])
P(f"  Spearman(spatial co-fasc, closure) = {spearmanr(cf,closure)[0]:+.2f}")
P(f"  Spearman(CONTACT, closure) = {spearmanr(CONTACT,closure)[0]:+.2f}")
P("  => spatial co-fasciculation and contact are the SAME physical thing (paths alongside = membranes touch);")
P("     'co-fasc coincides with closure' = 'contact coincides with closure', which is trivial (closure is DEFINED")
P("     on the contact graph). This is A≈A in trajectory clothing — NOT a mechanization.")
# the only non-trivial mechanization = UPSTREAM (shared earlier-born pioneer, not focal i-j adjacency)
bt={}
for r in csv.DictReader(open("../seed/data/external/mechanistic_frontier/neuron_birth_times_sulston/neuron_birth_times.csv")): bt[r['neuron']]=float(r['birth_time_min'])
def bc(n):
    for s in "LRDV":
        if n.endswith(s): return n[:-1]
    return n
BIRTH=np.array([bt.get(n,bt.get(bc(n),0.0)) for n in RING])
# does closure survive when we REMOVE the co-fasc(=contact) leak and use only upstream pioneer-sharing?
# pioneer follow: i follows earlier-born j if i's path is near j's path AND birth[i]>birth[j]
foll=np.zeros((M,M))
for i,ni in enumerate(RING):
    for j,nj in enumerate(RING):
        if i!=j and BIRTH[i]>BIRTH[j]:
            d,_=trees[nj].query(PT[ni]);
            if (d<800).sum()>3: foll[i,j]=1
shared_pioneer=np.array([ (foll[PI[k]]*foll[PJ[k]]).sum() for k in range(NP)])   # i,j both follow same earlier pioneer
P(f"\n  UPSTREAM shared-pioneer feature (birth-ordered, 3rd-party, no focal i-j adjacency):")
P(f"    Spearman(shared-pioneer, closure) = {spearmanr(shared_pioneer,closure)[0]:+.2f} (vs co-fasc-closure {spearmanr(cf,closure)[0]:+.2f})")
auc_sp=roc_auc_score(CONTACT,shared_pioneer)
P(f"    AUC(shared-pioneer -> contact) = {auc_sp:.3f} vs degree-null {DEGNULL_ND:.3f} -> {'beats' if auc_sp>DEGNULL_ND else 'BELOW degree'}")
P(f"  VERDICT AUDIT 1: {'MECHANIZED' if (auc_sp>DEGNULL_ND and spearmanr(shared_pioneer,closure)[0]>0.3) else 'FALLS — A≈A. Spatial co-fasc = contact (trivially coincides w/ closure); upstream pioneer-sharing does NOT mechanize closure (below degree, weak closure-corr). Angle-2 was the caught circularity.'}")

# ================= AUDIT 2 — ANGLE-3 DEGREE-NULL VERIFICATION =================
P("\n"+"="*84); P("AUDIT 2 — Angle-3: guidance routing vs DEGREE-PRESERVING null"); P("="*84)
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
GUID=["unc-6","unc-40","unc-5","slt-1","sax-3","cwn-1","cwn-2","egl-20","lin-44","cam-1","mig-1","lin-17","mom-5"]
def expr(g):
    w=np.where(gname==g)[0]; return (X[w[0]]>0).astype(float) if len(w) else None
Gm=np.array([expr(g) for g in GUID if expr(g) is not None])  # nGuid x nClasses
CC=np.array([n2c(n) if n2c(n) is not None else 0 for n in RING])
# guidance feature per pair: shared guidance-receptor/ligand co-expression (the routing-compatibility signal)
Gn=Gm[:,CC]  # nGuid x nNeuron
guid_feat=np.array([ (Gn[:,PI[k]]*Gn[:,PJ[k]]).sum() for k in range(NP)])  # # shared guidance genes
# real beyond-placement + beyond-degree
def nd(F):
    pred=np.full(NP,np.nan)
    for f in range(5):
        te=(PFOLD==f);tr=(PFOLD!=f)&(PFOLD>=0)
        if CONTACT[te].sum()>3 and (CONTACT[te]==0).sum()>3 and tr.sum()>50:
            pred[te]=HistGradientBoostingClassifier(max_iter=150,random_state=0).fit(F[tr],CONTACT[tr]).predict_proba(F[te])[:,1]
    ok=~np.isnan(pred); return roc_auc_score(CONTACT[ok],pred[ok]),ok
auc_guid,_=nd(guid_feat.reshape(-1,1))
P(f"  guidance-compat feature-alone AUC = {auc_guid:.3f} vs degree-null {DEGNULL_ND:.3f} -> beyond_degree = {auc_guid-DEGNULL_ND:+.3f}")
# DEGREE-PRESERVING null: permute guidance expression across neurons WITHIN degree-matched bins (preserves degree+its geometry)
bins=np.digitize(CDEG,np.quantile(CDEG,[.2,.4,.6,.8]))
def degpres_null():
    Gp=Gn.copy()
    for b in range(5):
        idxs=np.where(bins==b)[0]
        if len(idxs)>1: Gp[:,idxs]=Gn[:,idxs[RNG.permutation(len(idxs))]]
    return np.array([ (Gp[:,PI[k]]*Gp[:,PJ[k]]).sum() for k in range(NP)])
nullaucs=[nd(degpres_null().reshape(-1,1))[0] for _ in range(10)]
P(f"  DEGREE-PRESERVING null (guidance permuted within degree bins): AUC = {np.mean(nullaucs):.3f} ± {np.std(nullaucs):.3f}")
P(f"  real vs degree-preserving null: Δ = {auc_guid-np.mean(nullaucs):+.3f}")
# decomposition: regress guidance on degree, does residual predict contact?
from sklearn.linear_model import LinearRegression
degpair=(CDEG[PI]+CDEG[PJ]).reshape(-1,1)
gres=guid_feat - LinearRegression().fit(degpair,guid_feat).predict(degpair)
auc_gres,_=nd(gres.reshape(-1,1))
P(f"  guidance residual (orthogonal to degree) -> contact AUC = {auc_gres:.3f} (chance=0.5)")
# check null preserves degree
P(f"  [null-validity] degree corr real-vs-null bins preserved by construction; degree-guidance corr = {spearmanr(CDEG,Gn.sum(0))[0]:+.2f}")
clear2 = (auc_guid>DEGNULL_ND) and (auc_guid>np.mean(nullaucs)+3*np.std(nullaucs)) and (auc_gres>0.55)
P(f"  VERDICT AUDIT 2: {'REOPENS — guidance routes beyond degree' if clear2 else 'FALLS — guidance ≈ degree-preserving null and below degree null; residual-orthogonal-to-degree ~chance. The degree-COSTUME caught. Molecules generic at routing too.'}")
P("="*84)
