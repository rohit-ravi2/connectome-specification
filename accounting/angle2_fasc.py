import sys; sys.path.insert(0,'../accounting')
from traj_benchmark import *
from traj_benchmark import _nd_cv
import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score

N=len(RING)
PC=[np.array(list(SK[n]['coords'].values()),float) for n in RING]
TR=[cKDTree(p) for p in PC]
NN=np.array([len(p) for p in PC])
T=500.0  # co-running threshold (nm); contact 90th-pct min-dist ~707nm

# ---- directed co-run fraction matrix CF[i,p] = frac of i's nodes within T of p's cloud ----
CF=np.zeros((N,N))
for i in range(N):
    for p in range(N):
        if i==p: continue
        d,_=TR[p].query(PC[i])
        CF[i,p]=(d<T).mean()
np.save('../work/angle2_CF.npy',CF)

# ================= LEAKY CO-ROUTE feature (uses focal i-j mutual trajectory) =================
coroute_sym = np.array([max(CF[PI[k],PJ[k]],CF[PJ[k],PI[k]]) for k in range(NP)])
# min approach distance (definitional leak, for reference)
mind=np.array([min(TR[PJ[k]].query(PC[PI[k]])[0].min(), TR[PI[k]].query(PC[PJ[k]])[0].min()) for k in range(NP)])
corun_len=np.array([(TR[PJ[k]].query(PC[PI[k]])[0]<T).sum() for k in range(NP)])
LEAK_FEAT=np.column_stack([coroute_sym, np.exp(-mind/500.0), corun_len])

print("\n===== LEAKY CO-ROUTE (focal i-j mutual co-fasciculation) =====")
d_leak,rec_leak=score(LEAK_FEAT,'A2-fasciculation-COROUTE(leaky)',uses_focal_trajectory=True)

# ================= CLEAN SHARED-PIONEER feature =================
# i "follows" earlier-born p if CF[i,p] high AND BIRTH[p] < BIRTH[i]. Pioneer p != i,j.
# Feature for (i,j) = shared earlier-born pioneers both follow. Uses each focal's path vs THIRD
# parties, never the definitional i-j mutual proximity.
FOL=np.where(BIRTH[:,None] > BIRTH[None,:], CF, 0.0)  # FOL[i,p]>0 only if p born strictly before i
np.fill_diagonal(FOL,0.0)

sp_count=np.zeros(NP); sp_wsum=np.zeros(NP); sp_max=np.zeros(NP); sp_jac=np.zeros(NP)
THR=0.10  # a neuron 'follows' a pioneer if >=10% of its nodes co-run with it
for k in range(NP):
    i,j=PI[k],PJ[k]
    fi=FOL[i].copy(); fj=FOL[j].copy()
    fi[i]=fi[j]=0; fj[i]=fj[j]=0   # exclude i,j themselves as pioneers
    both=(fi>=THR)&(fj>=THR)
    sp_count[k]=both.sum()
    sp_wsum[k]=np.minimum(fi,fj)[both].sum()
    sp_max[k]=np.minimum(fi,fj).max() if both.any() else 0.0
    seti=(fi>=THR); setj=(fj>=THR); un=(seti|setj).sum()
    sp_jac[k]=both.sum()/un if un>0 else 0.0
CLEAN_FEAT=np.column_stack([sp_count,sp_wsum,sp_max,sp_jac])

print("\n===== CLEAN SHARED-PIONEER (birth-ordered, third-party pioneers only) =====")
d_clean,rec_clean=score(CLEAN_FEAT,'A2-fasciculation-SHARED-PIONEER(clean)',uses_focal_trajectory=False)
np.save('../work/angle2_recovered.npy',rec_clean)

# ================= KEY CROSS-TEST: fasciculation == triadic closure? =================
# closure(i,j) = # shared CONTACT-neighbors k (k!=i,j)
A=np.zeros((N,N),bool)
for k in range(NP):
    A[PI[k],PJ[k]]=CONTACT[k]; A[PJ[k],PI[k]]=CONTACT[k]
closure=np.zeros(NP)
for k in range(NP):
    i,j=PI[k],PJ[k]
    sh=A[i]&A[j]; sh[i]=sh[j]=False
    closure[k]=sh.sum()

from scipy.stats import spearmanr
print("\n===== CROSS-TEST: fasciculation vs triadic closure =====")
print("spearman(shared-pioneer count, closure) =", round(spearmanr(sp_count,closure)[0],3))
print("spearman(shared-pioneer wsum,  closure) =", round(spearmanr(sp_wsum,closure)[0],3))
print("spearman(coroute-leaky, closure)        =", round(spearmanr(coroute_sym,closure)[0],3))
# overlap of top-predicted pairs
def topset(x,frac=0.10):
    n=int(len(x)*frac); return set(np.argsort(-x)[:n])
for nm,x in [('shared-pioneer',sp_wsum),('coroute-leaky',coroute_sym)]:
    a=topset(x); b=topset(closure)
    jac=len(a&b)/len(a|b)
    print(f"top-10% overlap({nm} vs closure) jaccard={jac:.3f}")
# does closure ADD beyond shared-pioneer (are they the same signal)?
print("\n-- closure as its own feature (for reference) --")
score(closure,'A2-closure-ref',uses_focal_trajectory=False)
# does shared-pioneer add BEYOND closure? (if fasc IS closure, delta~0)
print("\n-- shared-pioneer ON TOP of placement+closure --")
Fpc=np.hstack([PLACEMENT,closure[:,None]])
pc,okpc=_nd_cv(Fpc)
auc_pc=roc_auc_score(CONTACT[okpc],pc[okpc])
Fpcs=np.hstack([PLACEMENT,closure[:,None],CLEAN_FEAT])
pcs,okpcs=_nd_cv(Fpcs); auc_pcs=roc_auc_score(CONTACT[okpcs],pcs[okpcs])
print(f"place+closure AUC={auc_pc:.3f}  place+closure+sharedpioneer AUC={auc_pcs:.3f}  delta={auc_pcs-auc_pc:+.3f}")

# ---- variance sanity: how universal is pioneer-sharing? ----
print("\n-- shared-pioneer variance --")
print("sp_count distribution pct[10,50,90,99]", np.percentile(sp_count,[10,50,90,99]))
print("frac pairs with >=1 shared pioneer:", round((sp_count>=1).mean(),3))

# ================= REFINED CLEAN: same-pioneer-ARC (strongest fair shot) =================
# i,j co-fasciculate if they hug the SAME STRETCH of an earlier pioneer p (overlap of p-node
# indices near i and near j). Meeting locus = p's arc (third party), birth-ordered. Note ~2T
# co-location softness (both within T of a common p-node => within ~2T of each other there).
arc_ov=np.zeros(NP)
# per (neuron n, pioneer p): boolean over p's nodes that are within T of n  -> cache
NEAR={}
def near_pnodes(n,p):
    key=(n,p)
    if key not in NEAR:
        d=TR[n].query(PC[p])[0]   # for each p-node, dist to nearest n-node
        NEAR[key]=(d<T)
    return NEAR[key]
for k in range(NP):
    i,j=PI[k],PJ[k]; best=0; lo=min(BIRTH[i],BIRTH[j])
    for p in range(N):
        if p==i or p==j or BIRTH[p]>=lo: continue
        if CF[i,p]<THR or CF[j,p]<THR: continue
        ov=int((near_pnodes(i,p)&near_pnodes(j,p)).sum())  # shared stretch of pioneer p
        if ov>best: best=ov
    arc_ov[k]=best
print("\n===== REFINED CLEAN: same-pioneer-ARC overlap =====")
score(arc_ov,'A2-fasciculation-PIONEER-ARC(clean-ish, ~2T soft)',uses_focal_trajectory=False)
print("spearman(pioneer-arc, closure)=",round(__import__('scipy.stats',fromlist=['spearmanr']).spearmanr(arc_ov,closure)[0],3))
