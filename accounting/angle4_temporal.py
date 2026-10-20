"""ANGLE 4 — TEMPORAL / SEQUENTIAL ROUTING.
Hypothesis: beyond-placement contact remainder = history-dependent routing. Same soma positions +
different birth-ORDER -> different routing, because late neurites navigate around/along the scaffold of
earlier-born neurites present at their outgrowth time.

LEAK DISCIPLINE:
 - Placement baseline ALREADY has raw BIRTH[i], BIRTH[j], |birth diff|. So a pure co-natality/birth-time
   feature is already used; Angle 4 must add birth-ORDER-dependent ROUTING structure beyond that.
 - The FOCAL pair's own adult trajectory = the leak (contact = focal adult adjacency). We NEVER use member
   i's or j's adult neurite cloud to predict contact of (i,j).
 - Scaffold density uses THIRD-PARTY (k != i,j) earlier-born neurites' adult clouds as a proxy for the
   developmental tissue-state. Declared honestly; not the focal pair's path.
"""
import sys; sys.path.insert(0,'../accounting')
from traj_benchmark import *
import numpy as np
from scipy.spatial import cKDTree

rng=np.random.default_rng(0)
M=len(RING)

# ---- per-neuron adult point clouds (for THIRD-PARTY scaffold only) + arbor length (obs-window control) ----
CLOUD=[]; ARBOR_LEN=np.zeros(M); NPTS=np.zeros(M)
for i,n in enumerate(RING):
    co=SK[n]['coords']; pts=np.array([co[k] for k in co],float)
    CLOUD.append(pts); ARBOR_LEN[i]=SK[n]['length']; NPTS[i]=len(pts)
# downsample clouds for speed
CLOUDds=[c[rng.integers(0,len(c),min(len(c),60))] if len(c)>60 else c for c in CLOUD]

birth=BIRTH.copy()
brank=np.argsort(np.argsort(birth))  # birth rank 0..M-1

# =========================================================================================
# TIER 1 — CLEAN birth-order features (NO trajectory at all). uses_focal_trajectory=False, truly clean.
# =========================================================================================
bi=birth[PI]; bj=birth[PJ]
bmin=np.minimum(bi,bj); bmax=np.maximum(bi,bj)         # scaffold(early) vs navigator(late)
bdiff=np.abs(bi-bj)
rmin=np.minimum(brank[PI],brank[PJ]); rmax=np.maximum(brank[PI],brank[PJ])
conat=np.exp(-bdiff/40.0)                               # co-natality kernel (~40 min)
somad=np.array([np.linalg.norm(SOMA[PI[k]]-SOMA[PJ[k]]) for k in range(NP)])
# history interaction: co-natal AND soma-near -> route together through same young tissue
conat_near=conat*np.exp(-somad/8000.0)
# navigator-constrained: later neuron's rank (how much prior structure existed at its outgrowth)
CLEAN=np.column_stack([bmin,bmax,rmin,rmax,conat,conat_near])

d1,rec1=score(CLEAN,'A4-temporal-CLEAN(birth-order only)',uses_focal_trajectory=False)

# =========================================================================================
# TIER 2 — SCAFFOLD density from THIRD-PARTY earlier neurites (declared: uses OTHER neurons' adult clouds)
# For each pair, at t=max(birth_i,birth_j): scaffold = neurons k!=i,j with birth_k < t.
# corridor = midpoint of the two somata. density = # scaffold points within radius R of midpoint.
# Also: shared-cohort routing = # of k co-natal with the pair whose cloud is near the corridor.
# =========================================================================================
somai=SOMA[PI]; somaj=SOMA[PJ]; mid=(somai+somaj)/2.0
R=6000.0
scaf_dense=np.zeros(NP); scaf_conat=np.zeros(NP); scaf_all=np.zeros(NP)
# precompute per-neuron KD trees on downsampled clouds
trees=[cKDTree(c) for c in CLOUDds]
for k in range(NP):
    i,j=PI[k],PJ[k]; t=bmax[k]; m=mid[k]
    dens=0.0; cohort=0.0; alld=0.0
    for kk in range(M):
        if kk==i or kk==j: continue
        # count this third-party neurite's points near the corridor midpoint
        cnt=len(trees[kk].query_ball_point(m,R))
        alld+=cnt
        if birth[kk] < t: dens+=cnt                     # already present at outgrowth
        if abs(birth[kk]-((bi[k]+bj[k])/2)) < 30: cohort+=cnt  # co-natal cohort near corridor
    scaf_dense[k]=dens; scaf_conat[k]=cohort; scaf_all[k]=alld
# fraction of local structure already present at outgrowth (history-order signal, corridor-normalized)
scaf_frac=scaf_dense/(scaf_all+1.0)
np.save('../work/a4_scaf.npy',np.column_stack([scaf_dense,scaf_conat,scaf_all,scaf_frac]))

SCAF=np.column_stack([scaf_dense,scaf_conat,scaf_frac])
d2,rec2=score(SCAF,'A4-temporal-SCAFFOLD(3rd-party clouds)',uses_focal_trajectory=False)

FULL=np.column_stack([CLEAN,SCAF])
d3,rec3=score(FULL,'A4-temporal-FULL',uses_focal_trajectory=False)

# =========================================================================================
# OBSERVATION-WINDOW CONTROL: partial out arbor length / npts. Does the temporal signal survive?
# Regress each temporal feature on (arbor_i,arbor_j,npts_i,npts_j,somad) and score the RESIDUAL.
# =========================================================================================
from numpy.linalg import lstsq
ai=ARBOR_LEN[PI]; aj=ARBOR_LEN[PJ]; ni=NPTS[PI]; nj=NPTS[PJ]
OBS=np.column_stack([np.ones(NP),ai,aj,ai*aj,ni,nj,somad,somad**2])
def resid(x):
    b,_,_,_=lstsq(OBS,x,rcond=None); return x-OBS@b
CLEAN_r=np.column_stack([resid(CLEAN[:,c]) for c in range(CLEAN.shape[1])])
SCAF_r=np.column_stack([resid(SCAF[:,c]) for c in range(SCAF.shape[1])])
FULL_r=np.column_stack([CLEAN_r,SCAF_r])
d4,rec4=score(FULL_r,'A4-temporal-FULL(obs-window residualized)',uses_focal_trajectory=False)

# =========================================================================================
# Save recovered mask (best clean beyond-placement recovery)
# =========================================================================================
best=max([(d1,rec1),(d2,rec2),(d3,rec3),(d4,rec4)],key=lambda x:x[0]['BEYOND_PLACEMENT_delta'])
np.save('../work/angle4_recovered.npy', best[1])
print('\nSAVED best recovered:',best[0]['name'],'delta',best[0]['BEYOND_PLACEMENT_delta'],'n_rec',int(np.nansum(best[1])))

# ---- correlation of temporal scaffold feature with degree/arbor (confound diagnostics) ----
print('\nCONFOUND DIAGNOSTICS:')
print(' corr(scaf_dense, CDEG_i*CDEG_j) =',round(np.corrcoef(scaf_dense,CDEG[PI]*CDEG[PJ])[0,1],3))
print(' corr(scaf_dense, arbor_i*arbor_j)=',round(np.corrcoef(scaf_dense,ai*aj)[0,1],3))
print(' corr(bmax, arbor_later)          =',round(np.corrcoef(bmax,np.maximum(ai,aj))[0,1],3))
print(' corr(conat, contact)             =',round(np.corrcoef(conat,CONTACT)[0,1],3))
print(' corr(scaf_frac, contact)         =',round(np.corrcoef(scaf_frac,CONTACT)[0,1],3))
