"""ANGLE 3 — Chemotactic / gradient guidance (the molecular reopening).
Do neurite trajectories follow guidance-molecule concentration gradients (netrin/slit/Wnt)?
Gradient FIELD = upstream (source-cell expression x soma positions). Two tests:
  (T1 ROUTING, honest focal-traj flag): does a receptor+ neurite's own trajectory align with its ligand's gradient?
  (T2 PAIR-CONTACT, clean upstream): do i,j (given receptor expr) route toward same field region -> contact?
"""
import sys; sys.path.insert(0,'../accounting')
from traj_benchmark import *
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from bulk_benchmark import _n2c

rng=np.random.default_rng(0)

# ---------- expression ----------
cg=pd.read_csv('../data/celegans/data/expression/cengen/thresholded/021821_liberal_threshold1.csv')
GENECOL=cg.columns[1]; CLASSES=list(cg.columns[3:])
def expr_vec(gene):
    """expression level per RING neuron (mapped via class), 0 if absent."""
    row=cg[cg[GENECOL]==gene]
    if len(row)==0: return np.zeros(len(RING))
    d=dict(zip(CLASSES,row[CLASSES].values[0].astype(float)))
    out=np.zeros(len(RING))
    for i,n in enumerate(RING):
        ci=_n2c(n)
        cls=CLASSES[ci] if ci is not None else None
        out[i]=d.get(cls,0.0)
    return out

LIG=['unc-6','slt-1','cwn-1','cwn-2','egl-20','lin-44']
# ligand -> (receptor, sign): sign +1 attractive (route UP gradient), -1 repulsive (route DOWN)
SYS={ # guidance system: ligand, receptor, sign, label
 'netrin_attr':('unc-6','unc-40',+1),
 'netrin_rep' :('unc-6','unc-5', -1),
 'slit'       :('slt-1','sax-3', -1),   # slit/robo repulsive
 'wnt_cwn1'   :('cwn-1','cam-1', +1),   # Wnt receptors: treat as attractive toward source (exploratory sign)
 'wnt_cwn2'   :('cwn-2','mig-1', +1),
 'wnt_egl20'  :('egl-20','lin-17',+1),
 'wnt_lin44'  :('lin-44','mom-5', +1),
}
LX={g:expr_vec(g) for g in set([v[0] for v in SYS.values()])}
RX={v[1]:expr_vec(v[1]) for v in SYS.values()}

# ---------- gradient field from source-cell expression x soma positions (UPSTREAM) ----------
LAMBDA=7000.0  # nm; ~ soma spacing / neurite reach scale
def field_and_grad(ligvec, x):
    """F_L(x)=sum_c w_c exp(-d/lam); grad points toward sources. x:(3,) -> (F, grad(3,))."""
    src=np.where(ligvec>0)[0]
    if len(src)==0: return 0.0,np.zeros(3)
    S=SOMA[src]; w=np.log1p(ligvec[src])
    diff=x[None,:]-S            # (k,3)  x - source
    d=np.linalg.norm(diff,axis=1)+1e-6
    k=np.exp(-d/LAMBDA)*w
    F=k.sum()
    # dF/dx = sum k * (-(x-S)/(d*lam))  -> points toward sources (up-gradient)
    g=-(k[:,None]*diff/(d[:,None]*LAMBDA)).sum(0)
    return F, g

# precompute gradient at each soma for each ligand, excluding self as source
def grad_at_somas(ligvec):
    G=np.zeros((len(RING),3)); Fv=np.zeros(len(RING))
    for i in range(len(RING)):
        lv=ligvec.copy(); lv[i]=0.0   # exclude self
        Fv[i],G[i]=field_and_grad(lv,SOMA[i])
    return Fv,G
LIGFIELD={g:grad_at_somas(LX[g]) for g in LX}

# ============================================================
# T1: ROUTING TEST (reopening). Uses focal trajectory (honest).
# For neuron n expressing receptor R: does its neurite trajectory move along ligand gradient (sign)?
# alignment = dot(mean neurite direction, sign*grad_at_soma).  Compare receptor+ vs receptor- cells.
# ============================================================
def neurite_dir(n):
    co=np.array(list(SK[n]['coords'].values()),float); s=SOMA[RING.index(n)]
    v=(co-s).mean(0); nv=np.linalg.norm(v)
    return v/nv if nv>0 else np.zeros(3)
NDIR=np.array([neurite_dir(n) for n in RING])

print("="*70); print("T1 ROUTING TEST (per-neurite gradient alignment; uses focal trajectory)")
print("="*70)
routing_rows=[]
for name,(lig,rec,sign) in SYS.items():
    Fv,G=LIGFIELD[lig]
    gn=np.linalg.norm(G,axis=1); ghat=np.where(gn[:,None]>0,G/(gn[:,None]+1e-9),0.0)
    align=sign*(NDIR*ghat).sum(1)          # signed alignment with expected routing direction
    rexpr=RX[rec]
    pos=rexpr>0; neg=~pos
    if pos.sum()<5: continue
    ap=align[pos]; an=align[neg]
    # null: shuffle which neurons are 'receptor+' 1000x, compare mean alignment
    obs=ap.mean()
    null=np.array([align[rng.permutation(len(align))[:pos.sum()]].mean() for _ in range(2000)])
    p=(np.sum(null>=obs)+1)/(len(null)+1) if obs>=null.mean() else (np.sum(null<=obs)+1)/(len(null)+1)
    routing_rows.append((name,rec,int(pos.sum()),round(obs,3),round(an.mean(),3),round(float(null.mean()),3),round(float(p),4)))
print(f"{'system':12s} {'recep':6s} {'n+':>4s} {'align+':>7s} {'align-':>7s} {'nullmean':>8s} {'p':>7s}")
for r in routing_rows: print(f"{r[0]:12s} {r[1]:6s} {r[2]:4d} {r[3]:7.3f} {r[4]:7.3f} {r[5]:8.3f} {r[6]:7.4f}")

# ============================================================
# T2: CLEAN PAIR-CONTACT FEATURE (upstream only, no focal trajectory).
# Predicted routing target for neuron i under system s: soma_i + sign*ghat_i * REACH.
# Pair features per system: co-receptor expr, routing-direction parallelism, predicted-target proximity,
# shared field-value. Do i,j route to same region -> contact.
# ============================================================
REACH=9000.0
def pair_features():
    feats=[]; names=[]
    for name,(lig,rec,sign) in SYS.items():
        Fv,G=LIGFIELD[lig]
        gn=np.linalg.norm(G,axis=1); ghat=np.where(gn[:,None]>0,G/(gn[:,None]+1e-9),0.0)
        rdir=sign*ghat                     # routing direction per neuron
        tgt=SOMA+rdir*REACH                # predicted target region per neuron
        rexpr=RX[rec]; rl=np.log1p(rexpr)
        # per pair
        corec=np.sqrt(rl[PI]*rl[PJ])                          # both express receptor
        par=(rdir[PI]*rdir[PJ]).sum(1)*corec                  # parallel routing weighted by co-receptor
        tgtd=np.linalg.norm(tgt[PI]-tgt[PJ],axis=1)           # predicted target proximity
        tgtprox=np.exp(-tgtd/REACH)*corec
        fvd=np.abs(Fv[PI]-Fv[PJ])                             # shared field value (co-migrate to same level)
        fvsim=np.exp(-fvd/(Fv.std()+1e-9))*corec
        feats+= [par,tgtprox,fvsim,corec]; names+=[f'{name}_par',f'{name}_tgt',f'{name}_fv',f'{name}_corec']
    return np.column_stack(feats),names
PF,PFN=pair_features()
print("\n"+"="*70); print(f"T2 PAIR-CONTACT: {PF.shape[1]} upstream gradient features across {len(SYS)} systems")
print("="*70)

# score the full gradient feature block (clean: upstream, no focal traj)
d,rec=score(PF,'A3-gradient-all',uses_focal_trajectory=False)

# also score each guidance FAMILY separately to see which system (if any) carries signal
fam={'netrin':[i for i,n in enumerate(PFN) if n.startswith('netrin')],
     'slit':[i for i,n in enumerate(PFN) if n.startswith('slit')],
     'wnt':[i for i,n in enumerate(PFN) if n.startswith('wnt')]}
print("\n--- per-family (beyond placement / beyond degree) ---")
famres={}
for f,idx in fam.items():
    dd,_=score(PF[:,idx],f'A3-{f}',uses_focal_trajectory=False)
    famres[f]=dd

np.save('../work/angle3_recovered.npy', rec)
print("\nsaved recovered mask, sum=",int(rec.sum()))

# summary
print("\n"+"="*70); print("ANGLE-3 SUMMARY"); print("="*70)
print("degree_null_ND=",round(DEGNULL_ND,3),"placement_only=",round(d['placement_only_auc'],3))
print("all-systems: beyond_placement=",d['BEYOND_PLACEMENT_delta']," feature_alone=",d['feature_alone_auc']," BEYOND_DEGREE=",d['beyond_degree'])
for f in famres: print(f, "beyond_placement=",famres[f]['BEYOND_PLACEMENT_delta'],"feature_alone=",famres[f]['feature_alone_auc'],"BEYOND_DEGREE=",famres[f]['beyond_degree'])
