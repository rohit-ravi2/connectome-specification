#!/usr/bin/env python3
"""Round3 FORMAL branch -- Morphospace Funnel (constraint entropy reduction).

Formalizes 'specified information = reduction of the space of possible connectomes'.
For directed chemical edges among nerve-ring neurons, build independent-edge ensembles with
increasing constraints and measure:
  H(E)          = ensemble entropy in bits = size of reachable morphospace
  NLL_observed  = bits to encode the OBSERVED strong-core under each ensemble (lower=more typical)
  funnel        = how sharply constraints shrink morphospace toward the observed connectome

Ensembles (per candidate directed pair, independent-edge with prob p_e, Σp_e = K):
  E0 uniform | E1 degree-preserved (Chung-Lu) | E2 contact-gated | E3 contact-area-weighted
  E4 +arrival-compat

Key test: if under E3 the observed strong-core NLL ~ H(E3), the connectome is a TYPICAL sample of
the contact-area-constrained ensemble -> no molecular 'specification' beyond geometry. Also test
whether the WEAK-HALO (1-animal edges) is a max-entropy sample given contact area.
"""
import openpyxl, csv, re, os, math, json
import numpy as np
from collections import defaultdict

DATA="../data/celegans/data"
OUT=os.path.expanduser("../seed/data/processed/formal_branch")
CONTACT=f"{DATA}/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx"
EDGES=f"{DATA}/connectome/witvliet2020/witvliet_combined_edges.csv"
LIN="../lineage/jtauber_lineage/cells.json"

tree=json.load(open(LIN)); depth={}
def walk(n,d):
    depth[n["name"]]=d
    for c in n.get("children",[]): walk(c,d+1)
walk(tree,0)

wb=openpyxl.load_workbook(CONTACT,read_only=True,data_only=True); ws=wb['adult nerve ring neighbors']
rows=list(ws.iter_rows(values_only=True)); alln=[h for h in rows[0][1:]]; idx={n:i for i,n in enumerate(alln)}; M=len(alln)
A=np.zeros((M,M))
for row in rows[1:]:
    rn=row[0]
    if rn is None or rn not in idx: continue
    for j,val in enumerate(row[1:]):
        if val in (None,''): continue
        try: A[idx[rn],j]=float(val)
        except: pass
A=np.maximum(A,A.T)

# directed reproducible strong-core (both adults) + halo (1 animal), chemical
pres=defaultdict(set)
for r in csv.DictReader(open(EDGES)):
    ds=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
    if ds not in (7,8) or r['type']=='electrical': continue
    pres[(r['pre'],r['post'])].add(ds)

ring=[n for n in alln if n in depth]           # neurons with contact + lineage
ri={n:i for i,n in enumerate(ring)}; R=len(ring)
# candidate directed edges = ordered pairs with CONTACT (undirected area>0)
cand=[]; areav=[]; arrv=[]
for a in ring:
    for b in ring:
        if a==b: continue
        i,j=idx[a],idx[b]
        if A[i,j]<=0: continue                 # only contacting pairs are candidates for E2+
        cand.append((a,b)); areav.append(A[i,j]); arrv.append(abs(depth[a]-depth[b]))
cand_all=[(a,b) for a in ring for b in ring if a!=b]     # for E0/E1 universe
areav=np.array(areav); arrv=np.array(arrv)
strong=np.array([1 if len(pres.get(e,()))==2 else 0 for e in cand])
halo  =np.array([1 if len(pres.get(e,()))==1 else 0 for e in cand])
K=int(strong.sum()); Ph=len(cand_all); Pc=len(cand)
print(f"ring neurons {R} | candidate contacting directed pairs {Pc} | all directed pairs {Ph}")
print(f"strong-core directed edges K={K} ({100*strong.mean():.1f}% of contacts) | halo {halo.sum()}")

def entropy_bits(p):
    p=np.clip(p,1e-12,1-1e-12); return float(-(p*np.log2(p)+(1-p)*np.log2(1-p)).sum())
def nll_bits(y,p):
    p=np.clip(p,1e-12,1-1e-12); return float(-(y*np.log2(p)+(1-y)*np.log2(1-p)).sum())
def scale_to_K(w,K):
    """independent-edge probs p=min(1,lambda*w) with sum p = K via bisection on lambda."""
    w=np.asarray(w,float)
    lo,hi=0.0, K/ w[w>0].min() if (w>0).any() else 1.0
    for _ in range(100):
        lam=(lo+hi)/2; p=np.minimum(1.0,lam*w)
        if p.sum()>K: hi=lam
        else: lo=lam
    return np.minimum(1.0,lam*w)

# ----- ensembles over the STRONG-CORE target (K edges among contacts) -----
# E0 uniform over ALL directed pairs
pE0=np.full(Pc, K/Ph)                          # uniform prob per pair (restricted view on contacts)
# E1 degree-preserved (Chung-Lu) using strong-core degrees
dout=defaultdict(float); din=defaultdict(float)
for (a,b),s in zip(cand,strong):
    if s: dout[a]+=1; din[b]+=1
wcl=np.array([dout[a]*din[b] for a,b in cand])
pE1=scale_to_K(wcl+1e-9,K)
# E2 contact-gated uniform (all candidates are contacts): uniform over contacts
pE2=np.full(Pc, K/Pc)
# E3 contact-area-weighted
pE3=scale_to_K(areav,K)
# E4 area * arrival-compat (closer arrival favored): weight = area * exp(-arr/mean)
warr=areav*np.exp(-arrv/ (arrv.mean()+1e-9))
pE4=scale_to_K(warr,K)

ens=[("E0 uniform(all pairs)",pE0),("E1 degree-preserved",pE1),("E2 contact-gated",pE2),
     ("E3 contact-area-weighted",pE3),("E4 +arrival-compat",pE4)]
print("\n============ MORPHOSPACE FUNNEL (target: strong-core directed edges) ============")
print(f"{'ensemble':26s} {'H(E) bits':>10s} {'ΔH':>8s} {'NLL_obs bits':>13s} {'NLL/edge':>9s}")
prevH=None
funnel=[]
for name,p in ens:
    H=entropy_bits(p); nll=nll_bits(strong,p); dH=(prevH-H) if prevH is not None else 0.0
    funnel.append((name,H,dH,nll)); prevH=H
    print(f"{name:26s} {H:>10.0f} {dH:>8.0f} {nll:>13.0f} {nll/K:>9.3f}")
print(f"\n  H(E0)->H(E3): morphospace shrinks {funnel[0][1]-funnel[3][1]:.0f} bits by contact+area constraints")
print(f"  observed strong-core NLL under E3 = {funnel[3][3]:.0f} bits vs H(E3)={funnel[3][1]:.0f} bits")
print(f"  NLL/H ratio under E3 = {funnel[3][3]/funnel[3][1]:.3f}  (~1 => observed is a TYPICAL E3 sample)")

# ----- weak-halo max-entropy test: is halo an area-weighted random sample given strong-core removed? -----
print("\n============ WEAK-HALO = MAX-ENTROPY GIVEN GEOMETRY? ============")
# among contacts NOT in strong-core, model halo prob ∝ area; compare halo area-dependence to null
noncore=strong==0
Kh=int(halo[noncore].sum())
pHalo=scale_to_K(areav[noncore],Kh)
# calibration: bin by area decile, compare observed halo rate vs predicted
a_nc=areav[noncore]; h_nc=halo[noncore]
dec=np.clip((np.argsort(np.argsort(a_nc))*10//len(a_nc)),0,9)
print(f"  halo among non-core contacts: {Kh}/{noncore.sum()} ({100*h_nc.mean():.1f}%)")
print(f"  {'area decile':11s} {'obs halo%':>9s} {'pred%(area-wt)':>14s}")
obs_all=[]; pred_all=[]
for dd in range(10):
    mm=dec==dd
    obs=100*h_nc[mm].mean(); pred=100*pHalo[mm].mean()
    obs_all.append(h_nc[mm].mean()); pred_all.append(pHalo[mm].mean())
    print(f"  {dd:>11d} {obs:>9.1f} {pred:>14.1f}")
# chi-like agreement
obs_all=np.array(obs_all); pred_all=np.array(pred_all)
corr=np.corrcoef(obs_all,pred_all)[0,1]
print(f"  obs-vs-area-weighted halo rate correlation across deciles: r={corr:.3f}")
print("  => high r: weak-halo is well-described as area-weighted (max-entropy) random given geometry.")

with open(f"{OUT}/morphospace_funnel.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["ensemble","H_bits","deltaH","NLL_obs_bits","NLL_per_edge"])
    for name,H,dH,nll in funnel: w.writerow([name,round(H,1),round(dH,1),round(nll,1),round(nll/K,4)])
    w.writerow([]); w.writerow(["halo_area_decile_corr",round(corr,4)])
print(f"\nsaved -> {OUT}/morphospace_funnel.csv")
