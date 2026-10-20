"""PIONEER-REPRODUCIBILITY TEST. Fork: are pioneers (earliest-born) reproducible-but-unexplained (real
non-placement determinant, located) or not-reproducible (noise floor, physical account holds)? Measure: cross-animal
reproducibility of each neuron's frame-invariant co-routing NEIGHBOR SET across the 4 L1 replicate animals (Witvliet
Datasets 1-4), noise-corrected (Stage-1 logic), pioneers vs followers vs random. Cook has no skeletons -> trajectory
reproducibility is within-Witvliet only (flagged).
"""
import warnings,sys,json,csv; warnings.simplefilter("ignore")
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu
def P(*a): print(*a); sys.stdout.flush()
RNG=np.random.default_rng(0)
SKD="../seed/data/external/mechanistic_frontier/witvliet2021_developmental/data/skeletons"
L1SETS=[1,2,3,4]; ADULT=[7,8]
def load(ds):
    d=json.load(open(f"{SKD}/Dataset{ds}_skeletons.json"))
    out={}
    for n,v in d.items():
        if v.get('coords'):
            pts=np.array(list(v['coords'].values()),float)
            out[n]=pts[RNG.choice(len(pts),min(60,len(pts)),replace=False)] if len(pts)>60 else pts
    return out
SK={ds:load(ds) for ds in L1SETS+ADULT}
# birth times
bt={}
for r in csv.DictReader(open("../seed/data/external/mechanistic_frontier/neuron_birth_times_sulston/neuron_birth_times.csv")): bt[r['neuron']]=float(r['birth_time_min'])
def bc(n):
    for s in "LRDV":
        if n.endswith(s): return n[:-1]
    return n
def bof(n): return bt.get(n,bt.get(bc(n),np.nan))
# neurons present in ALL 4 L1 animals + birth time
common=set(SK[1])
for ds in L1SETS[1:]: common&=set(SK[ds])
neurons=sorted([n for n in common if not np.isnan(bof(n))])
BIRTH=np.array([bof(n) for n in neurons]); M=len(neurons)
P(f"neurons in all 4 L1 animals with birth time: {M}")
# per-animal neighbor sets (frame-invariant identity sets; within-animal proximity)
def neighbors(ds,THR):
    present=[n for n in neurons if n in SK[ds]]
    trees={n:cKDTree(SK[ds][n]) for n in present}
    nb={}
    for i,ni in enumerate(present):
        s=set()
        for nj in present:
            if ni!=nj:
                d,_=trees[nj].query(SK[ds][ni]);
                if d.min()<THR: s.add(nj)
        nb[ni]=s
    return nb
def jac(a,b): return len(a&b)/max(1,len(a|b))
def repro(nb_map):
    R=np.full(M,np.nan)
    for i,n in enumerate(neurons):
        js=[jac(nb_map[a].get(n,set()),nb_map[b].get(n,set())) for a in L1SETS for b in L1SETS if a<b]
        R[i]=np.mean(js)
    return R
pio=BIRTH<=np.nanquantile(BIRTH,0.35); fol=BIRTH>=np.nanquantile(BIRTH,0.65)
P("\n"+"="*80); P("PIONEER-REPRODUCIBILITY FORK — THR sweep (noise-corrected co-routing reproducibility)"); P("="*80)
results={}
for THR in [500.0,800.0,1200.0]:
    NB={ds:neighbors(ds,THR) for ds in L1SETS}
    ss=np.mean([np.mean([len(NB[ds].get(n,set())) for ds in L1SETS]) for n in neurons])
    Robs=repro(NB)
    # noise floor: permute neighbor identities preserving set size
    accs=[]
    for _ in range(12):
        NBp={}
        for ds in L1SETS:
            pool=[n for n in neurons if n in SK[ds]]
            NBp[ds]={n:set(RNG.choice(pool,min(len(NB[ds].get(n,set())),len(pool)),replace=False)) for n in pool}
        accs.append(repro(NBp))
    Rnull=np.nanmean(accs,0); Rcorr=Robs-Rnull
    pr=np.nanmean(Rcorr[pio]); fr=np.nanmean(Rcorr[fol])
    u,pval=mannwhitneyu(Rcorr[pio][~np.isnan(Rcorr[pio])],Rcorr[fol][~np.isnan(Rcorr[fol])],alternative='greater')
    results[THR]=(ss,pr,fr,pval)
    P(f"  THR={THR:.0f}nm (mean set {ss:.0f}): corrected-repro pioneers={pr:.3f} followers={fr:.3f} | pio>fol p={pval:.2f}")
pio_repro=np.mean([results[t][1] for t in results]); more_than_followers=all(results[t][3]<0.05 and results[t][1]>results[t][2] for t in results)
Rcorr_all=np.nanmean([results[t][1] for t in results])
P("\n"+"="*80)
if pio_repro<=0.02:
    P("READ: pioneers NOT reproducible (corrected≈0) -> NOISE FLOOR. Pioneer routing is developmental noise that")
    P("  followers amplify into the residual. Physical account holds; the residual is pioneer-noise-inheritance.")
    P("  Clean DEFLATIONARY result: no non-placement specification found; located the noise floor.")
elif more_than_followers:
    P("READ: pioneers REPRODUCIBLE and MORE than followers (noise-corrected) -> a real reproducible non-placement")
    P("  determinant is LOCATED on the pioneer population. Proceed to Phase 2 (what routes them).")
else:
    P(f"READ: pioneers reproducible ({pio_repro:.3f}) but NOT specially more than followers (p={pval:.2f}) -> pioneers")
    P("  are as reproducible as anything (the whole connectome is ~reproducible); no SPECIAL pioneer specification.")
    P("  The reproducibility is general (geometry is reproducible), not a pioneer-concentrated determinant.")
P("="*80)
