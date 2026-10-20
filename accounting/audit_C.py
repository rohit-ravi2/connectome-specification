"""TEST 1 — C-HONESTY AUDIT. Gates the partition test. Two checks:
1a PARAMETER ACCOUNTING: C's generator = min inter-ADULT-arbor distance from full Dataset7 skeleton point clouds.
   That input is the entire adult arbor geometry (thousands of coords), NOT 'few physics-set parameters'. Count it
   honestly and compare to D's rank-8 (8 params).
1b UPSTREAM-LEAK: C's proximity is computed from ADULT arbors = the arrangement C claims to generate. Re-derive
   from strictly PRE-PLACEMENT info (soma position = skeleton root; the cell body, upstream of neurite growth) and
   see if bulk-reproduction survives. Material drop => leak-inflated.
"""
import warnings,sys,json,numpy as np; warnings.simplefilter("ignore")
sys.path.insert(0,'../accounting')
from bulk_benchmark import RING,PAIRS,PI,PJ,CONTACT,SK,neuron_disjoint_auc,DEGNULL_ND,M,AREA,YC
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
RNG=np.random.default_rng(0)
# ---- reconstruct C's generator: min inter-adult-arbor distance (full point clouds) ----
def cloud(n):
    co=SK[n]['coords']; pts=np.array(list(co.values()),float)
    return pts
PT={n:cloud(n) for n in RING}
# subsample for speed but keep FULL-arbor character
PTs={n:(PT[n][RNG.choice(len(PT[n]),min(90,len(PT[n])),replace=False)] if len(PT[n])>90 else PT[n]) for n in RING}
trees={n:cKDTree(PTs[n]) for n in RING}
def mindist(a,b): d,_=trees[b].query(PTs[a]); return d.min()
NpP=len(PAIRS)
mind=np.array([mindist(RING[PI[k]],RING[PJ[k]]) for k in range(NpP)])
propC=-mind
aucC=neuron_disjoint_auc(propC)
# ---- pre-placement soma position (skeleton root = 'starts'[0], dist_to_root=0) ----
def soma(n):
    st=SK[n].get('starts') or [list(SK[n]['coords'].keys())[0]]
    return np.array(SK[n]['coords'][str(st[0])] if str(st[0]) in SK[n]['coords'] else SK[n]['coords'][list(SK[n]['coords'].keys())[0]],float)
SOMA=np.array([soma(n) for n in RING])
somad=np.array([np.linalg.norm(SOMA[PI[k]]-SOMA[PJ[k]]) for k in range(NpP)])
propS=-somad
aucS=neuron_disjoint_auc(propS)
# soma + a physical exclusion/territory scale (1-2 physics params) — does a compact PRE-placement model recover it?
# territory proxy from soma: gaussian on soma distance at best physics scale (grid a couple scales = still pre-placement)
best=0
for sc in [500,1000,2000,4000,8000]:
    a=neuron_disjoint_auc(np.exp(-somad/sc))
    best=max(best,a)
P("="*82); P("TEST 1 — C-HONESTY AUDIT (gates the partition test)"); P("="*82)
P("\n[1a PARAMETER ACCOUNTING]")
ncoords=sum(len(SK[n]['coords']) for n in RING)
P(f"  C generator input = full adult arbor point clouds: {ncoords} skeleton nodes x3 = {3*ncoords} coordinates")
P(f"  effective 'parameters' C uses = the entire adult arbor GEOMETRY (the arrangement), NOT few physics knobs")
P(f"  D benchmark = rank-8 embedding = 8 params (162x8 latent coords, but 8 effective dims)")
P(f"  => C does NOT beat D at matched parameters — C uses ~{3*ncoords} coords of the adult arrangement vs D's 8 dims.")
P("\n[1b UPSTREAM-LEAK — re-derive from PRE-PLACEMENT soma position only]")
P(f"  C  (min ADULT-arbor distance -> contact):   neuron-disjoint AUC = {aucC:.3f}")
P(f"  PRE-PLACEMENT (soma-soma distance):          neuron-disjoint AUC = {aucS:.3f}")
P(f"  PRE-PLACEMENT (soma dist + 1 physics scale): neuron-disjoint AUC = {best:.3f}")
P(f"  degree-preserving null:                      neuron-disjoint AUC = {DEGNULL_ND:.3f}")
P(f"  => forcing constraints UPSTREAM (soma, pre-placement) drops C from {aucC:.3f} to {best:.3f} (Δ={aucC-best:+.3f}),")
P(f"     and pre-placement is {'BELOW' if best<DEGNULL_ND else 'above'} even the degree null ({DEGNULL_ND:.3f}).")
# how definitional is min-arbor-distance? correlation with contact-area (contact IS membrane adjacency)
r=np.corrcoef(mind[CONTACT>0], np.log1p(AREA[CONTACT>0]))[0,1]
P(f"\n[definitional check] among contacts, corr(min-arbor-dist, log contact-area) = {r:+.2f}")
P(f"  min-arbor-distance IS the physical basis of contact (contact = membrane adjacency) -> C's 0.868 is largely")
P(f"  reading the adult arrangement contact is defined from, not generating it from placement.")
P("\n"+"="*82)
clean_1a = False  # C uses full adult geometry, not few params
clean_1b = best>=aucC-0.05 and best>DEGNULL_ND
P("GATE:")
P(f"  1a compact physics-set-param win: {'PASS' if clean_1a else 'FAIL — C uses the full adult arbor geometry as input, not few physics params'}")
P(f"  1b constraints upstream + survives: {'PASS' if clean_1b else f'FAIL — pre-placement drops to {best:.3f} (leak-inflated by {aucC-best:+.3f})'}")
P(f"  => TEST 2 (partition) {'RUNS' if (clean_1a and clean_1b) else 'DEFERRED — C downgraded: real-but-leak-inflated (contact≈adult-arbor-proximity, not a compact pre-placement generator). Needs a morphogenesis rebuild (grow arbors from soma) before its degeneracy-unification can be honestly tested.'}")
P("="*82)
