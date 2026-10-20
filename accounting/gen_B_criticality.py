"""GENERATOR B — CRITICALITY. Does a critical-point organizing principle (scale-free degree,
near-percolation) GENERATE contact geometry beyond a degree-preserving null, compactly, out-of-sample?
Honest flag: B is most likely TRUE-BUT-EMPTY. Held to the HIGHER-ORDER bar."""
import sys; sys.path.insert(0,'../accounting')
import numpy as np, powerlaw, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from bulk_benchmark import *

print("="*72)
print("PART 1a — DEGREE DISTRIBUTION: scale-free vs lognormal vs exponential (Clauset)")
print("="*72)
deg = CDEG.copy()
print(f"contact-degree: N={len(deg)} mean={deg.mean():.2f} max={deg.max():.0f} min={deg.min():.0f} var={deg.var():.1f}")
# Clauset-style fit on the degree sequence (discrete)
fit = powerlaw.Fit(deg, discrete=True)
print(f"POWER-LAW fit: xmin={fit.power_law.xmin:.1f}  alpha={fit.power_law.alpha:.3f}  sigma={fit.power_law.sigma:.3f}")
print(f"  n_tail (deg>=xmin) = {(deg>=fit.power_law.xmin).sum()} / {len(deg)}")
for alt in ['lognormal','exponential','truncated_power_law']:
    R,p = fit.distribution_compare('power_law', alt, normalized_ratio=True)
    win = 'power_law' if R>0 else alt
    print(f"  LR power_law vs {alt:20s}: R={R:+.3f} p={p:.3f}  -> favors {win}{'  (sig)' if p<0.05 else '  (n.s.)'}")
# also lognormal vs exponential
R,p = fit.distribution_compare('lognormal','exponential', normalized_ratio=True)
print(f"  LR lognormal vs exponential   : R={R:+.3f} p={p:.3f}  -> favors {'lognormal' if R>0 else 'exponential'}")

print("\n"+"="*72)
print("PART 1b — PERCOLATION POSITION (Molloy-Reed): near critical point?")
print("="*72)
k1 = deg.mean(); k2 = (deg**2).mean()
kappa = k2/k1                       # Molloy-Reed branching factor <k^2>/<k>
pc = k1/(k2-k1)                     # bond-percolation threshold (occupation prob) for config model
# empirical giant component under random bond thinning
def giant_frac(p, reps=20, seed=0):
    rng=np.random.default_rng(seed); adj=CADJ.astype(bool)
    ei,ej=np.where(np.triu(adj,1)); fr=[]
    for r in range(reps):
        keep=rng.random(len(ei))<p
        # union-find
        par=list(range(M))
        def find(x):
            while par[x]!=x: par[x]=par[par[x]]; x=par[x]
            return x
        for a,b in zip(ei[keep],ej[keep]):
            ra,rb=find(a),find(b)
            if ra!=rb: par[ra]=rb
        from collections import Counter
        c=Counter(find(x) for x in range(M))
        fr.append(max(c.values())/M)
    return np.mean(fr)
ps=np.linspace(0.02,1.0,15); gc=[giant_frac(p) for p in ps]
print(f"<k>={k1:.2f}  <k^2>={k2:.1f}  Molloy-Reed kappa=<k^2>/<k>={kappa:.2f}  (giant comp iff kappa>2)")
print(f"bond-percolation threshold p_c = <k>/(<k^2>-<k>) = {pc:.4f}")
print(f"actual bond occupation = 1.0  -> operating at {1.0/pc:.1f}x above p_c (supercritical margin)")
print("giant-component fraction vs bond-occupation p:")
for p,g in zip(ps,gc): print(f"    p={p:.2f}  GC={g:.3f}")
print(f"VERDICT percolation: p_c={pc:.3f} << 1  => graph is DEEP in connected phase, NOT poised at percolation." )

print("\n"+"="*72)
print("PART 2 — GENERATE: Chung-Lu / configuration propensity from critical degree seq (deg_i*deg_j)")
print("="*72)
prop_CL = CDEG[PI]*CDEG[PJ]
d2 = score(prop_CL, 3, 'B-Criticality (Chung-Lu deg_i*deg_j)')
print("  params=(xmin,alpha,threshold)~3 ; expected margin~0 BY CONSTRUCTION (criticality = degree statement)")

print("\n"+"="*72)
print("PART 3 — THE REAL TEST: higher-order criticality BEYOND degree > 0.700 ?")
print("="*72)
# global assortativity coefficient r (scalar organizing parameter, NOT per-pair contact read)
di,dj=CDEG[PI],CDEG[PJ]; e=CONTACT>0
# Pearson degree assortativity over contact edges
r_assort=np.corrcoef(np.concatenate([di[e],dj[e]]),np.concatenate([dj[e],di[e]]))[0,1]
print(f"degree assortativity coefficient r = {r_assort:+.3f}  (global scalar organizing parameter)")

# (3a) FAIR higher-order: propensity = pure degree FUNCTIONS only (no per-pair contact read).
#      Fit small logistic on degree-derived features, neuron-disjoint CV. Tests if degree
#      CORRELATION structure (assortative mixing) adds over the raw product.
def nd_cv_logit(Xcols, name):
    X=np.column_stack(Xcols); aus=[]
    for f in range(5):
        te=(PFOLD==f); tr=(PFOLD!=f)&(PFOLD>=0)
        if not(CONTACT[te].sum()>3 and (CONTACT[te]==0).sum()>3): continue
        m=LogisticRegression(max_iter=500,C=1.0).fit(X[tr],CONTACT[tr])
        aus.append(roc_auc_score(CONTACT[te],m.predict_proba(X[te])[:,1]))
    a=float(np.mean(aus)); print(f"  {name:52s} ND-AUC={a:.3f}  margin={a-DEGNULL_ND:+.3f}"); return a
Ld=np.log1p(di*dj)
a_prod = nd_cv_logit([Ld], "degree product only (log di*dj)")
a_deg  = nd_cv_logit([np.log1p(di),np.log1p(dj)], "degrees separately")
a_asrt = nd_cv_logit([Ld, np.abs(di-dj), di+dj], "+ assortativity terms |di-dj|, di+dj")
best_fair=max(a_prod,a_deg,a_asrt)

# (3b) CLUSTERING / triangle term — criticality-motivated higher-order, BUT this READS the
#      contact adjacency per pair (common neighbors), so it is a LEAK, not a compact principle.
#      Report it flagged, to show whether even memorizing triangles helps.
CN = CADJ @ CADJ                              # common-neighbor count (reads contact graph)
prop_AA = np.array([CN[PI[k],PJ[k]] for k in range(NP)]).astype(float)
a_cluster = neuron_disjoint_auc(prop_AA)
a_cl_comb = neuron_disjoint_auc(prop_CL + prop_AA*prop_CL.mean()/(prop_AA.mean()+1e-9))
print(f"  [LEAK-FLAGGED] common-neighbor (triangle/clustering) propensity ND-AUC={a_cluster:.3f}  margin={a_cluster-DEGNULL_ND:+.3f}")
print(f"     -> reads contact adjacency per-pair; NOT a compact criticality principle (memorizes graph).")

# global clustering coefficient (scalar) for reporting
tri=np.trace(CADJ@CADJ@CADJ)/6
triads=sum(CDEG[i]*(CDEG[i]-1)/2 for i in range(M))
Cglob=3*tri/triads if triads>0 else 0
print(f"  global transitivity C={Cglob:.3f}  (scalar; random-graph expectation ~ <k>/N = {k1/M:.3f})")

print("\n"+"="*72)
print("DECISIVE SUMMARY")
print("="*72)
print(f"degree-null (Chung-Lu) ND-AUC to beat = {DEGNULL_ND:.3f}")
print(f"best FAIR higher-order (degree-function/assortativity) ND-AUC = {best_fair:.3f}  margin={best_fair-DEGNULL_ND:+.3f}")
print(f"clustering (LEAK-flagged) ND-AUC = {a_cluster:.3f}")
