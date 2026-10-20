"""Layer C: NT x receptor -> synapse SIGN (mechanism->structure).
NON-CIRCULAR: structure = Randi 2023 MEASURED functional propagation sign (independent of expression);
mechanism = presynaptic glutamate x postsynaptic receptor CLASS (GluCl=inhibitory Cl- channel vs
iGluR=excitatory cation). Rule: does the post cell expressing an inhibitory GluCl predict a NEGATIVE
functional edge, beyond base rate? Power: 383 glutamate-source nonzero edges, 37 negative.
"""
import warnings; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from scipy.stats import fisher_exact
RNG=np.random.default_rng(0)
D="../data/celegans/data"
cg=pd.read_csv(D+"/expression/cengen/thresholded/021821_liberal_threshold1.csv")
classes=list(cg.columns[3:]); X=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values
gname=cg["gene_name"].astype(str).str.lower().values; cls_idx={c:k for k,c in enumerate(classes)}
def bc(n):
    b=n
    for s in ("L","R","D","V"):
        if b.endswith(s) and b[:-1] in cls_idx: return b[:-1]
    return b
def cclass(n): return n if n in cls_idx else (bc(n) if bc(n) in cls_idx else None)
def level(genes,c):  # summed graded TPM of gene set in class c (relative receptor load)
    k=cls_idx.get(c)
    if k is None: return None
    rows=[i for i,g in enumerate(gname) if g in genes]
    return float(sum(X[i,k] for i in rows))
GLUCL={"avr-14","avr-15","glc-1","glc-2","glc-3","glc-4"}                    # inhibitory Cl-
IGLUR={"glr-1","glr-2","glr-3","glr-4","glr-5","glr-6","glr-7","glr-8","nmr-1","nmr-2"}  # excitatory cation

# NT of presynaptic
t5=np.load("../data/tier5/connectome.npz",allow_pickle=True)
nt={str(n):str(v) for n,v in zip(t5["names"],t5["nt_primary"])}
def is_glu(n): return str(nt.get(n,nt.get(bc(n),"?"))).startswith("Glu")

wt=pd.read_csv(D+"/randi2023/randi_funconn_wildty.tsv",sep="\t",names=["pre","post","w"])
edges=[]
for _,r in wt.iterrows():
    if r.w==0 or not is_glu(r.pre): continue
    cpost=cclass(r.post)
    if cpost is None: continue
    gc=level(GLUCL,cpost); ig=level(IGLUR,cpost)
    if gc is None: continue
    edges.append((r.pre,r.post,r.w,gc,ig))
E=pd.DataFrame(edges,columns=["pre","post","w","glucl","iglur"])
E["neg"]=(E.w<0).astype(int)
E["inh_frac"]=E.glucl/(E.glucl+E.iglur+1e-9)   # fraction of glutamate receptor load that is inhibitory Cl-
print(f"glutamate-source edges: {len(E)} | negative: {E.neg.sum()} ({100*E.neg.mean():.0f}%)")
print(f"inhibitory-fraction distribution: min {E.inh_frac.min():.2f} med {E.inh_frac.median():.2f} max {E.inh_frac.max():.2f}")

from scipy.stats import spearmanr, mannwhitneyu
# H: higher inhibitory receptor fraction -> more negative functional weight / more likely negative
rho,pr=spearmanr(E.inh_frac, E.w)   # expect negative rho (more inhibitory -> lower w)
print(f"\nSpearman(inhibitory-fraction, functional weight) rho={rho:+.3f} p={pr:.3f}  (expect NEGATIVE)")
hi=E[E.inh_frac>E.inh_frac.median()]; lo=E[E.inh_frac<=E.inh_frac.median()]
print(f"neg-edge rate: high-inhibitory-fraction {hi.neg.mean():.2f} ({hi.neg.sum()}/{len(hi)}) vs low {lo.neg.mean():.2f} ({lo.neg.sum()}/{len(lo)})")
# permutation on the rate difference
obs=hi.neg.mean()-lo.neg.mean(); med=E.inh_frac.median()
nul=[]
for _ in range(5000):
    p=RNG.permutation(E.neg.values)
    h=p[(E.inh_frac>med).values]; l=p[(E.inh_frac<=med).values]
    nul.append(h.mean()-l.mean())
nul=np.array(nul); z=(obs-nul.mean())/(nul.std()+1e-9)
print(f"neg-rate diff (hi-lo inhibitory): obs {obs:+.3f}  null {nul.mean():+.3f}±{nul.std():.3f}  z={z:+.1f}  p={(nul>=obs).mean():.3f}")
print("\n"+"="*72)
sig = (rho<0 and pr<0.05) or (z>2 and obs>0)
print("VERDICT:", "NT x receptor PREDICTS functional sign (mechanism->structure holds at sign layer)" if sig
      else "NT x receptor does NOT predict functional sign beyond base rate — bulk class receptor load is too coarse a proxy for per-synapse sign (or genuinely no signal)")
print("="*72)
