import sys, csv, re, numpy as np
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); m0=np.load(f"{SC}/mode0.npz")
I,J,y7,y8,neurons=z['I'],z['J'],z['y7'],z['y8'],list(z['neurons']); idx={n:i for i,n in enumerate(neurons)}
M=len(neurons); N=len(I); yrep=((y7==1)&(y8==1)).astype(int); agree=m0['agree']; L1c=m0['L1c']
var=(y7!=y8)  # halo/variable
# ---- #10 HODGE: node potential from divergence; edge gradient-consistency ----
A=np.zeros((M,M))
for k in np.where((y7==1)|(y8==1))[0]: A[I[k],J[k]]=1
deg=(A+A.T).sum(1); L=np.diag(deg)-(A+A.T)
div=A.sum(1)-A.sum(0)  # out - in per node
phi=np.linalg.pinv(L)@div  # Hodge potential (hierarchy)
grad=phi[I]-phi[J]  # edge flows downhill if >0 (gradient/feedforward-consistent)
core=((y7==1)&(y8==1)); 
P(f"[#10 HODGE directed-flow] node-potential gradient-consistency (|grad|, higher=more feedforward-aligned):")
P(f"  reproducible-CORE edges: mean|grad|={np.abs(grad[core]).mean():.3f} frac-downhill={ (grad[core]>0).mean():.3f}")
P(f"  VARIABLE/halo edges:     mean|grad|={np.abs(grad[var]).mean():.3f} frac-downhill={ (grad[var]>0).mean():.3f}")
# is core more gradient (feedforward) and halo more cyclic (near-zero potential diff)?
P(f"  => core |grad| {'>' if np.abs(grad[core]).mean()>np.abs(grad[var]).mean() else '<='} halo |grad| ({'core more feedforward, halo more cyclic/harmonic' if np.abs(grad[core]).mean()>np.abs(grad[var]).mean()+0.05 else 'no clean flow separation'})")
# ---- #12 NUCLEATION: do ADDED edges cluster around L1 seeds? ----
GL1=np.zeros((M,M))
with open("../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']=='chemical':
            d=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
            if d in (1,2,3,4) and r['pre'] in idx and r['post'] in idx: GL1[idx[r['pre']],idx[r['post']]]=1
# local L1 density around each pair = # L1 edges among {i,j}'s neighborhood (i's out + j's in partners)
Cff=GL1@GL1  # i->k->j L1 paths = seed support
added=((L1c==0)&(yrep==1)&agree); notadd=((L1c==0)&(yrep==0)&agree)
P(f"\n[#12 NUCLEATION/percolation] L1 seed-support (i->k->j paths at L1) around ADDED vs not-added (eligible):")
P(f"  ADDED edges (n={added.sum()}):     mean L1 seed-paths={Cff[I,J][added].mean():.2f} | frac with >=1 seed={ (Cff[I,J][added]>0).mean():.3f}")
P(f"  NOT-added (n={notadd.sum()}): mean L1 seed-paths={Cff[I,J][notadd].mean():.2f} | frac with >=1 seed={ (Cff[I,J][notadd]>0).mean():.3f}")
# does closure support form a percolating structure? giant component of closure-added edges
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
G2=GL1.copy()
for k in np.where(added)[0]: G2[I[k],J[k]]=1
nc,lab=connected_components(csr_matrix(((G2+G2.T)>0).astype(int)),directed=False)
sizes=np.bincount(lab); P(f"  L1+additions graph: {nc} components, giant={sizes.max()}/{M} neurons ({'PERCOLATES (nucleation cascade)' if sizes.max()>0.8*M else 'fragmented'})")
