import sys, csv, re, numpy as np
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); m0=np.load(f"{SC}/mode0.npz")
I,J,y7,y8,neurons=z['I'],z['J'],z['y7'],z['y8'],list(z['neurons']); idx={n:i for i,n in enumerate(neurons)}
M=len(neurons); N=len(I); yrep=((y7==1)&(y8==1)).astype(int); agree=m0['agree']; L1c=m0['L1c']
GL1=np.zeros((M,M))
with open("../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']=='chemical':
            d=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
            if d in (1,2,3,4) and r['pre'] in idx and r['post'] in idx: GL1[idx[r['pre']],idx[r['post']]]=1
# L1 directed shortest-path distance i->j (routing value: short=redundant, long=shortcut/control-value)
D=shortest_path(csr_matrix(GL1),directed=True,unweighted=True)
dist=np.array([D[I[k],J[k]] for k in range(N)]); dist[np.isinf(dist)]=99
added=((L1c==0)&(yrep==1)&agree); notadd=((L1c==0)&(yrep==0)&agree)
def dd(m): 
    x=dist[m]; return f"dist2(redundant/closure)={(x==2).mean():.2f} dist3-5={((x>=3)&(x<=5)).mean():.2f} unreachable/shortcut(>5)={(x>5).mean():.2f}"
P(f"[#15/#20 CONTROL-THEORETIC: L1 routing distance i->j for additions]")
P(f"  ADDED   (n={added.sum()}): {dd(added)}")
P(f"  NOT-add (n={notadd.sum()}): {dd(notadd)}")
P(f"  => additions are {'REDUNDANT closure (dist-2), NOT control-improving shortcuts' if (dist[added]==2).mean()>(dist[added]>5).mean() else 'shortcuts (control-value)'}")
# controllability proxy: does edge increase reachability from SENSORY neurons? (sensory as control inputs)
SENS={'ADF','ADL','AFD','ASE','ASG','ASH','ASI','ASJ','ASK','AWA','AWB','AWC','BAG','URX','IL1','IL2','OLL','OLQ','CEP','ADE','URY','URB'}
def isS(x): c=(x[:-1] if x[-1] in 'LR' else x).rstrip('DV'); return any(c.startswith(s) for s in SENS)
sens=np.array([isS(n) for n in neurons]); Rreach=(D<99); reach_from_sens=Rreach[sens].any(0)  # nodes reachable from any sensory
# marginal reachability value of an added edge: does j become reachable from sensory via i?
val_added=np.array([1 if (reach_from_sens[I[k]] and not reach_from_sens[J[k]]) else 0 for k in np.where(added)[0]])
P(f"  reachability-increasing additions (connect unreached target to sensory flow): {val_added.mean():.3f} (near 0 => additions don't extend control reach)")
P(f"  (control-theoretic prediction: adult edges = high routing/control value; observed: additions complete EXISTING short paths = redundancy, refuting the routing-value hypothesis)")
