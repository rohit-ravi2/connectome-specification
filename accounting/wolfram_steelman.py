import sys, csv, re, numpy as np
from collections import defaultdict
from scipy.stats import fisher_exact
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); z2=np.load(f"{SC}/resid2.npz")
I,J,y7,y8,neurons=z['I'],z['J'],z['y7'],z['y8'],list(z['neurons'])
M=len(neurons); N=len(I); yrep=((y7==1)&(y8==1)).astype(int); idx={n:i for i,n in enumerate(neurons)}; r1=z2['r1']
# per-DATASET chemical synapse presence
ds_present=defaultdict(set)  # (i,j) -> set of datasets with a synapse
with open("../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']!='chemical': continue
        d=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
        if r['pre'] in idx and r['post'] in idx: ds_present[(idx[r['pre']],idx[r['post']])].add(d)
def L1n(k): return len(ds_present[(I[k],J[k])] & {1,2,3,4})   # # of 4 L1 animals with a synapse
def ADn(k): return len(ds_present[(I[k],J[k])] & {7,8})       # # of 2 adults
l1=np.array([L1n(k) for k in range(N)]); ad=np.array([ADn(k) for k in range(N)])
L1syn=(l1>=2).astype(int); L2=np.array([1 if 5 in ds_present[(I[k],J[k])] else 0 for k in range(N)]); L3=np.array([1 if 6 in ds_present[(I[k],J[k])] else 0 for k in range(N)])
# developmental classes among contacting pairs
PRUNED=(L1syn==1)&(yrep==0)   # synapse at L1, gone by adult  (over-production+elimination = RESTRAINT)
ADDED=(L1syn==0)&(yrep==1)    # no L1 synapse, present adult  (late formation)
STABLE=(L1syn==1)&(yrep==1)
P(f"Developmental edge census (contacting pairs, N={N}):")
P(f"  L1 synapses (>=2/4 animals): {L1syn.sum()} | adult-core synapses: {yrep.sum()}")
P(f"  STABLE(L1+adult)={STABLE.sum()} | PRUNED(L1 not adult)={PRUNED.sum()} | ADDED(adult not L1)={ADDED.sum()}")
P(f"  => over-production ratio: {L1syn.sum()/max(yrep.sum(),1):.2f}x L1 synapses vs adult; {PRUNED.sum()/max(L1syn.sum(),1)*100:.0f}% of L1 synapses PRUNED")
# KEY: are FP (co-fasc-no-synapse hard core) enriched for having been PRUNED?
FP=(r1<-0.3); FN=(r1>0.3); hard=np.abs(r1)>0.3
def enr(mask,cls,lab):
    a=(mask&cls).sum(); b=(mask&~cls).sum(); c=(~mask&cls).sum(); d=(~mask&~cls).sum()
    orr,p=fisher_exact([[a,b],[c,d]]); P(f"    {lab}: rate in group={a/(a+b):.3f} vs out={c/(c+d):.3f}  OR={orr:.2f} p={p:.1e}")
P(f"\n[STEELMAN test: is the RESTRAINT = developmental PRUNING?]")
P(f"  FP (surprising NON-connections, touch-no-synapse-adult), n={FP.sum()}:")
enr(FP,PRUNED,"  were PRUNED (had L1 synapse)")
P(f"  FN (surprising connections, wire despite doubt), n={FN.sum()}:")
enr(FN,ADDED,"  were ADDED (no L1 synapse)")
enr(FN,L1syn.astype(bool),"  had L1 synapse (early-established)")
# does add+prune reproduce adult better than add-only baseline? compare L1->adult transition predictability
P(f"\n[replace-rule vs add-only]: of adult-core edges, {STABLE.sum()/max(yrep.sum(),1)*100:.0f}% already present at L1 (stable), {ADDED.sum()/max(yrep.sum(),1)*100:.0f}% added later")
P(f"  pruning removes {PRUNED.sum()} L1 edges that a monotonic ADD-only rule can NEVER remove = the restraint naive rewrite lacked")
