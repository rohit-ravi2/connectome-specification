import sys, csv, re, numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); m0=np.load(f"{SC}/mode0.npz")
I,J,y7,y8,base,neurons=z['I'],z['J'],z['y7'],z['y8'],z['base'],list(z['neurons']); idx={n:i for i,n in enumerate(neurons)}
M=len(neurons); N=len(I); yrep=((y7==1)&(y8==1)).astype(int); L1c=m0['L1c']
GL1=np.zeros((M,M)); 
with open("../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']=='chemical':
            d=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
            if d in (1,2,3,4) and r['pre'] in idx and r['post'] in idx: GL1[idx[r['pre']],idx[r['post']]]=1
# adjacency index of eligible (contacting) pairs
pairmap={(I[k],J[k]):k for k in range(N)}; area=base[:,0]
L1e=(L1c>=2).astype(float)  # L1 edge present (robust)
# fit addition hazard on real L1->adult additions (eligible, un-wired at L1): zero identity
Cff0=GL1@GL1; feat_add=np.column_stack([area,np.log1p(Cff0[I,J])])
elig_add=(L1e==0); lr=LogisticRegression(max_iter=1000).fit(feat_add[elig_add],yrep[elig_add])
a,b=lr.coef_[0]; c=lr.intercept_[0]; P(f"addition hazard coefs: contact={a:.2f} closure={b:.2f} intercept={c:.2f}")
prune_rate=0.20  # observed: ~20% of L1 synapses pruned (stochastic operator, zero identity)
def sig(x): return 1/(1+np.exp(-np.clip(x,-30,30)))
# REAL fingerprint
real_edges=int((y7==1).sum()+(y8==1).sum())/2; real_prune=int(((L1e==1)&(yrep==0)).sum()); real_add=int(((L1e==0)&(yrep==1)).sum())
P(f"REAL: adult edges/animal~{real_edges:.0f} | L1-stable-pruned={real_prune} added={real_add}")
# GENERATE ensemble
rng=np.random.default_rng(0); ENS=[]; Iarr,Jarr=I,J
for s in range(60):
    G=GL1.copy()
    for t in range(3):  # L1->L2->L3->adult
        Cff=G@G; ce=np.log1p(Cff[Iarr,Jarr])
        padd=sig(a*area+b*ce+c)
        cur=G[Iarr,Jarr]>0
        add=(rng.random(N)<padd)&(~cur)
        # stochastic prune of L1-origin edges
        isL1=(GL1[Iarr,Jarr]>0)&cur
        prune=(rng.random(N)<(prune_rate/3.0))&isL1
        for k in np.where(add)[0]: G[Iarr[k],Jarr[k]]=1
        for k in np.where(prune)[0]: G[Iarr[k],Jarr[k]]=0
    ENS.append(G[Iarr,Jarr]>0)
ENS=np.array(ENS).astype(float)  # (60, N)
theta=ENS.mean(0)  # per-pair generated propensity
gen_edges=ENS.sum(1).mean()
P(f"GEN: adult edges/animal={gen_edges:.0f} (real ~{real_edges:.0f})")
# theta distribution: bimodality (fraction near 0 / near 1 / intermediate halo)
near0=(theta<0.1).mean(); near1=(theta>0.9).mean(); halo=((theta>=0.2)&(theta<=0.8)).mean()
P(f"GEN theta distribution: near0={near0:.2f} near1={near1:.2f} intermediate-halo={halo:.2f} (bimodal if near0+near1 dominate)")
# dispersion on generated ensemble (large-ensemble truth): obs disagreement / 2p(1-p)
# sample pairs of synthetic animals, compute disagreement
di=0; cnt=0
for s in range(0,60,2):
    a1=ENS[s]; a2=ENS[s+1]; di+=(a1!=a2).mean(); cnt+=1
gen_disagree=di/cnt; gen_disp=gen_disagree/(2*theta*(1-theta)).mean()
P(f"GEN dispersion ratio={gen_disp:.2f} (real target ~0.47; 1.0=pure Bernoulli sampling)")
# add/prune split among generated
gen_add=((GL1[Iarr,Jarr]==0)&(theta>0.5)).sum(); gen_prune=((GL1[Iarr,Jarr]>0)&(theta<0.5)).sum()
P(f"GEN add/prune: added={gen_add} pruned={gen_prune} (real added={real_add} pruned={real_prune})")
