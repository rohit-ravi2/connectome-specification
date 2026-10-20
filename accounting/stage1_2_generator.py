import sys, csv, re, numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); m0=np.load(f"{SC}/mode0.npz")
I,J,y7,y8,neurons=z['I'],z['J'],z['y7'],z['y8'],list(z['neurons']); idx={n:i for i,n in enumerate(neurons)}
M=len(neurons); N=len(I); yrep=((y7==1)&(y8==1)).astype(int); L1c=m0['L1c']; agree=m0['agree']; coinflip=m0['coinflip']
cd=np.load(f"{SC}/contactdyn_raw.npz",allow_pickle=True); udk=cd['UD_keys']; udv=cd['UD_vals']; L1md={}
for r in range(len(udk)): a,b=int(udk[r][0]),int(udk[r][1]); L1md[(min(a,b),max(a,b))]=udv[r][0]
L1elig=np.array([ (lambda v:(not np.isnan(v)) and v<2000)(L1md.get((min(I[k],J[k]),max(I[k],J[k])),np.nan)) for k in range(N)])
GL1=np.zeros((M,M))
with open("../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']!='chemical': continue
        d=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
        if d in (1,2,3,4) and r['pre'] in idx and r['post'] in idx: GL1[idx[r['pre']],idx[r['post']]]=1

# ===== STAGE 1: derive the closure ADD rule (global, 2 params, ZERO identity) =====
Cff0=GL1@GL1
addable=(L1c==0)&L1elig                     # non-synaptic-at-L1 contacting pairs (opportunity)
l1edge=(L1c>=1)                             # L1 synapses (prunable)
# fit global closure rule: P(adult edge | closure support) on addable pairs — 2 params a,b, no identity
lr=LogisticRegression().fit(np.log1p(Cff0[I,J][addable]).reshape(-1,1), yrep[addable])
a,b=float(lr.intercept_[0]),float(lr.coef_[0][0]); P(f"closure ADD rule (global): logit p = {a:.2f} + {b:.2f}*log1p(2paths)  [zero identity]")
# observed rates to calibrate scale (global)
obs_add=yrep[addable].sum(); prune_rate=(yrep[l1edge]==0).mean()   # stochastic prune (Gate A failed)
P(f"targets: additions={obs_add} (rate {yrep[addable].mean():.3f}) | prune-rate(global)={prune_rate:.3f}")
addI,addJ=I[addable],J[addable]; prI,prJ=I[l1edge],J[l1edge]
# calibrate per-step add scale so 3 iterated steps reproduce ~obs_add
def gen_one(seed, T=3, scale=1.0):
    rng=np.random.default_rng(seed); G=GL1.copy()
    active=set(zip(np.where(GL1>0)[0],np.where(GL1>0)[1]))
    for t in range(T):
        Cff=G@G
        p=1/(1+np.exp(-(a+b*np.log1p(Cff[addI,addJ])))) * scale
        add=rng.random(len(addI))<(p/T)
        # stochastic prune of current L1-origin edges
        pr=rng.random(len(prI))<(prune_rate/T)
        for k in np.where(add)[0]: G[addI[k],addJ[k]]=1; active.add((addI[k],addJ[k]))
        for k in np.where(pr)[0]:
            if G[prI[k],prJ[k]]>0: G[prI[k],prJ[k]]=0; active.discard((prI[k],prJ[k]))
    return G
# calibrate scale
for scale in [0.6,0.8,1.0,1.2,1.5]:
    G=gen_one(0,scale=scale); na=int(G[addI,addJ].sum())
    if abs(na-obs_add)<60: break
P(f"calibrated add-scale={scale} -> generated additions={int(gen_one(0,scale=scale)[addI,addJ].sum())} (target {obs_add})")

# ===== STAGE 2: roll forward ENSEMBLE, test fingerprint =====
Nsim=200; ens=np.zeros((Nsim,len(addI)))       # generated edge presence over addable pairs
ensL1=np.zeros((Nsim,len(prI)))                # generated L1-edge survival (prune side)
for s in range(Nsim):
    G=gen_one(s,scale=scale); ens[s]=G[addI,addJ]; ensL1[s]=G[prI,prJ]
theta_gen=ens.mean(0)                           # generator propensity for addable pairs
P("\n=== FINGERPRINT ===")
# (1) edge-prediction fit vs REAL adult (must be ~0.867 rule-trajectory, NOT 0.901 leak)
auc_gen=roc_auc_score(yrep[addable], theta_gen); P(f"(1) generated theta predicts REAL adult additions AUC={auc_gen:.3f}  (target rule-trajectory ~0.867; >0.90 would be leak)")
# (2) bimodal theta?
bim=((theta_gen<0.2)|(theta_gen>0.8)).mean(); P(f"(2) theta bimodality (frac <0.2 or >0.8) = {bim:.3f}  (real gate bimodal ~0.93)")
# (3) DISPERSION of the GENERATED ensemble (does under-dispersion 0.47 EMERGE from iterated closure?)
#     dispersion = mean pairwise disagreement between 2 synthetic animals / 2 theta(1-theta). Estimator note:
#     generated ensemble is LARGE (theta precise from 200); we sample animal PAIRS -> matches the 2-real-animal noisy side.
rng=np.random.default_rng(1); pairs=[(rng.integers(Nsim),rng.integers(Nsim)) for _ in range(400)]; pairs=[(x,y) for x,y in pairs if x!=y]
allpres=np.concatenate([ens,ensL1],axis=1); th_all=allpres.mean(0)
dis=np.mean([np.abs(allpres[x]-allpres[y]) for x,y in pairs],axis=0)   # per-pair mean disagreement across animal-pairs
obs_dis=dis.mean(); pred_dis=(2*th_all*(1-th_all)).mean(); ratio=obs_dis/pred_dis
P(f"(3) GENERATED-ensemble dispersion ratio = {ratio:.3f}  (real target 0.47; Part2 predicts ~1 if edges independent-in-the-end)")
# also the true-generative control (sample independently from theta_gen)
indep=(rng.random((Nsim,len(th_all)))<th_all).astype(float); dind=np.mean([np.abs(indep[x]-indep[y]) for x,y in pairs],axis=0)
P(f"    independent-sampling control (same theta): ratio={dind.mean()/pred_dis:.3f}  (==1 by construction check)")
# (4) add/prune split reproduction
gen_addrate=theta_gen.mean(); gen_prunerate=(1-ensL1.mean()); P(f"(4) generated add-rate={gen_addrate:.3f} (real {yrep[addable].mean():.3f}) | generated prune-rate={gen_prunerate:.3f} (real {prune_rate:.3f})")
P("\nVERDICT: under-dispersion EMERGES from zero-identity iterated closure? -> "+("YES (ratio<0.6)" if ratio<0.6 else "NO (ratio~1) = under-dispersion is per-edge determinism the rule lacks (Part2 confirmed)"))
