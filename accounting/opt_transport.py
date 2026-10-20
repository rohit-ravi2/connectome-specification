import sys, csv, re, numpy as np
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); m0=np.load(f"{SC}/mode0.npz")
I,J,y7,y8,base,neurons=z['I'],z['J'],z['y7'],z['y8'],z['base'],list(z['neurons']); idx={n:i for i,n in enumerate(neurons)}
M=len(neurons); N=len(I); yrep=((y7==1)&(y8==1)).astype(int); agree=m0['agree']; area=base[:,0]
GL1=np.zeros((M,M))
with open("../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']=='chemical':
            d=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
            if d in (1,2,3,4) and r['pre'] in idx and r['post'] in idx: GL1[idx[r['pre']],idx[r['post']]]=1
Cff=GL1@GL1
# contact+closure "attraction" on contacting pairs; OT redistributes under per-neuron budget constraints
Amat=np.full((M,M),-1e6); Clo=np.zeros((M,M))
for k in range(N): Amat[I[k],J[k]]=area[k]; Clo[I[k],J[k]]=np.log1p(Cff[I[k],J[k]])
attract=Amat+1.0*Clo  # higher=more likely (negative cost)
mask_pair=Amat>-1e5
# supply/demand = adult out/in degree (budgets); test if OT allocation under budgets beats unconstrained
out_t=np.zeros(M); in_t=np.zeros(M)
for k in np.where(yrep==1)[0]: out_t[I[k]]+=1; in_t[J[k]]+=1
a_sup=out_t+0.5; b_dem=in_t+0.5; a_sup*=b_dem.sum()/a_sup.sum()
# Sinkhorn
K=np.where(mask_pair,np.exp((attract-attract[mask_pair].max())/1.5),0.0)
u=np.ones(M); v=np.ones(M)
for _ in range(200):
    u=a_sup/(K@v+1e-12); v=b_dem/(K.T@u+1e-12)
Pmat=(u[:,None]*K*v[None,:])
ot=np.array([Pmat[I[k],J[k]] for k in range(N)])         # OT budget-constrained allocation
unc=np.array([attract[I[k],J[k]] for k in range(N)])      # unconstrained contact+closure
ma=agree
P(f"[#4 OPTIMAL TRANSPORT / allocation, agree pairs n={ma.sum()}]")
P(f"  unconstrained contact+closure AUC={roc_auc_score(yrep[ma],unc[ma]):.3f}")
P(f"  budget-constrained OT allocation AUC={roc_auc_score(yrep[ma],ot[ma]):.3f}")
d=roc_auc_score(yrep[ma],ot[ma])-roc_auc_score(yrep[ma],unc[ma])
P(f"  => OT budget constraint {'ADDS' if d>0.01 else 'adds ~nothing'} over unconstrained (Δ{d:+.3f}) [budget already NULL in MDL/degree-gate; OT is the steelman]")
