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
Amat=np.full((M,M),-1e6); Clo=np.zeros((M,M))
for k in range(N): Amat[I[k],J[k]]=area[k]; Clo[I[k],J[k]]=np.log1p(Cff[I[k],J[k]])
mask_pair=Amat>-1e5
def sinkhorn(attract,a_sup,b_dem):
    K=np.where(mask_pair,np.exp((attract-attract[mask_pair].max())/1.5),0.0); u=np.ones(M); v=np.ones(M)
    for _ in range(200): u=a_sup/(K@v+1e-12); v=b_dem/(K.T@u+1e-12)
    return u[:,None]*K*v[None,:]
def pair(Pm): return np.array([Pm[I[k],J[k]] for k in range(N)])
ma=agree; att=Amat+1.0*Clo
# adult degrees (LEAKY), L1 degrees (fair), random attraction control
o_ad=np.zeros(M);i_ad=np.zeros(M)
for k in np.where(yrep==1)[0]: o_ad[I[k]]+=1;i_ad[J[k]]+=1
o_l1=GL1.sum(1);i_l1=GL1.sum(0)
def norm(a,b): a=a+0.5;b=b+0.5; return a*b.sum()/a.sum(),b
unc=roc_auc_score(yrep[ma],pair(att)[ma])
ad=roc_auc_score(yrep[ma],pair(sinkhorn(att,*norm(o_ad,i_ad)))[ma])
l1=roc_auc_score(yrep[ma],pair(sinkhorn(att,*norm(o_l1,i_l1)))[ma])
rng=np.random.default_rng(0); randatt=np.where(mask_pair,rng.random((M,M)),-1e6)
adrand=roc_auc_score(yrep[ma],pair(sinkhorn(randatt,*norm(o_ad,i_ad)))[ma])
P(f"[#4 OT corrected, agree pairs]")
P(f"  unconstrained contact+closure           AUC={unc:.3f}")
P(f"  OT + ADULT-degree budgets (LEAKY)        AUC={ad:.3f}")
P(f"  OT + ADULT-degree budgets + RANDOM attr  AUC={adrand:.3f}  <- degree-sequence readout floor")
P(f"  OT + L1-degree budgets (FAIR, pre-adult) AUC={l1:.3f}  (Δ vs unconstrained {l1-unc:+.3f})")
P(f"\n  => the +0.067 was DEGREE-SEQUENCE READOUT (random-attr+adult-budgets alone hits {adrand:.3f}). Fair L1-budget OT adds {l1-unc:+.3f} => budget/allocation {'REAL' if l1-unc>0.01 else 'adds ~nothing (consistent with budget NULL)'}")
