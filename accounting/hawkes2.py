import sys, csv, re, numpy as np
from sklearn.linear_model import LogisticRegression
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); m0=np.load(f"{SC}/mode0.npz")
I,J,y7,y8,base,neurons=z['I'],z['J'],z['y7'],z['y8'],z['base'],list(z['neurons']); idx={n:i for i,n in enumerate(neurons)}
M=len(neurons); N=len(I); yrep=((y7==1)&(y8==1)).astype(int); L1c=m0['L1c']; area=base[:,0]
GL1=np.zeros((M,M))
with open("../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']=='chemical':
            d=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
            if d in (1,2,3,4) and r['pre'] in idx and r['post'] in idx: GL1[idx[r['pre']],idx[r['post']]]=1
L1e=(L1c>=2).astype(float)
Cff0=GL1@GL1; lr=LogisticRegression(max_iter=1000).fit(np.column_stack([area,np.log1p(Cff0[I,J])])[L1e==0],yrep[L1e==0])
a,b=lr.coef_[0]; c0=lr.intercept_[0]
def sig(x): return 1/(1+np.exp(-np.clip(x,-30,30)))
real_rate=(y7==1).mean()  # target directed edge rate
# real dispersion (2 animals) for reference
di_real=(y7!=y8).mean(); th_real=(y7.astype(float)+y8)/2; disp_real=di_real/(2*th_real*(1-th_real)+1e-9).mean()
# ---- fair generator: single calibrated hazard pass to hit edge rate, then measure dispersion ----
# also report the HAZARD sharpness (bimodal p => can give under-dispersion; graded p => cannot)
def gen(cadj,steps=1,prune=0.0):
    rng=np.random.default_rng(1); ENS=[]
    for s in range(80):
        G=GL1.copy()
        for t in range(steps):
            Cff=G@G; padd=sig(a*area+b*np.log1p(Cff[I,J])+c0+cadj); cur=G[I,J]>0
            add=(rng.random(N)<padd)&(~cur)
            for k in np.where(add)[0]: G[I[k],J[k]]=1
            if prune>0:
                isL1=(GL1[I,J]>0)&(G[I,J]>0); pr=(rng.random(N)<prune)&isL1
                for k in np.where(pr)[0]: G[I[k],J[k]]=0
        ENS.append(G[I,J]>0)
    return np.array(ENS).astype(float)
# calibrate cadj so mean edge rate ~ real
lo,hi=-6,3
for _ in range(18):
    mid=(lo+hi)/2; r=gen(mid,steps=1).mean()
    if r>real_rate: hi=mid
    else: lo=mid
cadj=(lo+hi)/2; ENS=gen(cadj,steps=1); theta=ENS.mean(0)
gen_di=np.mean([(ENS[s]!=ENS[s+1]).mean() for s in range(0,80,2)]); gen_disp=gen_di/(2*theta*(1-theta)+1e-9).mean()
# hazard p distribution at calibration
phaz=sig(a*area+b*np.log1p(Cff0[I,J])+c0+cadj)
P(f"REAL: edge rate={real_rate:.3f} dispersion={disp_real:.2f}")
P(f"CALIBRATED GEN (1-step, rate-matched): edge rate={ENS.mean():.3f} dispersion={gen_disp:.2f}")
P(f"  hazard p distribution: frac p<0.1={ (phaz<0.1).mean():.2f} | 0.1-0.9(graded)={((phaz>=0.1)&(phaz<=0.9)).mean():.2f} | p>0.9={(phaz>0.9).mean():.2f}")
P(f"  theta bimodality: near0={(theta<0.1).mean():.2f} halo={((theta>=0.2)&(theta<=0.8)).mean():.2f} near1={(theta>0.9).mean():.2f}")
P(f"\n[VERDICT] real under-dispersion={disp_real:.2f}; rate-matched zero-identity process gives dispersion={gen_disp:.2f}")
P(f"  => {'process REPRODUCES under-dispersion' if gen_disp<0.7 else 'process CANNOT reproduce under-dispersion (memoryless hazard is graded not near-deterministic; per-pair determinism NOT captured by contact+closure)'}")
# what dispersion would a NEAR-DETERMINISTIC hazard give? (set p to 0/1 by thresholding at real rate)
thr=np.quantile(phaz,1-real_rate); pdet=(phaz>thr).astype(float)
rng=np.random.default_rng(2); ENSd=np.array([(rng.random(N)<np.clip(pdet*0.95+0.025,0,1)) for _ in range(80)]).astype(float); thd=ENSd.mean(0)
did=np.mean([(ENSd[s]!=ENSd[s+1]).mean() for s in range(0,80,2)]); dispd=did/(2*thd*(1-thd)+1e-9).mean()
P(f"  (for contrast: a NEAR-DETERMINISTIC per-pair hazard gives dispersion={dispd:.2f} -> under-dispersion REQUIRES per-pair near-determinism, which contact+closure hazard lacks)")
