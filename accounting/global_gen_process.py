import sys, numpy as np
from sklearn.linear_model import LogisticRegression
from global_inv import build_A, vecify
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
b=np.load(f"{SC}/global_base.npz",allow_pickle=True); m0=np.load(f"{SC}/mode0.npz")
I,J,y7,y8,GL1,area=b['I'],b['J'],b['y7'],b['y8'],b['GL1'],b['area']; M=181; N=len(I)
L1c=m0['L1c']; yrep=((y7==1)&(y8==1)).astype(int)
def sig(x): return 1/(1+np.exp(-np.clip(x,-30,30)))
L1e=(L1c>=2).astype(float); Cff0=GL1@GL1
lr=LogisticRegression(max_iter=1000).fit(np.column_stack([area,np.log1p(Cff0[I,J])])[L1e==0],yrep[L1e==0])
a,bc=lr.coef_[0]; c0=lr.intercept_[0]
real_rate=(y7==1).mean()
def gen1(cadj,rng,prune=0.2):
    G=GL1.copy(); Cff=G@G; padd=sig(a*area+bc*np.log1p(Cff[I,J])+c0+cadj); cur=G[I,J]>0
    add=(rng.random(N)<padd)&(~cur)
    for k in np.where(add)[0]: G[I[k],J[k]]=1
    pr=(rng.random(N)<prune)&(GL1[I,J]>0)
    for k in np.where(pr)[0]: G[I[k],J[k]]=0
    return (G[I,J]>0).astype(int)
# calibrate cadj to match real edge rate
lo,hi=-6.,3.
for _ in range(16):
    mid=(lo+hi)/2; r=np.mean([gen1(mid,np.random.default_rng(q)).mean() for q in range(6)])
    if r>real_rate: hi=mid
    else: lo=mid
cadj=(lo+hi)/2
B=64
V=np.zeros((B,40))
rng=np.random.default_rng(100)
rates=[]
for i in range(B):
    a_i=gen1(cadj,np.random.default_rng(1000+i)); rates.append(a_i.mean())
    V[i],KEYS=vecify(build_A(a_i,I,J,M))
    if (i+1)%16==0: P(f"  proc gen {i+1}/{B}")
v7,KEYS=vecify(build_A(y7,I,J,M)); v8,_=vecify(build_A(y8,I,J,M))
np.savez(f"{SC}/global_process.npz",v7=v7,v8=v8,V=V,KEYS=np.array(KEYS,dtype=object))
P(f"calibrated cadj={cadj:.3f} real_rate={real_rate:.3f} synth mean rate={np.mean(rates):.3f}")
P("saved global_process.npz")
