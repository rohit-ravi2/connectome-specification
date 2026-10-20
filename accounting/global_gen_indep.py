import sys, numpy as np
from global_inv import build_A, vecify
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
b=np.load(f"{SC}/global_base.npz",allow_pickle=True)
I,J,y7,y8,yn=b['I'],b['J'],b['y7'],b['y8'],b['yn']; M=181; N=len(I)
# ---- real invariant vectors ----
v7,KEYS=vecify(build_A(y7,I,J,M)); v8,_=vecify(build_A(y8,I,J,M)); vn,_=vecify(build_A(yn,I,J,M))
# ---- independent-edge null matched to 2-adult marginals ----
p2=(y7.astype(float)+y8.astype(float))/2.0     # 0,0.5,1  (variable edges=coinflip=0.5)
p3=(y7.astype(float)+y8.astype(float)+yn.astype(float))/3.0  # 0,1/3,2/3,1
B=140
rng=np.random.default_rng(0)
V2=np.zeros((B,len(KEYS))); V3=np.zeros((B,len(KEYS)))
for i in range(B):
    a2=(rng.random(N)<p2).astype(int); V2[i],_=vecify(build_A(a2,I,J,M))
    a3=(rng.random(N)<p3).astype(int); V3[i],_=vecify(build_A(a3,I,J,M))
    if (i+1)%20==0: P(f"  gen {i+1}/{B}")
np.savez(f"{SC}/global_indep.npz",v7=v7,v8=v8,vn=vn,V2=V2,V3=V3,KEYS=np.array(KEYS,dtype=object),
         p2mean=p2.mean(),p3mean=p3.mean())
P("saved global_indep.npz | keys:",len(KEYS))
P("real n_edges: y7,y8,yn =",v7[KEYS.index('n_edges')],v8[KEYS.index('n_edges')],vn[KEYS.index('n_edges')])
