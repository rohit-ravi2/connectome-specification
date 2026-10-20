import sys, numpy as np
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
d=np.load(f"{SC}/global_indep.npz",allow_pickle=True)
v7,v8,vn,V2,V3=d['v7'],d['v8'],d['vn'],d['V2'],d['V3']; KEYS=list(d['KEYS'])
B=V2.shape[0]; rng=np.random.default_rng(1)

# ---- 2-animal test (clean Witvliet 7,8 vs independent null p=(y7+y8)/2) ----
real2=np.abs(v7-v8)                       # realized 2-animal spread per invariant
# null pair spreads
pa=rng.integers(0,B,4000); pb=rng.integers(0,B,4000); m=pa!=pb; pa,pb=pa[m],pb[m]
null2=np.abs(V2[pa]-V2[pb])               # (npairs, ninv)
mu2,sd2=null2.mean(0),null2.std(0)
z2=(real2-mu2)/np.where(sd2>0,sd2,1)
q2=np.array([(null2[:,k]<=real2[k]).mean() for k in range(len(KEYS))])  # frac null tighter-or-equal

# ---- 3-animal test (Witvliet7,8 + White; independent null p=(y7+y8+yn)/3) ----
R=np.vstack([v7,v8,vn])
real3=R.std(0)                            # real spread across 3 (batch-inflated => conservative)
tri=[rng.choice(B,3,replace=False) for _ in range(3000)]
null3=np.array([V3[t].std(0) for t in tri])
mu3,sd3=null3.mean(0),null3.std(0)
z3=(real3-mu3)/np.where(sd3>0,sd3,1)
q3=np.array([(null3[:,k]<=real3[k]).mean() for k in range(len(KEYS))])

P("=== INDEPENDENT-EDGE MARGINAL-MATCHED NULL: is REAL spread TIGHTER (negative z) than independent flipping? ===")
P("2-animal = clean Witvliet 7,8 (DECISIVE) | 3-animal = +White N2U (batch-inflated=conservative)")
P(f"{'invariant':16s} {'real2':>10s} {'nullmu2':>10s} {'z2':>7s} {'q2':>6s} | {'z3':>7s} {'q3':>6s}")
order=np.argsort(z2)
for k in order:
    flag=''
    if z2[k]<-2: flag=' <== REAL TIGHTER (2sig)'
    if z2[k]>2: flag=' (real MORE dispersed)'
    P(f"{KEYS[k]:16s} {real2[k]:10.3f} {mu2[k]:10.3f} {z2[k]:7.2f} {q2[k]:6.3f} | {z3[k]:7.2f} {q3[k]:6.3f}{flag}")
P("")
tight2=[KEYS[k] for k in range(len(KEYS)) if z2[k]<-2]
tight3=[KEYS[k] for k in range(len(KEYS)) if z3[k]<-2]
P(f"2-animal invariants where REAL under-dispersed (<-2sig): {tight2}")
P(f"3-animal invariants where REAL under-dispersed (<-2sig): {tight3}")
