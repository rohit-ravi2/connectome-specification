import sys, numpy as np
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
d=np.load(f"{SC}/global_process.npz",allow_pickle=True)
v7,v8,V=d['v7'],d['v8'],d['V']; KEYS=list(d['KEYS']); B=V.shape[0]
rng=np.random.default_rng(2)
real=np.abs(v7-v8)
pa=rng.integers(0,B,4000); pb=rng.integers(0,B,4000); m=pa!=pb; pa,pb=pa[m],pb[m]
null=np.abs(V[pa]-V[pb]); mu,sd=null.mean(0),null.std(0)
z=(real-mu)/np.where(sd>0,sd,1)
q=np.array([(null[:,k]<=real[k]).mean() for k in range(len(KEYS))])
P("=== HAWKES PROCESS ENSEMBLE (contact+persistence+closure): real spread vs synthetic spread ===")
P("(NOTE: process is per-edge OVER-dispersed vs real ~0.66, so aggregate tightness is the marginal shadow of KNOWN per-edge determinism, not a new global object)")
P(f"{'invariant':16s} {'real':>10s} {'synthmu':>10s} {'z':>7s} {'q(frac null<=real)':>8s}")
for k in np.argsort(z):
    fl=' <== real TIGHTER' if z[k]<-2 else (' (real MORE dispersed)' if z[k]>2 else '')
    P(f"{KEYS[k]:16s} {real[k]:10.3f} {mu[k]:10.3f} {z[k]:7.2f} {q[k]:8.3f}{fl}")
tight=[KEYS[k] for k in range(len(KEYS)) if z[k]<-2]
P(f"\nprocess-ensemble: real under-dispersed (<-2sig) on {len(tight)} invariants: {tight}")
