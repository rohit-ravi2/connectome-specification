import sys, numpy as np
from global_inv import build_A
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
b=np.load(f"{SC}/global_base.npz",allow_pickle=True)
I,J,y7,y8,yn=b['I'],b['J'],b['y7'],b['y8'],b['yn']; M=181; N=len(I)
def extra(A):
    Au=((A+A.T)>0).astype(float); dt=Au.sum(1)
    ii,jj=np.where(Au>0)
    if len(ii)>2 and dt[ii].std()>0 and dt[jj].std()>0:
        assort=np.corrcoef(dt[ii],dt[jj])[0,1]
    else: assort=0.0
    L=np.diag(dt)-Au; ev=np.linalg.eigvalsh(L); ev=np.clip(ev,0,None)
    p=ev/ev.sum() if ev.sum()>0 else ev; p=p[p>0]
    spec_ent=float(-(p*np.log(p)).sum())
    # weighted degree gini
    d=np.sort(dt); n=len(d); gini=(2*np.arange(1,n+1)-n-1).dot(d)/(n*d.sum()) if d.sum()>0 else 0
    return np.array([assort,spec_ent,gini])
e7,e8,en=extra(build_A(y7,I,J,M)),extra(build_A(y8,I,J,M)),extra(build_A(yn,I,J,M))
p2=(y7.astype(float)+y8.astype(float))/2.0
rng=np.random.default_rng(0)
E=np.array([extra(build_A((rng.random(N)<p2).astype(int),I,J,M)) for _ in range(120)])
real=np.abs(e7-e8)
rr=np.random.default_rng(4); pa=rr.integers(0,120,3000); pb=rr.integers(0,120,3000); mk=pa!=pb
nd=np.abs(E[pa[mk]]-E[pb[mk]]); mu,sd=nd.mean(0),nd.std(0); z=(real-mu)/sd
names=['assortativity','spectral_entropy','degree_gini']
P("=== EXTRA invariants vs independent-edge null (2-adult) ===")
for k in range(3):
    fl=' <== REAL TIGHTER' if z[k]<-2 else (' (real MORE dispersed)' if z[k]>2 else '')
    P(f"{names[k]:18s} real|d|={real[k]:.4f} nullmu={mu[k]:.4f} z={z[k]:+.2f}{fl}")
P(f"raw values: assort y7/y8/N2U={e7[0]:.3f}/{e8[0]:.3f}/{en[0]:.3f}  specent={e7[1]:.3f}/{e8[1]:.3f}/{en[1]:.3f}  gini={e7[2]:.3f}/{e8[2]:.3f}/{en[2]:.3f}")
