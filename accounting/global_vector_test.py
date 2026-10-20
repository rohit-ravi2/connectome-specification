import sys, numpy as np
from sklearn.linear_model import LogisticRegression
from global_inv import build_A
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
b=np.load(f"{SC}/global_base.npz",allow_pickle=True); m0=np.load(f"{SC}/mode0.npz")
I,J,y7,y8,yn,GL1,area=b['I'],b['J'],b['y7'],b['y8'],b['yn'],b['GL1'],b['area']; M=181; N=len(I)
L1c=m0['L1c']; yrep=((y7==1)&(y8==1)).astype(int)

def vecs(A):
    Au=((A+A.T)>0).astype(float)
    do=np.sort(A.sum(1))[::-1]; di=np.sort(A.sum(0))[::-1]; dt=np.sort(Au.sum(1))[::-1]
    eva=np.sort(np.abs(np.linalg.eigvals(A)))[::-1][:20]
    evl=np.linalg.eigvalsh(np.diag(Au.sum(1))-Au)          # 181 laplacian eigs
    A2=A@A
    motif=np.array([ (A*A.T).sum()/2, np.trace(A@A@A)/3, (A2*A).sum(), A2.sum()-np.trace(A2),
                     np.trace(np.linalg.matrix_power(Au,3))/6 ])  # recip,3cyc,transtri,2path,tri
    return {'degtot':dt,'degout':do,'degin':di,'adjspec':eva,'lapspec':evl,'motif':motif}

VNAMES=['degtot','degout','degin','adjspec','lapspec','motif']
def dist(a,b_):
    return {k:np.linalg.norm(a[k]-b_[k]) for k in VNAMES}

r7,r8,rn=vecs(build_A(y7,I,J,M)),vecs(build_A(y8,I,J,M)),vecs(build_A(yn,I,J,M))
real2=dist(r7,r8)

# ---- independent-edge null (2-adult marginal-matched) ----
p2=(y7.astype(float)+y8.astype(float))/2.0
rng=np.random.default_rng(0); nA=60
IV=[vecs(build_A((rng.random(N)<p2).astype(int),I,J,M)) for _ in range(nA)]
# ---- process ensemble ----
def sig(x): return 1/(1+np.exp(-np.clip(x,-30,30)))
L1e=(L1c>=2).astype(float); Cff0=GL1@GL1
lr=LogisticRegression(max_iter=1000).fit(np.column_stack([area,np.log1p(Cff0[I,J])])[L1e==0],yrep[L1e==0])
aa,bc=lr.coef_[0]; c0=lr.intercept_[0]; real_rate=(y7==1).mean()
def gen1(cadj,rg):
    G=GL1.copy(); Cff=G@G; padd=sig(aa*area+bc*np.log1p(Cff[I,J])+c0+cadj); cur=G[I,J]>0
    add=(rg.random(N)<padd)&(~cur)
    for k in np.where(add)[0]: G[I[k],J[k]]=1
    pr=(rg.random(N)<0.2)&(GL1[I,J]>0)
    for k in np.where(pr)[0]: G[I[k],J[k]]=0
    return (G[I,J]>0).astype(int)
lo,hi=-6.,3.
for _ in range(14):
    mid=(lo+hi)/2; r=np.mean([gen1(mid,np.random.default_rng(q)).mean() for q in range(5)])
    hi,lo=(mid,lo) if r>real_rate else (hi,mid)
cadj=(lo+hi)/2
PV=[vecs(build_A(gen1(cadj,np.random.default_rng(5000+i)),I,J,M)) for i in range(48)]

def report(name,ENS):
    B=len(ENS); rr=np.random.default_rng(3)
    pa=rr.integers(0,B,3000); pb=rr.integers(0,B,3000); mk=pa!=pb; pa,pb=pa[mk],pb[mk]
    P(f"\n=== {name}: multivariate vector spread (real y7-y8 distance vs synthetic pair distance) ===")
    P(f"{'vector':10s} {'realdist':>10s} {'nullmu':>10s} {'nullsd':>8s} {'z':>7s} {'q':>6s}")
    for k in VNAMES:
        nd=np.array([np.linalg.norm(ENS[pa[t]][k]-ENS[pb[t]][k]) for t in range(len(pa))])
        z=(real2[k]-nd.mean())/nd.std(); q=(nd<=real2[k]).mean()
        fl=' <== REAL TIGHTER' if z<-2 else ''
        P(f"{k:10s} {real2[k]:10.3f} {nd.mean():10.3f} {nd.std():8.3f} {z:7.2f} {q:6.3f}{fl}")
report("INDEPENDENT-EDGE NULL (2-adult marginal-matched, DECISIVE)",IV)
report("HAWKES PROCESS ENSEMBLE",PV)
