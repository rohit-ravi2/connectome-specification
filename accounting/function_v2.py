"""WHAT DOES THE CONNECTOME COMPUTE — v2. Complete E/I signing (CeNGEN NT-machinery, full coverage). KEY NULL:
SIGN-PERMUTATION on FIXED topology (permute E/I labels, keep wiring) -> isolates whether inhibition is PLACED
to build computations, beyond generic clustering. Vectorized signed-motif census + inhibitory-placement + the
dynamical spectrum the sign structure affords.
"""
import warnings,sys,openpyxl; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
def P(*a): print(*a); sys.stdout.flush()
RNG=np.random.default_rng(0); D="../data/celegans/data"
# ---- complete E/I from CeNGEN NT-machinery ----
cg=pd.read_csv(f"{D}/expression/cengen/thresholded/021821_liberal_threshold1.csv")
classes=list(cg.columns[3:]); X=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values
gname=cg["gene_name"].astype(str).str.lower().values; cls_idx={c:k for k,c in enumerate(classes)}
def expr(g,c):
    w=np.where(gname==g)[0]; return X[w[0],c]>0 if len(w) else False
def nt_of_class(c):
    if expr("unc-25",c) or expr("unc-47",c): return -1     # GABA inhibitory
    if expr("eat-4",c): return +1                          # Glu (excitatory default, field convention)
    if expr("unc-17",c) or expr("cha-1",c): return +1      # ACh excitatory
    if any(expr(g,c) for g in ("tph-1","cat-2","tbh-1","tdc-1","bas-1")): return 0  # monoamine modulatory
    return 0
CLSIGN={c:nt_of_class(k) for k,c in enumerate(classes)}
def n2c(n):
    if n in cls_idx: return n
    b=n
    for s in "LR":
        if b.endswith(s): b=b[:-1]; break
    if b in cls_idx: return b
    for s in "DV":
        if b.endswith(s) and b[:-1] in cls_idx: return b[:-1]
    return None
def sign_neuron(n):
    c=n2c(n); return CLSIGN.get(c,0) if c else 0
# ---- Cook directed connectome ----
wb=openpyxl.load_workbook(f"{D}/connectome/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
A=list(wb['hermaphrodite chemical'].iter_rows(values_only=True))
def looks(n): return isinstance(n,str) and len(n)>=2 and n[0].isalpha()
hr=max(range(len(A)),key=lambda r:sum(looks(x) for x in A[r])); cols=[(str(x).strip() if x is not None else None) for x in A[hr]]
hc=max(range(min(6,len(A[0]))),key=lambda c:sum(looks(A[r][c]) for r in range(len(A))))
E=[]; nset=set()
for r in range(hr+1,len(A)):
    pre=A[r][hc]; pre=pre.strip() if isinstance(pre,str) else None
    if not pre: continue
    for c in range(hc+1,len(A[r])):
        post=cols[c]; v=A[r][c]
        if post and v not in (None,''):
            try:
                if float(v)>0: E.append((pre,post)); nset|={pre,post}
            except: pass
# keep only signed neurons (E/I), drop modulatory/unknown for the E/I motif analysis
neur=sorted([n for n in nset if sign_neuron(n)!=0]); idx={n:i for i,n in enumerate(neur)}; M=len(neur)
Adj=np.zeros((M,M))
for a,b in E:
    if a in idx and b in idx and a!=b: Adj[idx[a],idx[b]]=1
sg=np.array([sign_neuron(n) for n in neur],float)
mE=(sg>0).astype(float); mI=(sg<0).astype(float)
P(f"Cook signed connectome (E/I only): {M} neurons ({int(mE.sum())} E, {int(mI.sum())} I), {int(Adj.sum())} edges")
# ---- vectorized signed-motif census ----
def census(A,sg):
    mE=(sg>0).astype(float); mI=(sg<0).astype(float)
    AE=A*mE[None,:]; AI=A*mI[None,:]                      # A->B kept if B is E / I
    coh=float(((AE@A)*A).sum()); inco=float(((AI@A)*A).sum())   # FFL coherent(B exc) / incoherent(B inh)=feedforward inhibition
    AIs=mI[:,None]*A; disinh=float(((AIs@AIs)*(1-np.eye(len(A)))).sum())   # A-|B-|C
    recip=A*A.T
    mutI=float((recip*mI[:,None]*mI[None,:]).sum())/2; mutE=float((recip*mE[:,None]*mE[None,:]).sum())/2
    recEI=float((recip*mI[:,None]*mE[None,:]).sum())     # directed E-I reciprocal
    return dict(ffi=inco,coh=coh,disinh=disinh,mutI=mutI,mutE=mutE,recEI=recEI)
obs=census(Adj,sg)
# SIGN-PERMUTATION null (fixed topology, permute E/I labels preserving counts)
def perm_null(K=200):
    from collections import defaultdict
    acc=defaultdict(list)
    for _ in range(K):
        sp=sg[RNG.permutation(M)]; c=census(Adj,sp)
        for k,v in c.items(): acc[k].append(v)
    return acc
nul=perm_null()
P("\n"+"="*84); P("SIGN-PERMUTATION NULL (topology fixed) — is INHIBITION PLACED to build computations?"); P("="*84)
lab={'ffi':'feedforward inhibition (inhib intermediate: gain-control/timing)','coh':'coherent FFL (excit intermediate: filtering)',
     'disinh':'disinhibition (I-|I-|X: gating)','mutI':'mutual inhibition (winner-take-all)','mutE':'mutual excitation (amplifier)',
     'recEI':'reciprocal E->I->E (feedback gain-control)'}
for k in ['ffi','disinh','mutI','recEI','coh','mutE']:
    m=np.mean(nul[k]); s=np.std(nul[k])+1e-9; z=(obs[k]-m)/s
    tag="OVER (inhibition placed FOR this)" if z>2 else ("UNDER (inhibition avoids this)" if z<-2 else "~random E/I placement")
    P(f"  {lab[k]:56s} obs={obs[k]:7.0f} null={m:8.1f} z={z:+6.1f}  {tag}")
# ---- inhibitory placement: where do the ~few inhibitory neurons sit? ----
outdeg=Adj.sum(1); indeg=Adj.sum(0)
P(f"\n  inhibitory neurons: out-degree {outdeg[mI>0].mean():.1f} vs excitatory {outdeg[mE>0].mean():.1f} | in-degree {indeg[mI>0].mean():.1f} vs {indeg[mE>0].mean():.1f}")
P(f"  (are inhibitory neurons hubs? high-out = broadcast inhibition; high-in = integrate-then-inhibit)")
# ---- dynamical spectrum the sign structure affords ----
W=Adj*sg[:,None]        # W[i,j] = +/- 1 depending on presynaptic sign
ev=np.linalg.eigvals(W); rad=np.max(np.abs(ev)); pos=int((ev.real>0.5).sum()); cplx=float((np.abs(ev.imag)>0.5).mean())
P(f"\n  DYNAMICAL SPECTRUM (signed connectivity eigenvalues): spectral radius {rad:.1f} | modes with Re>0.5 (unstable/integrating) {pos} | complex fraction (oscillatory) {cplx:.2f}")
P("  vs sign-permuted spectrum:")
radN=[np.max(np.abs(np.linalg.eigvals(Adj*sg[RNG.permutation(M)][:,None]))) for _ in range(20)]
P(f"    real spectral radius {rad:.1f} vs sign-permuted {np.mean(radN):.1f}±{np.std(radN):.1f} (lower real => E/I placed to STABILIZE / balance)")
P("="*84)
P("[read] the sign-permutation null isolates COMPUTATION from clustering: motifs OVER here = inhibition")
P("  specifically placed to build that primitive. Spectral radius vs permuted = is E/I arranged for stability")
P("  (balanced) or excitability. Opens: locate the circuits, sensory->motor flow, per-behavior sub-networks.")
P("="*84)
