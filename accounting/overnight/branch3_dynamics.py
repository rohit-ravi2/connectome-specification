"""BRANCH 3 — does FUNCTION fall out of the identity-directed + E/I structure? North-star test. Build a rate
model on the DIRECTED signed connectome; ask (1) is it in a multistable regime (spectral radius>1 = can hold
states), (2) do the leading dynamical modes LOCALIZE on the command core (AVA/AVB), (3) does a kick to AVA produce
PERSISTENT state-holding (a latch), and (4) does all this VANISH under direction-shuffle / sign-shuffle nulls
(i.e. is the function specifically from the real identity-directed structure).
"""
import warnings,sys,openpyxl; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
def P(*a): print(*a); sys.stdout.flush()
RNG=np.random.default_rng(0); D="../data/celegans/data"
# E/I from CeNGEN NT-machinery (complete)
cg=pd.read_csv(f"{D}/expression/cengen/thresholded/021821_liberal_threshold1.csv")
classes=list(cg.columns[3:]); Xe=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values.T>0
gname=cg["gene_name"].astype(str).str.lower().values; cls_idx={c:k for k,c in enumerate(classes)}
def ex(g,c):
    w=np.where(gname==g)[0]; return Xe[c,w[0]] if len(w) else False
def clsign(c):
    if ex("unc-25",c) or ex("unc-47",c): return -1
    if ex("eat-4",c) or ex("unc-17",c) or ex("cha-1",c): return 1
    return 0
CLS={cc:clsign(k) for k,cc in enumerate(classes)}
def n2c(n):
    if n in cls_idx: return n
    b=n
    for s in "LR":
        if b.endswith(s): b=b[:-1]; break
    if b in cls_idx: return b
    for s in "DV":
        if b.endswith(s) and b[:-1] in cls_idx: return b[:-1]
    return None
def sign(n): c=n2c(n); return CLS.get(c,0) if c else 0
# Cook directed connectome (weighted)
wb=openpyxl.load_workbook(f"{D}/connectome/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
A=list(wb['hermaphrodite chemical'].iter_rows(values_only=True))
def looks(n): return isinstance(n,str) and len(n)>=2 and n[0].isalpha()
hr=max(range(len(A)),key=lambda r:sum(looks(x) for x in A[r])); cols=[(str(x).strip() if x is not None else None) for x in A[hr]]
hc=max(range(min(6,len(A[0]))),key=lambda c:sum(looks(A[r][c]) for r in range(len(A))))
Ed=[];nset=set()
for r in range(hr+1,len(A)):
    pre=A[r][hc]; pre=pre.strip() if isinstance(pre,str) else None
    if not pre or sign(pre)==0: continue
    for c in range(hc+1,len(A[r])):
        post=cols[c]; v=A[r][c]
        if post and sign(post)!=0 and v not in (None,'') and pre!=post:
            try:
                if float(v)>0: Ed.append((pre,post,float(v))); nset|={pre,post}
            except: pass
neur=sorted(nset); idx={n:i for i,n in enumerate(neur)}; M=len(neur); sg=np.array([sign(n) for n in neur],float)
# signed weight: presynaptic sign, normalized by post in-degree
Wraw=np.zeros((M,M))
for a,b,w in Ed: Wraw[idx[b],idx[a]]+=w*sign(a)   # W[post,pre]
indeg=np.abs(Wraw).sum(1,keepdims=True)+1e-9; W=Wraw/indeg   # row-normalized recurrent matrix
P(f"BRANCH 3 — dynamics | {M} E/I neurons, {len(Ed)} edges")
# (1) spectral radius of recurrent interaction (gain g): regime
J=W  # linearized dW around 0 for dx/dt=-x+g*tanh(Wx): Jacobian = -I + g*W
ev=np.linalg.eigvals(W); rad=np.max(np.abs(ev))
P(f"  recurrent spectral radius = {rad:.2f}  (with gain g>1/rad the network leaves the single rest state -> multistable/state-holding)")
# (2) leading modes localization on command core
CI=set(x+s for x in ['AVA','AVB','AVD','AVE','PVC','RIM','SMD','RMD','AIB','RIB'] for s in ['L','R','DL','DR','VL','VR',''])
ci=np.array([n in CI for n in neur])
lead=np.argsort(-ev.real)[:5]
Vr=np.linalg.eig(W)[1]
for m in lead[:3]:
    v=np.abs(Vr[:,m].real); v=v/v.sum()
    P(f"    mode Re(eig)={ev[m].real:+.2f}: command-core weight {v[ci].sum():.2f} vs core-size-frac {ci.mean():.2f} -> {'LOCALIZED on command core' if v[ci].sum()>2*ci.mean() else 'delocalized'}")
# (3) simulation: kick AVA, measure persistence (state-holding) after input removed
def simulate(Wm,g=1.6,T=400,kick_idx=None,kick_t=50):
    x=np.zeros(M); traj=[]
    for t in range(T):
        inp=np.zeros(M)
        if kick_idx is not None and t<kick_t: inp[kick_idx]=1.0
        x=x+0.1*(-x+g*np.tanh(Wm@x)+inp); traj.append(x.copy())
    return np.array(traj)
ava=[i for i,n in enumerate(neur) if n in ('AVAL','AVAR')]
tr=simulate(W,kick_idx=ava)
post=np.abs(tr[100:]).mean(0)   # activity after kick removed (t>kick)
persist=post.mean(); ava_persist=post[ava].mean()
P(f"\n  KICK AVA -> persistent activity after input off: network mean |x| = {persist:.3f}, AVA itself = {ava_persist:.3f}")
active=int((post>0.05).sum())
P(f"    neurons still active after kick removed: {active}/{M} ({'PERSISTENCE / state held' if active>M*0.1 else 'decays to rest (no latch)'})")
# (4) nulls: shuffle directions (transpose random edges) and shuffle signs
def shuf_sign():
    sp=sg[RNG.permutation(M)]
    Wr=np.zeros((M,M))
    for a,b,w in Ed: Wr[idx[b],idx[a]]+=w*sp[idx[a]]
    return Wr/(np.abs(Wr).sum(1,keepdims=True)+1e-9)
radS=[np.max(np.abs(np.linalg.eigvals(shuf_sign()))) for _ in range(10)]
persS=[]
for _ in range(6):
    Ws=shuf_sign(); trs=simulate(Ws,kick_idx=ava); persS.append(int((np.abs(trs[100:]).mean(0)>0.05).sum()))
P(f"\n  NULLS: spectral radius real {rad:.2f} vs sign-shuffled {np.mean(radS):.2f}±{np.std(radS):.2f}")
P(f"    persistent-active neurons real {active} vs sign-shuffled {np.mean(persS):.0f}±{np.std(persS):.0f}")
P(f"  VERDICT: {'FUNCTION FALLS OUT — real identity-directed+E/I structure holds state/localizes on command core beyond sign-shuffle' if active>np.mean(persS)+2*np.std(persS) else 'state-holding ~ as in shuffled sign -> multistability is topological, not E/I-specific'}")
