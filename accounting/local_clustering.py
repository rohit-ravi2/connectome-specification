import sys, numpy as np
from collections import defaultdict
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); m0=np.load(f"{SC}/mode0.npz")
I,J,y7,y8,neurons=z['I'],z['J'],z['y7'],z['y8'],list(z['neurons']); M=len(neurons); N=len(I)
FN=m0['FN']; FP=m0['FP']; det=m0['det_core']; agree=m0['agree']
pmap={(int(I[k]),int(J[k])):k for k in range(N)}
has_rev=np.array([ (int(J[k]),int(I[k])) in pmap for k in range(N)])  # reverse is a contacting pair
revk=np.array([ pmap.get((int(J[k]),int(I[k])),-1) for k in range(N)])
rng=np.random.default_rng(0)
def reciprocity(mask,lab):
    pool=has_rev  # fair pool: pairs whose reverse is also a contacting pair
    sel=mask&pool
    obs=np.mean([mask[revk[k]] for k in np.where(sel)[0]]) if sel.sum() else 0
    # null: shuffle the mask among pool pairs, keep count
    nl=[]
    poolidx=np.where(pool)[0]; cnt=int((mask&pool).sum())
    for _ in range(300):
        sh=np.zeros(N,bool); sh[rng.choice(poolidx,cnt,replace=False)]=True
        s2=sh&pool; nl.append(np.mean([sh[revk[k]] for k in np.where(s2)[0]]) if s2.sum() else 0)
    nl=np.array(nl); z=(obs-nl.mean())/(nl.std()+1e-9)
    P(f"  [{lab}] reciprocity: obs reverse-also-residual = {obs:.3f} vs null {nl.mean():.3f}±{nl.std():.3f}  z={z:+.1f}")
def bundling(mask,lab):
    idxs=np.where(mask)[0]; deg=defaultdict(int)
    for k in idxs: deg[int(I[k])]+=1; deg[int(J[k])]+=1
    shared=np.mean([ (deg[int(I[k])]+deg[int(J[k])])>2 for k in idxs])   # shares an endpoint with another residual edge
    conc=np.var([deg[n] for n in range(M)]); mx=max(deg.values()) if deg else 0
    # degree-stratified null: shuffle residual labels among agree pairs within source-degree strata
    src_deg=np.array([sum(1 for kk in range(N) if I[kk]==I[k]) for k in range(N)])  # slow; approximate w/ contacting degree
    nl_sh=[]; poolidx=np.where(agree)[0]; cnt=int((mask&agree).sum())
    for _ in range(200):
        sh=np.zeros(N,bool); sh[rng.choice(poolidx,cnt,replace=False)]=True
        d2=defaultdict(int)
        for k in np.where(sh)[0]: d2[int(I[k])]+=1; d2[int(J[k])]+=1
        nl_sh.append(np.mean([ (d2[int(I[k])]+d2[int(J[k])])>2 for k in np.where(sh)[0]]))
    nl_sh=np.array(nl_sh); zc=(shared-nl_sh.mean())/(nl_sh.std()+1e-9)
    P(f"  [{lab}] endpoint-bundling: {shared:.3f} of residual edges share a neuron w/ another vs null {nl_sh.mean():.3f}±{nl_sh.std():.3f} z={zc:+.1f} | max residual-edges-on-one-neuron={mx}")
P(f"residual sets: FN(surprising-connections)={FN.sum()} FP(surprising-non-conn)={FP.sum()} det_core={det.sum()}")
P("\n[RECIPROCITY — is the hard core organized into mutual pairs?]")
for m,l in [(FN,'FN'),(FP,'FP'),(det,'det_core')]: reciprocity(m,l)
P("\n[ENDPOINT-BUNDLING — do residual edges cluster on shared neurons beyond chance?]")
for m,l in [(FN,'FN'),(FP,'FP'),(det,'det_core')]: bundling(m,l)
