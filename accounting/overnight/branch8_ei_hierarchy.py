"""BRANCH 8 — is INHIBITION the FEEDBACK signal? Canonical stable-computation motif: feedforward EXCITATION +
feedback INHIBITION. We have the trophic hierarchy (low=sensory-source -> high=motor-sink) and E/I signs (NT
machinery). Test per chemical edge a->b: is it up-hierarchy (feedforward, h_b>h_a) or down (feedback)? Are
INHIBITORY edges enriched for the feedback (down) direction relative to EXCITATORY? Null = sign-shuffle (topology
& hierarchy fixed, only E/I labels permuted). If inhibition is disproportionately feedback, the connectome wears a
classic feedforward-excite / feedback-inhibit control architecture.
"""
import warnings,sys,openpyxl; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
def P(*a): print(*a); sys.stdout.flush()
RNG=np.random.default_rng(0); D="../data/celegans/data"
cg=pd.read_csv(f"{D}/expression/cengen/thresholded/021821_liberal_threshold1.csv")
classes=list(cg.columns[3:]); Xe=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values.T>0
gname=cg["gene_name"].astype(str).str.lower().values; cls_idx={c:k for k,c in enumerate(classes)}
def ex(g,c):
    w=np.where(gname==g)[0]; return bool(Xe[c,w[0]]) if len(w) else False
def clsign(c):
    if ex("unc-25",c) or ex("unc-47",c): return -1        # GABA
    if ex("eat-4",c) or ex("unc-17",c) or ex("cha-1",c): return 1  # Glu/ACh (excitatory-typed)
    return 0
def n2c(n):
    if not isinstance(n,str): return None
    if n in cls_idx: return n
    b=n
    for s in "LR":
        if b.endswith(s): b=b[:-1]; break
    if b in cls_idx: return b
    for s in "DV":
        if b.endswith(s) and b[:-1] in cls_idx: return b[:-1]
    return None
CLS={cc:clsign(k) for k,cc in enumerate(classes)}
# Cook chemical directed weighted
wb=openpyxl.load_workbook(f"{D}/connectome/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
A=list(wb['hermaphrodite chemical'].iter_rows(values_only=True))
def looks(n): return isinstance(n,str) and len(n)>=2 and n[0].isalpha()
hr=max(range(len(A)),key=lambda r:sum(looks(x) for x in A[r])); cols=[(str(x).strip() if x is not None else None) for x in A[hr]]
hc=max(range(min(6,len(A[0]))),key=lambda c:sum(looks(A[r][c]) for r in range(len(A))))
Ed=[]
for r in range(hr+1,len(A)):
    cpre=n2c(A[r][hc])
    if not cpre: continue
    for c in range(hc+1,len(A[r])):
        cpost=n2c(cols[c]); v=A[r][c]
        if cpost and cpost!=cpre and v not in (None,''):
            try:
                if float(v)>0: Ed.append((cpre,cpost,float(v)))
            except: pass
neur=sorted({a for a,b,w in Ed}|{b for a,b,w in Ed}); idx={n:i for i,n in enumerate(neur)}; M=len(neur)
W=np.zeros((M,M))
for a,b,w in Ed: W[idx[a],idx[b]]+=w
indeg=W.sum(0); outdeg=W.sum(1)
h=np.linalg.lstsq(np.diag(indeg+outdeg)-(W+W.T)+np.ones((M,M))/M, indeg-outdeg, rcond=None)[0]; h=h-h.min()
# edges with a known presynaptic E/I sign, aggregate to unique directed class-pairs (weight-summed)
from collections import defaultdict
ew=defaultdict(float)
for a,b,w in Ed: ew[(a,b)]+=w
sgn={n:CLS.get(n,0) for n in neur}
E=[]  # (delta, presyn_sign, weight)
for (a,b),w in ew.items():
    s=sgn[a]
    if s==0: continue
    E.append((h[idx[b]]-h[idx[a]], s, w))
E=np.array(E)  # cols: delta, sign, weight
delta,sign,wt=E[:,0],E[:,1],E[:,2]
exc=sign>0; inh=sign<0
P(f"BRANCH 8 — {int(exc.sum())} excitatory + {int(inh.sum())} inhibitory signed edges (of {len(E)})")
def fb_frac(mask,weighted=True):  # fraction of (weighted) edges going DOWN hierarchy (feedback)
    m=mask&(np.abs(delta)>1e-6)
    if weighted: return float((wt[m]*(delta[m]<0)).sum()/wt[m].sum())
    return float((delta[m]<0).mean())
fe,fi=fb_frac(exc),fb_frac(inh)
me,mi=np.average(delta[exc],weights=wt[exc]),np.average(delta[inh],weights=wt[inh])
P(f"\n  FEEDBACK (down-hierarchy) fraction of synaptic weight:")
P(f"    excitatory edges: {fe:.3f}   inhibitory edges: {fi:.3f}   (inhibition more feedback by {fi-fe:+.3f})")
P(f"  mean signed hierarchy step (h_post - h_pre): excitatory {me:+.3f} (up=feedforward)  inhibitory {mi:+.3f}")
# NULL: sign-shuffle (permute E/I labels across neurons, topology+hierarchy fixed)
signed=[n for n in neur if sgn[n]!=0]; svals=np.array([sgn[n] for n in signed])
diff_null=[]
for _ in range(1000):
    perm=dict(zip(signed,svals[RNG.permutation(len(svals))]))
    ss=np.array([perm.get(a,0) for (a,b),w in ew.items()])
    ex2=ss>0; ih2=ss<0
    dd=np.array([h[idx[b]]-h[idx[a]] for (a,b) in ew.keys()]); ww=np.array(list(ew.values()))
    fe2=(ww[ex2]*(dd[ex2]<0)).sum()/ww[ex2].sum(); fi2=(ww[ih2]*(dd[ih2]<0)).sum()/ww[ih2].sum()
    diff_null.append(fi2-fe2)
z=((fi-fe)-np.mean(diff_null))/np.std(diff_null)
P(f"\n  NULL (sign-shuffle, topology fixed): inhibition-feedback-excess {fi-fe:+.3f} vs null {np.mean(diff_null):+.3f}±{np.std(diff_null):.3f} -> z={z:+.1f}")
P(f"\n  VERDICT: {'FEEDBACK-INHIBITION motif PRESENT — inhibitory synapses run down the hierarchy (feedback) significantly more than excitatory, beyond sign-shuffle: the connectome wears a feedforward-excite / feedback-inhibit control architecture' if z>2 else ('FEEDFORWARD-INHIBITION / no feedback-inhibition asymmetry (z=%.1f) — E and I distribute similarly on the hierarchy'%z)}.")
