"""BRANCH 13 — is the feedforward flow LAYERED or SHALLOW/BROADCAST? Each neuron has an identity-rank s. For each
directed edge, the rank-gap |s_pre - s_post| tells us whether synapses connect ADJACENT ranks (layered cascade,
sensory->inter->motor step by step) or SKIP across the whole axis (shallow broadcast, sensory straight to motor).
Compare the real edge rank-gap distribution to a DEGREE-PRESERVING rewire null (same out/in degree sequence, random
partners). Smaller-than-null gaps => layered/local-in-rank; larger-or-equal => broadcast/shallow. Also: does contact
GEOMETRY constrain the gap (spatial adjacency ~ rank adjacency)?
"""
import warnings,sys,openpyxl; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
def P(*a): print(*a); sys.stdout.flush()
RNG=np.random.default_rng(0); D="../data/celegans/data"
cg=pd.read_csv(f"{D}/expression/cengen/thresholded/021821_liberal_threshold1.csv")
classes=list(cg.columns[3:]); Xf=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values.T>0
br=Xf.mean(0); Xc=Xf[:,(br>0.05)&(br<0.95)].astype(float); cls_idx={c:k for k,c in enumerate(classes)}
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
chem=set((a,b) for a,b,w in Ed); asym=[(a,b) for (a,b) in chem if (b,a) not in chem]
F=[];y=[]
for a,b in asym:
    d=Xc[cls_idx[a]]-Xc[cls_idx[b]]; F+=[d,-d]; y+=[1,0]
w=LogisticRegression(max_iter=1000,C=0.5).fit(np.array(F),np.array(y)).coef_[0]
neur=sorted({a for a,b,w_ in Ed}|{b for a,b,w_ in Ed})
s={n:float(Xc[cls_idx[n]]@w) for n in neur}
sv=np.array([s[n] for n in neur]); srng=sv.max()-sv.min()
# normalize ranks to [0,1] by percentile so "gap" is fraction-of-axis
from scipy.stats import rankdata
pr={neur[i]:rankdata(sv)[i]/len(sv) for i in range(len(neur))}
edges=[(a,b) for a,b,w_ in Ed]
gaps=np.array([pr[a]-pr[b] for a,b in edges])   # signed: >0 = down-rank (feedforward)
absgap=np.abs(gaps)
P(f"BRANCH 13 — {len(edges)} directed edges | rank normalized to [0,1] percentile")
P(f"  mean |rank-gap| of real edges = {absgap.mean():.3f} (0=adjacent rank, 1=full-axis skip)")
P(f"  median = {np.median(absgap):.3f}; fraction long-range (|gap|>0.5) = {(absgap>0.5).mean():.3f}")
# degree-preserving null: keep each edge's PRE, shuffle POST among all neurons weighted by in-degree (config-model-ish)
posts=[b for a,b in edges]; pres=[a for a,b in edges]
nullmeans=[]
for _ in range(500):
    perm=RNG.permutation(len(posts)); ng=np.abs(np.array([pr[pres[i]]-pr[posts[perm[i]]] for i in range(len(edges))]))
    nullmeans.append(ng.mean())
z=(absgap.mean()-np.mean(nullmeans))/np.std(nullmeans)
P(f"\n  degree-preserving rewire null: mean|gap| {np.mean(nullmeans):.3f}±{np.std(nullmeans):.3f}; real {absgap.mean():.3f} -> z={z:+.1f}")
P(f"    -> {'LAYERED: edges connect nearer-in-rank than chance (local cascade)' if z<-2 else ('BROADCAST/long-range: edges span the rank axis as much as chance' if abs(z)<2 else 'edges span WIDER than chance')}")
# does geometry (contact area) constrain gap? contacting-pair rank-gaps vs all-pair
wb0=openpyxl.load_workbook(f"{D}/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx",read_only=True,data_only=True)
ws=wb0['adult nerve ring neighbors']; dd=list(ws.iter_rows(values_only=True)); hdr=[h for h in dd[0][1:]]
from collections import defaultdict
ca=defaultdict(float)
for r in dd[1:]:
    rn=r[0]
    if rn is None: continue
    for jj,v in enumerate(r[1:]):
        cn=hdr[jj]
        if cn and v not in (None,'') and rn!=cn:
            try: ca[(min(str(rn),str(cn)),max(str(rn),str(cn)))]=float(v)
            except: pass
contact_pairs=[(a,b) for (a,b) in ca if n2c(a) in [n2c(x) for x in neur] and n2c(a) is not None and n2c(b) is not None and n2c(a) in s and n2c(b) in s]
cg_gap=np.array([abs(s[n2c(a)]-s[n2c(b)]) for a,b in contact_pairs if n2c(a) in s and n2c(b) in s])
allpairs_gap=np.array([abs(sv[i]-sv[j]) for i in range(len(neur)) for j in range(i+1,len(neur))])
P(f"\n  rank-gap (raw s): contacting pairs {np.median(cg_gap):.2f} vs all pairs {np.median(allpairs_gap):.2f} (contact does {'compress' if np.median(cg_gap)<np.median(allpairs_gap)*0.9 else 'NOT compress'} rank-gap)")
P(f"\n  VERDICT: the feedforward flow is {'a LOCAL CASCADE (layered — synapses connect adjacent ranks more than chance)' if z<-2 else 'SHALLOW/BROADCAST (synapses skip across the rank axis as freely as chance — sensory reaches many depths directly, few strict layers)'}.")
