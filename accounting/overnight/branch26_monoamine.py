"""BRANCH 26 — does the EXTRASYNAPTIC monoamine network share the chemical connectome's feedforward/rank
organization, or is it differently structured? B25 showed functional flow decouples from the chemical arrow;
Randi attributes function to extrasynaptic/neuromodulatory signaling. So examine the MONOAMINE network directly.
Build it from Bentley 2016 (neuron releases amine X -> any neuron expressing an X-receptor), with the textbook
amine->receptor pairing. Tests: (1) is it feedforward (trophic-coherent) like chemical? (2) does the chemical
identity-rank predict monoamine-edge direction? If the monoamine network is NOT feedforward / NOT rank-aligned,
that explains B25 — function runs on an extrasynaptic network with a different (broadcast) architecture.
"""
import warnings,sys,openpyxl,csv; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
D="../data/celegans/data"
cg=pd.read_csv(f"{D}/expression/cengen/thresholded/021821_liberal_threshold1.csv")
classes=list(cg.columns[3:]); Xf=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values.T>0
br=Xf.mean(0); Xc=Xf[:,(br>0.05)&(br<0.95)].astype(float); cls_idx={c:k for k,c in enumerate(classes)}
def n2c(n):
    if not isinstance(n,str): return None
    n=n.strip().upper()
    if n in cls_idx: return n
    b=n
    for s in "LR":
        if b.endswith(s): b=b[:-1]; break
    if b in cls_idx: return b
    for s in "DV":
        if b.endswith(s) and b[:-1] in cls_idx: return b[:-1]
    return None
# identity rank from chemical
wb=openpyxl.load_workbook(f"{D}/connectome/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
A=list(wb['hermaphrodite chemical'].iter_rows(values_only=True))
def looks(n): return isinstance(n,str) and len(n)>=2 and n[0].isalpha()
hr=max(range(len(A)),key=lambda r:sum(looks(x) for x in A[r])); cols=[(str(x).strip() if x is not None else None) for x in A[hr]]
hc=max(range(min(6,len(A[0]))),key=lambda c:sum(looks(A[r][c]) for r in range(len(A))))
chem=set()
for r in range(hr+1,len(A)):
    cpre=n2c(A[r][hc])
    if not cpre: continue
    for c in range(hc+1,len(A[r])):
        cpost=n2c(cols[c]); v=A[r][c]
        if cpost and cpost!=cpre and v not in (None,''):
            try:
                if float(v)>0: chem.add((cpre,cpost))
            except: pass
asym=[(a,b) for (a,b) in chem if (b,a) not in chem]
Fr=[];yr=[]
for a,b in asym:
    d=Xc[cls_idx[a]]-Xc[cls_idx[b]]; Fr+=[d,-d]; yr+=[1,0]
w=LogisticRegression(max_iter=1000,C=0.5).fit(np.array(Fr),np.array(yr)).coef_[0]
s={c:float(Xc[cls_idx[c]]@w) for c in classes if c in cls_idx}
# Bentley: amine sources + receptor expression
PAIR={"Dopamine":["DOP-1","DOP-2","DOP-3","DOP-4","DOP-5","DOP-6"],
      "Serotonin":["SER-1","SER-4","SER-5","SER-7","MOD-1"],
      "Tyramine":["SER-2","TYRA-2","TYRA-3","LGC-55"],
      "Octopamine":["SER-3","SER-6","OCTR-1"]}
releases={}; recept={}
with open(f"{D}/expression/bentley2016/Bentley_et_al_2016_expression.csv") as f:
    for r in csv.DictReader(f):
        e1=r['Entity1'].strip(); rel=r['Relationship'].strip(); e2=r['Entity2'].strip()
        if rel=="Neurotransmitter" and e2 in PAIR: releases.setdefault(e1,set()).add(e2)
        if rel=="Receptor": recept.setdefault(e1,set()).add(e2.upper())
# build monoamine edges A->B: A releases X, B has an X-receptor
mono=set()
for a,amines in releases.items():
    ca=n2c(a)
    if ca is None: continue
    for b,recs in recept.items():
        cb=n2c(b)
        if cb is None or cb==ca: continue
        for X in amines:
            if any(rr in recs for rr in PAIR[X]): mono.add((ca,cb)); break
P(f"BRANCH 26 — monoamine edges {len(mono)} | {len(releases)} amine-source neurons, {len(recept)} receptor-expressing")
# (1) trophic coherence of monoamine network
neur=sorted({a for a,b in mono}|{b for a,b in mono}); idx={n:i for i,n in enumerate(neur)}; M=len(neur)
W=np.zeros((M,M))
for a,b in mono: W[idx[a],idx[b]]+=1
indeg=W.sum(0); outdeg=W.sum(1)
h=np.linalg.lstsq(np.diag(indeg+outdeg)-(W+W.T)+np.ones((M,M))/M, indeg-outdeg, rcond=None)[0]; h=h-h.min()
r,c=np.nonzero(W); F0=float(((h[c]-h[r]-1)**2).mean())
RNG=np.random.default_rng(0); F0n=[float(((h[RNG.permutation(M)][c]-h[RNG.permutation(M)][r]-1)**2).mean()) for _ in range(200)]
P(f"  (1) monoamine trophic incoherence F0={F0:.2f} vs shuffled {np.mean(F0n):.2f} (chemical F0~0.4 strongly feedforward)")
P(f"      -> {'monoamine net IS feedforward-coherent' if F0<np.mean(F0n)*0.7 else 'monoamine net is NOT strongly feedforward (more recurrent/broadcast than chemical)'}")
# (2) does chemical identity-rank predict monoamine-edge direction?
masym=[(a,b) for (a,b) in mono if (b,a) not in mono and a in s and b in s]
if len(masym)>20:
    yv=[];sc=[]
    for a,b in masym: yv+=[1,0]; sc+=[s[a]-s[b], s[b]-s[a]]
    au=roc_auc_score(yv,sc); dn=np.mean([1 if s[a]>s[b] else 0 for a,b in masym])
    P(f"  (2) chemical identity-rank -> monoamine-edge direction: {len(masym)} asym pairs, down-rank {dn:.3f}, AUC {au:.3f}")
# (3) overlap with chemical
ov=len(mono & chem)/len(mono) if mono else 0
P(f"  (3) monoamine edges also present as chemical synapses: {ov:.2f}")
P(f"\n  VERDICT: {'the monoamine (extrasynaptic) network is DIFFERENTLY organized than the chemical feedforward one — %s, %s: consistent with B25, function runs on a broadcast neuromodulatory network decoupled from the chemical arrow'%(('less feedforward F0=%.2f'%F0) if F0>np.mean(F0n)*0.7 else 'feedforward too', ('rank-decoupled AUC=%.2f'%au if len(masym)>20 and abs(au-0.5)<0.15 else 'rank-partly-aligned')) if (F0>np.mean(F0n)*0.7 or (len(masym)>20 and abs(au-0.5)<0.15)) else 'monoamine network resembles the chemical feedforward/rank organization'}.")
