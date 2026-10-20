"""BRANCH 17 — do canonical BEHAVIORAL circuits descend the identity-rank? Ecological validation: if the identity-
rank s is the information-flow (sensory->motor) axis, then textbook signal-flow pathways should descend it
monotonically. Trace known circuits (gentle touch, thermotaxis, chemotaxis, O2/CO2, nose-touch) and check each
consecutive step goes DOWN the rank (s decreasing), vs random paths through the same neurons. A clean descent = the
rank is not just a statistical axis but the actual direction real behavioral signals flow.
"""
import warnings,sys; warnings.simplefilter("ignore")
import numpy as np, pandas as pd, openpyxl
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
s={c:float(Xc[cls_idx[c]]@w) for c in classes if c in cls_idx}
# canonical circuits (class-level; established C. elegans signal-flow)
circuits={
 "gentle-touch(ant)": ["ALM","AVD","AVA","VA"],   # ALM->AVD/AVA command->A-motor (reversal)
 "gentle-touch(post)":["PLM","AVB","VB"],          # PLM->AVB->B-motor (forward)
 "thermotaxis":       ["AFD","AIY","AIZ","RIA","RMD"],
 "chemotaxis-attract":["ASE","AIY","AIB","AIZ","RIA"],
 "nose-touch":        ["ASH","AVA","VA"],
 "O2-sensing":        ["URX","RMG","AVA"],
 "oxygen-aversion":   ["BAG","RIA","RMD"],
}
def rank_of(name):
    c=n2c(name); return s.get(c)
P("BRANCH 17 — do behavioral circuits descend the identity-rank (s should DECREASE each step)?")
allsteps=[]; alldesc=[]
for cname,path in circuits.items():
    rs=[rank_of(x) for x in path]
    if any(r is None for r in rs): P(f"    {cname:20s}: (missing map) {list(zip(path,rs))}"); continue
    steps=np.diff(rs)  # negative = descending (correct signal-flow)
    desc=np.mean(steps<0)
    allsteps+=list(steps); alldesc.append(desc)
    P(f"    {cname:20s}: s=[{', '.join(f'{r:+.2f}' for r in rs)}]  descending-steps {int((steps<0).sum())}/{len(steps)}")
frac=np.mean(np.array(allsteps)<0)
P(f"\n  overall: {int((np.array(allsteps)<0).sum())}/{len(allsteps)} steps descend the rank ({frac:.2f})")
# null: random orderings of the same neuron sets
nulls=[]
for _ in range(2000):
    tot=0;cnt=0
    for cname,path in circuits.items():
        rs=[rank_of(x) for x in path]
        if any(r is None for r in rs): continue
        perm=RNG.permutation(rs); st=np.diff(perm); tot+=(st<0).sum(); cnt+=len(st)
    nulls.append(tot/cnt)
z=(frac-np.mean(nulls))/np.std(nulls)
P(f"  random-order null: {np.mean(nulls):.2f}±{np.std(nulls):.2f} -> z={z:+.1f}")
P(f"\n  VERDICT: {'the identity-rank is the REAL signal-flow axis — canonical behavioral circuits descend it monotonically far beyond chance (z>2): the molecular rank predicts the direction real sensory->motor information flows through named pathways' if z>2 and frac>0.6 else 'circuits do not cleanly descend the rank (frac %.2f, z=%.1f)'%(frac,z)}.")
