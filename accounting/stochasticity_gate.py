"""STOCHASTICITY GATE (gates Phase 2 interpretation). Is the geometric algorithm's ~49% gap cross-animal
REPRODUCIBLE (structured -> a real target, activity-dependence is a live generator) or IDIOSYNCRATIC (white
noise -> ungenerable by any process incl. a running network -> capstone = geometric computation + irreducible
developmental stochasticity)? Decisive metric: reproducibility of the GEOMETRY-RESIDUAL across Witvliet & Cook.
"""
import warnings,sys,json,csv,re; warnings.simplefilter("ignore")
import numpy as np, pandas as pd, openpyxl
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from scipy.stats import pearsonr
def P(*a): print(*a); sys.stdout.flush()
D="../data/celegans/data"
pj=json.load(open("../data/network_positions.json"))
pos={c["cell"]:np.array([c["x"],c["y"],c["z"]],float) for c in pj["cells"] if None not in (c.get("x"),c.get("y"),c.get("z"))}
cg=pd.read_csv(D+"/expression/cengen/thresholded/021821_liberal_threshold1.csv"); classes=list(cg.columns[3:]); cls_idx={c:k for k,c in enumerate(classes)}
def n2c(n):
    if n in cls_idx: return cls_idx[n]
    b=n
    for s in "LR":
        if b.endswith(s): b=b[:-1]; break
    if b in cls_idx: return cls_idx[b]
    for s in "DV":
        if b.endswith(s) and b[:-1] in cls_idx: return cls_idx[b[:-1]]
    return cls_idx.get(b,None)
wb0=openpyxl.load_workbook(f"{D}/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx",read_only=True,data_only=True)
ws=wb0['adult nerve ring neighbors']; dd=list(ws.iter_rows(values_only=True)); hdr=[h for h in dd[0][1:]]
ca=defaultdict(float)
for r in dd[1:]:
    rn=r[0]
    if rn is None: continue
    for jj,v in enumerate(r[1:]):
        cn=hdr[jj]
        if cn and v not in (None,'') and rn!=cn:
            try: ca[(min(rn,cn),max(rn,cn))]=max(ca[(min(rn,cn),max(rn,cn))],float(v))
            except: pass
ring=[n for n in hdr if n in pos and n2c(n) is not None]; M=len(ring); idx={n:i for i,n in enumerate(ring)}
pairs=[(ring[i],ring[j]) for i in range(M) for j in range(i+1,M)]; Np=len(pairs)
area=np.array([ca.get((min(a,b),max(a,b)),0.0) for a,b in pairs])
# contact graph 2-paths (geometry-derived closure feature, NO synapses)
Cadj=np.zeros((M,M))
for k,(a,b) in enumerate(pairs):
    if area[k]>0: Cadj[idx[a],idx[b]]=1; Cadj[idx[b],idx[a]]=1
c2=Cadj@Cadj; twop=np.array([c2[idx[a],idx[b]] for a,b in pairs])
# Witvliet reproducible synapse
syn=defaultdict(int)
with open(f"{D}/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']!='chemical': continue
        m=re.search(r'witvliet_2020_(\d)',r['source_file'])
        if m and int(m.group(1)) in (7,8): syn[(min(r['pre'],r['post']),max(r['pre'],r['post']))]+=1
yW=np.array([1 if syn.get((min(a,b),max(a,b)),0)>=2 else 0 for a,b in pairs])
# Cook synapse
wb=openpyxl.load_workbook(f"{D}/connectome/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
A=list(wb['hermaphrodite chemical'].iter_rows(values_only=True)); nset=set(ring)
def nm(seq): return sum(1 for x in seq if isinstance(x,str) and x.strip() in nset)
hrow=int(np.argmax([nm(r) for r in A])); cols=[(str(x).strip() if x is not None else None) for x in A[hrow]]
hcol=int(np.argmax([nm([A[r][c] for r in range(len(A))]) for c in range(min(6,len(A[0])))]))
GC=np.zeros((M,M))
for r in range(hrow+1,len(A)):
    rn=A[r][hcol]; rn=rn.strip() if isinstance(rn,str) else None
    if rn not in idx: continue
    for c in range(hcol+1,len(A[r])):
        cn=cols[c]; v=A[r][c]
        if cn in idx and v not in (None,''):
            try:
                if float(v)>0: GC[idx[rn],idx[cn]]=1
            except: pass
yC=np.array([1 if (GC[idx[a],idx[b]] or GC[idx[b],idx[a]]) else 0 for a,b in pairs])
elig=area>0
P(f"pairs {Np} | contact-eligible {elig.sum()} | Witvliet syn {yW.sum()} | Cook syn {yC.sum()}")

# STRONG geometry control: contact strength + shared-neighbors + contact-degrees + soma distance + L4 contact
cdeg=Cadj.sum(1)
somad=np.array([np.linalg.norm(pos[a]-pos[b]) for a,b in pairs])
# L4 (earlier-stage) contact as extra geometry
wsL=wb0['L4 nerve ring neighbors']; dl=list(wsL.iter_rows(values_only=True)); hl=[h for h in dl[0][1:]]; caL=defaultdict(float)
for r in dl[1:]:
    rn=r[0]
    if rn is None: continue
    for jj,v in enumerate(r[1:]):
        cn=hl[jj]
        if cn and v not in (None,'') and rn!=cn:
            try: caL[(min(rn,cn),max(rn,cn))]=max(caL[(min(rn,cn),max(rn,cn))],float(v))
            except: pass
areaL=np.array([caL.get((min(a,b),max(a,b)),0.0) for a,b in pairs])
Fg=np.column_stack([np.log1p(area),np.log1p(twop),cdeg[[idx[a] for a,b in pairs]],cdeg[[idx[b] for a,b in pairs]],somad,np.log1p(areaL)])
skf=StratifiedKFold(5,shuffle=True,random_state=0)
pgeo=cross_val_predict(HistGradientBoostingClassifier(max_iter=250,random_state=0),Fg,yW,cv=skf,method="predict_proba")[:,1]
from sklearn.metrics import roc_auc_score
P(f"  [strong geometry control AUC(pgeo->yW) = {roc_auc_score(yW[area>0],pgeo[area>0]):.3f}]")
m=elig
P("\n"+"="*80); P("STOCHASTICITY GATE — is the geometric-algorithm gap cross-animal reproducible?"); P("="*80)
# (1) raw connectome reproducibility (Witvliet vs Cook)
rraw=pearsonr(yW[m],yC[m])[0]
# (2) CLEAN residual reproducibility: residualize EACH animal on geometry SEPARATELY (no shared-pgeo artifact)
pgeoC=cross_val_predict(HistGradientBoostingClassifier(max_iter=200,random_state=0),Fg,yC,cv=skf,method="predict_proba")[:,1]
resW=yW-pgeo; resC=yC-pgeoC     # each residualized on its OWN geometry projection
rres=pearsonr(resW[m],resC[m])[0]
# also formal partial correlation yW,yC | pgeo (shared geometry basis)
def pcorr(x,y,z):
    rxy=pearsonr(x,y)[0]; rxz=pearsonr(x,z)[0]; ryz=pearsonr(y,z)[0]
    return (rxy-rxz*ryz)/np.sqrt((1-rxz**2)*(1-ryz**2))
rpart=pcorr(yW[m],yC[m],pgeo[m])
P(f"  raw connectome reproducibility  corr(yW, yC | eligible)          = {rraw:+.3f}")
P(f"  CLEAN residual reproducibility corr(yW-pgeoW, yC-pgeoC)          = {rres:+.3f}  [each residualized separately]")
P(f"  partial corr(yW, yC | geometry)                                  = {rpart:+.3f}  [decisive: shared-geometry removed]")
P(f"  fraction of reproducibility surviving geometry-removal           = {rpart/rraw:.2f}")
# (3) among synapses the geometric algorithm MISSES on Witvliet, is Cook also positive?
miss=(yW==1)&(pgeo<np.quantile(pgeo,0.5))&m         # real Witvliet synapse, geometry ranks low
P(f"  geometry-MISSED Witvliet synapses (n={miss.sum()}): Cook-synapse rate = {yC[miss].mean():.2f}  vs base {yC[m].mean():.2f}")
false=(yW==0)&(pgeo>np.quantile(pgeo,0.85))&m       # geometry predicts synapse, Witvliet absent
P(f"  geometry-FALSE Witvliet (n={false.sum()}): Cook ALSO absent rate = {1-yC[false].mean():.2f} vs base {1-yC[m].mean():.2f}")
# (4) does any static feature predict the |residual|? (structured if yes)
from sklearn.model_selection import cross_val_predict as cvp
absres=np.abs(resW)
degW=np.zeros(M)
for k in np.where(yW>0)[0]: a,b=pairs[k]; degW[idx[a]]+=1; degW[idx[b]]+=1
Fr=np.column_stack([np.log1p(area),np.log1p(twop),degW[[idx[a] for a,b in pairs]],degW[[idx[b] for a,b in pairs]]])
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
pr=cvp(HistGradientBoostingRegressor(max_iter=150,random_state=0),Fr[m],absres[m],cv=5)
P(f"  static features predict |residual| R2 = {r2_score(absres[m],pr):.3f} (>0.05 = some structure)")
P("\n"+"="*80)
structured = rpart>0.12 and (rpart/rraw)>0.25
if structured:
    P("GATE: STRUCTURED — the geometry-residual is cross-animal reproducible. Something specifies the gap;")
    P("  activity-dependence (Phase 2) is a live generator. Phase-2 output would be interpretable as signal.")
else:
    P("GATE: STRUCTURELESS / IDIOSYNCRATIC — the geometry-residual does NOT reproduce across animals. The gap is")
    P("  developmental NOISE, ungenerable by any process INCLUDING a running network (activity makes structured")
    P("  wiring, not noise). => Phase 2 CANNOT close it; a ceiling-break would be same-data overfit. HONEST")
    P("  CAPSTONE: geometric computation + IRREDUCIBLE developmental stochasticity. Do NOT force a dynamical story.")
P("="*80)
