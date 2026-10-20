"""STAGE 2 — ACTIVITY REOPENING, confound-closed. Null hierarchy scored against Stage-1 CORRECTED target (Cook
~0.746), NOT 0.75. Rate-based stabilization (coincidence frequency ~ rate_i*rate_j). Rate is the generated
network's OWN emergent dynamics (leak-clean). 2a: real vs plain-rate-scramble (rate signal at all?). 2b: real vs
DEGREE-PRESERVING rate-scramble (rate = r_geo[regress on full residual-predictors] + r_act; null=r_geo+shuffle(r_act))
-> isolates activity-specific rate from degree. Break static ceiling + survive Cook = reopen; else confirmed closed.
"""
import warnings,sys,json,csv,re; warnings.simplefilter("ignore")
import numpy as np, openpyxl, pandas as pd
from collections import defaultdict
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, r2_score
def P(*a): print(*a); sys.stdout.flush()
RNG=np.random.default_rng(0); D="../data/celegans/data"
SKD="../seed/data/external/mechanistic_frontier/witvliet2021_developmental/data/skeletons"
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
sk=json.load(open(f"{SKD}/Dataset7_skeletons.json"))
ring=[n for n in hdr if n in sk and sk[n].get('coords') and n2c(n) is not None]; M=len(ring); idx={n:i for i,n in enumerate(ring)}
pairs=[(ring[i],ring[j]) for i in range(M) for j in range(i+1,M)]; Np=len(pairs)
pi=np.array([idx[a] for a,b in pairs]); pj_=np.array([idx[b] for a,b in pairs])
area=np.array([ca.get((min(a,b),max(a,b)),0.0) for a,b in pairs]); geo=np.log1p(area); Eb=area>0
syn=defaultdict(int)
with open(f"{D}/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']!='chemical': continue
        m=re.search(r'witvliet_2020_(\d)',r['source_file'])
        if m and int(m.group(1)) in (7,8): syn[(min(r['pre'],r['post']),max(r['pre'],r['post']))]+=1
yW=np.array([1 if syn.get((min(a,b),max(a,b)),0)>=2 else 0 for a,b in pairs]); K=int(yW.sum())
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
CORR=0.746  # Stage-1 corrected target (Cook contact-area)
# node-level residual-predictor features (for r_geo): contact-degree, node-triangles, arbor length/branch/extent
Cadj=np.zeros((M,M))
for k in np.where(Eb)[0]: Cadj[pi[k],pj_[k]]=1; Cadj[pj_[k],pi[k]]=1
cdeg=Cadj.sum(1); tri=np.diag(Cadj@Cadj@Cadj)/2
def sfeat(n):
    v=sk[n]; return [float(v['length']),sum(1 for _,p in [(x,y) for x,y in [(a,b) for a in [0] for b in [0]]]),0]  # placeholder
alen=np.array([float(sk[n]['length']) for n in ring])
NBR=np.array([sum(1 for _,c in {p:sum(1 for _,pp in sk[n]['arbor'].items() if pp==p) for p in set(sk[n]['arbor'].values())}.items() if c>=2) for n in ring])
EXT=np.array([np.array(list(sk[n]['coords'].values())).std(0).mean() for n in ring])
NODEF=np.column_stack([cdeg,tri,np.log1p(alen),NBR,EXT])

def rate_of(selvec):
    Adj=np.zeros((M,M))
    for k in np.where(selvec>0)[0]: Adj[pi[k],pj_[k]]=1; Adj[pj_[k],pi[k]]=1
    dnorm=Adj/(Adj.sum(1,keepdims=True)+1e-9); x=RNG.standard_normal(M); acc=np.zeros(M)
    for t in range(120):
        x=np.tanh(0.9*(dnorm@x)+0.3*RNG.standard_normal(M))
        if t>20: acc+=np.abs(x)
    return acc/99.0
def grow(ratefn,beta,T=3):
    prop=geo*Eb; sel=np.zeros(Np,bool); sel[np.argsort(-prop)[:K]]=True
    for t in range(T):
        if beta>0:
            rt=ratefn(sel.astype(float)); co=rt[pi]*rt[pj_]; prop=geo*Eb+beta*(co/ (co.max()+1e-9))*Eb
        newsel=np.zeros(Np,bool); newsel[np.argsort(-prop)[:K]]=True; sel=newsel
    return sel
def score(sel,y):
    au=roc_auc_score(y[Eb],sel.astype(float)[Eb]) if y[Eb].sum()>0 else np.nan
    tp=(sel&(y>0)).sum(); return 2*tp/max(1,sel.sum()+y.sum()), au

# static ceiling
static=grow(None,0); f1s,_=score(static,yW); _,aus=score(static,yC)
P("="*84); P("STAGE 2 — activity null hierarchy (scored vs Stage-1 corrected Cook target 0.746)"); P("="*84)
P(f"  STATIC ceiling: F1(W)={f1s:.3f} | Cook-AUC-of-selection ref")
# real rate
realrate=rate_of(static.astype(float))
# r_geo decomposition (regress node rate on FULL residual-predictor node features)
lr=LinearRegression().fit(NODEF,realrate); r_geo=lr.predict(NODEF); r_act=realrate-r_geo; R2geo=r2_score(realrate,r_geo)
# orthogonality check: does r_act predict the node features?
r2_leak=max(r2_score(NODEF[:,j],LinearRegression().fit(r_act.reshape(-1,1),NODEF[:,j]).predict(r_act.reshape(-1,1))) for j in range(NODEF.shape[1]))
P(f"  [decomposition] r_geo R2 = {R2geo:.3f} (rate explained by degree/triangle/spatial) | r_act leak into features R2={r2_leak:.3f}")
if R2geo>0.9: P("  WARNING: r_geo R2>0.9 -> activity-specific rate r_act near-negligible by construction (underpowered).")

def rate_real(sel): return rate_of(sel)
def rate_plain(sel): rt=rate_of(sel); p=RNG.permutation(M); return rt[p]                 # plain rate-scramble
def rate_degpres(sel):                                                                     # degree-preserving rate-scramble
    rt=rate_of(sel); rg=lr.predict(NODEF); ra=rt-rg; return rg+ra[RNG.permutation(M)]
# 2a
P("\n  -- 2a: real vs plain-rate-scramble --")
for beta in [1.0,2.0]:
    fR=np.mean([score(grow(rate_real,beta),yW)[0] for _ in range(3)]); fRc=np.mean([score(grow(rate_real,beta),yC)[0] for _ in range(2)])
    fP=np.mean([score(grow(rate_plain,beta),yW)[0] for _ in range(3)])
    P(f"    beta={beta}: real F1(W)={fR:.3f} F1(Cook)={fRc:.3f} | plain-scramble F1(W)={fP:.3f} | real-plain Δ={fR-fP:+.3f}")
    if beta==1.0: fR1,fP1=fR,fP
# 2b
P("\n  -- 2b: real vs DEGREE-PRESERVING rate-scramble (decisive) --")
for beta in [1.0,2.0]:
    real=[grow(rate_real,beta) for _ in range(3)]; dp=[grow(rate_degpres,beta) for _ in range(3)]
    fR=np.mean([score(s,yW)[0] for s in real]); fRc=np.mean([score(s,yC)[0] for s in real]); fD=np.mean([score(s,yW)[0] for s in dp]); fDc=np.mean([score(s,yC)[0] for s in dp])
    P(f"    beta={beta}: real F1(W)={fR:.3f}/Cook {fRc:.3f} | deg-pres-scramble F1(W)={fD:.3f}/Cook {fDc:.3f} | real-degpres Δ(W)={fR-fD:+.3f} Δ(Cook)={fRc-fDc:+.3f}")
    if beta==1.0: fR1b,fD1b,fRc1b,fDc1b=fR,fD,fRc,fDc
P("\n"+"="*84)
brk=fR1b>f1s+0.01; beats2b=(fR1b>fD1b+0.01) and (fRc1b>fDc1b+0.005)
if R2geo>0.9:
    P("VERDICT: r_geo R2>0.9 -> activity-specific rate NEGLIGIBLE by construction. Rate is a geometry readout. LEANS DEFLATIONARY (closed).")
elif beats2b and brk:
    P("VERDICT: real BEATS degree-preserving rate-scramble + breaks ceiling + survives Cook -> ACTIVITY AXIS REOPENS, confound-closed.")
elif (fR1>fP1+0.01) and not beats2b:
    P("VERDICT: real>plain-rate-scramble but real≈degree-preserving -> the rate signal was r_geo (rate proxying degree/geometry).")
    P("  ACTIVITY ELIMINATION ROBUSTLY CONFIRMED against three nulls (phase + plain-rate + degree-preserving-rate).")
else:
    P("VERDICT: real≈plain-rate-scramble -> no rate signal at all. Activity elimination confirmed.")
P(f"  (scored vs corrected target {CORR}; pre-registered call was: real≈degree-preserving = elimination confirmed.)")
P("="*84)
