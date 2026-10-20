"""WAVE 2 GATES — each template runs ONLY if its null passes a random-gene calibration (random gene-sets give
max|z|<3 on real structures). If the null false-positives on random genes, the template is UNCALIBRATABLE and
does not run — report the null-design failure honestly rather than screening with a broken null.
C=gradient (must be separable from raw position), D=threshold (must beat degree-matched null on random genes),
E=temporal (event-ordering must be separable from observation-window/confirmation-count).
"""
import warnings,sys,csv,re,json; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
def P(*a): print(*a); sys.stdout.flush()
RNG=np.random.default_rng(0); SC="../work"; D="../data/celegans/data"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); m0=np.load(f"{SC}/mode0.npz")
I,J,y7,y8,neurons=z['I'],z['J'],z['y7'],z['y8'],list(z['neurons']); N=len(I); idx={n:i for i,n in enumerate(neurons)}
yrep=((y7==1)&(y8==1)).astype(int); agree=m0['agree']; r1=np.load(f"{SC}/resid2.npz")['r1']
cg=pd.read_csv(D+"/expression/cengen/thresholded/021821_liberal_threshold1.csv")
classes=list(cg.columns[3:]); Xt=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values
gname=cg["gene_name"].astype(str).str.lower().values; cls_idx={c:k for k,c in enumerate(classes)}; nCls=len(classes)
def n2c(n):
    if n in cls_idx: return cls_idx[n]
    b=n
    for s in "LR":
        if b.endswith(s): b=b[:-1]; break
    if b in cls_idx: return cls_idx[b]
    for s in "DV":
        if b.endswith(s) and b[:-1] in cls_idx: return cls_idx[b[:-1]]
    return cls_idx.get(b,None)
Ci=np.array([n2c(neurons[i]) if n2c(neurons[i]) is not None else -1 for i in I])
Cj=np.array([n2c(neurons[j]) if n2c(neurons[j]) is not None else -1 for j in J]); valid=(Ci>=0)&(Cj>=0)
pj=json.load(open("../data/network_positions.json"))
pos={c["cell"]:np.array([c["x"],c["y"],c["z"]],float) for c in pj["cells"] if None not in (c.get("x"),c.get("y"),c.get("z"))}
npos=np.array([pos.get(n,[np.nan]*3) for n in neurons]); hasp=~np.isnan(npos[:,0])
somad=np.array([np.linalg.norm(npos[I[k]]-npos[J[k]]) if hasp[I[k]] and hasp[J[k]] else np.nan for k in range(N)])
Xb=(Xt>0).astype(float)

# ===================== GATE C — GRADIENT =====================
P("="*80); P("GATE C — gradient: separable from raw position? random-gene false positives?"); P("="*80)
def grad_diff(gi):
    e=Xb[gi]; ex=[c for c in range(nCls) if e[c]>0]
    # class positions
    cp={c:np.nanmean([npos[idx[n]] for n in neurons if n2c(n)==c and hasp[idx[n]]] or [[np.nan]*3],0) for c in set(list(Ci[valid])+list(Cj[valid]))}
    exok=[c for c in ex if c in cp and not np.isnan(cp[c][0])]
    if len(exok)<4: return None
    P0=np.array([cp[c] for c in exok]); axis=np.linalg.svd(P0-P0.mean(0))[2][0]
    coord={c:(cp[c]@axis if c in cp and not np.isnan(cp[c][0]) else np.nan) for c in cp}
    gd=np.array([abs(coord.get(Ci[k],np.nan)-coord.get(Cj[k],np.nan)) if valid[k] else np.nan for k in range(N)])
    return gd
m=valid&~np.isnan(somad)&(agree>0)
tgt=(yrep>0).astype(float)  # structure = synapse (gradients/guidance shape wiring)
corrs=[]; zrs=[]
for s in range(8):
    gi=np.random.default_rng(s).choice(np.where(Xb.mean(1)>0.05)[0]); gd=grad_diff(gi)
    if gd is None: continue
    mm=m&~np.isnan(gd)
    corrs.append(abs(np.corrcoef(gd[mm],somad[mm])[0,1]))
    # z of grad_diff on synapse AFTER residualizing soma distance
    from numpy.polynomial import polynomial as _
    b=np.polyfit(somad[mm],tgt[mm],1); res=tgt[mm]-np.polyval(b,somad[mm])
    o=np.corrcoef(gd[mm],res)[0,1]; nul=[np.corrcoef(RNG.permutation(gd[mm]),res)[0,1] for _ in range(300)]
    zrs.append((o-np.mean(nul))/(np.std(nul)+1e-9))
P(f"  random-gene gradient: mean|corr(grad_diff, soma_dist)|={np.mean(corrs):.2f} (>0.85 => grad==position)")
P(f"  random-gene gradient false-positive z on synapse|dist: max|z|={np.nanmax(np.abs(zrs)):.1f} (want<3)")
gateC = np.mean(corrs)<0.85 and np.nanmax(np.abs(zrs))<3
P(f"  => GATE C: {'PASS (runnable)' if gateC else 'VOID — gradient inseparable from geometry / null false-positives'}")

# ===================== GATE D — THRESHOLD =====================
P("\n"+"="*80); P("GATE D — threshold: random-gene load beat degree-matched null? (budget-confound lesson)"); P("="*80)
cdeg=np.zeros(nCls)  # contact-degree per class
for k in np.where(valid)[0]: cdeg[Ci[k]]+=1; cdeg[Cj[k]]+=1
zs=[]
for s in range(8):
    g=np.random.default_rng(s).choice(len(gname),30,replace=False); load=Xb[g].sum(0)  # per-class total load
    cc=[c for c in range(nCls) if cdeg[c]>0]
    o=np.corrcoef(load[cc],cdeg[cc])[0,1]
    nul=[np.corrcoef(RNG.permutation(load[cc]),cdeg[cc])[0,1] for _ in range(500)]
    zs.append((o-np.mean(nul))/(np.std(nul)+1e-9))
P(f"  random-gene load vs contact-degree: max|z|={np.nanmax(np.abs(zs)):.1f}  mean z={np.nanmean(zs):+.1f} (want max<3)")
gateD = np.nanmax(np.abs(zs))<3
P(f"  => GATE D: {'PASS (runnable)' if gateD else 'VOID — load trivially predicts degree; threshold inseparable from degree confound'}")

# ===================== GATE E — TEMPORAL =====================
P("\n"+"="*80); P("GATE E — temporal: event-ordering separable from observation-window/confirmation-count?"); P("="*80)
stages=defaultdict=None
appear=defaultdict_=None
import collections
app=collections.defaultdict(set)
with open(f"{D}/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']!='chemical': continue
        mm=re.search(r'witvliet_2020_(\d)',r['source_file'])
        if not mm: continue
        d=int(mm.group(1))
        if r['pre'] in idx and r['post'] in idx:
            a,b=idx[r['pre']],idx[r['post']]; app[(min(a,b),max(a,b))].add(d)
form=np.full(N,np.nan); nobs=np.full(N,np.nan)
for k in range(N):
    key=(min(int(I[k]),int(J[k])),max(int(I[k]),int(J[k])))
    if key in app: form[k]=min(app[key]); nobs[k]=len(app[key])
mm=~np.isnan(form)
c_fn=np.corrcoef(form[mm],nobs[mm])[0,1]
P(f"  corr(formation-stage, #stages-observed) = {c_fn:+.2f}  (|.|>0.9 => collinear, cannot separate)")
# does formation predict residual r1 BEYOND nobs? partial check
from numpy import polyfit,polyval
b=polyfit(nobs[mm],form[mm],1); form_res=form[mm]-polyval(b,nobs[mm])  # formation orthogonalized to nobs
var_left=np.var(form_res)/ (np.var(form[mm])+1e-9)
P(f"  variance of formation-stage independent of #stages-observed: {var_left:.2f} (>0.2 => separable signal exists)")
gateE = abs(c_fn)<0.9 and var_left>0.2
P(f"  => GATE E: {'PASS (runnable — build on developmental connectome, not adult CeNGEN)' if gateE else 'VOID — event-ordering inseparable from confirmation-count'}")
P("\n"+"="*80)
P("WAVE-2 GATE SUMMARY:", f"C={'run' if gateC else 'void'} | D={'run' if gateD else 'void'} | E={'run' if gateE else 'void'}")
P("="*80)
