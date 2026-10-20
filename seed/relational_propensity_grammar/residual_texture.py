#!/usr/bin/env python3
"""RESIDUAL TEXTURE (Part2 per-edge vs Part3 smooth — resolve the disagreement). ONE question:
is the SURVIVING residual (reproducibility after contact + timing + symmetry) spatially AUTOCORRELATED in
arbor-coordinate space, or spatially WHITE?
Key: coordinate for each EDGE = the APPOSITION LOCATION (nearest point between the two arbors, from
skeletons) — arbor-level, not soma centroid. Moran's I on the RESIDUAL (not the propensity).
Contrast: Moran's I of PROPENSITY (should reproduce Part3 ~0.15 smooth) vs Moran's I of RESIDUAL.
Report texture, hold both outcomes open."""
import csv, re, os, json, math
import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

C="../data/celegans/data"
OUT=os.path.expanduser("../seed/data/processed/relational_propensity_grammar")
MASTER=os.path.expanduser("../seed/data/processed/contact_decomposition/contact_decomposition_master.csv")
EDGES=f"{C}/connectome/witvliet2020/witvliet_combined_edges.csv"
SKJSON=os.path.expanduser("../seed/data/external/mechanistic_frontier/witvliet2021_developmental/data/skeletons/Dataset8_skeletons.json")
SKCACHE=os.path.expanduser("../seed/data/processed/subcellular_tiling/_skeleton_coords.json")

ae=defaultdict(set)
for r in csv.DictReader(open(EDGES)):
    if r['type']=='electrical': continue
    ae[int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))].add((r['pre'],r['post']))
W7,W8=ae[7],ae[8]
def pres(a,b,S): return (a,b) in S or (b,a) in S

# ---- skeleton 3D coords (arbor-level) ----
coords={}
if os.path.exists(SKCACHE):
    sk=json.load(open(SKCACHE))
    for n,s in sk.items():
        xyz=np.array(s.get('xyz',[]),float)
        if len(xyz)>0: coords[n]=xyz
if not coords:
    sk=json.load(open(SKJSON))
    for n,d in (sk.items() if isinstance(sk,dict) else []):
        cc=d.get('coords') if isinstance(d,dict) else None
        if cc: coords[n]=np.array(list(cc.values()),float)
print(f"neurons with skeleton coords: {len(coords)}")
# subsample nodes for speed + build trees
trees={}; capped={}
for n,xyz in coords.items():
    if len(xyz)>300:
        idx=np.random.default_rng(0).choice(len(xyz),300,replace=False); xyz=xyz[idx]
    capped[n]=xyz; trees[n]=cKDTree(xyz)
soma={n:xyz.mean(0) for n,xyz in capped.items()}   # centroid (Part3-style node coord)

def mirror(n):
    if n and n[-1]=='L': return n[:-1]+'R'
    if n and n[-1]=='R': return n[:-1]+'L'
    return n

# ---- build pair records: contact-positive, both have skeleton ----
def fnum(r,k):
    try: return float(r[k])
    except: return np.nan
rows=[r for r in csv.DictReader(open(MASTER)) if r.get('c_adult') and float(r['c_adult'])>0]
recs=[]
for r in rows:
    a,b=r['na'],r['nb']
    if a not in trees or b not in trees: continue
    # apposition location = nearest point between arbors
    dd,ii=trees[a].query(capped[b]); j=int(np.argmin(dd)); pb=capped[b][j]
    d2,i2=trees[a].query(pb); pa=capped[a][int(i2)]
    app=(pa+pb)/2.0; appdist=float(np.linalg.norm(pa-pb))
    theta=(1.0*pres(a,b,W7)+1.0*pres(a,b,W8))/2.0
    ma,mb=mirror(a),mirror(b)
    mtheta=(1.0*pres(ma,mb,W7)+1.0*pres(ma,mb,W8))/2.0 if (ma in trees and mb in trees) else np.nan
    recs.append(dict(a=a,b=b,app=app,appdist=appdist,theta=theta,
                     lc=math.log1p(float(r['c_adult'])),
                     lcL1=math.log1p(fnum(r,'c_L1')),
                     birth=np.nanmax([fnum(r,'pre_birth'),fnum(r,'post_birth')]),
                     cstage={'L1':0,'L2':1,'L3':2,'adult':3}.get(r.get('first_contact_stage','L1'),0),
                     mtheta=mtheta))
print(f"contact-positive pairs with apposition coords: {len(recs)} | median apposition dist {np.median([x['appdist'] for x in recs]):.0f} nm")

theta=np.array([x['theta'] for x in recs])
birthmed=np.nanmedian([x['birth'] for x in recs]); mthmed=np.nanmedian([x['mtheta'] for x in recs])
def feat(x):
    b=x['birth'] if not math.isnan(x['birth']) else birthmed
    mt=x['mtheta'] if not math.isnan(x['mtheta']) else mthmed
    return [x['lc'],x['lcL1'],b/1000.0,x['cstage'],mt]  # contact + timing + SYMMETRY(mirror theta)
X=np.array([feat(x) for x in recs])
# neuron-disjoint CV propensity p (regression on theta)
neu=sorted(set(x['a'] for x in recs)|set(x['b'] for x in recs))
rng=np.random.default_rng(0); fold=np.array_split(rng.permutation(neu),5); nf={n:i for i,ff in enumerate(fold) for n in ff}
pr=[(nf[x['a']],nf[x['b']]) for x in recs]
p=np.full(len(recs),np.nan)
for k in range(5):
    te=np.array([fa==k and fb==k for fa,fb in pr]); tr=np.array([fa!=k and fb!=k for fa,fb in pr])
    if te.sum()<10: continue
    m=make_pipeline(StandardScaler(),GradientBoostingRegressor(random_state=0)); m.fit(X[tr],theta[tr]); p[te]=m.predict(X[te])
ok=~np.isnan(p)
resid=theta-p   # residual: reproducibility after contact+timing+symmetry
App=np.array([x['app'] for x in recs])

# ---- Moran's I with kNN spatial weights in APPOSITION (arbor) space ----
def morans_I(values, XYZ, k=8, nperm=999):
    n=len(values); z=values-values.mean()
    tr=cKDTree(XYZ); _,nb=tr.query(XYZ,k=k+1); nb=nb[:,1:]  # k nearest (excl self)
    num=sum(z[i]*z[nb[i]].sum() for i in range(n)); den=(z*z).sum(); S0=n*k
    I=(n/S0)*(num/den)
    perms=[]
    for s in range(nperm):
        zp=np.random.default_rng(s).permutation(z); perms.append((n/S0)*(sum(zp[i]*zp[nb[i]].sum() for i in range(n))/den))
    perms=np.array(perms); pval=(np.sum(perms>=I)+1)/(nperm+1)
    return I, pval, float(perms.mean()), float(perms.std())

print("\n=== MORAN'S I (kNN=8) in ARBOR/APPOSITION coordinate space ===")
Ip,pp,mp,sp=morans_I(p[ok],App[ok]); print(f"  PROPENSITY p (Part3-style, expect smooth): I={Ip:.3f} p={pp:.3g} (null {mp:+.3f}±{sp:.3f})")
Ir,prv,mr,sr=morans_I(resid[ok],App[ok]); print(f"  RESIDUAL (theta - contact-timing-symmetry): I={Ir:.3f} p={prv:.3g} (null {mr:+.3f}±{sr:.3f})  <-- THE ANSWER")
# node-level (soma centroid) residual, for comparison with Part3 soma coords
nodeR=defaultdict(list)
for x,rv,o in zip(recs,resid,ok):
    if o: nodeR[x['a']].append(rv); nodeR[x['b']].append(rv)
nn=[n for n in nodeR if n in soma]; nr=np.array([np.mean(nodeR[n]) for n in nn]); ns=np.array([soma[n] for n in nn])
In,pn,mn2,sn=morans_I(nr,ns,k=6)
print(f"  node-mean RESIDUAL over SOMA centroids (Part3-comparable): I={In:.3f} p={pn:.3g}")

# ---- correlogram: residual spatial covariance vs apposition distance (length scale) ----
print("\n=== correlogram: residual product vs apposition-point distance (texture / length scale) ===")
zr=resid[ok]-resid[ok].mean(); AP=App[ok]
rngc=np.random.default_rng(1); N=ok.sum(); nsamp=min(80000,N*(N-1)//2)
ia=rngc.integers(0,N,nsamp); ib=rngc.integers(0,N,nsamp); msk=ia!=ib; ia,ib=ia[msk],ib[msk]
dst=np.linalg.norm(AP[ia]-AP[ib],axis=1); prod=zr[ia]*zr[ib]
bins=[0,1000,2000,4000,8000,16000,1e9]
for lo,hi in zip(bins[:-1],bins[1:]):
    s=(dst>=lo)&(dst<hi)
    if s.sum()>50:
        lbl=f"{lo/1000:.0f}-{hi/1000:.0f}um" if hi<1e9 else f">{lo/1000:.0f}um"
        print(f"  {lbl:>10s}: mean z_a*z_b = {prod[s].mean():+.4f}  (n={s.sum()})  [>0 = autocorrelated at this scale]")
with open(f"{OUT}/residual_texture.csv","w",newline="") as fo:
    w=csv.writer(fo); w.writerow(["object","morans_I","pval"])
    w.writerow(["propensity_apposition",round(Ip,4),f"{pp:.3g}"]); w.writerow(["RESIDUAL_apposition",round(Ir,4),f"{prv:.3g}"])
    w.writerow(["residual_node_soma",round(In,4),f"{pn:.3g}"])
print("\nresidual texture done ->",OUT)
