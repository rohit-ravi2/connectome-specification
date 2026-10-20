"""STAGE 2 — single-hypothesis lock. Each Stage-1 survivor pulled out, sign pre-registered from its structure:
FP-hit -> predict enriched-in-FP, FLAT-on-FN (avoidance direction); FN-hit -> enriched-FN, flat-FP (defiance).
Locks iff: clears expr-perm AND region null (z>2) on hit structure, directional prediction holds (other structure
flat), AND survives contact-strength stratification (the geometry-leakage kill-shot). Report LOCKED / dissolves-under-X.
"""
import warnings,sys,csv,re,json; warnings.simplefilter("ignore")
import numpy as np, pandas as pd, openpyxl
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
def P(*a): print(*a); sys.stdout.flush()
RNG=np.random.default_rng(1)
SC="../work"
D="../data/celegans/data"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); m0=np.load(f"{SC}/mode0.npz")
I,J,y7,y8,neurons=z['I'],z['J'],z['y7'],z['y8'],list(z['neurons'])
N=len(I); yrep=((y7==1)&(y8==1)).astype(int); agree=m0['agree']; FN=m0['FN']; FPa=m0['fp_absent']
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
def by_name(g):
    w=np.where(gname==g)[0]; return (Xt[w[0]]>0).astype(float) if len(w) else None
Ci=np.array([ (n2c(neurons[i]) if n2c(neurons[i]) is not None else -1) for i in I])
Cj=np.array([ (n2c(neurons[j]) if n2c(neurons[j]) is not None else -1) for j in J])
valid=(Ci>=0)&(Cj>=0); ci=np.where(valid,Ci,0); cj=np.where(valid,Cj,0)
def comp(l,r): return (((l[ci]*r[cj])+(r[ci]*l[cj]))>0)&valid
syn=(yrep==1)&agree&valid; nons=(yrep==0)&agree&valid
STR={'FN':(syn,FN.astype(float)),'FP':(nons,FPa.astype(float))}
# region blocks
pj=json.load(open("../data/network_positions.json"))
pos={c["cell"]:np.array([c["x"],c["y"],c["z"]],float) for c in pj["cells"] if None not in (c.get("x"),c.get("y"),c.get("z"))}
clspos=np.zeros((nCls,3)); have=np.zeros(nCls,bool); acc={}
for n in neurons:
    c=n2c(n)
    if c is not None and n in pos: acc.setdefault(c,[]).append(pos[n])
for c,v in acc.items(): clspos[c]=np.mean(v,0); have[c]=True
lab=KMeans(8,n_init=5,random_state=0).fit(clspos[have]).labels_
class2block=np.full(nCls,-1); class2block[have]=lab; nb=8
for c in np.where(~have)[0]: class2block[c]=nb; nb+=1
blocks=[np.where(class2block==b)[0] for b in range(nb)]
def rperm(e):
    out=e.copy()
    for idx in blocks:
        if len(idx)>1: out[idx]=e[idx][RNG.permutation(len(idx))]
    return out
def gperm(e): return e[RNG.permutation(nCls)]
# contact area
wb0=openpyxl.load_workbook(D+"/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx",read_only=True,data_only=True)
ws=wb0['adult nerve ring neighbors']; data=list(ws.iter_rows(values_only=True)); hdr=[h for h in data[0][1:]]
ca=defaultdict(float)
for r in data[1:]:
    rn=r[0]
    if rn is None: continue
    for jj,val in enumerate(r[1:]):
        cn=hdr[jj]
        if cn and val not in (None,'') and rn!=cn:
            try: ca[(min(rn,cn),max(rn,cn))]=max(ca[(min(rn,cn),max(rn,cn))],float(val))
            except: pass
area=np.array([ca.get((min(neurons[I[k]],neurons[J[k]]),max(neurons[I[k]],neurons[J[k]])),0.0) for k in range(N)])
def enr(ind,U,tgt):
    m=ind&U
    if m.sum()<6 or (U&~ind).sum()<6: return None
    return tgt[m].mean()-tgt[U&~ind].mean()
def zz(l,r,U,tgt,perm,reps):
    ind=comp(l,r); o=enr(ind,U,tgt)
    if o is None: return None
    nl=[enr(comp(perm(l),perm(r)),U,tgt) for _ in range(reps)]; nl=[x for x in nl if x is not None]
    return (o-np.mean(nl))/(np.std(nl)+1e-9) if len(nl)>30 else None
def z_contact(l,r,U,tgt):  # residualize target on contact area, re-test expr-perm
    yy=tgt[U].astype(int); la=np.log1p(area[U]).reshape(-1,1)
    if yy.sum()<6: return None
    pc=LogisticRegression(max_iter=500).fit(la,yy).predict_proba(la)[:,1]
    res=np.zeros(N); res[U]=yy-pc
    ind=comp(l,r); o=enr(ind,U,res)
    nl=[enr(comp(gperm(l),gperm(r)),U,res) for _ in range(1500)]; nl=[x for x in nl if x is not None]
    return (o-np.mean(nl))/(np.std(nl)+1e-9) if len(nl)>30 else None

CAND=[  # (name, geneA, geneB, hit_structure)  hit -> predict enriched there, flat on the other
 ("Notch-Delta LIN-12~LAG-2","lin-12","lag-2","FP"),
 ("MIG-13~MIG-6","mig-13","mig-6","FP"),
 ("NDNF-1~EGL-15","ndnf-1","egl-15","FP"),
 ("Slit-Robo SLT-1~SAX-3","slt-1","sax-3","FN"),
]
P("="*86); P("STAGE 2 — single-hypothesis lock (pre-registered direction, dual null + contact kill-shot)"); P("="*86)
for name,ga,gb,hit in CAND:
    la,lb=by_name(ga),by_name(gb)
    if la is None or lb is None: P(f"{name}: gene missing"); continue
    oth='FN' if hit=='FP' else 'FP'
    Uh,th=STR[hit]; Uo,to=STR[oth]
    zEh=zz(la,lb,Uh,th,gperm,2000); zRh=zz(la,lb,Uh,th,rperm,500)
    zEo=zz(la,lb,Uo,to,gperm,2000); zc=z_contact(la,lb,Uh,th)
    n=int((comp(la,lb)&Uh).sum())
    dir_ok = (zEo is None or zEo<2)   # flat on the other structure
    lock = (zEh is not None and zEh>2) and (zRh is not None and zRh>2) and (zc is not None and zc>2) and dir_ok
    P(f"\n{name}  (pre-registered: enriched-{hit}, flat-{oth})  n={n}")
    P(f"   {hit}: expr-perm-z={zEh if zEh is None else round(zEh,1)}  region-z={zRh if zRh is None else round(zRh,1)}  contact-resid-z={zc if zc is None else round(zc,1)}")
    P(f"   {oth} (should be flat): expr-perm-z={zEo if zEo is None else round(zEo,1)}")
    if lock: P(f"   => LOCKED (both nulls>2, direction holds, survives contact)")
    else:
        why=[]
        if not(zEh and zEh>2): why.append("expr-perm")
        if not(zRh and zRh>2): why.append("region")
        if not(zc and zc>2): why.append("contact")
        if not dir_ok: why.append("direction(other-structure-not-flat)")
        P(f"   => DISSOLVES under: {', '.join(why)}")
P("="*86)
