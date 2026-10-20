"""HELD-OUT REPLICATION on Cook 2019 hermaphrodite (independent animal + reconstruction + lab).
Rebuild the zero-identity residual on Cook chemical synapses (contact[shared geometry] + Cook-closure[2-paths]
+ Cook-mirror), then test whether the Stage-2 candidates replicate on Cook's residual, SAME structure:
  FN_cook (Cook synapse the process missed): Slit-Robo SLT-1~SAX-3 should re-hit.
  FP_cook (contact but no Cook synapse):     Notch-Delta LIN-12~LAG-2, MIG-13~MIG-6 should re-hit.
Replicates iff same structure + survives expr-perm + region + contact nulls out-of-sample.
"""
import warnings,sys,json; warnings.simplefilter("ignore")
import numpy as np, pandas as pd, openpyxl
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression
def P(*a): print(*a); sys.stdout.flush()
RNG=np.random.default_rng(2)
SC="../work"
D="../data/celegans/data"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True)
I,J,neurons=z['I'],z['J'],list(z['neurons']); N=len(I); M=len(neurons); nset=set(neurons); idx={n:i for i,n in enumerate(neurons)}
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

# ---- Cook herm chemical adjacency ----
wb=openpyxl.load_workbook(f"{D}/connectome/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
A=list(wb['hermaphrodite chemical'].iter_rows(values_only=True))
def nmatch(seq): return sum(1 for x in seq if isinstance(x,str) and x.strip() in nset)
hr=int(np.argmax([nmatch(row) for row in A]))                       # header row (col names)
cols=[ (str(x).strip() if x is not None else None) for x in A[hr]]
hc=int(np.argmax([nmatch([A[r][c] for r in range(len(A))]) for c in range(min(6,len(A[0])))]))
GC=np.zeros((M,M))
for r in range(hr+1,len(A)):
    rn=A[r][hc]; rn=rn.strip() if isinstance(rn,str) else None
    if rn not in idx: continue
    for c in range(hc+1,len(A[r])):
        cn=cols[c]; v=A[r][c]
        if cn in idx and v not in (None,'') :
            try:
                if float(v)>0: GC[idx[rn],idx[cn]]=1
            except: pass
covered=sorted(set(np.where(GC.sum(1)+GC.sum(0)>0)[0]))
P(f"Cook herm chemical: {int(GC.sum())} directed edges among {len(covered)}/{M} mapped neurons")
yC=np.array([1 if (GC[I[k],J[k]] or GC[J[k],I[k]]) else 0 for k in range(N)])

# ---- contact (shared geometry) ----
wb0=openpyxl.load_workbook(f"{D}/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx",read_only=True,data_only=True)
ws=wb0['adult nerve ring neighbors']; dat=list(ws.iter_rows(values_only=True)); hdr=[h for h in dat[0][1:]]
ca=defaultdict(float)
for r in dat[1:]:
    rn=r[0]
    if rn is None: continue
    for jj,val in enumerate(r[1:]):
        cn=hdr[jj]
        if cn and val not in (None,'') and rn!=cn:
            try: ca[(min(rn,cn),max(rn,cn))]=max(ca[(min(rn,cn),max(rn,cn))],float(val))
            except: pass
area=np.array([ca.get((min(neurons[I[k]],neurons[J[k]]),max(neurons[I[k]],neurons[J[k]])),0.0) for k in range(N)])
contact=(area>0)
# ---- zero-identity residual on Cook: GBM(contact + Cook-closure + Cook-mirror) ----
GCu=((GC+GC.T)>0).astype(float); twop=GCu@GCu
deg=GCu.sum(1)
def mir(n):
    if n and n[-1] in 'LR':
        m=n[:-1]+('R' if n[-1]=='L' else 'L'); return m if m in idx else None
    return None
mrep=np.zeros(N)
for k in range(N):
    mi,mj=mir(neurons[I[k]]),mir(neurons[J[k]])
    if mi and mj: mrep[k]=GCu[idx[mi],idx[mj]]
feat=np.column_stack([np.log1p(area),deg[I],deg[J],np.log1p(twop[I,J]),mrep])
cv=StratifiedKFold(5,shuffle=True,random_state=0)
pC=cross_val_predict(GradientBoostingClassifier(random_state=0),feat,yC,cv=cv,method='predict_proba')[:,1]
rC=yC-pC
synC=(yC==1)&valid&contact; FNc=synC&(rC>0.3)
nonC=(yC==0)&valid&contact; FPc=nonC&(rC<-0.3)
P(f"Cook residual: synapses(mapped,contacting)={synC.sum()} FN_cook={FNc.sum()} | non-syn={nonC.sum()} FP_cook={FPc.sum()}")

# ---- region blocks + nulls ----
pj=json.load(open("../data/network_positions.json"))
pos={c["cell"]:np.array([c["x"],c["y"],c["z"]],float) for c in pj["cells"] if None not in (c.get("x"),c.get("y"),c.get("z"))}
clspos=np.zeros((nCls,3)); have=np.zeros(nCls,bool); acc={}
for n in neurons:
    c=n2c(n)
    if c is not None and n in pos: acc.setdefault(c,[]).append(pos[n])
for c,v in acc.items(): clspos[c]=np.mean(v,0); have[c]=True
lab=KMeans(8,n_init=5,random_state=0).fit(clspos[have]).labels_
c2b=np.full(nCls,-1); c2b[have]=lab; nb=8
for c in np.where(~have)[0]: c2b[c]=nb; nb+=1
blocks=[np.where(c2b==b)[0] for b in range(nb)]
def rperm(e):
    o=e.copy()
    for ix in blocks:
        if len(ix)>1: o[ix]=e[ix][RNG.permutation(len(ix))]
    return o
def gperm(e): return e[RNG.permutation(nCls)]
def enr(ind,U,t):
    m=ind&U
    if m.sum()<6 or (U&~ind).sum()<6: return None
    return t[m].mean()-t[U&~ind].mean()
def Z(l,r,U,t,perm,reps):
    o=enr(comp(l,r),U,t)
    if o is None: return None
    nl=[enr(comp(perm(l),perm(r)),U,t) for _ in range(reps)]; nl=[x for x in nl if x is not None]
    return (o-np.mean(nl))/(np.std(nl)+1e-9) if len(nl)>30 else None
def Zc(l,r,U,t):
    yy=t[U].astype(int); la=np.log1p(area[U]).reshape(-1,1)
    if yy.sum()<6: return None
    pc=LogisticRegression(max_iter=500).fit(la,yy).predict_proba(la)[:,1]; res=np.zeros(N); res[U]=yy-pc
    o=enr(comp(l,r),U,res); nl=[enr(comp(gperm(l),gperm(r)),U,res) for _ in range(1500)]; nl=[x for x in nl if x is not None]
    return (o-np.mean(nl))/(np.std(nl)+1e-9) if len(nl)>30 else None

CAND=[("Notch-Delta LIN-12~LAG-2","lin-12","lag-2","FP",(nonC,FPc.astype(float))),
      ("MIG-13~MIG-6","mig-13","mig-6","FP",(nonC,FPc.astype(float))),
      ("Slit-Robo SLT-1~SAX-3","slt-1","sax-3","FN",(synC,FNc.astype(float)))]
P("\n"+"="*82); P("HELD-OUT REPLICATION ON COOK — do Stage-2 candidates re-hit the same structure?"); P("="*82)
for name,ga,gb,hit,(U,t) in CAND:
    la,lb=by_name(ga),by_name(gb)
    zE=Z(la,lb,U,t,gperm,2000); zR=Z(la,lb,U,t,rperm,500); zc=Zc(la,lb,U,t); n=int((comp(la,lb)&U).sum())
    rep = zE is not None and zE>2 and zR is not None and zR>2 and zc is not None and zc>2
    P(f"\n{name} @ {hit}_cook  n={n}")
    P(f"   expr-perm-z={zE if zE is None else round(zE,1)}  region-z={zR if zR is None else round(zR,1)}  contact-resid-z={zc if zc is None else round(zc,1)}")
    P(f"   => {'REPLICATES out-of-sample (survives all 3 nulls on Cook)' if rep else 'does NOT replicate on Cook'}")
P("="*82)
