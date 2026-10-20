"""TEMPLATE B — COMBINATORIAL (barcode vector-distance). Native form of self-avoidance (Dscam/Pcdh):
cells sharing too much barcode repel -> contact but no synapse (FP). Distributed code: no single pair carries
it (why per-rule matching nulled it); only barcode-SIMILARITY can see it.
Pre-registered direction: barcode similarity -> FP enrichment. Native null: permute barcodes across neurons
(preserve family stats, break neuron<->barcode). Region-null standard. Cook replication BUILT IN.
Enumerate named self-avoidance families + DATA-DERIVED co-expression modules (data picks the barcode).
"""
import warnings,sys,json; warnings.simplefilter("ignore")
import numpy as np, pandas as pd, openpyxl
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
def P(*a): print(*a); sys.stdout.flush()
RNG=np.random.default_rng(0)
SC="../work"; D="../data/celegans/data"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); m0=np.load(f"{SC}/mode0.npz")
I,J,y7,y8,neurons=z['I'],z['J'],z['y7'],z['y8'],list(z['neurons']); N=len(I); M=len(neurons); idx={n:i for i,n in enumerate(neurons)}
yrep=((y7==1)&(y8==1)).astype(int); agree=m0['agree']; FN=m0['FN']; FPa=m0['fp_absent']; DET=m0['det_core']
cg=pd.read_csv(D+"/expression/cengen/thresholded/021821_liberal_threshold1.csv")
classes=list(cg.columns[3:]); Xt=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values
gname=cg["gene_name"].astype(str).str.lower().values; cls_idx={c:k for k,c in enumerate(classes)}; nCls=len(classes)
Xb=(Xt>0).astype(np.float32)  # genes x classes binary
def n2c(n):
    if n in cls_idx: return cls_idx[n]
    b=n
    for s in "LR":
        if b.endswith(s): b=b[:-1]; break
    if b in cls_idx: return cls_idx[b]
    for s in "DV":
        if b.endswith(s) and b[:-1] in cls_idx: return cls_idx[b[:-1]]
    return cls_idx.get(b,None)
Ci=np.array([ (n2c(neurons[i]) if n2c(neurons[i]) is not None else -1) for i in I])
Cj=np.array([ (n2c(neurons[j]) if n2c(neurons[j]) is not None else -1) for j in J])
valid=(Ci>=0)&(Cj>=0); ciV=np.where(valid,Ci,0); cjV=np.where(valid,Cj,0)

# contact + Cook residual (for FP_cook replication) ----
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
area=np.array([ca.get((min(neurons[I[k]],neurons[J[k]]),max(neurons[I[k]],neurons[J[k]])),0.0) for k in range(N)]); contact=area>0
wb=openpyxl.load_workbook(f"{D}/connectome/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
A=list(wb['hermaphrodite chemical'].iter_rows(values_only=True)); nset=set(neurons)
def nmatch(seq): return sum(1 for x in seq if isinstance(x,str) and x.strip() in nset)
hrow=int(np.argmax([nmatch(r) for r in A])); cols=[(str(x).strip() if x is not None else None) for x in A[hrow]]
hcol=int(np.argmax([nmatch([A[r][c] for r in range(len(A))]) for c in range(min(6,len(A[0])))]))
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
yC=np.array([1 if (GC[I[k],J[k]] or GC[J[k],I[k]]) else 0 for k in range(N)])
GCu=((GC+GC.T)>0).astype(float); twop=GCu@GCu; deg=GCu.sum(1)
def mir(n):
    if n and n[-1] in 'LR':
        mm=n[:-1]+('R' if n[-1]=='L' else 'L'); return mm if mm in idx else None
    return None
mrep=np.array([ (GCu[idx[mir(neurons[I[k]])],idx[mir(neurons[J[k]])]] if mir(neurons[I[k]]) and mir(neurons[J[k]]) else 0) for k in range(N)])
feat=np.column_stack([np.log1p(area),deg[I],deg[J],np.log1p(twop[I,J]),mrep])
pC=cross_val_predict(GradientBoostingClassifier(random_state=0),feat,yC,cv=StratifiedKFold(5,shuffle=True,random_state=0),method='predict_proba')[:,1]
rC=yC-pC; FPc=(yC==0)&valid&contact&(rC<-0.3); nonC=(yC==0)&valid&contact

# universes
nonW=(yrep==0)&agree&valid&contact   # Witvliet FP universe
STRUCT={'FP':(nonW,FPa.astype(float)),'FN':((yrep==1)&agree&valid,FN.astype(float)),'det':(agree&valid,DET.astype(float))}
# region blocks
pj=json.load(open("../data/network_positions.json"))
pos={c["cell"]:np.array([c["x"],c["y"],c["z"]],float) for c in pj["cells"] if None not in (c.get("x"),c.get("y"),c.get("z"))}
clspos=np.zeros((nCls,3)); have=np.zeros(nCls,bool); acc={}
for n in neurons:
    c=n2c(n)
    if c is not None and n in pos: acc.setdefault(c,[]).append(pos[n])
for c,v in acc.items(): clspos[c]=np.mean(v,0); have[c]=True
c2b=np.full(nCls,-1); c2b[have]=KMeans(8,n_init=5,random_state=0).fit(clspos[have]).labels_; nb=8
for c in np.where(~have)[0]: c2b[c]=nb; nb+=1
blocks=[np.where(c2b==b)[0] for b in range(nb)]

def simvec(Bmod, perm=None):
    """barcode similarity per pair = # module genes co-expressed. Bmod: g x classes. perm: class permutation."""
    Bp=Bmod[:,perm] if perm is not None else Bmod
    return (Bp[:,ciV]*Bp[:,cjV]).sum(0)   # shared-count similarity
def obs_enr(s,U,tgt):
    m=U&(tgt>0); b=U&(tgt==0)
    if m.sum()<6 or b.sum()<6: return None
    return s[m].mean()-s[b].mean()
def screen(Bmod,U,tgt,reps_bc=250,reps_rg=250,perm_kind='barcode'):
    s=simvec(Bmod); o=obs_enr(s,U,tgt)
    if o is None: return None,None
    nb_=[]
    for _ in range(reps_bc):
        p=RNG.permutation(nCls) if perm_kind=='barcode' else None
        nb_.append(obs_enr(simvec(Bmod,p),U,tgt))
    nb_=[x for x in nb_ if x is not None]; zB=(o-np.mean(nb_))/(np.std(nb_)+1e-9) if len(nb_)>30 else None
    # region: within-block permutation of class labels
    nr=[]
    for _ in range(reps_rg):
        p=np.arange(nCls)
        for ix in blocks:
            if len(ix)>1: p[ix]=ix[RNG.permutation(len(ix))]
        nr.append(obs_enr(simvec(Bmod,p),U,tgt))
    nr=[x for x in nr if x is not None]; zR=(o-np.mean(nr))/(np.std(nr)+1e-9) if len(nr)>30 else None
    return zB,zR

# ---- modules ----
def genes_pref(prefs): return [i for i in range(len(gname)) if any(gname[i].startswith(p) for p in prefs)]
named={'zig(2-Ig)':genes_pref(['zig-']),'casy':genes_pref(['casy-']),'rig':genes_pref(['rig-']),
       'IgSF(oig/igcm/igdb/lron)':genes_pref(['oig-','igcm-','igdb-','lron-']),'cadherin':genes_pref(['cdh-']),
       'innexin':genes_pref(['inx-','unc-7','unc-9'])}
named={k:v for k,v in named.items() if len(v)>=4}
# data-derived co-expression modules from contrast genes
brd=Xb.mean(1); contrast=np.where((brd>=0.05)&(brd<=0.60))[0]
Z=Xt[contrast]; Z=np.log1p(Z); Z=(Z-Z.mean(1,keepdims=True))/(Z.std(1,keepdims=True)+1e-9)
klab=KMeans(100,n_init=3,random_state=0).fit(Z).labels_
dmods={}
for k in range(100):
    g=contrast[klab==k]
    if 8<=len(g)<=40: dmods[f"mod{k}"]=list(g)
P(f"modules: {len(named)} named + {len(dmods)} data-derived (size 8-40) = {len(named)+len(dmods)}")

# ---- CALIBRATION ----
P("\n[CALIBRATION Template B]")
# synthetic positive: classes with high FP-degree share a synthetic barcode -> similarity must predict FP
fpdeg=np.zeros(nCls)
for k in np.where(nonW&(FPa>0))[0]: fpdeg[Ci[k]]+=1; fpdeg[Cj[k]]+=1
hi=np.argsort(-fpdeg)[:20]; synB=np.zeros((5,nCls),np.float32); synB[:,hi]=1
zB,zR=screen(synB,nonW,FPa.astype(float),reps_bc=200,reps_rg=100)
P(f"  synthetic (FP-hub shared barcode): barcode-z={zB:+.1f} region-z={zR:+.1f} (want>3)")
rz=[]
for s in range(8):
    g=np.random.default_rng(s).choice(len(gname),15,replace=False); rz.append(screen(Xb[g],nonW,FPa.astype(float),reps_bc=150,reps_rg=1)[0])
P(f"  random 15-gene modules -> FP: max|barcode-z|={np.nanmax(np.abs(rz)):.1f} (want<3)")

# ---- SCREEN (barcode null first; region + Cook for survivors) ----
P("\n"+"="*90); P("TEMPLATE B SCREEN — barcode-similarity -> FP (pre-registered self-avoidance direction)"); P("="*90)
allmods={**named,**dmods}; cand=[]
for nmk,g in allmods.items():
    B=Xb[g]; s=simvec(B); o=obs_enr(s,nonW,FPa.astype(float))
    if o is None: continue
    nb_=[obs_enr(simvec(B,RNG.permutation(nCls)),nonW,FPa.astype(float)) for _ in range(150)]; nb_=[x for x in nb_ if x is not None]
    zB=(o-np.mean(nb_))/(np.std(nb_)+1e-9) if len(nb_)>30 else None
    if zB is not None and zB>2: cand.append((nmk,g,zB,o))
cand.sort(key=lambda x:-x[2])
P(f"modules with barcode-z>2 on FP: {len(cand)} (of {len(allmods)})")
locked=[]
for nmk,g,zB,o in cand[:15]:
    _,zR=screen(Xb[g],nonW,FPa.astype(float),reps_bc=1,reps_rg=300)
    # Cook replication
    sC=simvec(Xb[g]); oC=obs_enr(sC,nonC,FPc.astype(float))
    zCk=None
    if oC is not None:
        nck=[obs_enr(simvec(Xb[g],RNG.permutation(nCls)),nonC,FPc.astype(float)) for _ in range(200)]; nck=[x for x in nck if x is not None]
        zCk=(oC-np.mean(nck))/(np.std(nck)+1e-9) if len(nck)>30 else None
    rep = zR is not None and zR>2 and zCk is not None and zCk>2
    tag="REPLICATES(banked)" if rep else ("region-fails" if (zR is None or zR<=2) else "Cook-fails(per-animal)")
    P(f"   {nmk:26s} barcode-z={zB:+.1f} region-z={zR if zR is None else round(zR,1)} COOK-z={zCk if zCk is None else round(zCk,1)} genes={len(g)}  {tag}")
    if rep: locked.append(nmk)
P("\n"+"="*90)
P(f"TEMPLATE B RESULT: {len(locked)} module(s) survive barcode-null + region-null + COOK replication -> {locked if locked else 'NONE'}")
P("  => "+("FP HAS a combinatorial self-avoidance code (replicates)" if locked else
   "FP is NOT a replicable combinatorial self-avoidance code — consistent with per-animal noise (matching-template Notch-Delta/MIG failed Cook too)"))
P("="*90)
