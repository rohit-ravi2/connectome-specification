"""STAGE 1 — data-derived candidate screen over the annotated interactome (Nawrocka; NPP-GPCR not on disk).
Pre-filter: criterion 3 (contrast: gene expressed 5-60% of classes) + criterion 2 (complementary: ligand/
receptor cross-class corr low). Screen survivors x 4 structures {FN, FP(fp_absent), det_core, GJ}, complementary
config, readout = structure-of-max-signal. Columns: expr-perm-z + REGION-null-z (within-spatial-block permutation
preserving block expression+residual density, breaking only pair<->residual link). Calibrated. Nothing banked here.
"""
import warnings,sys,csv,re,json; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
def P(*a): print(*a); sys.stdout.flush()
RNG=np.random.default_rng(0)
SC="../work"
D="../data/celegans/data"; SUP="../data/adhesion/nawrocka_supp"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); m0=np.load(f"{SC}/mode0.npz")
I,J,y7,y8,neurons=z['I'],z['J'],z['y7'],z['y8'],list(z['neurons'])
N=len(I); yrep=((y7==1)&(y8==1)).astype(int); agree=m0['agree']
FN=m0['FN']; FPa=m0['fp_absent']; DET=m0['det_core']
cg=pd.read_csv(D+"/expression/cengen/thresholded/021821_liberal_threshold1.csv")
classes=list(cg.columns[3:]); Xt=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values
gname=cg["gene_name"].astype(str).str.lower().values; wbid=cg["Wormbase_ID"].astype(str).values
cls_idx={c:k for k,c in enumerate(classes)}; nCls=len(classes)
def n2c(n):
    if n in cls_idx: return cls_idx[n]
    b=n
    for s in "LR":
        if b.endswith(s): b=b[:-1]; break
    if b in cls_idx: return cls_idx[b]
    for s in "DV":
        if b.endswith(s) and b[:-1] in cls_idx: return cls_idx[b[:-1]]
    return cls_idx.get(b,None)
def by_wb(w):
    i=np.where(wbid==w)[0]; return (Xt[i[0]]>0).astype(float) if len(i) else None
def by_name(g):
    w=np.where(gname==g)[0]; return (Xt[w[0]]>0).astype(float) if len(w) else None
Ci=np.array([ (n2c(neurons[i]) if n2c(neurons[i]) is not None else -1) for i in I])
Cj=np.array([ (n2c(neurons[j]) if n2c(neurons[j]) is not None else -1) for j in J])
valid=(Ci>=0)&(Cj>=0); ci=np.where(valid,Ci,0); cj=np.where(valid,Cj,0)
def comp(l,r): return (((l[ci]*r[cj])+(r[ci]*l[cj]))>0)&valid

# ---- residual structures (contact-controlled universes) ----
syn=(yrep==1)&agree&valid; nons=(yrep==0)&agree&valid; detU=agree&valid
GJ=np.zeros(N); gj=set()
with open(f"{D}/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']=='electrical':
            mm=re.search(r'witvliet_2020_(\d)',r['source_file'])
            if mm and int(mm.group(1)) in (7,8) and r['pre'] in neurons and r['post'] in neurons:
                a,b=neurons.index(r['pre']),neurons.index(r['post']); gj.add((min(a,b),max(a,b)))
for k in range(N):
    if (min(int(I[k]),int(J[k])),max(int(I[k]),int(J[k]))) in gj: GJ[k]=1
STRUCT={'FN':(syn,FN.astype(float)),'FP':(nons,FPa.astype(float)),'det_core':(detU,DET.astype(float)),'GJ':(detU,GJ)}

# ---- region blocks: spatial k-means on class soma positions ----
pj=json.load(open("../data/network_positions.json"))
pos={c["cell"]:np.array([c["x"],c["y"],c["z"]],float) for c in pj["cells"] if None not in (c.get("x"),c.get("y"),c.get("z"))}
clspos=np.zeros((nCls,3)); have=np.zeros(nCls,bool)
acc={}
for n in neurons:
    c=n2c(n)
    if c is not None and n in pos: acc.setdefault(c,[]).append(pos[n])
for c,v in acc.items(): clspos[c]=np.mean(v,0); have[c]=True
km=KMeans(8,n_init=5,random_state=0).fit(clspos[have])
class2block=np.full(nCls,-1); class2block[have]=km.labels_
# classes without position -> own singleton blocks
nb=8
for c in np.where(~have)[0]: class2block[c]=nb; nb+=1
blocks=[np.where(class2block==b)[0] for b in range(nb)]
def region_perm(e):
    out=e.copy()
    for idx in blocks:
        if len(idx)>1: out[idx]=e[idx][RNG.permutation(len(idx))]
    return out
def global_perm(e): return e[RNG.permutation(nCls)]

def enrich(ind,U,tgt):
    m=ind&U
    if m.sum()<6 or (U&~ind).sum()<6: return None,0,0
    return tgt[m].mean()-tgt[U&~ind].mean(),int(m.sum()),1
def zboth(l,r,U,tgt,reps_e=400,reps_r=200):
    ind=comp(l,r); obs,n,ok=enrich(ind,U,tgt)
    if not ok: return None,None,0,0
    nE=[]
    for _ in range(reps_e):
        e=enrich(comp(global_perm(l),global_perm(r)),U,tgt)
        if e[2]: nE.append(e[0])
    nR=[]
    for _ in range(reps_r):
        e=enrich(comp(region_perm(l),region_perm(r)),U,tgt)
        if e[2]: nR.append(e[0])
    zE=(obs-np.mean(nE))/(np.std(nE)+1e-9) if len(nE)>30 else None
    zR=(obs-np.mean(nR))/(np.std(nR)+1e-9) if len(nR)>30 else None
    return zE,zR,n,obs

# ---- annotated pairs (Nawrocka) + expression ----
s1=pd.read_excel(f"{SUP}/media-1.xlsx",sheet_name="Table S1",header=1); prot2wb={};seq2wb={};nm={}
for _,r in s1.iterrows():
    w=str(r.get("WormBase Gene ID","")).strip()
    if w.startswith("WBGene"):
        pn=str(r.get("Protein name","")).strip().upper(); sq=str(r.get("Sequence ID","")).strip()
        if pn and pn!="NAN": prot2wb[pn]=w; nm[w]=pn
        if sq and sq!="nan": seq2wb[sq]=w
s3=pd.read_excel(f"{SUP}/media-3.xlsx",sheet_name="Table S3",header=1); bind=set()
for _,r in s3.iterrows():
    wa=prot2wb.get(str(r["Protein Name"]).strip().upper(),seq2wb.get(str(r["Sequence Name"]).strip()))
    wb=prot2wb.get(str(r["Protein Name.1"]).strip().upper(),seq2wb.get(str(r["Sequence Name.1"]).strip()))
    if wa and wb and wa!=wb: bind.add(frozenset((wa,wb)))
P(f"annotated interactome pairs (Nawrocka): {len(bind)} | NPP-GPCR: NOT on disk (coverage = adhesion/guidance only)")

# ---- PRE-FILTER ----
exprW={w:by_wb(w) for fs in bind for w in fs}
def breadth(w): e=exprW.get(w); return None if e is None else e.mean()
def corr(a,b):
    ea,eb=exprW[a],exprW[b]
    if ea is None or eb is None: return None
    if ea.std()==0 or eb.std()==0: return 1.0
    return float(np.corrcoef(ea,eb)[0,1])
both_expr=[fs for fs in bind if all(exprW.get(w) is not None for w in fs)]
# criterion 3 fixed for HETEROPHILIC pairs: the MORE-RESTRICTED partner must be contrast-bearing (5-60%);
# a broad receptor field is fine (it's the target field). Drops pan-neuronal-BOTH (e.g. sax-7~sax-7); keeps
# restricted-ligand/broad-receptor (Slit-Robo). Also neither partner near-silent.
contrast=[fs for fs in both_expr if 0.05<=min(breadth(w) for w in fs)<=0.60 and min(breadth(w) for w in fs)>=0.03]
shortlist=[fs for fs in contrast if corr(*tuple(fs))<0.20]
P(f"FUNNEL: {len(bind)} annotated -> {len(both_expr)} both-expressed -> {len(contrast)} contrast-bearing(min-partner 5-60%) -> {len(shortlist)} complementary(corr<0.20)")
slt,sax=by_name("slt-1"),by_name("sax-3")
P(f"  (sanity: Slit-Robo breadth slt-1={slt.mean():.2f} sax-3={sax.mean():.2f} corr={np.corrcoef(slt,sax)[0,1]:+.2f} -> in shortlist: {frozenset((prot2wb.get('SLT-1'),prot2wb.get('SAX-3'))) in shortlist})")

# ---- CALIBRATION ----
P("\n[CALIBRATION]")
fi=np.where(FN&syn)[0]; ni=np.where(syn&~FN)[0]
sp=np.zeros(nCls); # synthetic: a fake 'ligand'/'receptor' whose complementary set aligns to FN is hard; use direct pair-mask synthetic instead
# direct synthetic on FN: mark 80% FN via a pair-indicator, check enrich detects
synth=np.zeros(N,bool); synth[RNG.choice(fi,40,replace=False)]=True; synth[RNG.choice(ni,10,replace=False)]=True
o,n,_=enrich(synth,syn,FN.astype(float));
nE=[]
for _ in range(400):
    rr=np.zeros(N,bool); rr[RNG.choice(np.where(syn)[0],50,replace=False)]=True
    nE.append(enrich(rr,syn,FN.astype(float))[0])
P(f"  synthetic 80%-FN pair-rule: enrich={o:+.3f} z_vs_random={ (o-np.mean(nE))/np.std(nE):+.1f} (want>3)")
negmax=max(abs((enrich((lambda s:(np.zeros(N,bool)))(0),syn,FN.astype(float))[0] or 0)) for _ in [0])
rz=[]
for s in range(8):
    rr=np.zeros(N,bool); rr[np.random.default_rng(s).choice(np.where(syn)[0],50,replace=False)]=True
    rz.append((enrich(rr,syn,FN.astype(float))[0]-np.mean(nE))/np.std(nE))
P(f"  random pair-rules max|z|={np.abs(rz).max():.1f} (want<3)")
# innexin -> GJ (known positive on its structure)
inx=[g for g in ([f"inx-{i}" for i in range(1,23)]+["unc-7","unc-9","che-7"]) if by_name(g) is not None]
ig=sum(1 for g in inx if (lambda zz: zz is not None and zz>2)(zboth(by_name(g),by_name(g),detU,GJ,reps_e=200,reps_r=100)[0]))
P(f"  innexin->GJ positive control: {ig}/{len(inx)} homotypic rules expr-perm-z>2")

# ---- SCREEN ----
P("\n"+"="*94); P("STAGE 1 SCREEN — shortlisted complementary pairs x 4 structures (expr-perm-z / region-z)"); P("="*94)
rows=[]
for fs in shortlist:
    a,b=tuple(fs); la,lb=exprW[a],exprW[b]; name=f"{nm.get(a,a)}~{nm.get(b,b)}"
    best=(-9,None)
    rec={}
    for sname,(U,tgt) in STRUCT.items():
        zE,zR,n,ob=zboth(la,lb,U,tgt)
        rec[sname]=(zE,zR,n,ob)
        if zE is not None and zE>best[0]: best=(zE,sname)
    rows.append((name,best[1],best[0],rec))
rows.sort(key=lambda x:-(x[2] if x[2] is not None else -9))
P(f"{'pair':20s} {'max@':9s} {'FN(zE/zR)':14s} {'FP(zE/zR)':14s} {'detcore(zE/zR)':16s} {'GJ(zE/zR)':13s}")
def fmt(t):
    zE,zR,n,ob=t;
    return f"{('%.1f'%zE) if zE is not None else '-':>4}/{('%.1f'%zR) if zR is not None else '-':>4}"
for name,bst,bz,rec in rows:
    P(f"{name:20s} {str(bst):9s} {fmt(rec['FN']):14s} {fmt(rec['FP']):14s} {fmt(rec['det_core']):16s} {fmt(rec['GJ']):13s}")
# survivors of BOTH nulls (>2) on any structure
surv=[(name,s,rec[s]) for name,bst,bz,rec in rows for s in STRUCT
      if rec[s][0] is not None and rec[s][1] is not None and rec[s][0]>2 and rec[s][1]>2]
P(f"\nSURVIVE BOTH NULLS (expr-perm>2 AND region>2) on some structure: {len(surv)}")
for name,s,t in surv: P(f"   {name:20s} @ {s:9s} expr-perm-z={t[0]:+.1f} region-z={t[1]:+.1f} n={t[2]} enrich={t[3]:+.3f}  -> STAGE-2 CANDIDATE")
P("="*94)
