"""SLIT-ROBO directed lock (Part A) + repulsive-guidance-as-a-class (Part B). Pre-registered.
Part A: single hypothesis SLT-1xSAX-3 -> FN residual (missed synapses), dual null, DIRECTIONAL: FN not FP.
Part B: repulsive-guidance complementarity (Slit-Robo/Netrin-UNC5/Sema-Plexin/Ephrin-Eph) FN-enriched, and
        repulsive-FN-enrichment > attractive-FN-enrichment (Netrin-DCC/Syg/Nrx-Nlg/attractive CAMs).
"""
import warnings,sys,csv,re; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
def P(*a): print(*a); sys.stdout.flush()
RNG=np.random.default_rng(0)
SC="../work"
D="../data/celegans/data"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True)
I,J,y7,y8,neurons=z['I'],z['J'],z['y7'],z['y8'],list(z['neurons'])
N=len(I); yrep=((y7==1)&(y8==1)).astype(int); r1=np.load(f"{SC}/resid2.npz")['r1']; det=(y7==y8)
cg=pd.read_csv(D+"/expression/cengen/thresholded/021821_liberal_threshold1.csv")
classes=list(cg.columns[3:]); Xt=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values
gname=cg["gene_name"].astype(str).str.lower().values; cls_idx={c:k for k,c in enumerate(classes)}
def n2c(n):
    if n in cls_idx: return cls_idx[n]
    b=n
    for s in "LR":
        if b.endswith(s): b=b[:-1]; break
    if b in cls_idx: return cls_idx[b]
    for s in "DV":
        if b.endswith(s) and b[:-1] in cls_idx: return cls_idx[b[:-1]]
    return cls_idx.get(b,None)
def E(g):
    w=np.where(gname==g)[0]; return (Xt[w[0]]>0).astype(float) if len(w) else None
Ci=np.array([ (n2c(neurons[i]) if n2c(neurons[i]) is not None else -1) for i in I])
Cj=np.array([ (n2c(neurons[j]) if n2c(neurons[j]) is not None else -1) for j in J])
valid=(Ci>=0)&(Cj>=0); ci=np.where(valid,Ci,0); cj=np.where(valid,Cj,0)
def comp(lig,rec):  # complementary presentation, either orientation
    return (((lig[ci]*rec[cj])+(rec[ci]*lig[cj]))>0)&valid

syn=(yrep==1)&det&valid; fn=syn&(r1>0.3)
nsyn=(yrep==0)&det&valid; fp=nsyn&(r1<-0.3)
P(f"synapse universe {syn.sum()} (FN {fn.sum()}) | non-synapse universe {nsyn.sum()} (FP {fp.sum()})")

# ---- constrained degree-preserving null within a host universe ----
key=lambda k:(min(int(I[k]),int(J[k])),max(int(I[k]),int(J[k])))
def make_null(target_mask, host_mask, K=200):
    host=set(key(k) for k in np.where(host_mask)[0]); kof={key(k):k for k in np.where(host_mask)[0]}
    tgt=[key(k) for k in np.where(target_mask)[0]]; outs=[]; nsw=10*len(tgt)
    for _ in range(K):
        edges=list(tgt); es=set(edges); c=t=0
        while c<nsw and t<nsw*25:
            t+=1; i,j=int(RNG.integers(len(edges))),int(RNG.integers(len(edges)))
            if i==j: continue
            a,b=edges[i]; cc,d=edges[j]
            if len({a,b,cc,d})<4: continue
            e1=(min(a,d),max(a,d)); e2=(min(cc,b),max(cc,b))
            if e1 in host and e2 in host and e1 not in es and e2 not in es:
                es.discard((a,b)); es.discard((cc,d)); es.add(e1); es.add(e2); edges[i]=e1; edges[j]=e2; c+=1
        m=np.zeros(N,bool)
        for e in es:
            kk=kof.get(e)
            if kk is not None: m[kk]=True
        outs.append(m)
    return outs
FNN=make_null(fn,syn,K=200); FPN=make_null(fp,nsyn,K=200)
def zc(ind,tgt,U,NL):
    m=ind&U
    if m.sum()<6 or (U&~ind).sum()<6: return None,0,int(m.sum())
    obs=tgt[m].mean()-tgt[U&~ind].mean()
    ns=np.array([ (nm[m].mean()-nm[U&~ind].mean()) for nm in NL])
    return (obs-ns.mean())/(ns.std()+1e-9),obs,int(m.sum())
def z_exprperm(lig,rec,tgt,U,reps=3000):
    ind=comp(lig,rec); m=ind&U
    if m.sum()<6: return None
    obs=tgt[m].mean()-tgt[U&~ind].mean(); nul=[]
    for _ in range(reps):
        l2=lig[RNG.permutation(len(lig))]; r2=rec[RNG.permutation(len(rec))]; i2=comp(l2,r2)&U
        if i2.sum()>=6 and (U&~i2).sum()>=6: nul.append(tgt[i2].mean()-tgt[U&~i2].mean())
    nul=np.array(nul); return (obs-nul.mean())/(nul.std()+1e-9) if len(nul)>30 else None

# ---- calibration ----
P("\n[CALIBRATION on FN]")
fi=np.where(fn)[0]; ni=np.where(syn&~fn)[0]
sp=np.zeros(N,bool); sp[RNG.choice(fi,40,replace=False)]=True; sp[RNG.choice(ni,10,replace=False)]=True
P(f"  synthetic pos (80% FN): z={zc(sp,fn,syn,FNN)[0]:+.1f} (want >3)")
negz=[zc((lambda r:(np.zeros(N,bool)|False))(0) ,fn,syn,FNN)[0] for _ in range(0)]
nz=[]
for s in range(8):
    rr=np.zeros(N,bool); rr[np.random.default_rng(s).choice(np.where(syn)[0],50,replace=False)]=True
    v=zc(rr,fn,syn,FNN)[0]
    if v is not None: nz.append(v)
P(f"  random neg (n=50,8 draws): max|z|={np.abs(nz).max():.1f} (want <3)")
# innexin -> GJ positive control on this pair subset
GJ=np.zeros(N); gj=set()
with open(f"{D}/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']!='electrical': continue
        m=re.search(r'witvliet_2020_(\d)',r['source_file'])
        if m and int(m.group(1)) in (7,8) and r['pre'] in neurons and r['post'] in neurons:
            a,b=neurons.index(r['pre']),neurons.index(r['post']); gj.add((min(a,b),max(a,b)))
for k in range(N):
    if (min(int(I[k]),int(J[k])),max(int(I[k]),int(J[k]))) in gj: GJ[k]=1
GJN=make_null(GJ.astype(bool)&det&valid, det&valid, K=100)
inxz=[]
for g in [f"inx-{i}" for i in range(1,23)]+["unc-7","unc-9","che-7"]:
    e=E(g)
    if e is None: continue
    v=zc((comp(e,e))&det, GJ, det&valid, GJN)[0]
    if v is not None: inxz.append(v)
P(f"  innexin->GJ positive control: {sum(1 for v in inxz if v>2)} of {len(inxz)} homotypic-innexin rules z>2 (machinery detects real code)")

# ================= PART A =================
P("\n"+"="*76); P("PART A — SLIT-ROBO single pre-registered hypothesis (FN not FP)"); P("="*76)
slt,sax=E("slt-1"),E("sax-3"); ind=comp(slt,sax)
zf_d,of,nf=zc(ind,fn,syn,FNN); zf_e=z_exprperm(slt,sax,fn,syn)
zp_d,op,npp=zc(ind,fp,nsyn,FPN); zp_e=z_exprperm(slt,sax,fp,nsyn)
P(f"  FN (missed synapses):     deg-z={zf_d:+.1f}  expr-perm-z={zf_e:+.1f}  enrich={of:+.3f} n={nf}")
P(f"  FP (spurious predictions): deg-z={zp_d:+.1f}  expr-perm-z={zp_e if zp_e is None else round(zp_e,1)}  enrich={op:+.3f} n={npp}")
lockA = zf_d>2 and (zf_e is not None and zf_e>2) and (zp_e is None or zp_e<2)
P(f"  => {'LOCKS: survives both nulls on FN, correct direction (FN>>FP) = first BANKABLE molecular signal in the residual' if lockA else 'does not lock as directed — treat residual as molecularly dark'}")

# ================= PART B =================
P("\n"+"="*76); P("PART B — repulsive guidance as a CLASS vs attractive (pre-registered contrast)"); P("="*76)
REP=[("slt-1","sax-3"),("unc-6","unc-5"),
     ("mab-20","plx-1"),("mab-20","plx-2"),("smp-1","plx-1"),("smp-2","plx-1"),("smp-1","plx-2"),("smp-2","plx-2"),
     ("efn-1","vab-1"),("efn-2","vab-1"),("efn-3","vab-1"),("efn-4","vab-1")]
ATT=[("unc-6","unc-40"),("syg-2","syg-1"),("nrx-1","nlg-1"),
     ("sax-7","sax-7"),("hmr-1","hmr-1"),("casy-1","casy-1"),("rig-6","rig-6")]
def build(pairs):
    agg=np.zeros(N,bool); rows=[]
    for l,r in pairs:
        el,er=E(l),E(r)
        if el is None or er is None: rows.append((f"{l}~{r}",None)); continue
        ind=comp(el,er); agg|=ind
        zd=zc(ind,fn,syn,FNN)[0]; ze=z_exprperm(el,er,fn,syn)
        rows.append((f"{l}~{r}",(zd,ze,int((ind&syn).sum()))))
    return agg,rows
repagg,reprows=build(REP); attagg,attrows=build(ATT)
P("repulsive pairs (FN, dual null):")
for nm,v in reprows:
    if v and v[0] is not None: P(f"   {nm:16s} deg-z={v[0]:+.1f} expr-perm-z={'n/a' if v[1] is None else round(v[1],1)} n={v[2]}")
P("attractive pairs (FN, dual null):")
for nm,v in attrows:
    if v and v[0] is not None: P(f"   {nm:16s} deg-z={v[0]:+.1f} expr-perm-z={'n/a' if v[1] is None else round(v[1],1)} n={v[2]}")
zr,orr,nr=zc(repagg,fn,syn,FNN); za,oaa,na=zc(attagg,fn,syn,FNN)
P(f"\nAGGREGATE repulsive-complementarity  -> FN: deg-z={zr:+.1f} enrich={orr:+.3f} (marks {nr} syn)")
P(f"AGGREGATE attractive-complementarity -> FN: deg-z={za:+.1f} enrich={oaa:+.3f} (marks {na} syn)")
P(f"clean built-in discriminator: UNC-6xUNC-5 (repulsive) vs UNC-6xUNC-40 (attractive) above.")
P("\n"+"="*76)
P("PRE-REGISTERED CONTRAST (repulsive-FN > attractive-FN):",
  "HOLDS — process is blind specifically to repulsion-defying synapses" if (orr>oaa and zr>2 and za<zr)
  else "does NOT hold cleanly — either generic guidance (both enriched) or repulsive-only Slit-Robo (class doesn't generalize)")
P("="*76)
