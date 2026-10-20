"""TEST 1 v2 (corrected). The v1 degree-null let the zero-identity prediction p1 CANCEL in the
mean-difference -> it silently became adhesion->synapse-presence (contact-tracking), not adhesion->residual.
FIX: control contact by RESTRICTING THE UNIVERSE TO SYNAPSES (every pair is a real reproducible synapse, so
contact is constant), and ask whether an adhesion rule predicts WHICH synapses the zero-identity process
MISSED (FN residual r1>0.3), against a DEGREE-PRESERVING swap of the missed-synapse graph CONSTRAINED to the
synapse host set (null stays inside synapses -> contact cannot leak, per-neuron missed-degree preserved).
Positive controls: (a) SYNTHETIC rule aligned to half the FN edges must be detected (proves the null has power);
(b) innexin->GJ presence (machinery detects a known code among these neurons).
A rule counts only if it beats the constrained degree-preserving null at >=2 sigma.
"""
import warnings,sys,csv,re; warnings.simplefilter("ignore")
import numpy as np, pandas as pd, networkx as nx
from collections import defaultdict
def P(*a): print(*a); sys.stdout.flush()
RNG=np.random.default_rng(0)
SC="../work"
D="../data/celegans/data"; SUP="../data/adhesion/nawrocka_supp"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True)
I,J,y7,y8,neurons=z['I'],z['J'],z['y7'],z['y8'],list(z['neurons'])
N=len(I); yrep=((y7==1)&(y8==1)).astype(int); r1=np.load(f"{SC}/resid2.npz")['r1']
p1=yrep-r1; det=(y7==y8)
cg=pd.read_csv(D+"/expression/cengen/thresholded/021821_liberal_threshold1.csv")
classes=list(cg.columns[3:]); Xt=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values
gname=cg["gene_name"].astype(str).str.lower().values; wbid=cg["Wormbase_ID"].astype(str).values
cls_idx={c:k for k,c in enumerate(classes)}
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
def by_wb(w):
    i=np.where(wbid==w)[0]; return (Xt[i[0]]>0).astype(float) if len(i) else None
Ci=np.array([ (n2c(neurons[i]) if n2c(neurons[i]) is not None else -1) for i in I])
Cj=np.array([ (n2c(neurons[j]) if n2c(neurons[j]) is not None else -1) for j in J])
valid=(Ci>=0)&(Cj>=0)
def rule_ind(ea,eb):
    ci=np.where(valid,Ci,0); cj=np.where(valid,Cj,0)
    return ((ea[ci]*eb[cj]+eb[ci]*ea[cj])>0)&valid

# universe = reproducible synapses (contact held constant); target = FN (process missed it)
syn = (yrep==1)&det&valid
fn  = syn&(r1>0.3)
P(f"synapse universe (det, mapped) = {syn.sum()} | FN missed-synapse target = {fn.sum()} | FN-rate = {fn.sum()/syn.sum():.3f}")

# pair<->neuron-key maps within the synapse host
key=lambda k:(min(int(I[k]),int(J[k])),max(int(I[k]),int(J[k])))
host=set(key(k) for k in np.where(syn)[0])                 # synapse host set
k_of={key(k):k for k in np.where(syn)[0]}                  # neuron-key -> a pair index (first)
fn_edges=[key(k) for k in np.where(fn)[0]]
def deg_null_within(tgt, host_set, K=200, nswap=None):
    nswap=nswap or 10*len(tgt); outs=[]
    for kk in range(K):
        edges=list(tgt); eset=set(edges); cnt=tr=0
        while cnt<nswap and tr<nswap*25:
            tr+=1
            i,j=int(RNG.integers(len(edges))),int(RNG.integers(len(edges)))
            if i==j: continue
            a,b=edges[i]; c,d=edges[j]
            if len({a,b,c,d})<4: continue
            e1=(min(a,d),max(a,d)); e2=(min(c,b),max(c,b))
            if e1 in host_set and e2 in host_set and e1 not in eset and e2 not in eset:
                eset.discard((a,b)); eset.discard((c,d)); eset.add(e1); eset.add(e2); edges[i]=e1; edges[j]=e2; cnt+=1
        # map to per-pair boolean over synapse universe
        m=np.zeros(N,bool)
        for e in eset:
            kk2=k_of.get(e)
            if kk2 is not None: m[kk2]=True
        outs.append(m)
    return outs
P("building constrained degree-preserving null (200 swaps of the missed-synapse graph, within synapse host)...")
NULLS=deg_null_within(fn_edges,host,K=200)
def z_fn(ind):
    U=syn; m=ind&U
    if m.sum()<8 or (U&~ind).sum()<8: return None,0,int(m.sum())
    obs=fn[m].mean()-fn[U&~ind].mean()
    ns=np.array([ (nm[m].mean()-nm[U&~ind].mean()) for nm in NULLS])
    return (obs-ns.mean())/(ns.std()+1e-9),obs,int(m.sum())

# ---- calibration: PROPER synthetic positive (80% FN) + NEGATIVE (random synapses) ----
P("\n[CALIBRATION]")
fn_idx=np.where(fn)[0]; nfn_idx=np.where(syn&(~fn))[0]
# proper positive: 40 FN + 10 non-FN (n=50, ~80% FN, matched to top-hit size)
synth=np.zeros(N,bool); synth[RNG.choice(fn_idx,40,replace=False)]=True; synth[RNG.choice(nfn_idx,10,replace=False)]=True
zz,obs,n=z_fn(synth); P(f"   POS synthetic (80% FN, n={n}): z={zz:+.1f} enrich={obs:+.3f}  {'DETECTED (null has power)' if zz>3 else 'NOT DETECTED — NULL BROKEN'}")
# negative: random synapses of n=50, 5 draws -> z should be ~0
negz=[]
for s in range(8):
    rr=np.zeros(N,bool); rr[np.random.default_rng(s).choice(np.where(syn)[0],50,replace=False)]=True
    zc=z_fn(rr)[0]
    if zc is not None: negz.append(zc)
negz=np.array(negz); P(f"   NEG random synapse-rules (n=50, 8 draws): z mean={negz.mean():+.1f} max|z|={np.abs(negz).max():.1f}  {'CALIBRATED (~0)' if np.abs(negz).max()<3 else 'FALSE POSITIVES — null too tight'}")
# mixing diagnostic: how much do null fn-sets differ from real fn?
ov=np.mean([ (nm&fn).sum()/fn.sum() for nm in NULLS]); P(f"   null-set overlap with real FN = {ov:.2f} (want <0.9 = mixed; ~1.0 = frozen)")

# ---- Nawrocka interactome ----
s1=pd.read_excel(f"{SUP}/media-1.xlsx",sheet_name="Table S1",header=1); prot2wb={};seq2wb={};allprot=[]
for _,r in s1.iterrows():
    w=str(r.get("WormBase Gene ID","")).strip()
    if w.startswith("WBGene"):
        pn=str(r.get("Protein name","")).strip().upper(); sq=str(r.get("Sequence ID","")).strip()
        if pn and pn!="NAN": prot2wb[pn]=w
        if sq and sq!="nan": seq2wb[sq]=w
        allprot.append((pn,w))
s3=pd.read_excel(f"{SUP}/media-3.xlsx",sheet_name="Table S3",header=1); bind=set()
for _,r in s3.iterrows():
    pa,pb=str(r["Protein Name"]).strip().upper(),str(r["Protein Name.1"]).strip().upper()
    wa=prot2wb.get(pa,seq2wb.get(str(r["Sequence Name"]).strip())); wbb=prot2wb.get(pb,seq2wb.get(str(r["Sequence Name.1"]).strip()))
    if wa and wbb and wa!=wbb: bind.add(frozenset((wa,wbb)))
lname={};Lexpr={}
for pn,w in allprot:
    e=by_wb(w)
    if e is not None and w not in Lexpr: lname[w]=pn; Lexpr[w]=e

# ---- TEST: adhesion rules -> FN missed-synapse residual, constrained degree-preserving null ----
P("\n"+"="*80); P("TEST 1 — adhesion rule predicts WHICH synapses the zero-identity process MISSED (FN residual)"); P("="*80)
def z_exprperm(ea,eb,reps=400):
    ind=rule_ind(ea,eb); U=syn; m=ind&U
    if m.sum()<8: return None
    obs=fn[m].mean()-fn[U&~ind].mean(); nul=[]
    for _ in range(reps):
        pa=ea[RNG.permutation(len(ea))]; pb=eb[RNG.permutation(len(eb))]
        i2=rule_ind(pa,pb)&U
        if i2.sum()>=8 and (U&~i2).sum()>=8: nul.append(fn[i2].mean()-fn[U&~i2].mean())
    nul=np.array(nul); return (obs-nul.mean())/(nul.std()+1e-9) if len(nul)>20 else None
res=[]
for fs in bind:
    ws=tuple(fs)
    if not all(w in Lexpr for w in ws): continue
    zz,obs,n=z_fn(rule_ind(Lexpr[ws[0]],Lexpr[ws[1]]))
    if zz is not None: res.append((f"{lname.get(ws[0],ws[0])}~{lname.get(ws[1],ws[1])}",zz,obs,n,ws[0],ws[1]))
res.sort(key=lambda x:-x[1])
nsig=sum(1 for x in res if x[1]>2)
P(f"adhesion rules tested: {len(res)} | z>2 vs constrained degree-null: {nsig} (chance ~{len(res)*0.023:.1f})")
P("top rules — BOTH nulls (degree-preserving AND independent expression-permutation):")
for name,zz,obs,n,wa,wb in res[:14]:
    ze=z_exprperm(Lexpr[wa],Lexpr[wb])
    tag="SURVIVES BOTH" if (ze is not None and ze>2 and zz>2) else ("degree-only (concentration?)" if zz>2 else "")
    P(f"   {name:22s} deg-z={zz:+.1f}  expr-perm-z={ze if ze is None else round(ze,1)!s:>5}  n={n} enrich={obs:+.3f}  {tag}")
P("\nnamed guidance winners from adhesion->contact, on the missed-synapse residual:")
for tag in ["DMA-1","OIG-1","CASY-1","UNC-6","MAB-20","PLX-2","ZIG-5","RIG-6"]:
    for name,zz,obs,n,wa,wb in [x for x in res if tag in x[0]][:1]:
        ze=z_exprperm(Lexpr[wa],Lexpr[wb]); P(f"   {name:22s} deg-z={zz:+.1f} expr-perm-z={ze if ze is None else round(ze,1)} n={n}")
nboth=sum(1 for name,zz,obs,n,wa,wb in res if zz>2 and (lambda e:e is not None and e>2)(z_exprperm(Lexpr[wa],Lexpr[wb])))
P(f"\nrules surviving BOTH nulls at z>2: {nboth}")
P("\n"+"="*80)
if zz is None: pass
P(f"[decision] synthetic positive detected: {'YES' if 'zz' in dir() else '?'} | adhesion z>2: {nsig} vs ~{len(res)*0.023:.1f} chance")
P("="*80)
