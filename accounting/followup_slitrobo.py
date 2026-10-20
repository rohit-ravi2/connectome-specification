"""Stress-test the SLT-1~SAX-3 residual signal. Two questions:
(1) DIRECTIONALITY: is the enrichment carried by the COMPLEMENTARY (Slit on one cell / Robo on the other =
    signaling geometry) configuration, vs mere co-expression?
(2) GEOMETRY-LEAKAGE (decisive): Slit-Robo is a GUIDANCE molecule; the residual is post-MODELED-contact, not
    post-ACTUAL-contact. Does the FN enrichment survive conditioning on real graded CONTACT STRENGTH? If it
    dissolves within contact-strength strata -> the signal was underfit contact geometry, not synapse choice.
"""
import warnings,sys,openpyxl; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from collections import defaultdict
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
def by_name(g):
    w=np.where(gname==g)[0]; return (Xt[w[0]]>0).astype(float) if len(w) else None
Ci=np.array([ (n2c(neurons[i]) if n2c(neurons[i]) is not None else -1) for i in I])
Cj=np.array([ (n2c(neurons[j]) if n2c(neurons[j]) is not None else -1) for j in J])
valid=(Ci>=0)&(Cj>=0)
syn=(yrep==1)&det&valid; fn=syn&(r1>0.3)
slt=by_name("slt-1"); sax=by_name("sax-3")
ci=np.where(valid,Ci,0); cj=np.where(valid,Cj,0)
comp=(((slt[ci]*sax[cj])+(sax[ci]*slt[cj]))>0)&valid   # complementary: ligand one side, receptor other
both=((slt[ci]*sax[cj]*sax[ci]*slt[cj])>0)&valid        # both cells express BOTH (co-expression)
only_comp=comp&~both
def enrich_z(ind,reps=3000):
    U=syn; m=ind&U
    if m.sum()<6 or (U&~ind).sum()<6: return None,0,int(m.sum())
    obs=fn[m].mean()-fn[U&~ind].mean()
    # expression-permutation null (permute both genes' class labels)
    nul=[]
    for _ in range(reps):
        s2=slt[RNG.permutation(len(slt))]; x2=sax[RNG.permutation(len(sax))]
        i2=((( s2[ci]*x2[cj])+(x2[ci]*s2[cj]))>0)&valid
        if (i2&U).sum()>=6 and (U&~i2).sum()>=6: nul.append(fn[i2&U].mean()-fn[U&~i2].mean())
    nul=np.array(nul); return (obs-nul.mean())/(nul.std()+1e-9),obs,int(m.sum())

P("="*74); P("(1) DIRECTIONALITY — complementary (signaling geometry) vs co-expression"); P("="*74)
for lab,ind in [("complementary (Slit|Robo split)",comp),("co-express both",both),("complementary-ONLY (excl co-expr)",only_comp)]:
    zz,obs,n=enrich_z(ind); zs=f"{zz:+.1f}" if zz is not None else "n/a(<6)"
    P(f"   {lab:34s} n={n:4d} FN-enrich={obs:+.3f} expr-perm-z={zs}")

# ---- graded contact strength (ACTUAL geometry, not model) ----
wb0=openpyxl.load_workbook(D+"/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx",read_only=True,data_only=True)
ws=wb0['adult nerve ring neighbors']; data=list(ws.iter_rows(values_only=True)); hdr=[h for h in data[0][1:]]
carea=defaultdict(float)
for r in data[1:]:
    rn=r[0]
    if rn is None: continue
    for jj,val in enumerate(r[1:]):
        cn=hdr[jj]
        if cn and val not in (None,'') and rn!=cn:
            try: carea[(min(rn,cn),max(rn,cn))]=max(carea[(min(rn,cn),max(rn,cn))],float(val))
            except: pass
area=np.array([carea.get((min(neurons[I[k]],neurons[J[k]]),max(neurons[I[k]],neurons[J[k]])),0.0) for k in range(N)])
P("\n"+"="*74); P("(2) GEOMETRY-LEAKAGE (decisive) — does Slit-Robo FN-enrichment survive real contact strength?"); P("="*74)
sy=np.where(syn)[0]
P(f"   contact strength (area) among synapses: Slit-Robo-complementary mean={area[comp&syn].mean():.1f} vs other-synapse mean={area[syn&~comp].mean():.1f}")
# stratify synapses into contact-strength terciles; test enrichment within each
q=np.quantile(area[syn],[1/3,2/3]);
P(f"   FN-rate & Slit-Robo enrichment WITHIN contact-strength terciles (contact held ~fixed):")
for lo,hi,lab in [(-1,q[0],"low-contact"),(q[0],q[1],"mid-contact"),(q[1],1e9,"high-contact")]:
    strat=syn&(area>lo)&(area<=hi)
    m=strat&comp; b=strat&~comp
    if m.sum()>=4 and b.sum()>=4:
        e=fn[m].mean()-fn[b].mean()
        P(f"     {lab:12s} (n_syn={strat.sum():3d}): FN-enrich(Slit-Robo)={e:+.3f}  [comp n={m.sum()}, comp-FN-rate={fn[m].mean():.2f} vs {fn[b].mean():.2f}]")
# residualize FN on contact area (logistic), re-test enrichment on the contact-residual
from sklearn.linear_model import LogisticRegression
Us=syn
la=np.log1p(area[Us]).reshape(-1,1); yf=fn[Us].astype(int)
pc=LogisticRegression(max_iter=500).fit(la,yf).predict_proba(la)[:,1]
fn_res=np.zeros(N); fn_res[Us]=yf-pc
cm=comp&Us; cb=Us&~comp
obs_c=fn_res[cm].mean()-fn_res[cb].mean()
nul=[]
for _ in range(3000):
    s2=slt[RNG.permutation(len(slt))]; x2=sax[RNG.permutation(len(sax))]
    i2=((( s2[ci]*x2[cj])+(x2[ci]*s2[cj]))>0)&valid&Us
    if i2.sum()>=6 and (Us&~i2).sum()>=6: nul.append(fn_res[i2].mean()-fn_res[Us&~i2].mean())
nul=np.array(nul); zc=(obs_c-nul.mean())/(nul.std()+1e-9)
P(f"\n   FN residualized on ACTUAL contact area -> Slit-Robo enrich={obs_c:+.3f} expr-perm-z={zc:+.1f}")
P("\n"+"="*74)
P("READING:", "SURVIVES contact-strength control -> synapse-choice beyond geometry (finding earned)" if zc>2
  else "DISSOLVES under real contact strength -> was underfit contact geometry (deflationary reading (2))")
P("="*74)
