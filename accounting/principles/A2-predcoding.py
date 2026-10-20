"""A2 — PREDICTIVE-CODING SIGNATURE in Randi functional connectivity.
Hypothesis: chemical synapses = fast feedforward prediction (down identity-rank);
neuropeptides = slow broadcast gain/precision.
EXPLORATORY: report the SHAPE of sign/magnitude patterns across chemical vs peptidergic layers.
"""
import warnings,sys,openpyxl; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr, mannwhitneyu
def P(*a): print(*a); sys.stdout.flush()
D="../data/celegans/data"

# ---------- identity-rank s (reuse branch10/25) ----------
cg=pd.read_csv(f"{D}/expression/cengen/thresholded/021821_liberal_threshold1.csv")
classes=list(cg.columns[3:]); Xf=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values.T>0
br=Xf.mean(0); Xc=Xf[:,(br>0.05)&(br<0.95)].astype(float); cls_idx={c:k for k,c in enumerate(classes)}
def n2c(n):
    if not isinstance(n,str): return None
    if n in cls_idx: return n
    b=n
    for s in "LR":
        if b.endswith(s): b=b[:-1]; break
    if b in cls_idx: return b
    for s in "DV":
        if b.endswith(s) and b[:-1] in cls_idx: return b[:-1]
    return None
wb=openpyxl.load_workbook(f"{D}/connectome/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
A=list(wb['hermaphrodite chemical'].iter_rows(values_only=True))
def looks(n): return isinstance(n,str) and len(n)>=2 and n[0].isalpha()
hr=max(range(len(A)),key=lambda r:sum(looks(x) for x in A[r])); cols=[(str(x).strip() if x is not None else None) for x in A[hr]]
hc=max(range(min(6,len(A[0]))),key=lambda c:sum(looks(A[r][c]) for r in range(len(A))))
chem=set()
for r in range(hr+1,len(A)):
    cpre=n2c(A[r][hc])
    if not cpre: continue
    for c in range(hc+1,len(A[r])):
        cpost=n2c(cols[c]); v=A[r][c]
        if cpost and cpost!=cpre and v not in (None,''):
            try:
                if float(v)>0: chem.add((cpre,cpost))
            except: pass
asym=[(a,b) for (a,b) in chem if (b,a) not in chem]
F=[];y=[]
for a,b in asym:
    d=Xc[cls_idx[a]]-Xc[cls_idx[b]]; F+=[d,-d]; y+=[1,0]
w=LogisticRegression(max_iter=1000,C=0.5).fit(np.array(F),np.array(y)).coef_[0]
s={c:float(Xc[cls_idx[c]]@w) for c in classes if c in cls_idx}
P(f"identity-rank s computed for {len(s)} classes; {len(asym)} asym chem pairs used to fit")

# ---------- Randi signed functional connectivity (individual-neuron level) ----------
# keep max-magnitude signed value per individual-neuron ordered pair
fc_neuron={}
with open(f"{D}/randi2023/randi_funconn_wildty.tsv") as f:
    for line in f:
        p=line.rstrip("\n").split('\t')
        if len(p)<3: continue
        i,j=p[0].strip(),p[1].strip()
        try: v=float(p[2])
        except: continue
        if i!=j and v!=0.0:
            if abs(v)>abs(fc_neuron.get((i,j),0)): fc_neuron[(i,j)]=v
P(f"\nRandi: {len(fc_neuron)} nonzero directed neuron-pairs among 188 neurons (coverage: 188 stimulated neurons)")

# map each edge to class-level s; keep only edges where both endpoints have an s and s differs
rows=[]
for (i,j),v in fc_neuron.items():
    ci,cj=n2c(i),n2c(j)
    if ci is None or cj is None: continue
    if ci not in s or cj not in s: continue
    if ci==cj: continue
    rows.append((i,j,ci,cj,s[ci],s[cj],v))
df=pd.DataFrame(rows,columns=["i","j","ci","cj","s_pre","s_post","fc"])
df["drank"]=df["s_pre"]-df["s_post"]  # >0 feedforward/down-rank
df["down"]=df["drank"]>0
df["mag"]=df["fc"].abs()
df["suppress"]=df["fc"]<0
P(f"Randi edges mapped to identity-rank: {len(df)} (endpoints with expression s, s_pre!=s_post)")

# =========================================================
P("\n================= (3) SIGN x IDENTITY-RANK STRATUM =================")
for lab,sub in [("DOWN-rank (feedforward s_pre>s_post)",df[df.down]),
                ("UP-rank   (feedback   s_pre<s_post)",df[~df.down])]:
    n=len(sub); fsup=sub.suppress.mean(); mmag=sub.mag.median(); mmean=sub.mag.mean()
    P(f"  {lab}: n={n:4d} | suppressive(neg) frac={fsup:.3f} activating frac={1-fsup:.3f} | |fc| median={mmag:.4f} mean={mmean:.4f}")
# overall base rate
P(f"  ALL edges: suppressive frac={df.suppress.mean():.3f} | |fc| median={df.mag.median():.4f}")
# test sign vs rank
from scipy.stats import chi2_contingency
ct=pd.crosstab(df.down,df.suppress)
chi2,pchi,_,_=chi2_contingency(ct)
P(f"  chi2 sign x down-rank: chi2={chi2:.2f} p={pchi:.3g}  (table rows=down T/F, cols=suppress T/F)\n  {ct.values.tolist()}")
# magnitude vs signed drank
rho,pr=spearmanr(df.drank,df.mag)
P(f"  Spearman(|fc|, drank magnitude signed) r={rho:+.3f} p={pr:.2g}")
rho2,pr2=spearmanr(df.drank.abs(),df.mag)
P(f"  Spearman(|fc|, |drank|) r={rho2:+.3f} p={pr2:.2g}  (does bigger rank-gap -> bigger influence?)")

# subset: edges that ALSO have an anatomical chemical synapse (class level)
df["haschem"]=[ (r.ci,r.cj) in chem for r in df.itertuples()]
dc=df[df.haschem]
P(f"\n  --- subset WITH anatomical chemical synapse (class-level): n={len(dc)} ---")
for lab,sub in [("DOWN",dc[dc.down]),("UP",dc[~dc.down])]:
    if len(sub)==0: continue
    P(f"    {lab}: n={len(sub):3d} suppress={sub.suppress.mean():.3f} |fc|med={sub.mag.median():.4f}")

# =========================================================
P("\n================= (4) PEPTIDERGIC LAYER (Ripoll long-range NPP) =================")
npp=pd.read_csv(f"{D}/ripollsanchez2023/01022024_neuropeptide_connectome_long_range_model.csv",index_col=0)
P(f"  NPP long-range matrix: {npp.shape}, values range {npp.values.min()}..{npp.values.max()}")
# build individual-neuron NPP edge dict
npp_edges={}
srcs=list(npp.index); tgts=list(npp.columns)
V=npp.values
for a in range(len(srcs)):
    for b in range(len(tgts)):
        v=V[a,b]
        if v>0 and srcs[a]!=tgts[b]:
            npp_edges[(srcs[a],tgts[b])]=float(v)
P(f"  NPP nonzero directed neuron-pairs: {len(npp_edges)}")
# intersect with Randi (individual-neuron level, same naming)
merge=[]
for (i,j),nw in npp_edges.items():
    if (i,j) in fc_neuron:
        merge.append((i,j,nw,fc_neuron[(i,j)]))
mg=pd.DataFrame(merge,columns=["i","j","npp","fc"])
mg["mag"]=mg["fc"].abs(); mg["suppress"]=mg["fc"]<0
P(f"  NPP edges also measured in Randi (both endpoints stimulated+recorded): {len(mg)}")
# Randi pairs with NO npp edge (but both endpoints in npp namespace + in randi)
npp_names=set(srcs)|set(tgts)
randi_in_npp=[(i,j,v) for (i,j),v in fc_neuron.items() if i in npp_names and j in npp_names]
rip=pd.DataFrame(randi_in_npp,columns=["i","j","fc"]); rip["mag"]=rip["fc"].abs(); rip["suppress"]=rip["fc"]<0
rip["hasnpp"]=[(r.i,r.j) in npp_edges for r in rip.itertuples()]
P(f"  Randi edges within NPP-namespace: {len(rip)}  ({rip.hasnpp.sum()} have NPP edge, {(~rip.hasnpp).sum()} none)")
P(f"  sign distribution on NPP-strong (has NPP) Randi edges: suppress frac={mg.suppress.mean():.3f} activating frac={1-mg.suppress.mean():.3f}")
P(f"  sign distribution on NO-NPP Randi edges (same namespace): suppress frac={rip[~rip.hasnpp].suppress.mean():.3f}")
P(f"  |fc| median: NPP-edge={mg.mag.median():.4f}  no-NPP={rip[~rip.hasnpp].mag.median():.4f}")
if len(mg)>5 and (~rip.hasnpp).sum()>5:
    U,pu=mannwhitneyu(mg.mag, rip[~rip.hasnpp].mag)
    P(f"  MannWhitney |fc| NPP-edge vs no-NPP: p={pu:.2g}")
# relationship NPP weight vs |functional influence|
if len(mg)>5:
    rho,pr=spearmanr(mg.npp,mg.mag)
    P(f"  Spearman(NPP weight, |fc|) over NPP-edge pairs: r={rho:+.3f} p={pr:.2g}, n={len(mg)}")
    rho_s,pr_s=spearmanr(mg.npp, mg.fc)  # signed
    P(f"  Spearman(NPP weight, signed fc): r={rho_s:+.3f} p={pr_s:.2g}")
    # binned
    mg["nppbin"]=pd.qcut(mg.npp,min(4,mg.npp.nunique()),duplicates="drop")
    P("  |fc| by NPP-weight quartile:")
    for b,gg in mg.groupby("nppbin",observed=True):
        P(f"    npp {str(b):20s} n={len(gg):3d} |fc|med={gg.mag.median():.4f} suppress={gg.suppress.mean():.3f}")

# cross-layer: do NPP-strong pairs align with identity-rank direction?
mg["ci"]=mg.i.map(n2c); mg["cj"]=mg.j.map(n2c)
mg2=mg[mg.ci.isin(s)&mg.cj.isin(s)&(mg.ci!=mg.cj)].copy()
mg2["down"]=mg2.ci.map(s)>mg2.cj.map(s)
if len(mg2)>5:
    P(f"\n  NPP-edge Randi pairs on identity-rank: n={len(mg2)} down-frac={mg2.down.mean():.3f}")
    P(f"    down-rank NPP pairs suppress={mg2[mg2.down].suppress.mean():.3f} | up-rank suppress={mg2[~mg2.down].suppress.mean():.3f}")

df.to_csv("../accounting/principles/A2-predcoding_chem_edges.csv",index=False)
mg.to_csv("../accounting/principles/A2-predcoding_npp_edges.csv",index=False)
P("\nDONE. wrote A2-predcoding_chem_edges.csv, A2-predcoding_npp_edges.csv")
