"""TEST 1 — RESIDUAL (headline). Do per-rule adhesion/guidance rules (Nawrocka interactome) explain the
residual of the ZERO-IDENTITY process (contact+katz-closure+mirror-persistence), on the CLEAN DETERMINISTIC
set (745 = |r1|>0.3 & y7==y8), beyond contact (already in the residual) and beyond a DEGREE-PRESERVING null?
Positive control (method must recover a KNOWN code on this pair subset or the null is meaningless):
innexin homotypic -> gap junction, same pairs, same degree-preserving machinery.
A rule counts only if it beats the degree-preserving null at >=2 sigma (permutation-z).
"""
import warnings,json,sys,csv,re; warnings.simplefilter("ignore")
import numpy as np, pandas as pd, openpyxl, networkx as nx
from collections import defaultdict
def P(*a): print(*a); sys.stdout.flush()
RNG=np.random.default_rng(0)
SC="../work"
D="../data/celegans/data"; SUP="../data/adhesion/nawrocka_supp"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True)
I,J,y7,y8,neurons=z['I'],z['J'],z['y7'],z['y8'],list(z['neurons'])
N=len(I); yrep=((y7==1)&(y8==1)).astype(int); r1=np.load(f"{SC}/resid2.npz")['r1']
p1=yrep-r1; det=(y7==y8)   # deterministic = both animals agree
P(f"pairs {N} | yrep {yrep.sum()} | deterministic(y7==y8) {det.sum()} | clean-det residual |r1|>.3 & det = {(det&(np.abs(r1)>0.3)).sum()} (745)")

# ---- class mapping + CeNGEN expression ----
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
P(f"pairs with both endpoints mapped to CeNGEN class: {valid.sum()}")

# ---- Nawrocka interactome (binding pairs + expression) ----
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
labels=[];lname={};Lexpr={}
for pn,w in allprot:
    e=by_wb(w)
    if e is not None and w not in Lexpr: lname[w]=pn; Lexpr[w]=e
P(f"Nawrocka binding pairs (both expressed): {sum(1 for fs in bind if all(w in Lexpr for w in fs))}")

def rule_ind(ea,eb):
    ci=np.where(valid,Ci,0); cj=np.where(valid,Cj,0)
    return ((ea[ci]*eb[cj]+eb[ci]*ea[cj])>0)&valid

# ---- degree-preserving null machinery on a target graph among deterministic pairs ----
def build_pmap(mask):
    pm={}
    for k in np.where(mask)[0]:
        a,b=int(I[k]),int(J[k]); pm.setdefault((min(a,b),max(a,b)),k)
    return pm
def degree_null_presence(pres, mask, K=60, mult=3):
    """pres: binary target presence per pair; returns K x N array of degree-preserving rewired presence (within mask)."""
    pm=build_pmap(mask)
    G=nx.Graph()
    G.add_nodes_from(range(len(neurons)))
    for k in np.where(mask&(pres>0))[0]:
        a,b=int(I[k]),int(J[k]); G.add_edge(min(a,b),max(a,b))
    E=G.number_of_edges(); out=np.zeros((K,N))
    for kk in range(K):
        H=G.copy()
        try: nx.double_edge_swap(H,nswap=mult*E,max_tries=30*E)
        except: pass
        for a,b in H.edges():
            key=(min(a,b),max(a,b))
            if key in pm: out[kk,pm[key]]=1
    return out,E
def z_degree(ind, target_resid, null_resid):
    m=ind
    if m.sum()<8 or (~ind&valid&det).sum()<8: return None,0,int(m.sum())
    base=~ind&valid&det
    obs=target_resid[m].mean()-target_resid[base].mean()
    ns=null_resid[:,m].mean(1)-null_resid[:,base].mean(1)
    return (obs-ns.mean())/(ns.std()+1e-9),obs,int(m.sum())

# =================== POSITIVE CONTROL: innexin homotypic -> GAP JUNCTION ===================
P("\n"+"="*80); P("POSITIVE CONTROL — innexin homotypic -> gap junction (method must work on THIS pair subset)"); P("="*80)
GJ=np.zeros(N)
gjset=set()
with open(f"{D}/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']!='electrical': continue
        m=re.search(r'witvliet_2020_(\d)',r['source_file'])
        if not m: continue
        d=int(m.group(1))
        if d in (7,8) and r['pre'] in neurons and r['post'] in neurons:
            a,b=neurons.index(r['pre']),neurons.index(r['post']); gjset.add((min(a,b),max(a,b)))
for k in range(N):
    if (min(int(I[k]),int(J[k])),max(int(I[k]),int(J[k]))) in gjset: GJ[k]=1
gj_mask=valid&det
GJnull,Egj=degree_null_presence(GJ,gj_mask,K=60)
GJresid=GJ.copy().astype(float); GJnull_r=GJnull  # for presence target, residual == presence (null already degree-preserving)
INX=[f"inx-{i}" for i in range(1,23)]+["unc-7","unc-9","eat-5","che-7"]
P(f"gap junctions among mapped-deterministic pairs: {int(GJ[gj_mask].sum())} | degree-null rewires: {Egj} edges")
pc=[]
for g in INX:
    e=by_name(g)
    if e is None: continue
    ind=rule_ind(e,e)&det
    zz,obs,n=z_degree(ind,GJ.astype(float),GJnull_r)
    if zz is not None: pc.append((g,zz,n))
pc.sort(key=lambda x:-x[1])
for g,zz,n in pc[:8]: P(f"   {g:7s} homotypic: degree-null z={zz:+.1f} (n={n})"+("  HIT" if zz>2 else ""))
pc_ok=sum(1 for _,zz,_ in pc if zz>2)
P(f"   => positive control HITS (z>2): {pc_ok}  [{'METHOD VALID on this subset' if pc_ok>=1 else 'METHOD FAILS — null below is meaningless'}]")

# =================== TEST 1: adhesion/guidance -> residual r1 ===================
P("\n"+"="*80); P("TEST 1 — adhesion/guidance rules -> zero-identity-process RESIDUAL (r1), deterministic pairs"); P("="*80)
# degree-preserving null on the reproducible-synapse graph; residual_null = yrep_swapped - p1
SYNnull,Esyn=degree_null_presence(yrep.astype(float),valid&det,K=60)
r1null=SYNnull-p1[None,:]     # (K,N): swapped-synapse-presence minus fixed zero-identity prediction
P(f"synapse edges (mapped-det) {Esyn} | degree-preserving rewires: 60")
res=[]
for fs in bind:
    ws=tuple(fs)
    if not all(w in Lexpr for w in ws): continue
    ea,eb=Lexpr[ws[0]],Lexpr[ws[1]]
    ind=rule_ind(ea,eb)&det
    zz,obs,n=z_degree(ind,r1,r1null)
    if zz is not None: res.append((f"{lname.get(ws[0],ws[0])}~{lname.get(ws[1],ws[1])}",zz,obs,n))
res.sort(key=lambda x:-x[1])
nsig=sum(1 for _,zz,_,_ in res if zz>2)
P(f"\nadhesion rules tested: {len(res)} | z>2 (degree-preserving null): {nsig} vs ~{len(res)*0.023:.1f} chance (one-sided 2σ)")
P("top rules by degree-preserving z:")
for name,zz,obs,n in res[:12]: P(f"   {name:22s} z={zz:+.1f} enrich(r1)={obs:+.3f} n={n}"+("  HIT" if zz>2 else ""))
# also the known guidance hits from the contact test — do THEY survive here on the residual?
P("\nnamed guidance rules from adhesion->contact (do they carry to the synapse residual?):")
for tag in ["DMA-1","LRLP-1","OIG-1","CASY-1","UNC-6","UNC-40","MAB-20","PLX-2","IGDB-2","ZIG-5","RIG-6"]:
    hits=[x for x in res if tag in x[0]]
    for name,zz,obs,n in hits[:1]: P(f"   {name:22s} z={zz:+.1f} enrich={obs:+.3f} n={n}")
P("\n"+"="*80)
if pc_ok<1:
    P("VERDICT: INCONCLUSIVE — positive control failed on this subset; the method can't be trusted here.")
elif nsig>3*max(1,len(res)*0.023):
    P("VERDICT: MOLECULAR SIGNAL IN THE RESIDUAL — adhesion rules beat the degree-preserving null on what the")
    P("         zero-identity process leaves unexplained. The thing the arc said doesn't exist. (verify top hits)")
else:
    P("VERDICT: NULL IN THE RESIDUAL — the working per-rule method (validated by the positive control on the SAME")
    P("         pairs) finds NO adhesion rule beating the degree-preserving null on the residual. Strongest null in")
    P("         the arc: the method that recovered DMA-1/netrin on CONTACT fails on what contact+closure can't explain.")
P("="*80)
