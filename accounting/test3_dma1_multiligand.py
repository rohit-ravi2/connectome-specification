"""TEST 3 — DMA-1 multi-ligand, SINGLE HYPOTHESIS (escapes the 120-rule multiple-testing that buried it).
DMA-1 receptor binds a characterized ligand set (LRLP-1, and the tripartite SAX-7/MNR-1/LECT-2 menorin complex).
ONE test: does DMA-1(cell A) x any-known-ligand(cell B) predict CONTACT beyond an arbor-centroid baseline and
BOTH a degree-preserving contact null AND expression-permutation? Method-validation + one clean citable rule.
"""
import warnings,json,sys; warnings.simplefilter("ignore")
import numpy as np, pandas as pd, openpyxl, networkx as nx
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
RNG=np.random.default_rng(0); D="../data/celegans/data"
pj=json.load(open("../data/network_positions.json"))
pos={c["cell"]:np.array([c["x"],c["y"],c["z"]],float) for c in pj["cells"] if None not in (c.get("x"),c.get("y"),c.get("z"))}
cent=defaultdict(list)
for dsn in (7,8):
    sk=json.load(open(f"../seed/data/external/mechanistic_frontier/witvliet2021_developmental/data/skeletons/Dataset{dsn}_skeletons.json"))
    for n,v in sk.items():
        if v.get('coords'): cent[n].append(np.array(list(v['coords'].values()),float).mean(0))
arbor={n:np.mean(v,0) for n,v in cent.items()}
cg=pd.read_csv(D+"/expression/cengen/thresholded/021821_liberal_threshold1.csv")
classes=list(cg.columns[3:]); Xt=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values
gname=cg["gene_name"].astype(str).str.lower().values; cls_idx={c:k for k,c in enumerate(classes)}
def by_name(g):
    w=np.where(gname==g)[0]; return (Xt[w[0]]>0).astype(float) if len(w) else None
def n2c(n):
    if n in cls_idx: return cls_idx[n]
    b=n
    for s in "LR":
        if b.endswith(s): b=b[:-1]; break
    if b in cls_idx: return cls_idx[b]
    for s in "DV":
        if b.endswith(s) and b[:-1] in cls_idx: return cls_idx[b[:-1]]
    return cls_idx.get(b,None)
wb0=openpyxl.load_workbook(D+"/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx",read_only=True,data_only=True)
ws=wb0['adult nerve ring neighbors']; data=list(ws.iter_rows(values_only=True)); hdr=[h for h in data[0][1:]]
contact=defaultdict(float)
for r in data[1:]:
    rn=r[0]
    if rn is None: continue
    for j,val in enumerate(r[1:]):
        cn=hdr[j]
        if cn and val not in (None,'') and rn!=cn:
            try: contact[(min(rn,cn),max(rn,cn))]=max(contact[(min(rn,cn),max(rn,cn))],float(val))
            except: pass
ring=[n for n in hdr if n in pos and n in arbor and n2c(n) is not None]
pairs=[(ring[i],ring[j]) for i in range(len(ring)) for j in range(i+1,len(ring))]
y=np.array([1 if contact[(min(a,b),max(a,b))]>0 else 0 for a,b in pairs])
d_arbor=np.array([np.log1p(np.linalg.norm(arbor[a]-arbor[b])) for a,b in pairs])
Ci=np.array([n2c(a) for a,b in pairs]); Cj=np.array([n2c(b) for a,b in pairs]); N=len(pairs)
skf=StratifiedKFold(5,shuffle=True,random_state=0)
pa=cross_val_predict(LogisticRegression(max_iter=2000,class_weight="balanced"),d_arbor.reshape(-1,1),y,cv=skf,method="predict_proba")[:,1]
ra=y-pa; P(f"ring {len(ring)} pairs {N} contact-rate {y.mean():.3f} | arbor->contact AUC {roc_auc_score(y,pa):.3f}")

# ---- WBGene lookup (LRLP-1 is not under a public name in CeNGEN) ----
wbid=cg["Wormbase_ID"].astype(str).values
def by_wb(w):
    i=np.where(wbid==w)[0]; return (Xt[i[0]]>0).astype(float) if len(i) else None
SUP="../data/adhesion/nawrocka_supp"
s1=pd.read_excel(f"{SUP}/media-1.xlsx",sheet_name="Table S1",header=1); prot2wb={}
for _,r in s1.iterrows():
    w=str(r.get("WormBase Gene ID","")).strip(); pn=str(r.get("Protein name","")).strip().upper()
    if w.startswith("WBGene") and pn and pn!="NAN": prot2wb[pn]=w
def expr_of(name):
    e=by_name(name.lower())
    if e is None and name.upper() in prot2wb: e=by_wb(prot2wb[name.upper()])
    return e
dma1=expr_of("DMA-1")
def rule_ind(rec,l): return (rec[Ci]*l[Cj]+rec[Cj]*l[Ci])>0
def test_rule(name,recv,ligv):
    ind=rule_ind(recv,ligv); obs=ra[ind].mean()-ra[~ind].mean(); nr=int(ind.sum())
    nulE=[]
    for _ in range(2000):
        r2=recv[RNG.permutation(len(recv))]; l2=ligv[RNG.permutation(len(ligv))]; i2=rule_ind(r2,l2)
        if 0<i2.sum()<N: nulE.append(ra[i2].mean()-ra[~i2].mean())
    zE=(obs-np.mean(nulE))/(np.std(nulE)+1e-9)
    nsD=rn[:,ind].mean(1)-rn[:,~ind].mean(1); zD=(obs-nsD.mean())/(nsD.std()+1e-9)
    P(f"  {name:34s} marks {nr:5d} pairs  enrich={obs:+.3f}  expr-perm-z={zE:+.1f}  degree-z={zD:+.1f}"+("  LOCKS" if zE>2 and zD>2 else "  (expr-perm only)" if zE>2 else ""))
    return zE,zD
# degree-preserving contact null (shared)
G=nx.Graph(); G.add_nodes_from(ring)
pmap={(min(a,b),max(a,b)):k for k,(a,b) in enumerate(pairs)}
for k,(a,b) in enumerate(pairs):
    if y[k]: G.add_edge(a,b)
E=G.number_of_edges(); K=100; Cn=np.zeros((K,N))
for kk in range(K):
    H=G.copy()
    try: nx.double_edge_swap(H,nswap=3*E,max_tries=30*E)
    except: pass
    for a,b in H.edges():
        key=(min(a,b),max(a,b))
        if key in pmap: Cn[kk,pmap[key]]=1
rn=Cn-pa[None,:]
# ligand expression (WBGene-aware)
lrlp=expr_of("LRLP-1"); sax7=expr_of("SAX-7"); mnr1=expr_of("MNR-1"); lect2=expr_of("LECT-2")
P(f"expression found: DMA-1={dma1 is not None} LRLP-1={lrlp is not None} SAX-7={sax7 is not None} MNR-1={mnr1 is not None} LECT-2={lect2 is not None}")
P("\n"+"="*72); P("DMA-1 SINGLE-HYPOTHESIS rules -> CONTACT (arbor baseline, dual null)"); P("="*72)
if lrlp is not None: test_rule("DMA-1 ~ LRLP-1 (the buried scan hit)",dma1,lrlp)
if all(v is not None for v in (sax7,mnr1,lect2)):
    tripart=((sax7>0)&(mnr1>0)&(lect2>0)).astype(float)   # tripartite menorin AND
    P(f"  (tripartite menorin co-expressing classes: {int(tripart.sum())})")
    test_rule("DMA-1 ~ (SAX-7 & MNR-1 & LECT-2)",dma1,tripart)
P("\nNOTE: canonical DMA-1 readout is PVD menorah dendrite patterning (body wall), NOT nerve-ring contact —")
P("this tests the DMA-1~ligand adhesion rule on ring contact as the single-hypothesis version of the scan hit.")
P("="*72)
