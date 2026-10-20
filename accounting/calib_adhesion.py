import warnings, json, sys; warnings.simplefilter("ignore")
import numpy as np, pandas as pd, openpyxl, networkx as nx
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
RNG=np.random.default_rng(0); D="../data/celegans/data"; SUP="../data/adhesion/nawrocka_supp"
pj=json.load(open("../data/network_positions.json"))
pos={c["cell"]:np.array([c["x"],c["y"],c["z"]],float) for c in pj["cells"] if None not in (c.get("x"),c.get("y"),c.get("z"))}
cent=defaultdict(list)
for dsn in (7,8):
    sk=json.load(open(f"../seed/data/external/mechanistic_frontier/witvliet2021_developmental/data/skeletons/Dataset{dsn}_skeletons.json"))
    for n,v in sk.items():
        if v.get('coords'): cent[n].append(np.array(list(v['coords'].values()),float).mean(0))
arbor={n:np.mean(v,0) for n,v in cent.items()}
cg=pd.read_csv(D+"/expression/cengen/thresholded/021821_liberal_threshold1.csv")
classes=list(cg.columns[3:]); Xt=(cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values>0).astype(float)
gname=cg["gene_name"].astype(str).str.lower().values; wbid=cg["Wormbase_ID"].astype(str).values
gwb={w:i for i,w in enumerate(wbid)}
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
wb0=openpyxl.load_workbook(D+"/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx",read_only=True,data_only=True)
ws=wb0['adult nerve ring neighbors']; data=list(ws.iter_rows(values_only=True)); hdr=[h for h in data[0][1:]]
contact=defaultdict(float)
for row in data[1:]:
    rn=row[0]
    if rn is None: continue
    for j,val in enumerate(row[1:]):
        cn=hdr[j]
        if cn and val not in (None,'') and rn!=cn:
            try: contact[(min(rn,cn),max(rn,cn))]=max(contact[(min(rn,cn),max(rn,cn))],float(val))
            except: pass
ring=[n for n in hdr if n in pos and n in arbor and n2c(n) is not None]
s1=pd.read_excel(f"{SUP}/media-1.xlsx",sheet_name="Table S1",header=1)
prot2wb={};seq2wb={};allprot=[]
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
def wbexpr(w): 
    i=gwb.get(w); return Xt[i] if i is not None else None
labels=[];lname={}
for pn,w in allprot:
    e=wbexpr(w)
    if e is not None and w not in lname: lname[w]=pn; labels.append(w)
pairs=[(ring[i],ring[j]) for i in range(len(ring)) for j in range(i+1,len(ring))]
y=np.array([1 if contact[(min(a,b),max(a,b))]>0 else 0 for a,b in pairs])
d_arbor=np.array([np.log1p(np.linalg.norm(arbor[a]-arbor[b])) for a,b in pairs])
Ci=np.array([n2c(a) for a,b in pairs]); Cj=np.array([n2c(b) for a,b in pairs]); N=len(pairs)
skf=StratifiedKFold(5,shuffle=True,random_state=0)
p_arbor=cross_val_predict(LogisticRegression(max_iter=2000,class_weight="balanced"),d_arbor.reshape(-1,1),y,cv=skf,method="predict_proba")[:,1]
ra=y-p_arbor
G=nx.Graph(); G.add_nodes_from(ring)
for k,(a,b) in enumerate(pairs):
    if y[k]: G.add_edge(a,b)
E=G.number_of_edges(); K=100; pmap={(min(a,b),max(a,b)):k for k,(a,b) in enumerate(pairs)}
Cnull=np.zeros((K,N))
for kk in range(K):
    H=G.copy()
    try: nx.double_edge_swap(H,nswap=3*E,max_tries=30*E)
    except: pass
    for a,b in H.edges():
        key=(min(a,b),max(a,b))
        if key in pmap: Cnull[kk,pmap[key]]=1
residnull=Cnull-p_arbor[None,:]
P(f"ring {len(ring)} pairs {N} contact {y.mean():.3f} | K={K} degree-rewires | binding pairs {len(bind)}")
def ind_of(ea,eb): return (ea[Ci]*eb[Cj]+eb[Ci]*ea[Cj])>0
def z_deg(ea,eb):
    ind=ind_of(ea,eb)
    if ind.sum()<10 or (~ind).sum()<10: return None
    obs=ra[ind].mean()-ra[~ind].mean(); ns=residnull[:,ind].mean(1)-residnull[:,~ind].mean(1)
    return (obs-ns.mean())/(ns.std()+1e-9)
def z_expr(ea,eb,reps=100):
    ind=ind_of(ea,eb)
    if ind.sum()<10 or (~ind).sum()<10: return None
    obs=ra[ind].mean()-ra[~ind].mean(); nul=[]
    for _ in range(reps):
        pa=ea[RNG.permutation(len(ea))]; pb=eb[RNG.permutation(len(eb))]; i2=ind_of(pa,pb)
        if i2.sum()>0 and (~i2).sum()>0: nul.append(ra[i2].mean()-ra[~i2].mean())
    nul=np.array(nul); return (obs-nul.mean())/(nul.std()+1e-9)
# REAL binding pairs
bd=[(f"{lname.get(tuple(fs)[0])}~{lname.get(tuple(fs)[1])}",z_deg(wbexpr(tuple(fs)[0]),wbexpr(tuple(fs)[1])),z_expr(wbexpr(tuple(fs)[0]),wbexpr(tuple(fs)[1]))) for fs in bind if wbexpr(tuple(fs)[0]) is not None and wbexpr(tuple(fs)[1]) is not None]
bd=[(n,zd,ze) for n,zd,ze in bd if zd is not None and ze is not None]
nb_deg=sum(1 for _,zd,_ in bd if zd>3); nb_expr=sum(1 for _,_,ze in bd if ze>3)
P(f"\nBINDING pairs (n={len(bd)}): DEGREE-null z>3 = {nb_deg}  |  EXPR-null z>3 = {nb_expr}")
ins=[(n,round(zd,1),round(ze,1)) for n,zd,ze in bd if 'INS' in n.upper() or 'DAF-28' in n.upper()]
P(f"  insulin-family binding pairs [name, z_deg, z_expr]: {ins}")
# CALIBRATION: random gene pairs (matched: sample any 2 CeNGEN genes), false-positive rate
ngene=Xt.shape[0]; rp_deg=0; rp_expr=0; ntested=0
for _ in range(400):
    a,b=RNG.integers(ngene),RNG.integers(ngene); ea,eb=Xt[a],Xt[b]
    zd=z_deg(ea,eb); ze=z_expr(ea,eb,reps=60)
    if zd is None or ze is None: continue
    ntested+=1; rp_deg+= (zd>3); rp_expr+= (ze>3)
P(f"\nCALIBRATION — {ntested} RANDOM gene pairs: DEGREE-null z>3 = {rp_deg} ({100*rp_deg/ntested:.1f}%)  |  EXPR-null z>3 = {rp_expr} ({100*rp_expr/ntested:.1f}%)")
P(f"  expected-by-chance at z>3 (well-calibrated) ~= {0.00135*ntested:.1f}")
P(f"\n[VERDICT] if DEGREE-null random-FP-rate ~ {nb_deg}/{len(bd)} scaled => the 9/120 was NULL INFLATION; if random-FP ~0 => binding+insulin signal is REAL and my dismissal was wrong")
