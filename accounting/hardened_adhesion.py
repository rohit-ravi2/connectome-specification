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
# ARBOR centroids from skeletons (the harder, correct geometric baseline)
cent=defaultdict(list)
for dsn in (7,8):
    sk=json.load(open(f"../seed/data/external/mechanistic_frontier/witvliet2021_developmental/data/skeletons/Dataset{dsn}_skeletons.json"))
    for n,v in sk.items():
        if v.get('coords'): cent[n].append(np.array(list(v['coords'].values()),float).mean(0))
arbor={n:np.mean(v,0) for n,v in cent.items()}
cg=pd.read_csv(D+"/expression/cengen/thresholded/021821_liberal_threshold1.csv")
classes=list(cg.columns[3:]); Xt=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values
gname=cg["gene_name"].astype(str).str.lower().values; wbid=cg["Wormbase_ID"].astype(str).values
def by_name(g):
    w=np.where(gname==g)[0]; return (Xt[w[0]]>0).astype(float) if len(w) else None
def by_wb(w):
    i=np.where(wbid==w)[0]; return (Xt[i[0]]>0).astype(float) if len(i) else None
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
ring=[n for n in hdr if n in pos and n in arbor and n2c(n) is not None]  # require arbor too
P(f"ring neurons w/ soma+arbor+class: {len(ring)}")
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
labels=[];lname={}
for pn,w in allprot:
    e=by_wb(w)
    if e is not None and w not in lname: lname[w]=pn; labels.append(w)
Lexpr={w:by_wb(w) for w in labels}
pairs=[(ring[i],ring[j]) for i in range(len(ring)) for j in range(i+1,len(ring))]
y=np.array([1 if contact[(min(a,b),max(a,b))]>0 else 0 for a,b in pairs])
d_soma=np.array([np.log1p(np.linalg.norm(pos[a]-pos[b])) for a,b in pairs])
d_arbor=np.array([np.log1p(np.linalg.norm(arbor[a]-arbor[b])) for a,b in pairs])
Ci=np.array([n2c(a) for a,b in pairs]); Cj=np.array([n2c(b) for a,b in pairs]); N=len(pairs)
skf=StratifiedKFold(5,shuffle=True,random_state=0)
def resid_of(dd): 
    p=cross_val_predict(LogisticRegression(max_iter=2000,class_weight="balanced"),dd.reshape(-1,1),y,cv=skf,method="predict_proba")[:,1]; return y-p, roc_auc_score(y,p)
rs,as_=resid_of(d_soma); ra,aa=resid_of(d_arbor)
P(f"pairs {N} contact-rate {y.mean():.3f} | baseline AUC: SOMA {as_:.3f}  ARBOR {aa:.3f}  (arbor should be stronger)")
def rule_ind(ea,eb): return (ea[Ci]*eb[Cj]+eb[Ci]*ea[Cj])>0
def z_expr(ea,eb,resid,reps=40):
    ind=rule_ind(ea,eb)
    if ind.sum()<10 or (~ind).sum()<10: return None,0,int(ind.sum())
    obs=resid[ind].mean()-resid[~ind].mean(); nul=[]
    for _ in range(reps):
        pa=ea[RNG.permutation(len(ea))]; pb=eb[RNG.permutation(len(eb))]; i2=(pa[Ci]*pb[Cj]+pb[Ci]*pa[Cj])>0
        if i2.sum()>0 and (~i2).sum()>0: nul.append(resid[i2].mean()-resid[~i2].mean())
    nul=np.array(nul); return (obs-nul.mean())/(nul.std()+1e-9),obs,int(ind.sum())
P("\n[POSITIVE CONTROL homotypic CAMs: does it survive ARBOR baseline?]")
for g in ["sax-7","rig-6","hmr-1","casy-1","oig-1"]:
    e=by_name(g)
    if e is None: continue
    zs=z_expr(e,e,rs)[0]; za=z_expr(e,e,ra)[0]
    if zs is None or za is None: continue
    P(f"   {g:8s} homotypic: z(soma)={zs:+.1f}  z(ARBOR)={za:+.1f}")
def scan(resid):
    r=[]
    for fs in bind:
        (a,b)=tuple(fs); ea,eb=Lexpr.get(a),Lexpr.get(b)
        if ea is None or eb is None: continue
        z,obs,n=z_expr(ea,eb,resid)
        if z is not None: r.append((f"{lname.get(a,a)}~{lname.get(b,b)}",n,obs,z))
    r.sort(key=lambda x:-x[3]); return r
P("\n[PER-RULE, expression-permutation null] soma vs ARBOR baseline:")
rsoma=scan(rs); rarbor=scan(ra)
P(f"   SOMA baseline (reproduce parallel): {sum(1 for x in rsoma if x[3]>3)}/{len(rsoma)} rules z>3 | top: {[(x[0],round(x[3],1)) for x in rsoma[:4]]}")
P(f"   ARBOR baseline (hardened):          {sum(1 for x in rarbor if x[3]>3)}/{len(rarbor)} rules z>3 | top: {[(x[0],round(x[3],1)) for x in rarbor[:4]]}")
# ---- DEGREE-PRESERVING contact null on the ARBOR residual ----
P("\n[DEGREE-PRESERVING null on ARBOR residual — generating rewires...]")
pidx={(min(ring[i],ring[j]),max(ring[i],ring[j])):k for i in range(len(ring)) for j in range(i+1,len(ring)) for k in [ len([1])] }
pmap={(min(a,b),max(a,b)):k for k,(a,b) in enumerate(pairs)}
G=nx.Graph(); G.add_nodes_from(ring)
for k,(a,b) in enumerate(pairs):
    if y[k]: G.add_edge(a,b)
E=G.number_of_edges(); K=40; Cnull=np.zeros((K,N))
for kk in range(K):
    H=G.copy()
    try: nx.double_edge_swap(H,nswap=3*E,max_tries=30*E)
    except: pass
    for a,b in H.edges():
        key=(min(a,b),max(a,b))
        if key in pmap: Cnull[kk,pmap[key]]=1
P(f"   {E} contact edges, {K} degree-preserving rewires done")
residnull=Cnull-(y-ra)[None,:]  # C' - p_arbor  (p_arbor = y - ra)
def z_deg(ea,eb):
    ind=rule_ind(ea,eb)
    if ind.sum()<10 or (~ind).sum()<10: return None
    obs=ra[ind].mean()-ra[~ind].mean()
    ns=residnull[:,ind].mean(1)-residnull[:,~ind].mean(1)
    return (obs-ns.mean())/(ns.std()+1e-9)
degres=[]
for fs in bind:
    (a,b)=tuple(fs); ea,eb=Lexpr.get(a),Lexpr.get(b)
    if ea is None or eb is None: continue
    zd=z_deg(ea,eb)
    if zd is not None: degres.append((f"{lname.get(a,a)}~{lname.get(b,b)}",zd))
degres.sort(key=lambda x:-x[1])
P(f"   ARBOR baseline + DEGREE-PRESERVING null: {sum(1 for x in degres if x[1]>3)}/{len(degres)} rules z>3 | top: {[(x[0],round(x[1],1)) for x in degres[:5]]}")
P("\n[VERDICT] soma+expr(their) -> arbor+expr -> arbor+degree: watch whether 3/120 survives the stronger baseline+null")
