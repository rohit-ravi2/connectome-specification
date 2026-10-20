"""Layer B: homophilic fasciculation CAM -> co-fasciculation (tract co-membership).
CAVEAT UP FRONT: Moyle 2021 neuropil strata are NOT on disk; the only derivable 'tract' structure is a
contact-COMMUNITY (greedy modularity on the contactome), which is DOWNSTREAM of contact -> this is a
coarser-target ROBUSTNESS check on the adhesion->contact positive, NOT an independent 5th layer.
Rule = homotypic CAM (both cells express same fasciculation CAM) -> same tract, beyond soma distance.
Positive controls: sax-7 (L1CAM), fmi-1 (flamingo/CELSR), rig-6 (contactin).
"""
import warnings,json; warnings.simplefilter("ignore")
import numpy as np, pandas as pd, openpyxl, networkx as nx
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
RNG=np.random.default_rng(0); D="../data/celegans/data"
pj=json.load(open("../data/network_positions.json"))
pos={c["cell"]:np.array([c["x"],c["y"],c["z"]],float) for c in pj["cells"] if None not in (c.get("x"),c.get("y"),c.get("z"))}
cg=pd.read_csv(D+"/expression/cengen/thresholded/021821_liberal_threshold1.csv")
classes=list(cg.columns[3:]); Xt=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values
gname=cg["gene_name"].astype(str).str.lower().values; cls_idx={c:k for k,c in enumerate(classes)}
def n2c(n):
    if n in cls_idx: return cls_idx[n]
    b=n
    for s in "LRDV":
        if b.endswith(s) and b[:-1] in cls_idx: return cls_idx[b[:-1]]
    return cls_idx.get(b,None)
# contactome
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
ring=[n for n in hdr if n in pos and n2c(n) is not None]
# contact-community tracts
G=nx.Graph()
G.add_nodes_from(ring)
for (a,b),w in contact.items():
    if a in ring and b in ring and w>0: G.add_edge(a,b,weight=w)
comm=nx.community.greedy_modularity_communities(G,weight="weight")
node2c={n:i for i,cset in enumerate(comm) for n in cset}
print(f"ring neurons {len(ring)} | contact-community tracts {len(comm)} (sizes {sorted(len(c) for c in comm)[-6:]})")
pairs=[(ring[i],ring[j]) for i in range(len(ring)) for j in range(i+1,len(ring))]
y=np.array([1 if node2c.get(a)==node2c.get(b) else 0 for a,b in pairs])
dist=np.array([np.log1p(np.linalg.norm(pos[a]-pos[b])) for a,b in pairs])
Ci=np.array([n2c(a) for a,b in pairs]); Cj=np.array([n2c(b) for a,b in pairs]); N=len(pairs)
skf=StratifiedKFold(5,shuffle=True,random_state=0)
resid=y-cross_val_predict(LogisticRegression(max_iter=1000,class_weight="balanced"),dist.reshape(-1,1),y,cv=skf,method="predict_proba")[:,1]
from sklearn.metrics import roc_auc_score
print(f"co-tract pairs {y.sum()}/{N} | soma-distance->co-tract AUC {roc_auc_score(y,-dist):.3f}")

CAMS=["sax-7","fmi-1","rig-6","casy-1","cdh-1","cdh-3","cdh-4","cdh-8","hmr-1","nlg-1","nrx-1",
      "lad-2","zig-3","zig-4","zig-8","rig-5","ptp-3","oig-1","wrk-1","dig-1","mig-6","unc-40","sax-3"]
def homo_ind(g):
    k=[i for i,gg in enumerate(gname) if gg==g]
    if not k: return None
    e=(Xt[k[0]]>0).astype(float)
    return (e[Ci]*e[Cj])>0
def rule_z(g,reps=1000):
    ind=homo_ind(g)
    if ind is None or ind.sum()<10 or (~ind).sum()<10: return None
    obs=resid[ind].mean()-resid[~ind].mean()
    nul=np.array([(lambda p: resid[p].mean()-resid[~p].mean())(RNG.permutation(ind)) for _ in range(reps)])
    return (obs-nul.mean())/(nul.std()+1e-9), obs, int(ind.sum())
print("\nPOSITIVE CONTROLS (homotypic fasciculation CAM -> co-tract):")
for g in ["sax-7","fmi-1","rig-6"]:
    r=rule_z(g)
    if r: print(f"  {g:7s}: z={r[0]:+.1f} enrich={r[1]:+.3f} (homotypic in {r[2]} pairs)"+("  HIT" if r[0]>3 else ""))
    else: print(f"  {g:7s}: N/A")
print("\nFULL CAM SCAN:")
res=[(g,)+rule_z(g) for g in CAMS if rule_z(g)]
res.sort(key=lambda t:-t[1])
for g,z,o,n in res: print(f"  {g:7s}: z={z:+.1f} enrich={o:+.3f} n={n}"+("  HIT" if z>3 else ""))
nsig=sum(1 for _,z,_,_ in res if z>3)
print("\n"+"="*72)
print(f"CAM rules z>3: {nsig}/{len(res)} vs ~{len(res)*0.00135:.2f} chance")
print("NOTE: co-tract is contact-derived -> this CONFIRMS/extends adhesion->contact at coarser granularity;")
print("      it is NOT an independent 5th layer (that needs Moyle strata, not on disk).")
print("="*72)
