"""Fix Part 2 matched control: the NN matcher was pairing each directed gap edge (i,j) with its own
transpose (j,i) [identical symmetric area/somad, rev=1 by construction]. Exclude transpose; caliper-match
on area+somad+endpoint-degree; report gap motif-completion vs proper matched non-edges + raw baselines."""
import warnings,sys,json,csv,re; warnings.simplefilter("ignore")
import numpy as np, pandas as pd, openpyxl
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from scipy.stats import mannwhitneyu
def P(*a): print(*a); sys.stdout.flush()
D="../data/celegans/data"
pj=json.load(open("../data/network_positions.json"))
pos={c["cell"]:np.array([c["x"],c["y"],c["z"]],float) for c in pj["cells"] if None not in (c.get("x"),c.get("y"),c.get("z"))}
cg=pd.read_csv(D+"/expression/cengen/thresholded/021821_liberal_threshold1.csv"); classes=list(cg.columns[3:]); cls_idx={c:k for k,c in enumerate(classes)}
def n2c(n):
    if n in cls_idx: return cls_idx[n]
    b=n
    for s in "LR":
        if b.endswith(s): b=b[:-1]; break
    if b in cls_idx: return cls_idx[b]
    for s in "DV":
        if b.endswith(s) and b[:-1] in cls_idx: return cls_idx[b[:-1]]
    return cls_idx.get(b,None)
wb0=openpyxl.load_workbook(f"{D}/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx",read_only=True,data_only=True)
ws=wb0['adult nerve ring neighbors']; dd=list(ws.iter_rows(values_only=True)); hdr=[h for h in dd[0][1:]]
ca=defaultdict(float)
for r in dd[1:]:
    rn=r[0]
    if rn is None: continue
    for jx,v in enumerate(r[1:]):
        cn=hdr[jx]
        if cn and v not in (None,'') and rn!=cn:
            try: ca[(min(rn,cn),max(rn,cn))]=max(ca[(min(rn,cn),max(rn,cn))],float(v))
            except: pass
ring=[n for n in hdr if n in pos and n2c(n) is not None]; M=len(ring); idx={n:i for i,n in enumerate(ring)}
dcount=defaultdict(set)
with open(f"{D}/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']!='chemical': continue
        m=re.search(r'witvliet_2020_(\d)',r['source_file'])
        if m and int(m.group(1)) in (7,8) and r['pre'] in idx and r['post'] in idx: dcount[(r['pre'],r['post'])].add(int(m.group(1)))
W=np.zeros((M,M),int)
for (a,b),ds in dcount.items():
    if {7,8}<=ds: W[idx[a],idx[b]]=1
Cadj=np.zeros((M,M),int)
for (a,b) in ca:
    if a in idx and b in idx and ca[(a,b)]>0: Cadj[idx[a],idx[b]]=1; Cadj[idx[b],idx[a]]=1
E=np.array([(i,j) for i in range(M) for j in range(M) if i!=j and Cadj[i,j]>0]); ii=E[:,0]; jj=E[:,1]
area=np.array([ca.get((min(ring[i],ring[j]),max(ring[i],ring[j])),0.0) for i,j in E])
somad=np.array([np.linalg.norm(pos[ring[i]]-pos[ring[j]]) for i,j in E])
yW=np.array([W[i,j] for i,j in E])
outd=W.sum(1); ind=W.sum(0); U=((W+W.T)>0).astype(int); Mut=((W*W.T)>0).astype(int); CN=U@U
rev=W.T[ii,jj]; mut_tri=(Mut@Mut)[ii,jj]; recip_wedge=((Mut@U)+(U@Mut))[ii,jj]; cn=CN[ii,jj]
Fa=np.column_stack([np.log1p(area),somad])
p_area=cross_val_predict(HistGradientBoostingClassifier(max_iter=200,random_state=0),np.column_stack([np.log1p(area),somad]),yW,cv=StratifiedKFold(5,shuffle=True,random_state=0),method="predict_proba")[:,1]
gap=(yW==1)&(p_area<np.quantile(p_area,0.5))
# transpose index map: for directed pair k=(i,j) find its transpose (j,i) in E
pos_map={(int(a),int(b)):k for k,(a,b) in enumerate(E)}
transpose=np.array([pos_map.get((int(jj[k]),int(ii[k])),-1) for k in range(len(E))])
gi=np.where(gap)[0]
# candidate matched pool: eligible NON-edges (yW==0) EXCLUDING the transpose of any gap edge
gap_transposes=set(int(transpose[k]) for k in gi if transpose[k]>=0)
pool=np.array([k for k in np.where(yW==0)[0] if k not in gap_transposes])
# caliper match on standardized [log area, somad, outdeg_i, indeg_j]
zfeat=lambda a: (a-a.mean())/(a.std()+1e-9)
Z=np.column_stack([zfeat(np.log1p(area)),zfeat(somad),zfeat(outd[ii].astype(float)),zfeat(ind[jj].astype(float))])
from sklearn.neighbors import NearestNeighbors
nn=NearestNeighbors(n_neighbors=1).fit(Z[pool]); dist,mind=nn.kneighbors(Z[gi]); matched=pool[mind[:,0]]
P("="*78); P("PART 2c — GAP MOTIF-COMPLETION, transpose-excluded caliper-matched control"); P("="*78)
P(f"gap n={len(gi)} | matched non-edges n={len(matched)} (excl transpose; matched on area,somad,outdeg_i,indeg_j; mean caliper dist={dist.mean():.2f})")
allne=np.where(yW==0)[0]
for nm_,x in [('rev(j->i exists)',rev),('mutual-triangle k',mut_tri),('reciprocal-wedge',recip_wedge),('common-neighbor',cn)]:
    g=x[gi].astype(float); mt=x[matched].astype(float); al=x[allne].astype(float)
    u,pu=mannwhitneyu(g,mt,alternative='two-sided')
    P(f"  {nm_:20s} gap={g.mean():.3f}  matched={mt.mean():.3f}  all-nonedge={al.mean():.3f}  | gap/matched={g.mean()/max(mt.mean(),1e-9):.2f}x  MWU p={pu:.2e}")
P(f"  >> gap reciprocated: {rev[gi].mean()*100:.0f}%  matched: {rev[matched].mean()*100:.0f}%  all-nonedge: {rev[allne].mean()*100:.0f}%")
P(f"  >> gap closes >=1 reciprocal-wedge: {(recip_wedge[gi]>0).mean()*100:.0f}%  matched: {(recip_wedge[matched]>0).mean()*100:.0f}%")
P("DONE3.")
