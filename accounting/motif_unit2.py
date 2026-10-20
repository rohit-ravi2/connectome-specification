"""Addendum: the over-represented cross-animal motifs are RECIPROCITY-rich (111,201,120,210,300),
NOT transitive FFL (030T z~0). Redo the gap motif-completion + head-to-head with completion metrics
matched to the ACTUAL over-represented objects (reciprocal dyad / mutual triangle)."""
import warnings,sys,json,csv,re; warnings.simplefilter("ignore")
import numpy as np, pandas as pd, openpyxl
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
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
wsL=wb0['L4 nerve ring neighbors']; dl=list(wsL.iter_rows(values_only=True)); hl=[h for h in dl[0][1:]]; caL=defaultdict(float)
for r in dl[1:]:
    rn=r[0]
    if rn is None: continue
    for jx,v in enumerate(r[1:]):
        cn=hl[jx]
        if cn and v not in (None,'') and rn!=cn:
            try: caL[(min(rn,cn),max(rn,cn))]=max(caL[(min(rn,cn),max(rn,cn))],float(v))
            except: pass
dcount=defaultdict(set)
with open(f"{D}/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']!='chemical': continue
        m=re.search(r'witvliet_2020_(\d)',r['source_file'])
        if m and int(m.group(1)) in (7,8) and r['pre'] in idx and r['post'] in idx:
            dcount[(r['pre'],r['post'])].add(int(m.group(1)))
W=np.zeros((M,M),int)
for (a,b),ds in dcount.items():
    if {7,8}<=ds: W[idx[a],idx[b]]=1
wb=openpyxl.load_workbook(f"{D}/connectome/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
A=list(wb['hermaphrodite chemical'].iter_rows(values_only=True)); nset=set(ring)
def nm(seq): return sum(1 for x in seq if isinstance(x,str) and x.strip() in nset)
hrow=int(np.argmax([nm(r) for r in A])); cols=[(str(x).strip() if x is not None else None) for x in A[hrow]]
hcol=int(np.argmax([nm([A[r][c] for r in range(len(A))]) for c in range(min(6,len(A[0])))]))
C=np.zeros((M,M),int)
for r in range(hrow+1,len(A)):
    rn=A[r][hcol]; rn=rn.strip() if isinstance(rn,str) else None
    if rn not in idx: continue
    for c in range(hcol+1,len(A[r])):
        cn=cols[c]; v=A[r][c]
        if cn in idx and v not in (None,''):
            try:
                if float(v)>0: C[idx[rn],idx[cn]]=1
            except: pass
Cadj=np.zeros((M,M),int)
for (a,b) in ca:
    if a in idx and b in idx and ca[(a,b)]>0: Cadj[idx[a],idx[b]]=1; Cadj[idx[b],idx[a]]=1
E=np.array([(i,j) for i in range(M) for j in range(M) if i!=j and Cadj[i,j]>0]); ii=E[:,0]; jj=E[:,1]
def au(i,j): a,b=ring[i],ring[j]; return ca.get((min(a,b),max(a,b)),0.0)
def auL(i,j): a,b=ring[i],ring[j]; return caL.get((min(a,b),max(a,b)),0.0)
area=np.array([au(i,j) for i,j in E]); areaL=np.array([auL(i,j) for i,j in E])
somad=np.array([np.linalg.norm(pos[ring[i]]-pos[ring[j]]) for i,j in E])
yW=np.array([W[i,j] for i,j in E]); yC=np.array([C[i,j] for i,j in E])

def recip_feats(Adj,ii,jj):
    """completion metrics for the ACTUALLY over-represented (reciprocity-rich) motifs, excl focal edge.
    rev: reverse edge j->i exists (adding i->j closes a 201/300 mutual dyad)
    mut_tri: # k with mutual dyads i<->k and j<->k (adding i<->j builds 300); wedge toward 210/300
    recip_wedge: # k with (i<->k) and (k->j or j->k) -> 111/120/210 closure
    sh_recip_parent/child on reciprocal subgraph """
    Mut=((Adj*Adj.T)>0).astype(int)     # mutual (reciprocal) adjacency
    U=((Adj+Adj.T)>0).astype(int)
    rev=Adj.T[ii,jj]                      # j->i present
    mut_tri=(Mut@Mut)[ii,jj]             # k mutually tied to both i and j
    recip_wedge=(Mut@U)[ii,jj]+(U@Mut)[ii,jj]  # one reciprocal leg + any leg
    cn_any=(U@U)[ii,jj]
    return np.column_stack([rev,mut_tri,recip_wedge,cn_any]).astype(float)

P("="*78); P("PART 2b — GAP MOTIF-COMPLETION using the ACTUAL over-rep (reciprocity) motifs"); P("="*78)
Fa=np.column_stack([np.log1p(area),np.log1p(areaL),somad])
p_area=cross_val_predict(HistGradientBoostingClassifier(max_iter=200,random_state=0),Fa,yW,cv=StratifiedKFold(5,shuffle=True,random_state=0),method="predict_proba")[:,1]
gap=(yW==1)&(p_area<np.quantile(p_area,0.5)); nonsyn=(yW==0)
Rf=recip_feats(W,ii,jj)  # rev, mut_tri, recip_wedge, cn_any
names=['rev(j->i exists)','mutual-triangle k','reciprocal-wedge','common-neighbor']
gi=np.where(gap)[0]; pi=np.where(nonsyn)[0]
Xg=np.column_stack([np.log1p(area),somad])
nn=NearestNeighbors(n_neighbors=1).fit(Xg[pi]); _,mind=nn.kneighbors(Xg[gi]); matched=pi[mind[:,0]]
P(f"gap edges n={gap.sum()} | matched non-edges n={len(matched)} (matched on area+somadist)")
for c,nm_ in enumerate(names):
    g=Rf[gi,c]; mt=Rf[matched,c]; al=Rf[pi,c]
    u,pu=mannwhitneyu(g,mt,alternative='greater')
    P(f"  {nm_:20s} gap={g.mean():.3f} matched={mt.mean():.3f} all-nonedge={al.mean():.3f} | ratio {g.mean()/max(mt.mean(),1e-9):.2f}x  MWU p={pu:.1e}")
P(f"  >> gap edges reciprocated (j->i exists): {Rf[gi,0].mean()*100:.0f}%  vs matched non-edges {Rf[matched,0].mean()*100:.0f}%  vs all-nonedge {Rf[pi,0].mean()*100:.0f}%")

P("\n"+"="*78); P("PART 3b — HEAD-TO-HEAD with reciprocity-motif block (neuron-disjoint + Cook held-out)"); P("="*78)
Fpair=np.column_stack([np.log1p(area),np.log1p(areaL),somad])
def global_feats(Adj):
    outd=Adj.sum(1); ind=Adj.sum(0); U=((Adj+Adj.T)>0).astype(int); CN=U@U
    return np.column_stack([outd[ii],ind[ii],outd[jj],ind[jj],CN[ii,jj]]).astype(float)
Fglob=global_feats(W); Fmot=recip_feats(W,ii,jj)
gnode=np.minimum(ii,jj); gkf=GroupKFold(5)
def cv_auc(Xf,y,groups):
    pr=np.zeros(len(y))
    for tr,te in gkf.split(Xf,y,groups):
        m=HistGradientBoostingClassifier(max_iter=250,random_state=0).fit(Xf[tr],y[tr]); pr[te]=m.predict_proba(Xf[te])[:,1]
    return roc_auc_score(y,pr)
amb=area<np.median(area)
P("neuron-disjoint 5-fold AUC (target=reproducible directed synapse):")
blocks=[('pairwise (contact area)',Fpair),('global topology',Fglob),('reciprocity-motif',Fmot),
        ('pairwise+global',np.column_stack([Fpair,Fglob])),('ALL',np.column_stack([Fpair,Fglob,Fmot]))]
for lab,Xf in blocks: P(f"  {lab:26s} full AUC={cv_auc(Xf,yW,gnode):.3f}   gap-regime AUC={cv_auc(Xf[amb],yW[amb],gnode[amb]):.3f}")
P("\nCook held-out (train Witvliet feats, test Cook feats from Cook graph, eval yC):")
FpairC=Fpair; FglobC=global_feats(C); FmotC=recip_feats(C,ii,jj)
for lab,Xtr,Xte in [('pairwise (contact area)',Fpair,FpairC),('global topology',Fglob,FglobC),
                    ('reciprocity-motif',Fmot,FmotC),('ALL',np.column_stack([Fpair,Fglob,Fmot]),np.column_stack([FpairC,FglobC,FmotC]))]:
    m=HistGradientBoostingClassifier(max_iter=250,random_state=0).fit(Xtr,yW); pc=m.predict_proba(Xte)[:,1]
    mg=HistGradientBoostingClassifier(max_iter=250,random_state=0).fit(Xtr[amb],yW[amb]); pcg=mg.predict_proba(Xte[amb])[:,1]
    P(f"  {lab:26s} Cook AUC={roc_auc_score(yC,pc):.3f}   Cook gap-regime AUC={roc_auc_score(yC[amb],pcg):.3f}")
# isolate reciprocity single feature (rev) predictive power in gap regime
P("\nSingle 'reverse-edge-exists' feature alone (Witvliet neuron-disjoint):")
P(f"  rev-only full AUC={cv_auc(Fmot[:,:1],yW,gnode):.3f}   gap-regime AUC={cv_auc(Fmot[amb,:1],yW[amb],gnode[amb]):.3f}")
P("DONE2.")
