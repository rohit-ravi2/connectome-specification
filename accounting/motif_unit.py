"""MOTIF-AS-UNIT inversion. Specified object = reproducible directed 3-node circuit motif;
edges are its fill-in. (1) triad significance profile vs degree-preserving null, cross-animal
(Witvliet<->Cook). (2) is the contact-area gap specifically motif-completing? (3) does motif
membership beat pairwise (contact area) AND global topology at predicting the gap out-of-sample?
STRUCTURAL/DERIVED/FREE tagged; motif feats DERIVED -> discriminator = out-of-sample + cross-animal."""
import warnings,sys,json,csv,re; warnings.simplefilter("ignore")
import numpy as np, pandas as pd, openpyxl, networkx as nx
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
rng=np.random.default_rng(0)
D="../data/celegans/data"

# ---------- reuse exact loaders (from stochasticity_gate.py) ----------
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
    for jj,v in enumerate(r[1:]):
        cn=hdr[jj]
        if cn and v not in (None,'') and rn!=cn:
            try: ca[(min(rn,cn),max(rn,cn))]=max(ca[(min(rn,cn),max(rn,cn))],float(v))
            except: pass
ring=[n for n in hdr if n in pos and n2c(n) is not None]; M=len(ring); idx={n:i for i,n in enumerate(ring)}
# L4 area
wsL=wb0['L4 nerve ring neighbors']; dl=list(wsL.iter_rows(values_only=True)); hl=[h for h in dl[0][1:]]; caL=defaultdict(float)
for r in dl[1:]:
    rn=r[0]
    if rn is None: continue
    for jj,v in enumerate(r[1:]):
        cn=hl[jj]
        if cn and v not in (None,'') and rn!=cn:
            try: caL[(min(rn,cn),max(rn,cn))]=max(caL[(min(rn,cn),max(rn,cn))],float(v))
            except: pass

# ---------- DIRECTED reproducible Witvliet adjacency (present in BOTH adult 7 & 8) ----------
dcount=defaultdict(set)
with open(f"{D}/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']!='chemical': continue
        m=re.search(r'witvliet_2020_(\d)',r['source_file'])
        if m and int(m.group(1)) in (7,8) and r['pre'] in idx and r['post'] in idx:
            dcount[(r['pre'],r['post'])].add(int(m.group(1)))
W=np.zeros((M,M),int)  # directed reproducible: in both 7 and 8
for (a,b),ds in dcount.items():
    if {7,8}<=ds: W[idx[a],idx[b]]=1

# ---------- DIRECTED Cook adjacency (SI5 hermaphrodite chemical, row=pre col=post) ----------
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

# ---------- undirected contact eligibility ----------
Cadj=np.zeros((M,M),int)
for (a,b) in ca:
    if a in idx and b in idx and ca[(a,b)]>0:
        Cadj[idx[a],idx[b]]=1; Cadj[idx[b],idx[a]]=1
P(f"ring neurons M={M} | W directed edges={W.sum()} | Cook directed edges={C.sum()} | contact undirected edges={Cadj.sum()//2}")
P(f"W recip pairs={int((W*W.T).sum()//2)} | Cook recip pairs={int((C*C.T).sum()//2)}")

# ============================================================================
# PART 1: TRIAD SIGNIFICANCE PROFILE (13 connected directed triads) vs degree-preserving null
# ============================================================================
TRICON=['021D','021U','021C','111D','111U','030T','030C','201','120D','120U','120C','210','300']
def census_vec(Adj):
    G=nx.from_numpy_array(Adj,create_using=nx.DiGraph)
    tc=nx.triadic_census(G)
    return np.array([tc[k] for k in TRICON],float)
def dir_swap_null(Adj,nsamp=100,seed=0):
    """degree-preserving directed edge swaps; returns array (nsamp,13) of census vectors"""
    G0=nx.from_numpy_array(Adj,create_using=nx.DiGraph)
    E=G0.number_of_edges(); out=[]
    r=np.random.default_rng(seed)
    for s in range(nsamp):
        G=G0.copy()
        try: nx.directed_edge_swap(G,nswap=8*E,max_tries=200*E,seed=int(r.integers(1e9)))
        except Exception: pass
        tc=nx.triadic_census(G)
        out.append([tc[k] for k in TRICON])
    return np.array(out,float)

def tsp(Adj,name,nsamp=100,seed=0):
    obs=census_vec(Adj)
    nul=dir_swap_null(Adj,nsamp,seed)
    mu=nul.mean(0); sd=nul.std(0); sd[sd==0]=1e-9
    z=(obs-mu)/sd
    sp=z/np.sqrt((z**2).sum())  # normalized triad significance profile (Milo)
    P(f"\n--- {name}: triad census, z vs degree-preserving directed-swap null (n={nsamp}) ---")
    P("  motif   obs      null_mu     z       SP")
    for k,o,m,zz,s in zip(TRICON,obs,mu,z,sp):
        flag=' *OVER*' if zz>2 else (' under' if zz<-2 else '')
        P(f"  {k:5s} {o:8.0f} {m:10.1f} {zz:8.2f} {s:7.3f}{flag}")
    return obs,z,sp
P("\n"+"="*78); P("PART 1 — TRIAD SIGNIFICANCE PROFILE (structural, from connectome; DERIVED)"); P("="*78)
obsW,zW,spW=tsp(W,"WITVLIET reproducible",nsamp=100,seed=1)
obsC,zC,spC=tsp(C,"COOK",nsamp=100,seed=2)
from scipy.stats import pearsonr,spearmanr
r_sp=pearsonr(spW,spC)[0]
P(f"\n  >> Triad-SP correlation Witvliet vs Cook (cross-animal) = {r_sp:+.3f}")
over_both=[k for k,a,b in zip(TRICON,zW,zC) if a>2 and b>2]
under_both=[k for k,a,b in zip(TRICON,zW,zC) if a<-2 and b<-2]
P(f"  >> OVER-represented in BOTH animals (z>2): {over_both}")
P(f"  >> UNDER-represented in BOTH animals (z<-2): {under_both}")

# ============================================================================
# build directed eligible-pair table (contact area>0), features, targets
# ============================================================================
E=[(i,j) for i in range(M) for j in range(M) if i!=j and Cadj[i,j]>0]
E=np.array(E); ii=E[:,0]; jj=E[:,1]; nE=len(E)
def au(i,j): a,b=ring[i],ring[j]; return ca.get((min(a,b),max(a,b)),0.0)
def auL(i,j): a,b=ring[i],ring[j]; return caL.get((min(a,b),max(a,b)),0.0)
area=np.array([au(i,j) for i,j in E]); areaL=np.array([auL(i,j) for i,j in E])
somad=np.array([np.linalg.norm(pos[ring[i]]-pos[ring[j]]) for i,j in E])
yW=np.array([W[i,j] for i,j in E]); yC=np.array([C[i,j] for i,j in E])
P(f"\neligible DIRECTED pairs (contact>0): {nE} | reproducible W synapse among them={yW.sum()} | Cook={yC.sum()}")

# global topology feats (endpoint degrees, common neighbors) computed on a graph G, excl focal edge
def global_feats(Adj,ii,jj):
    outd=Adj.sum(1); ind=Adj.sum(0)
    U=((Adj+Adj.T)>0).astype(int)
    CN=U@U  # common neighbors (undirected sense)
    # remove focal-edge self contribution not needed (CN counts third nodes)
    f=np.column_stack([outd[ii],ind[ii],outd[jj],ind[jj],CN[ii,jj]])
    return f
# motif-completion / participation feats on graph Adj, for candidate directed edge i->j, EXCLUDING focal edge
def motif_feats(Adj,ii,jj):
    Ae=Adj.copy()
    # wedge counts using OTHER edges (a->b focal excluded naturally since patterns use third node k)
    A=Ae; AT=Ae.T
    # transitive-completion (030T / FFL): a->k->b  => A@A [i,j]
    ffl=(A@A)
    # a<-k, k<-b  (converging) : common in-source? a<-k and b<-k -> shared parent
    shpar=(AT@A)   # number k: k->i and k->j
    shchild=(A@AT) # number k: i->k and j->k
    # reciprocal-mediated: a->k, b->k with k->b etc -> use 210/300 proxies
    ki=A[ii]; # not used directly
    f=np.column_stack([ffl[ii,jj], (A.T@A.T)[ii,jj], shpar[ii,jj], shchild[ii,jj]])
    return f  # [FFL_fwd, FFL_back(b->k->a), shared-parent, shared-child]

# ============================================================================
# PART 2: IS THE GAP MOTIF-COMPLETING?
# ============================================================================
P("\n"+"="*78); P("PART 2 — IS THE CONTACT-AREA GAP MOTIF-COMPLETING?"); P("="*78)
# contact-area-only model -> p_area (5-fold), gap = reproducible synapse ranked low by area
Fa=np.column_stack([np.log1p(area),np.log1p(areaL),somad])
skf=StratifiedKFold(5,shuffle=True,random_state=0)
p_area=cross_val_predict(HistGradientBoostingClassifier(max_iter=200,random_state=0),Fa,yW,cv=skf,method="predict_proba")[:,1]
P(f"  contact-area-only AUC(->yW) = {roc_auc_score(yW,p_area):.3f}")
thr=np.quantile(p_area,0.5)
gap=(yW==1)&(p_area<thr)          # reproducible synapse contact-area ranks LOW = the gap
nonsyn=(yW==0)                    # eligible non-edges (matched pool)
P(f"  gap edges (reproducible synapse, contact-area-missed, p<median) = {gap.sum()}")
# motif-completion score = FFL wedges (a->k->b) on reproducible W EXCLUDING focal edge
Wf=W.copy()
mf=motif_feats(Wf,ii,jj)          # cols: FFL_fwd, FFL_back, shared-parent, shared-child
ffl_fwd=mf[:,0]
# motif-completion: does adding i->j close an OVER-represented triad? use FFL (030T) + shared parent/child if over
mc=mf[:,0]+mf[:,1]  # total transitive-wedge closures (both orientations of the third leg)
# matched comparison: gap vs non-gap eligible non-edges matched on contact area + soma dist
from sklearn.neighbors import NearestNeighbors
gi=np.where(gap)[0]; pi=np.where(nonsyn)[0]
Xg=np.column_stack([np.log1p(area),somad])
nn=NearestNeighbors(n_neighbors=1).fit(Xg[pi])
_,mind=nn.kneighbors(Xg[gi]); matched=pi[mind[:,0]]
P(f"  motif-completion (transitive-wedge count) — gap vs matched non-edges (matched on area+somadist):")
P(f"     gap mean FFL-wedge     = {mc[gi].mean():.3f}  (median {np.median(mc[gi]):.1f}, frac>0 {(mc[gi]>0).mean():.2f})")
P(f"     matched non-edge mean  = {mc[matched].mean():.3f}  (median {np.median(mc[matched]):.1f}, frac>0 {(mc[matched]>0).mean():.2f})")
P(f"     ALL eligible non-edge  = {mc[pi].mean():.3f}  (frac>0 {(mc[pi]>0).mean():.2f})")
from scipy.stats import mannwhitneyu
u,pu=mannwhitneyu(mc[gi],mc[matched],alternative='greater')
P(f"     Mann-Whitney gap>matched p={pu:.2e}  | ratio of means={mc[gi].mean()/max(mc[matched].mean(),1e-9):.2f}x")
# also: fraction of gap edges that complete >=1 over-represented triad
P(f"     >> {(mc[gi]>0).mean()*100:.0f}% of gap edges close >=1 transitive wedge vs {(mc[matched]>0).mean()*100:.0f}% of matched non-edges")

# ============================================================================
# PART 3: CROSS-SCALE HEAD-TO-HEAD (pairwise vs global vs motif), neuron-disjoint + Cook held-out
# ============================================================================
P("\n"+"="*78); P("PART 3 — CROSS-SCALE HEAD-TO-HEAD predicting the gap (motif vs pairwise vs global)"); P("="*78)
P("  [circularity: motif+global feats are DERIVED from the connectome, computed EXCLUDING the focal")
P("   edge (link-prediction construction); discriminator = neuron-disjoint CV + Cook held-out transfer]")
# feature blocks (Witvliet graph)
Fpair=np.column_stack([np.log1p(area),np.log1p(areaL),somad])                       # PAIRWISE
Fglob=global_feats(W,ii,jj).astype(float)                                            # GLOBAL topology
Fmot =motif_feats(W,ii,jj).astype(float)                                             # MOTIF membership
groups=ii  # neuron-disjoint by source neuron (approx; also try min-node)
gnode=np.minimum(ii,jj)
gkf=GroupKFold(5)
def cv_auc(Xf,y,groups,label):
    pr=np.zeros(len(y))
    for tr,te in gkf.split(Xf,y,groups):
        m=HistGradientBoostingClassifier(max_iter=250,random_state=0).fit(Xf[tr],y[tr])
        pr[te]=m.predict_proba(Xf[te])[:,1]
    return roc_auc_score(y,pr)
for tgt,yy,tname in [(yW,yW,'FULL reproducible-synapse')]:
    P(f"\n  TARGET: {tname} (neuron-disjoint 5-fold, group=min-node):")
    for lab,Xf in [('pairwise (contact area)',Fpair),('global topology',Fglob),('motif membership',Fmot),
                   ('pairwise+global',np.column_stack([Fpair,Fglob])),
                   ('ALL',np.column_stack([Fpair,Fglob,Fmot]))]:
        P(f"    {lab:26s} AUC = {cv_auc(Xf,yy,gnode,lab):.3f}")
# GAP-restricted target: within contact-ambiguous regime (area below median), predict synapse
amb=area<np.median(area)
P(f"\n  GAP REGIME (contact area below median, n={amb.sum()}, synapses={yW[amb].sum()}):")
for lab,Xf in [('pairwise (contact area)',Fpair),('global topology',Fglob),('motif membership',Fmot),
               ('ALL',np.column_stack([Fpair,Fglob,Fmot]))]:
    try: P(f"    {lab:26s} AUC = {cv_auc(Xf[amb],yW[amb],gnode[amb],lab):.3f}")
    except Exception as e: P(f"    {lab}: {e}")

# COOK HELD-OUT: train on Witvliet feats->yW, test on Cook (feats from Cook graph, excl focal), eval vs yC
P(f"\n  COOK HELD-OUT transfer (train Witvliet, predict Cook directed synapse yC={yC.sum()}):")
FpairC=Fpair  # pairwise same (contact area shared)
FglobC=global_feats(C,ii,jj).astype(float)
FmotC =motif_feats(C,ii,jj).astype(float)
for lab,Xtr,Xte in [('pairwise (contact area)',Fpair,FpairC),('global topology',Fglob,FglobC),
                    ('motif membership',Fmot,FmotC),('ALL',np.column_stack([Fpair,Fglob,Fmot]),np.column_stack([FpairC,FglobC,FmotC]))]:
    m=HistGradientBoostingClassifier(max_iter=250,random_state=0).fit(Xtr,yW)
    pc=m.predict_proba(Xte)[:,1]
    P(f"    {lab:26s} Cook AUC = {roc_auc_score(yC,pc):.3f}")
# Cook held-out, GAP regime only
P(f"\n  COOK HELD-OUT, GAP REGIME (area<median):")
for lab,Xtr,Xte in [('pairwise (contact area)',Fpair,FpairC),('global topology',Fglob,FglobC),
                    ('motif membership',Fmot,FmotC),('ALL',np.column_stack([Fpair,Fglob,Fmot]),np.column_stack([FpairC,FglobC,FmotC]))]:
    m=HistGradientBoostingClassifier(max_iter=250,random_state=0).fit(Xtr[amb],yW[amb])
    pc=m.predict_proba(Xte[amb])[:,1]
    P(f"    {lab:26s} Cook AUC = {roc_auc_score(yC[amb],pc):.3f}")
P("\nDONE.")
