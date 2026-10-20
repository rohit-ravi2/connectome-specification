"""TOPOLOGY-AS-PRIMARY inversion. Treat reproducible graph-topological structure as the SPECIFIED
object; geometry (contact area) as a readout/trace. Three analyses, Cook held-out, capacity/degree nulls.
Predictor tags: STRUCTURAL (anatomy) / DERIVED (from connectome; circularity-risk) / FREE (fitted).
python: python
"""
import warnings,sys,json,csv,re; warnings.simplefilter("ignore")
import numpy as np, openpyxl, pandas as pd
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC, HistGradientBoostingRegressor as HGBR
from sklearn.metrics import roc_auc_score, r2_score
from scipy.stats import spearmanr, pearsonr
def P(*a): print(*a); sys.stdout.flush()
RNG=np.random.default_rng(0)
D="../data/celegans/data"

# ---------- neuron set + soma positions ----------
pj=json.load(open("../data/network_positions.json"))
pos={c["cell"]:np.array([c["x"],c["y"],c["z"]],float) for c in pj["cells"] if None not in (c.get("x"),c.get("y"),c.get("z"))}

# ---------- adult contact area (symmetric graded) ----------
wb0=openpyxl.load_workbook(f"{D}/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx",read_only=True,data_only=True)
data=list(wb0['adult nerve ring neighbors'].iter_rows(values_only=True))
neurons=[h for h in data[0][1:] if isinstance(h,str)]
neurons=[n for n in neurons if n in pos]           # keep those with soma coords
idx={n:i for i,n in enumerate(neurons)}; M=len(neurons)
A=np.zeros((M,M)); hdr=[h for h in data[0][1:]]
for row in data[1:]:
    rn=row[0]
    if rn in idx:
        for j,v in enumerate(row[1:]):
            cn=hdr[j]
            if cn in idx and v not in (None,''):
                try: A[idx[rn],idx[cn]]=float(v)
                except: pass
A=np.maximum(A,A.T); Ab=(A>0).astype(float)
P(f"neurons(ring, with soma) = {M} | contact edges (undirected) = {int(Ab.sum()/2)}")

# ---------- Witvliet directed chemical, datasets 7&8 ----------
W7=np.zeros((M,M)); W8=np.zeros((M,M))
with open(f"{D}/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']!='chemical': continue
        m=re.search(r'witvliet_2020_(\d)',r['source_file'])
        if not m: continue
        ds=int(m.group(1))
        if r['pre'] in idx and r['post'] in idx:
            if ds==7: W7[idx[r['pre']],idx[r['post']]]+=1
            elif ds==8: W8[idx[r['pre']],idx[r['post']]]+=1
Wrep=((W7>0)&(W8>0)).astype(float)     # DIRECTED reproducible synapse = present in BOTH adults

# ---------- Cook directed chemical ----------
wc=openpyxl.load_workbook(f"{D}/connectome/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
cd=list(wc['hermaphrodite chemical'].iter_rows(values_only=True))
def isneu(x): return bool(isinstance(x,str) and str(x).strip() in idx)
hr=max(range(6),key=lambda r:sum(isneu(x) for x in cd[r]))
colmap={c:cd[hr][c].strip() for c in range(len(cd[hr])) if isneu(cd[hr][c])}
rhc=max(range(6),key=lambda c:sum(isneu(cd[r][c]) for r in range(hr+1,len(cd)) if c<len(cd[r])))
C=np.zeros((M,M))
for r in range(hr+1,len(cd)):
    if rhc<len(cd[r]) and isneu(cd[r][rhc]):
        pi=idx[cd[r][rhc].strip()]
        for c,post in colmap.items():
            if c<len(cd[r]) and cd[r][c] not in (None,''):
                try:
                    if float(cd[r][c])>=1: C[pi,idx[post]]=1.0
                except: pass
P(f"Witvliet reproducible directed edges = {int(Wrep.sum())} | Cook directed edges = {int(C.sum())}")

# ---------- ordered pairs over contact-eligible (A[i,j]>0) ----------
OP=[(i,j) for i in range(M) for j in range(M) if i!=j and A[i,j]>0]
oi=np.array([p[0] for p in OP]); oj=np.array([p[1] for p in OP]); nO=len(OP)
soma=np.linalg.norm(np.array([pos[neurons[i]]-pos[neurons[j]] for i,j in OP]),axis=1)
logarea=np.log1p(A[oi,oj])
yW=Wrep[oi,oj]                         # directed reproducible synapse presence  (TOPOLOGY target)
yC=C[oi,oj]
P(f"ordered contact-eligible pairs = {nO} | Witvliet edge-rate={yW.mean():.3f} | Cook edge-rate={yC.mean():.3f}")

# ================= graph-feature builder (DERIVED). exclude=given adjacency; leave-pair-out degrees =========
def graph_feats(G, subtract_self=True):
    """G directed adjacency. per ordered pair (i,j): degrees/common-nbr/recip/triangles from G.
    subtract_self removes the (i->j) edge from i-outdeg & j-indeg to avoid trivial target leakage."""
    kout=G.sum(1); kin=G.sum(0); Gu=((G+G.T)>0).astype(float)
    comm=Gu@Gu                                    # common undirected neighbors
    ff=G@G                                         # i->k->j feed-forward paths
    tri=Gu@Gu                                       # (same as comm here) triangles use Gu
    outi=kout[oi]-(subtract_self*G[oi,oj]); ini=kin[oi]
    outj=kout[oj]; inj=kin[oj]-(subtract_self*G[oi,oj])
    recip=G[oj,oi]                                  # reverse edge present
    cn=comm[oi,oj]; f=ff[oi,oj]
    return np.column_stack([outi,ini,outj,inj,recip,cn,np.log1p(f)])   # 7 DERIVED feats

# geometry features (STRUCTURAL)
cdeg=Ab.sum(1); commC=Ab@Ab
Fgeo=np.column_stack([logarea, soma, cdeg[oi], cdeg[oj], commC[oi,oj]])  # 5 STRUCTURAL feats

# ---------- neuron-disjoint CV predict ----------
def ndcv_predict(Feat, y, task='clf', k=5, seed=0):
    rng=np.random.default_rng(seed); fold=rng.integers(0,k,M)
    pred=np.full(len(y),np.nan)
    for f in range(k):
        te_n=set(np.where(fold==f)[0])
        te=np.array([(oi[p] in te_n and oj[p] in te_n) for p in range(len(y))])
        tr=np.array([(oi[p] not in te_n and oj[p] not in te_n) for p in range(len(y))])
        if te.sum()==0 or tr.sum()==0: continue
        if task=='clf':
            if y[tr].sum()==0 or y[tr].sum()==tr.sum(): continue
            mdl=HGBC(max_iter=250,random_state=0).fit(Feat[tr],y[tr]); pred[te]=mdl.predict_proba(Feat[te])[:,1]
        else:
            mdl=HGBR(max_iter=250,random_state=0).fit(Feat[tr],y[tr]); pred[te]=mdl.predict(Feat[te])
    return pred

def auc(y,p): m=~np.isnan(p); return roc_auc_score(y[m],p[m]) if len(set(y[m].tolist()))>1 else np.nan
def r2(y,p): m=~np.isnan(p); return r2_score(y[m],p[m])

P("\n"+"="*84); P("ANALYSIS 1 — ASYMMETRY: topology<->geometry, neuron-disjoint CV, Cook held-out"); P("="*84)

# GEOMETRY -> TOPOLOGY : predict directed edge presence from geometry (STRUCTURAL)
p_g2t=ndcv_predict(Fgeo,yW,'clf'); auc_g2t=auc(yW,p_g2t)
# capacity null: matched-dim (5) random features
p_g2t_rnd=ndcv_predict(RNG.standard_normal((nO,5)),yW,'clf'); auc_g2t_rnd=auc(yW,p_g2t_rnd)
P(f"  [G->T] geometry(STRUCTURAL,5) -> Witvliet edge   AUC={auc_g2t:.3f}   (random-5 null {auc_g2t_rnd:.3f})")

# TOPOLOGY -> GEOMETRY : predict log contact-area from graph features (DERIVED)
Ft=graph_feats(Wrep)
p_t2g=ndcv_predict(Ft,logarea,'reg'); r2_t2g=r2(logarea,p_t2g)
p_t2g_rnd=ndcv_predict(RNG.standard_normal((nO,7)),logarea,'reg'); r2_t2g_rnd=r2(logarea,p_t2g_rnd)
sp_t2g=spearmanr(logarea[~np.isnan(p_t2g)],p_t2g[~np.isnan(p_t2g)]).statistic
P(f"  [T->G] topology(DERIVED,7)   -> log contact-area  R2={r2_t2g:.3f} (spearman {sp_t2g:.3f})  (random-7 null {r2_t2g_rnd:.3f})")

# ---- COMPARABLE continuous<->continuous (both R2), both directions ----
logsyn=np.log1p((W7+W8)[oi,oj])        # continuous topology intensity target
p_g2t_c=ndcv_predict(Fgeo,logsyn,'reg'); r2_g2t_c=r2(logsyn,p_g2t_c)
P(f"  [comparable, both R2]  G->T (geom->log synapse count) R2={r2_g2t_c:.3f}  |  T->G (topo->log area) R2={r2_t2g:.3f}")

# ---- COMPARABLE binary<->binary (both AUC): predict contact PRESENCE among ALL ordered pairs from topology ----
allP=[(i,j) for i in range(M) for j in range(M) if i!=j]
ai=np.array([p[0] for p in allP]); aj=np.array([p[1] for p in allP])
def gf_all(G):
    kout=G.sum(1); kin=G.sum(0); Gu=((G+G.T)>0).astype(float); comm=Gu@Gu; ff=G@G
    return np.column_stack([kout[ai],kin[ai],kout[aj],kin[aj],G[aj,ai],comm[ai,aj],np.log1p(ff[ai,aj])])
yContact=Ab[ai,aj]
# neuron-disjoint over all pairs
def ndcv_all(Feat,y,seed=0,k=5):
    rng=np.random.default_rng(seed); fold=rng.integers(0,k,M); pred=np.full(len(y),np.nan)
    for f in range(k):
        te_n=set(np.where(fold==f)[0])
        te=np.array([(ai[p] in te_n and aj[p] in te_n) for p in range(len(y))])
        tr=np.array([(ai[p] not in te_n and aj[p] not in te_n) for p in range(len(y))])
        if te.sum()==0 or tr.sum()==0 or y[tr].sum()==0: continue
        mdl=HGBC(max_iter=200,random_state=0).fit(Feat[tr],y[tr]); pred[te]=mdl.predict_proba(Feat[te])[:,1]
    return pred
p_t2contact=ndcv_all(gf_all(Wrep),yContact); auc_t2contact=auc(yContact,p_t2contact)
# reverse binary: geometry->edge already auc_g2t (both AUC, on contact-eligible). For all-pairs symmetric,
# geometry->edge on ALL pairs:
def geom_all():
    cd_=Ab.sum(1); commc=Ab@Ab
    return np.column_stack([np.log1p(A[ai,aj]),
                            np.linalg.norm(np.array([pos[neurons[i]]-pos[neurons[j]] for i,j in allP]),axis=1),
                            cd_[ai],cd_[aj],commc[ai,aj]])
p_g2edge_all=ndcv_all(geom_all(),Wrep[ai,aj]); auc_g2edge_all=auc(Wrep[ai,aj],p_g2edge_all)
P(f"  [comparable, both AUC]  T->G (topo->contact presence) AUC={auc_t2contact:.3f}  |  G->T (geom->edge) AUC={auc_g2edge_all:.3f}")

# ---- COOK held-out ----
FtC=graph_feats(C)                     # topology features from COOK
# T->G on Cook: fit topo(Cook)->area  (does cross-dataset topology still trace geometry?)
p_t2g_cook=ndcv_predict(FtC,logarea,'reg'); r2_t2g_cook=r2(logarea,p_t2g_cook)
# G->T on Cook: geometry -> Cook edge presence
p_g2t_cook=ndcv_predict(Fgeo,yC,'clf'); auc_g2t_cook=auc(yC,p_g2t_cook)
P(f"  [COOK held-out] T->G (Cook-topo->area) R2={r2_t2g_cook:.3f}  |  G->T (geom->Cook edge) AUC={auc_g2t_cook:.3f}")
# cross-fit: train topo->area on Witvliet-topo, test predicting area using Cook-topo features (true held-out generalization of the map)
mdl=HGBR(max_iter=250,random_state=0).fit(Ft,logarea); r2_cross=r2_score(logarea,mdl.predict(FtC))
P(f"  [COOK cross-map] map fit on Witvliet-topo, applied to Cook-topo -> area  R2={r2_cross:.3f}")

P("\n"+"="*84); P("ANALYSIS 2 — COMPACT SPATIAL-FREE GENERATIVE GRAPH MODEL (Witvliet directed)"); P("="*84)
kout=Wrep.sum(1); kin=Wrep.sum(0); m_edges=Wrep.sum()
# model 0: directed Chung-Lu (degree sequence ONLY) : p_ij ~ kout_i*kin_j/m
P0=np.outer(kout,kin)/m_edges; np.fill_diagonal(P0,0); P0=np.clip(P0,0,1)
# reciprocity fraction real
recip_real=(Wrep*Wrep.T).sum()/m_edges
# global clustering (directed->undirected)
Wu=((Wrep+Wrep.T)>0).astype(float)
def gclust(G):
    tri=np.trace(G@G@G); paths=(G@G).sum()-np.trace(G@G); return tri/paths if paths>0 else 0
Cr=gclust(Wu)
P(f"  real spec: N={M}  edges={int(m_edges)}  reciprocity-frac={recip_real:.3f}  global-clustering={Cr:.3f}")
# model 1: Chung-Lu + reciprocity boost + triangle(common-neighbor) boost   (FREE: 2 scalars)
Wu2=Wu@Wu
def gen_prob(a_recip,b_tri):
    Pm=P0*(1+a_recip*Wrep.T)*(1+b_tri*np.log1p(Wu2)); np.fill_diagonal(Pm,0)
    return Pm/Pm.sum()*m_edges
# fit a_recip,b_tri by matching reciprocity & clustering of expected ensemble (coarse grid)
def ensemble_stats(Pm,ns=8):
    rf=[]; cl=[]; degcorr=[]
    for s in range(ns):
        rng=np.random.default_rng(s); G=(rng.random((M,M))<np.clip(Pm,0,1)).astype(float); np.fill_diagonal(G,0)
        e=G.sum();
        if e==0: continue
        rf.append((G*G.T).sum()/e); cl.append(gclust(((G+G.T)>0).astype(float)))
        degcorr.append(pearsonr(G.sum(1)+G.sum(0),kout+kin)[0])
    return np.mean(rf),np.mean(cl),np.mean(degcorr)
best=None
for a in [0,2,5,10,20]:
    for b in [0,0.3,0.6,1.0]:
        Pm=gen_prob(a,b); rf,cl,dc=ensemble_stats(Pm,ns=5)
        err=abs(rf-recip_real)+abs(cl-Cr)
        if best is None or err<best[0]: best=(err,a,b,rf,cl,dc,Pm)
_,a_star,b_star,rf,cl,dc,Pbest=best
P(f"  compact fit: a_recip={a_star} b_tri={b_star}  -> ensemble reciprocity={rf:.3f} clustering={cl:.3f} deg-corr={dc:.3f}")
# how well does per-pair generative probability reproduce held-out real edges (AUC) & does GEOMETRY fall out?
def pair_vec(Pm): return Pm[oi,oj]
auc_gen0=roc_auc_score(yW,pair_vec(P0))
auc_gen1=roc_auc_score(yW,pair_vec(Pbest))
sp_area0=spearmanr(pair_vec(P0),A[oi,oj]).statistic
sp_area1=spearmanr(pair_vec(Pbest),A[oi,oj]).statistic
P(f"  reproduce real edges: Chung-Lu(degree-only) AUC={auc_gen0:.3f} | +recip+tri AUC={auc_gen1:.3f}")
P(f"  GEOMETRY falls out? corr(gen-prob, contact-area): degree-only spearman={sp_area0:.3f} | +recip+tri={sp_area1:.3f}")
# capacity check: does gen-prob correlate with area BEYOND what contact-degree alone gives?
# partial: regress area on cdeg product, correlate residual with gen-prob
cdp=np.log1p(cdeg[oi]*cdeg[oj]); ar=A[oi,oj]
b=np.polyfit(cdp,ar,1); ares=ar-np.polyval(b,cdp)
sp_area_partial=spearmanr(pair_vec(Pbest),ares).statistic
P(f"  geometry-beyond-contact-degree: partial spearman(gen-prob, area | contact-deg-product)={sp_area_partial:.3f}")

P("\n"+"="*84); P("ANALYSIS 3 — EDGE-DISJOINT: how much topological structure survives target-edge holdout"); P("="*84)
# split ordered edges+nonedges into k folds; test pair features computed from graph with ALL test-fold edges REMOVED.
def edge_disjoint(y_mat, k=5, seed=0):
    rng=np.random.default_rng(seed); assign=rng.integers(0,k,nO); pred=np.full(nO,np.nan)
    for f in range(k):
        te=assign==f; tr=~te
        Gtr=np.zeros((M,M))                        # graph with ONLY training edges
        for p in np.where((y_mat[oi,oj]>0)&tr)[0]: Gtr[oi[p],oj[p]]=1
        Ftr=graph_feats_from(Gtr)                  # features for ALL pairs from Gtr only
        if y_mat[oi,oj][tr].sum()==0: continue
        mdl=HGBC(max_iter=250,random_state=0).fit(Ftr[tr],y_mat[oi,oj][tr])
        pred[te]=mdl.predict_proba(Ftr[te])[:,1]
    return pred
def graph_feats_from(G):
    kout=G.sum(1); kin=G.sum(0); Gu=((G+G.T)>0).astype(float); comm=Gu@Gu; ff=G@G
    return np.column_stack([kout[oi],kin[oi],kout[oj],kin[oj],G[oj,oi],comm[oi,oj],np.log1p(ff[oi,oj])])
p_ed=edge_disjoint(Wrep); auc_ed=roc_auc_score(yW,p_ed)
# capacity null: random 7-dim features, same edge-disjoint scheme
def edge_disjoint_rnd(k=5,seed=1):
    rng=np.random.default_rng(seed); assign=rng.integers(0,k,nO); pred=np.full(nO,np.nan); Rf=RNG.standard_normal((nO,7))
    for f in range(k):
        te=assign==f; tr=~te
        if yW[tr].sum()==0: continue
        mdl=HGBC(max_iter=250,random_state=0).fit(Rf[tr],yW[tr]); pred[te]=mdl.predict_proba(Rf[te])[:,1]
    return pred
auc_ed_rnd=roc_auc_score(yW,edge_disjoint_rnd())
# neuron-disjoint edge-disjoint (both held out): strictest
p_ed_cook=edge_disjoint(C); auc_ed_cook=roc_auc_score(yC,p_ed_cook)   # Cook edge-disjoint self
# cross: features from Witvliet-train predict Cook edges (held-out animal)
Gfull=Wrep.copy()
mdlx=HGBC(max_iter=250,random_state=0).fit(graph_feats_from(Gfull),yW)
auc_wit2cook=roc_auc_score(yC,mdlx.predict_proba(graph_feats_from(Gfull))[:,1])
P(f"  edge-disjoint (features from OTHER edges only) predict held-out Witvliet edge AUC={auc_ed:.3f}  (random-7 null {auc_ed_rnd:.3f})")
P(f"  edge-disjoint on Cook (self) AUC={auc_ed_cook:.3f}")
P(f"  Witvliet-topology features predict COOK edges AUC={auc_wit2cook:.3f}  [cross-animal held-out survival]")
# also: how much of the reproducible topology is 'the rest of the graph' — fraction of edges recovered at fixed budget
thr=np.sort(p_ed)[::-1][int(yW.sum())]; recov=((p_ed>=thr)&(yW>0)).sum()/yW.sum()
P(f"  at real-edge budget, edge-disjoint recovers {recov:.2f} of reproducible edges from the rest of the graph")
P("\n[TAGS] geometry feats=STRUCTURAL; graph/degree/recip/triangle feats=DERIVED(circularity-flagged);"
  " a_recip,b_tri & any model coeffs=FREE. All claims: neuron-disjoint or edge-disjoint + capacity null + Cook.")
