"""PART 6 — is the connectome KOLMOGOROV-COMPRESSIBLE to the lineage?
Practical proxies (Kolmogorov uncomputable). Description lengths in BITS for the
Cook 2019 hermaphrodite chemical connectome (binary directed, neuron level).

Schemes:
  (RAW)   M bits, M = N(N-1) possible directed edges.
  (ER)    Erdos-Renyi entropy: structure-free MLE code.
  (CONFIG)Chung-Lu configuration model = RANDOM graph w/ same in/out degree seq.
  (ZLIB/LZMA) generic byte compressors on packed adjacency.
  (LINEAGE) rule model: logistic edge prob from lineage-only pair features
            (tree distance, depth, founder, birth early/late).  In-sample AND
            neuron-grouped cross-validated (generative: predict unseen neuron's wiring).
  (LIN+CONTACT) add adult nerve-ring contact area (NOTE: anatomy, not pure lineage).
  (CONFIG+LINEAGE) degree + lineage, incremental compression from lineage over degree.

Discipline: LINEAGE model must GENERATE edges from lineage/rules. CV (group by
presyn neuron) => rule generalises to held-out neurons, not read from adult graph.
Contact is flagged as anatomy-derived (partial cheat).
"""
import warnings,sys,zlib,lzma,csv,math
warnings.simplefilter("ignore")
import numpy as np, pandas as pd, openpyxl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
def P(*a): print(*a); sys.stdout.flush()
D="../data/celegans/data"
np.random.seed(0)

# ---------- Cook hermaphrodite chemical, NEURON level, binary directed ----------
wb=openpyxl.load_workbook(f"{D}/connectome/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
A=list(wb['hermaphrodite chemical'].iter_rows(values_only=True))
def looks(n): return isinstance(n,str) and len(n)>=2 and n[0].isalpha()
hr=max(range(len(A)),key=lambda r:sum(looks(x) for x in A[r]))
cols=[(str(x).strip() if x is not None else None) for x in A[hr]]
hc=max(range(min(6,len(A[0]))),key=lambda c:sum(looks(A[r][c]) for r in range(len(A))))
raw_edges=[]  # (pre,post,weight) individual neuron
for r in range(hr+1,len(A)):
    pre=A[r][hc]
    if not looks(pre): continue
    pre=str(pre).strip()
    for c in range(hc+1,len(A[r])):
        post=cols[c]; v=A[r][c]
        if post and looks(post) and post!=pre and v not in (None,''):
            try:
                if float(v)>0: raw_edges.append((pre,post,float(v)))
            except: pass

# ---------- lineage paths ----------
lf=pd.read_csv(f"{D}/processed/developmental_rules/lineage_features.csv")
path={r.neuron:str(r.lineage_path) for r in lf.itertuples() if isinstance(r.lineage_path,str)}
depth={r.neuron:int(r.depth) for r in lf.itertuples()}
bt=pd.read_csv(f"{D}/processed/developmental_rules/birthtime_earlylate.csv")
born={r.neuron:(1 if r.born=='early' else 0) for r in bt.itertuples()}

# neuron set = intersection of connectome & lineage
conn_neurons=sorted({a for a,b,w in raw_edges}|{b for a,b,w in raw_edges})
neurons=sorted([n for n in conn_neurons if n in path])
P(f"Cook chem neurons {len(conn_neurons)} | with lineage path {len(neurons)}")
idx={n:i for i,n in enumerate(neurons)}; N=len(neurons)
Adj=np.zeros((N,N),dtype=np.uint8)
for a,b,w in raw_edges:
    if a in idx and b in idx: Adj[idx[a],idx[b]]=1
E=int(Adj.sum()); M=N*(N-1)
P(f"N={N} neurons, E={E} directed edges, M={M} possible, density p={E/M:.4f}")

def H2(p):
    if p<=0 or p>=1: return 0.0
    return -(p*math.log2(p)+(1-p)*math.log2(1-p))

# ---------------- (RAW) ----------------
raw_bits=M  # 1 bit/cell (exclude diagonal)
# ---------------- (ER) ----------------
p=E/M; er_bits=M*H2(p)
# param cost of ER: encode p at 1/M precision ~ 0.5*log2(M)
er_bits_total=er_bits+0.5*math.log2(M)

# ---------------- (CONFIG / Chung-Lu = random same-degree) ----------------
outdeg=Adj.sum(1).astype(float); indeg=Adj.sum(0).astype(float)
Q=np.outer(outdeg,indeg)/E
np.fill_diagonal(Q,0.0)
Q=np.clip(Q,1e-9,1-1e-9)
mask=~np.eye(N,dtype=bool)
def dl_bits(prob,adj,mask):
    pr=np.clip(prob[mask],1e-12,1-1e-12); a=adj[mask].astype(float)
    return float(-(a*np.log2(pr)+(1-a)*np.log2(1-pr)).sum())
config_data=dl_bits(Q,Adj,mask)
# param cost: 2N degrees, each ~log2(N) bits (crude)
config_param=2*N*math.log2(N)
config_total=config_data+config_param
P(f"\n-- CONFIG (Chung-Lu, random same-degree) data {config_data:.0f} + params {config_param:.0f} = {config_total:.0f} bits")

# ---------------- (ZLIB / LZMA) on packed adjacency ----------------
packed=np.packbits(Adj[mask])  # off-diagonal bits packed
zl=len(zlib.compress(packed.tobytes(),9))*8
xz=len(lzma.compress(packed.tobytes(),preset=9))*8
# also row-major full matrix bytes (0/1 uint8) so compressor sees structure
xz_full=len(lzma.compress(Adj.tobytes(),preset=9))*8

# ---------------- pairwise feature construction ----------------
def treedist(pi,pj):
    a=pi.split('/'); b=pj.split('/'); k=0
    while k<len(a) and k<len(b) and a[k]==b[k]: k+=1
    return (len(a)-k)+(len(b)-k)
# founder = element index 1..2 of path (the AB/MS/... branch); use 2-level branch label
def branch(pi):
    parts=pi.split('/'); return parts[2] if len(parts)>2 else parts[-1]
rows_i,rows_j=np.where(mask)
feat=[]
for i,j in zip(rows_i,rows_j):
    ni,nj=neurons[i],neurons[j]
    td=treedist(path[ni],path[nj])
    same_br=1.0 if branch(path[ni])==branch(path[nj]) else 0.0
    di,dj=depth.get(ni,0),depth.get(nj,0)
    bi,bj=born.get(ni,0),born.get(nj,0)
    feat.append([td, same_br, di, dj, bi, bj, bi*bj, abs(di-dj)])
feat=np.array(feat,dtype=float)
yv=Adj[mask].astype(int)
groups=rows_i  # group by presynaptic neuron for CV
fnames=["tree_dist","same_branch","depth_i","depth_j","born_i","born_j","born_ixj","|dep_i-dep_j|"]

# standardize continuous
Fs=feat.copy()
for c in [0,2,3,7]:
    Fs[:,c]=(Fs[:,c]-Fs[:,c].mean())/(Fs[:,c].std()+1e-9)

def model_dl(Ffull, tag, cv=True):
    clf=LogisticRegression(max_iter=2000,C=1.0)
    clf.fit(Ffull,yv)
    p_in=clf.predict_proba(Ffull)[:,1]
    dl_in=float(-(yv*np.log2(np.clip(p_in,1e-12,1))+(1-yv)*np.log2(np.clip(1-p_in,1e-12,1))).sum())
    param=0.5*Ffull.shape[1]*math.log2(M)
    out={"in_data":dl_in,"param":param,"in_total":dl_in+param}
    if cv:
        gk=GroupKFold(5)
        p_cv=cross_val_predict(LogisticRegression(max_iter=2000,C=1.0),Ffull,yv,cv=gk,groups=groups,method='predict_proba')[:,1]
        dl_cv=float(-(yv*np.log2(np.clip(p_cv,1e-12,1))+(1-yv)*np.log2(np.clip(1-p_cv,1e-12,1))).sum())
        out["cv"]=dl_cv; out["p_cv"]=p_cv
    out["clf"]=clf; out["p_in"]=p_in
    P(f"-- {tag}: in-sample data {dl_in:.0f} +param {param:.0f} = {dl_in+param:.0f}"+(f" | CV(neuron-grouped) {out.get('cv',0):.0f}" if cv else ""))
    return out

P("\n== LINEAGE-RULE MODELS (bits) ==")
lin=model_dl(Fs, "LINEAGE-only (tree/depth/founder/birth)")

# contact model
wc=openpyxl.load_workbook(f"{D}/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx",read_only=True,data_only=True)
cs=list(wc['adult nerve ring neighbors'].iter_rows(values_only=True))
cnames=[str(x).strip() if x else None for x in cs[0]]
cidx={n:k for k,n in enumerate(cnames) if n}
contact=np.zeros((N,N))
for r in range(1,len(cs)):
    rn=str(cs[r][0]).strip() if cs[r][0] else None
    if rn in idx:
        for c in range(1,len(cs[r])):
            cn=cnames[c]; v=cs[r][c]
            if cn in idx and isinstance(v,(int,float)):
                contact[idx[rn],idx[cn]]=v
lc=np.log1p(contact)
lc_feat=lc[mask].reshape(-1,1)
lc_feat=(lc_feat-lc_feat.mean())/(lc_feat.std()+1e-9)
Fs_c=np.hstack([Fs,lc_feat])
P("\n== + CONTACT (anatomy-derived, PARTIAL CHEAT flagged) ==")
linc=model_dl(Fs_c, "LINEAGE + adult contact area")

# config(degree)+lineage : include log expected degree product as feature
deg_feat=np.log(np.clip(Q[mask],1e-9,None)).reshape(-1,1)
deg_feat=(deg_feat-deg_feat.mean())/(deg_feat.std()+1e-9)
Fs_d=np.hstack([Fs,deg_feat])
P("\n== CONFIG(degree)+LINEAGE (does lineage add over degree?) ==")
lind=model_dl(Fs_d, "degree + lineage")
Fs_donly=deg_feat
P("degree-only reference:")
degm=model_dl(Fs_donly,"degree-only (Chung-Lu as logistic)")

# lineage tree encoding cost (the 'interpreter'): compress the lineage paths
paths_blob=("\n".join(path[n] for n in neurons)).encode()
tree_bits=len(lzma.compress(paths_blob,preset=9))*8

# ---------------- SUMMARY TABLE ----------------
P("\n"+"="*70)
P("DESCRIPTION LENGTH SUMMARY (bits) for Cook chem, N=%d E=%d"%(N,E))
P("="*70)
tab=[
 ("RAW adjacency (1 bit/cell)", raw_bits, ""),
 ("ER entropy (density only)", er_bits_total, "structure-free"),
 ("zlib on packed adjacency", zl, "generic"),
 ("lzma on packed adjacency", xz, "generic"),
 ("lzma on full uint8 matrix", xz_full, "generic"),
 ("CONFIG = random same-degree", config_total, "RANDOM baseline (3)"),
 ("  config data-only", config_data, ""),
 ("LINEAGE-only in-sample", lin['in_total'], "generative, fit"),
 ("LINEAGE-only CV(neuron)", lin['cv'], "generative, held-out"),
 ("LINEAGE+CONTACT in-sample", linc['in_total'], "contact=anatomy"),
 ("LINEAGE+CONTACT CV", linc['cv'], "contact=anatomy"),
 ("degree+lineage CV", lind['cv'], ""),
 ("degree-only CV", degm['cv'], "= config-ish"),
]
for name,b,note in tab:
    P(f"  {name:34s} {b:12.0f}   {b/E:8.2f} bits/edge   {note}")
P(f"\n  lineage TREE encoding cost (lzma on paths): {tree_bits:.0f} bits  ({tree_bits/N:.1f}/neuron)")
P(f"  => LINEAGE-only CV + tree = {lin['cv']+tree_bits:.0f} bits total generative code")

# ---------------- (5) PARTIAL: which edges best lineage-described ----------------
P("\n"+"="*70); P("(5) PARTIAL — per-edge surprisal under LINEAGE-only CV model")
pcv=lin['p_cv']
edge_mask=yv==1
surp=-np.log2(np.clip(pcv,1e-12,1))  # bits to encode presence
edge_surp=surp[edge_mask]
tdcol=feat[:,0][edge_mask]; samebr=feat[:,1][edge_mask]
P(f"  edges: mean surprisal {edge_surp.mean():.2f} bits (vs -log2(density)={-math.log2(p):.2f})")
lo=edge_surp<np.quantile(edge_surp,0.25); hi=edge_surp>np.quantile(edge_surp,0.75)
P(f"  best-predicted 25% edges: mean tree_dist {tdcol[lo].mean():.1f}, same_branch frac {samebr[lo].mean():.2f}")
P(f"  worst-predicted 25% edges: mean tree_dist {tdcol[hi].mean():.1f}, same_branch frac {samebr[hi].mean():.2f}")

# Witvliet stability: are edges present across many stages better lineage-predicted?
from collections import defaultdict
stage=defaultdict(set)
with open(f"{D}/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']=='chemical':
            stage[r['source_file']].add((r['pre'],r['post']))
allstages=sorted(stage)
persist={}
for st,edges in stage.items():
    for e in edges: persist[e]=persist.get(e,0)+1
# map to our neuron pairs
pers_arr=np.full(len(yv),-1)
pi=list(zip(rows_i,rows_j))
for k,(i,j) in enumerate(pi):
    e=(neurons[i],neurons[j]); pers_arr[k]=persist.get(e,0)
have=pers_arr>=0
present=edge_mask & have & (pers_arr>0)
stable=present & (pers_arr>=6); variable=present & (pers_arr<=2)
if stable.sum()>5 and variable.sum()>5:
    P(f"\n  Witvliet stability (of Cook edges also in Witvliet):")
    P(f"    STABLE (>=6/{len(allstages)} stages, n={int(stable.sum())}): mean surprisal {surp[stable].mean():.2f} bits")
    P(f"    VARIABLE (<=2 stages, n={int(variable.sum())}): mean surprisal {surp[variable].mean():.2f} bits")
    P(f"    -> {'stable edges MORE lineage-compressible' if surp[stable].mean()<surp[variable].mean() else 'no compressibility gap'}")

np.savez("../accounting/principles/A6-lineage-compression.npz",
         summary=np.array([[t[0],t[1]] for t in tab],dtype=object), tree_bits=tree_bits, N=N,E=E)
P("\nDONE.")
