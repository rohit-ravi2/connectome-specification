"""GLOBAL-OBJECTIVE / CONSTRAINT-SATISFACTION inversion.
Is the reproducible C. elegans nerve-ring connectome at an EXTREMUM of a global objective
vs degree-preserving rewiring? And does per-edge objective-CONSTRAINT (criticality) map onto
per-edge REPRODUCIBILITY (present in both Witvliet adults / in Cook)?
Directed synapse graph among ring neurons. All nulls = directed degree-preserving rewiring.
"""
import warnings,sys,json,csv,re; warnings.simplefilter("ignore")
import numpy as np, pandas as pd, openpyxl, networkx as nx
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.linalg import expm
from scipy.stats import pearsonr, spearmanr, pointbiserialr
def P(*a): print(*a); sys.stdout.flush()
rng=np.random.default_rng(0)
D="../data/celegans/data"

# ---- ring neuron set + positions (loader parity) ----
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
nset=set(ring)

# ---- per-dataset DIRECTED chemical edges (Witvliet adults 7 & 8) ----
d7=defaultdict(int); d8=defaultdict(int)
with open(f"{D}/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']!='chemical': continue
        pre,post=r['pre'],r['post']
        if pre not in nset or post not in nset: continue
        m=re.search(r'witvliet_2020_(\d)',r['source_file'])
        if not m: continue
        ds=int(m.group(1)); w=int(r['synapses'])
        if ds==7: d7[(pre,post)]+=w
        elif ds==8: d8[(pre,post)]+=w
E7=set(d7); E8=set(d8)
union=E7|E8
repro=E7&E8                       # present in BOTH adults
variable=union-repro              # present in one adult only
P(f"ring nodes {M} | dir edges: W7 {len(E7)}  W8 {len(E8)}  union {len(union)}  reproducible {len(repro)}  variable {len(variable)}")

# ---- Cook directed chemical adjacency (held-out reproducibility target) ----
wb=openpyxl.load_workbook(f"{D}/connectome/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
A=list(wb['hermaphrodite chemical'].iter_rows(values_only=True))
def nm(seq): return sum(1 for x in seq if isinstance(x,str) and x.strip() in nset)
hrow=int(np.argmax([nm(r) for r in A])); cols=[(str(x).strip() if x is not None else None) for x in A[hrow]]
hcol=int(np.argmax([nm([A[r][c] for r in range(len(A))]) for c in range(min(6,len(A[0])))]))
cookE=set()
for r in range(hrow+1,len(A)):
    rn=A[r][hcol]; rn=rn.strip() if isinstance(rn,str) else None
    if rn not in idx: continue
    for c in range(hcol+1,len(A[r])):
        cn=cols[c]; v=A[r][c]
        if cn in idx and v not in (None,''):
            try:
                if float(v)>0: cookE.add((rn,cn))
            except: pass
P(f"Cook directed edges among ring: {len(cookE)}")

# ================= OBJECTIVES on a directed graph given by edge set =================
def to_adj(edges):
    Aadj=np.zeros((M,M))
    for (a,b) in edges: Aadj[idx[a],idx[b]]=1.0
    return Aadj
def largest_wcc_frac(Aadj):
    G=nx.from_numpy_array(Aadj,create_using=nx.DiGraph)
    if G.number_of_edges()==0: return 0.0
    return max(len(c) for c in nx.weakly_connected_components(G))/M
def global_efficiency_dir(Aadj):
    sp=shortest_path(csr_matrix(Aadj),directed=True,unweighted=True)
    inv=1.0/sp; inv[~np.isfinite(inv)]=0.0; np.fill_diagonal(inv,0.0)
    return inv.sum()/(M*(M-1))
def char_path_len(Aadj):
    sp=shortest_path(csr_matrix(Aadj),directed=True,unweighted=True)
    m=np.isfinite(sp)&(sp>0)
    return sp[m].mean()
def fiedler(Aadj):
    Au=((Aadj+Aadj.T)>0).astype(float)          # underlying undirected
    d=Au.sum(1); L=np.diag(d)-Au
    ev=np.linalg.eigvalsh(L); ev=np.sort(ev)
    return ev[1]                                  # algebraic connectivity
def spectral_gap(Aadj):
    Au=((Aadj+Aadj.T)>0).astype(float)
    ev=np.sort(np.abs(np.linalg.eigvals(Au)))[::-1]
    return (ev[0]-ev[1]) if len(ev)>1 else ev[0]
def total_communicability(Aadj):
    Au=((Aadj+Aadj.T)>0).astype(float)
    Au=Au/(np.abs(np.linalg.eigvals(Au)).max()+1e-9)   # normalize spectral radius
    return float(expm(Au).sum())
def wiring_cost_soma(Aadj):
    r,c=np.where(Aadj>0)
    return float(sum(np.linalg.norm(pos[ring[i]]-pos[ring[j]]) for i,j in zip(r,c)))
def wiring_cost_area(Aadj):
    # cost = summed soma distance / contact area proxy; here use total soma-distance (economy) — area is capacity not cost
    return wiring_cost_soma(Aadj)
def robustness_targeted(Aadj):
    # global efficiency retention after removing top-10% highest total-degree nodes
    deg=Aadj.sum(0)+Aadj.sum(1)
    k=max(1,int(0.1*M)); kill=np.argsort(deg)[::-1][:k]
    keep=np.array([i for i in range(M) if i not in set(kill)])
    sub=Aadj[np.ix_(keep,keep)]
    return global_efficiency_dir_sub(sub)
def global_efficiency_dir_sub(Aadj):
    n=Aadj.shape[0]
    if n<2: return 0.0
    sp=shortest_path(csr_matrix(Aadj),directed=True,unweighted=True)
    inv=1.0/sp; inv[~np.isfinite(inv)]=0.0; np.fill_diagonal(inv,0.0)
    return inv.sum()/(n*(n-1))
def robustness_random(Aadj,reps=20):
    k=max(1,int(0.1*M)); vals=[]
    for _ in range(reps):
        kill=set(rng.choice(M,k,replace=False))
        keep=np.array([i for i in range(M) if i not in kill])
        vals.append(global_efficiency_dir_sub(Aadj[np.ix_(keep,keep)]))
    return float(np.mean(vals))

OBJ={
 'fiedler_lambda2':fiedler,
 'spectral_gap':spectral_gap,
 'global_efficiency':global_efficiency_dir,
 'char_path_length':char_path_len,
 'wiring_cost_soma':wiring_cost_soma,
 'total_communicability':total_communicability,
 'robustness_targeted':robustness_targeted,
 'robustness_random':robustness_random,
}

def rewire_null(edges,nswap_mult=10):
    G=nx.DiGraph(); G.add_nodes_from(range(M))
    G.add_edges_from([(idx[a],idx[b]) for a,b in edges])
    try:
        nx.directed_edge_swap(G,nswap=nswap_mult*G.number_of_edges(),max_tries=200*G.number_of_edges(),seed=int(rng.integers(1e9)))
    except Exception:
        pass
    return {(ring[u],ring[v]) for u,v in G.edges()}

def screen(edges,label,nnull=100):
    P("\n"+"="*78); P(f"OBJECTIVE SCREEN — {label}  ({len(edges)} directed edges) vs {nnull} degree-preserving rewired nulls"); P("="*78)
    Areal=to_adj(edges)
    real={k:f(Areal) for k,f in OBJ.items()}
    nulls={k:[] for k in OBJ}
    for s in range(nnull):
        ne=rewire_null(edges)
        An=to_adj(ne)
        for k,f in OBJ.items(): nulls[k].append(f(An))
    P(f"{'objective':22s} {'real':>10s} {'null_mean':>10s} {'null_sd':>9s} {'z':>8s}")
    zres={}
    for k in OBJ:
        nu=np.array(nulls[k]); mu=nu.mean(); sd=nu.std()+1e-12
        z=(real[k]-mu)/sd; zres[k]=z
        P(f"{k:22s} {real[k]:>10.4f} {mu:>10.4f} {sd:>9.4f} {z:>+8.2f}")
    return zres

# ---------- (1) screen the REPRODUCIBLE connectome ----------
z_repro=screen(repro,"REPRODUCIBLE (both Witvliet adults)",nnull=100)
# also union & Cook for reference
z_union=screen(union,"UNION (either Witvliet adult)",nnull=60)
z_cook=screen(cookE,"COOK (held-out)",nnull=60)

# ================= (2) DISCRIMINATOR: per-edge constraint <-> reproducibility =================
P("\n"+"="*78); P("DISCRIMINATOR — does per-edge OBJECTIVE-CONSTRAINT map onto REPRODUCIBILITY?"); P("="*78)
# Work on the UNION graph (contains both reproducible + variable edges).
Aun=to_adj(union)
ulist=sorted(union)
# per-edge criticality on the objective(s) the network is extremal on: use global efficiency (routing/communication)
# and edge betweenness (routing degeneracy). criticality = drop in global efficiency when edge removed.
base_eff=global_efficiency_dir(Aun)
Gun=nx.DiGraph(); Gun.add_nodes_from(range(M)); Gun.add_edges_from([(idx[a],idx[b]) for a,b in ulist])
ebc=nx.edge_betweenness_centrality(Gun)   # routing criticality (fraction of shortest paths through edge)
# communicability criticality via Estrada: derivative approx = product of endpoint subgraph-centralities
Au=((Aun+Aun.T)>0).astype(float); Au_n=Au/(np.abs(np.linalg.eigvals(Au)).max()+1e-9)
EXP=expm(Au_n); commu_node=EXP.diagonal()
crit_eff=np.zeros(len(ulist)); crit_bc=np.zeros(len(ulist)); crit_comm=np.zeros(len(ulist))
for e,(a,b) in enumerate(ulist):
    i,j=idx[a],idx[b]
    A2=Aun.copy(); A2[i,j]=0.0
    crit_eff[e]=base_eff-global_efficiency_dir(A2)         # positive = removing hurts efficiency = constrained
    crit_bc[e]=ebc[(i,j)]
    crit_comm[e]=commu_node[i]*commu_node[j]               # communicability weight of the edge
y_repro=np.array([1 if e in repro else 0 for e in ulist]) # reproducibility label on union edges
y_cook =np.array([1 if (e in cookE or (e[1],e[0]) in cookE) else 0 for e in ulist])
# endpoint degrees (to control: are constrained edges just high-degree?)
degtot=Aun.sum(0)+Aun.sum(1)
edeg=np.array([degtot[idx[a]]+degtot[idx[b]] for a,b in ulist])
ew7=np.array([d7.get(e,0)+d8.get(e,0) for e in ulist])     # synapse weight (count)

def report_corr(crit,name):
    r_r,p_r=pointbiserialr(y_repro,crit)
    r_c,p_c=pointbiserialr(y_cook,crit)
    # partial correlation controlling endpoint-degree (regress both on edeg, correlate residuals)
    def resid(v):
        b=np.polyfit(edeg,v,1); return v-np.polyval(b,edeg)
    rp=pearsonr(resid(y_repro.astype(float)),resid(crit))[0]
    P(f"  {name:20s}  repro r={r_r:+.3f} (p={p_r:.1e}) | Cook r={r_c:+.3f} (p={p_c:.1e}) | repro|degree r={rp:+.3f}")
    P(f"      mean {name} : reproducible={crit[y_repro==1].mean():.4g}  variable={crit[y_repro==0].mean():.4g}")
P(f"union edges {len(ulist)}: reproducible {y_repro.sum()}  variable {(y_repro==0).sum()} | Cook+ {y_cook.sum()}")
report_corr(crit_eff,"eff-criticality")
report_corr(crit_bc,"betweenness")
report_corr(crit_comm,"communicability")
report_corr(edeg.astype(float),"endpoint-degree")
report_corr(ew7.astype(float),"synapse-weight")

# also: are reproducible edges the ones the objective PINS? compare criticality distributions
from scipy.stats import mannwhitneyu
for crit,name in [(crit_eff,"eff-criticality"),(crit_bc,"betweenness"),(crit_comm,"communicability")]:
    u,pu=mannwhitneyu(crit[y_repro==1],crit[y_repro==0],alternative='greater')
    P(f"  MannWhitney {name}: reproducible>variable  p={pu:.2e}")
P("\nDONE")
