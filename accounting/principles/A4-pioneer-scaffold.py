"""
A4 — do PIONEERS tile the sensory->motor identity-rank?

Pioneer = earliest-born neuron per soma-position cluster (tract/bundle), following
grow-worm stage1 identify_pioneers (KMeans on authored birth-position proxy bpos,
earliest birth_time_min per cluster). We ALSO try anatomical region_group as bundles.

Identity-rank s_n and trophic h_n recomputed from Cook 2019 chemical exactly as
overnight/branch10_rank.py (logistic antisym on asymmetric directed pairs; MacKay
trophic lstsq). NO generative use of the adult connectome — s/h are DESCRIPTIVE
axes we place pioneers onto; we never claim pioneers generate wiring.

Tests:
 (3) is pioneer s-set more UNIFORM across the rank than random same-N?
       spacing-CV (std/mean of sorted gaps; low=tiling), KS vs uniform, range coverage,
       vs (a) N earliest-born overall, (b) random N from full, (c) random N from early pool.
 (4) spatial: do pioneers at different trophic h tile different anatomical positions (bpos_ap)?
 (5) temporal: within pioneers, does birth order relate to s / h (do motor pioneers lead sensory)?
 (6) gaps: levels lacking a pioneer -> more variable Witvliet wiring?
"""
import warnings, sys, openpyxl; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from scipy.stats import spearmanr, kstest, mannwhitneyu, kruskal
def P(*a): print(*a); sys.stdout.flush()
RNG = np.random.default_rng(0)
D = "../data/celegans/data"
OUT = "../accounting/principles"

# ---------- identity-rank s and trophic h from Cook chemical (branch10 pattern) ----------
cg = pd.read_csv(f"{D}/expression/cengen/thresholded/021821_liberal_threshold1.csv")
classes = list(cg.columns[3:])
Xf = cg[classes].apply(pd.to_numeric, errors="coerce").fillna(0).values.T > 0
br = Xf.mean(0); Xc = Xf[:, (br>0.05)&(br<0.95)].astype(float)
cls_idx = {c:k for k,c in enumerate(classes)}
def n2c(n):
    if not isinstance(n,str): return None
    if n in cls_idx: return n
    b=n
    for s in "LR":
        if b.endswith(s): b=b[:-1]; break
    if b in cls_idx: return b
    for s in "DV":
        if b.endswith(s) and b[:-1] in cls_idx: return b[:-1]
    return None
wb = openpyxl.load_workbook(f"{D}/connectome/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx", read_only=True, data_only=True)
A = list(wb['hermaphrodite chemical'].iter_rows(values_only=True))
def looks(n): return isinstance(n,str) and len(n)>=2 and n[0].isalpha()
hr = max(range(len(A)), key=lambda r: sum(looks(x) for x in A[r]))
cols = [(str(x).strip() if x is not None else None) for x in A[hr]]
hc = max(range(min(6,len(A[0]))), key=lambda c: sum(looks(A[r][c]) for r in range(len(A))))
Ed=[]
for r in range(hr+1,len(A)):
    cpre=n2c(A[r][hc])
    if not cpre: continue
    for c in range(hc+1,len(A[r])):
        cpost=n2c(cols[c]); v=A[r][c]
        if cpost and cpost!=cpre and v not in (None,''):
            try:
                if float(v)>0: Ed.append((cpre,cpost,float(v)))
            except: pass
chem=set((a,b) for a,b,w in Ed); asym=[(a,b) for (a,b) in chem if (b,a) not in chem]
Fd=[];y=[]
for (a,b) in asym:
    ea,eb=Xc[cls_idx[a]],Xc[cls_idx[b]]; Fd+=[ea-eb,eb-ea]; y+=[1,0]
Fd=np.array(Fd); y=np.array(y)
w=LogisticRegression(max_iter=1000,C=0.5).fit(Fd,y).coef_[0]
neur=sorted({a for a,b,_ in Ed}|{b for a,b,_ in Ed})
s_cls={n:float(Xc[cls_idx[n]]@w) for n in neur}         # identity-rank per CeNGEN class
# trophic h per class
idx={n:i for i,n in enumerate(neur)}; M=len(neur); Wm=np.zeros((M,M))
for a,b,wt in Ed: Wm[idx[a],idx[b]]+=wt
indeg=Wm.sum(0); outdeg=Wm.sum(1)
h=np.linalg.lstsq(np.diag(indeg+outdeg)-(Wm+Wm.T)+np.ones((M,M))/M, indeg-outdeg, rcond=None)[0]; h=h-h.min()
h_cls={n:float(h[idx[n]]) for n in neur}
P(f"classes with s: {len(s_cls)}   s range [{min(s_cls.values()):.2f},{max(s_cls.values()):.2f}]")

# ---------- seed table (birth time + authored soma bpos) ----------
seed = pd.read_csv("../data/grow-worm/stages/stage0_seed.csv")
seed["cls"] = seed["neuron"].map(n2c)
seed["s"] = seed["cls"].map(s_cls)
seed["h"] = seed["cls"].map(h_cls)
have = seed.dropna(subset=["s","birth_time_min"]).copy()   # need s and birth time for pioneer logic
P(f"neurons with s & birthtime: {len(have)}  (of {len(seed)})")

# ---------- pioneer identification: KMeans on bpos, earliest born per bundle ----------
def identify_pioneers(df, k, seed_rs=0):
    X = df[["bpos_ap","bpos_lr","bpos_dv"]].values.astype(float)
    lab = KMeans(n_clusters=k, random_state=seed_rs, n_init=10).fit_predict(X)
    df = df.assign(bundle=lab)
    pion=[]
    for b in sorted(set(lab)):
        sub = df[df.bundle==b].sort_values(["birth_time_min","neuron"])
        pion.append(sub.iloc[0]["neuron"])
    return pion, df

# ---------- uniformity / tiling metrics ----------
def spacing_cv(vals):
    """CV of gaps between sorted values. low = evenly tiled; high = clumped."""
    v=np.sort(np.asarray(vals,float));
    if len(v)<3: return np.nan
    g=np.diff(v);
    return float(g.std()/g.mean()) if g.mean()>0 else np.nan
def coverage(vals, lo, hi):
    return (max(vals)-min(vals))/(hi-lo)
def ks_uniform(vals, lo, hi):
    u=(np.asarray(vals,float)-lo)/(hi-lo)
    return kstest(u,'uniform').statistic

s_all = have["s"].values; slo,shi = s_all.min(), s_all.max()

P("\n================ (3) TILING vs RANDOM ================")
for k in [8,10,12,16,20]:
    pion, dfk = identify_pioneers(have, k)
    pv = have.set_index("neuron").loc[pion,"s"].values
    cv = spacing_cv(pv); cov = coverage(pv,slo,shi); ks = ks_uniform(pv,slo,shi)
    N=len(pion)
    # null A: random N from full s-having pool
    cvA=[]; ksA=[]; covA=[]
    for _ in range(2000):
        r=RNG.choice(s_all,N,replace=False); cvA.append(spacing_cv(r)); ksA.append(ks_uniform(r,slo,shi)); covA.append(coverage(r,slo,shi))
    cvA=np.array(cvA)
    # null C: random N from N-earliest-born pool (2N earliest)
    eb = have.sort_values("birth_time_min").head(max(2*N, N+8))
    ebs = eb["s"].values
    cvC=[]
    for _ in range(2000):
        r=RNG.choice(ebs,N,replace=False); cvC.append(spacing_cv(r))
    cvC=np.array(cvC)
    # null B reference: the N earliest-born overall (deterministic "early but not tract-spread")
    ebN = have.sort_values("birth_time_min").head(N)["s"].values
    pA = (cvA<=cv).mean()   # fraction of random with CV <= pioneer (lower=more uniform); small p => pioneers unusually uniform
    P(f"\n k={k:2d} N={N} pioneers={sorted(pion)}")
    P(f"   pioneer spacing-CV={cv:.3f}  cov={cov:.2f}  KSuni={ks:.3f}")
    P(f"   randomFull CV mean={cvA.mean():.3f}±{cvA.std():.3f}  P(rand<=pio)={pA:.3f}   (low P => pioneers MORE uniform)")
    P(f"   earlyPool CV mean={cvC.mean():.3f}  |  N-earliest-born CV={spacing_cv(ebN):.3f}")
    P(f"   randomFull cov mean={np.mean(covA):.2f}  pioneer cov={cov:.2f}")

# ---------- fix k=12 (stage1 default) for detailed spatial/temporal ----------
K=12
pion, dfk = identify_pioneers(have, K)
pio_df = have.set_index("neuron").loc[pion].reset_index()
P("\n================ pioneer table (k=12) ================")
P(pio_df[["neuron","cls","modality","birth_time_min","s","h","bpos_ap","bpos_lr","bpos_dv"]].sort_values("s").to_string(index=False))

P("\n================ (4) SPATIAL: trophic level vs anatomical position ================")
# do high-h (motor) pioneers sit at different bpos_ap than low-h (sensory)?
rho_ap,p_ap = spearmanr(pio_df["h"], pio_df["bpos_ap"])
rho_s_ap,p_s_ap = spearmanr(pio_df["s"], pio_df["bpos_ap"])
P(f"  pioneer trophic h  vs bpos_ap : Spearman r={rho_ap:+.3f} p={p_ap:.3f}")
P(f"  pioneer identity s vs bpos_ap : Spearman r={rho_s_ap:+.3f} p={p_s_ap:.3f}")
# across ALL neurons for reference
rho_all,p_all = spearmanr(have["h"], have["bpos_ap"])
P(f"  (all neurons) h vs bpos_ap    : r={rho_all:+.3f} p={p_all:.1e}")

P("\n================ (5) TEMPORAL: birth order vs hierarchy WITHIN pioneers ================")
rho_bs,p_bs = spearmanr(pio_df["birth_time_min"], pio_df["s"])
rho_bh,p_bh = spearmanr(pio_df["birth_time_min"], pio_df["h"])
P(f"  pioneer birthtime vs s (identity-rank): r={rho_bs:+.3f} p={p_bs:.3f}")
P(f"  pioneer birthtime vs h (trophic)      : r={rho_bh:+.3f} p={p_bh:.3f}")
# all-neuron reference
rho_abs,p_abs = spearmanr(have["birth_time_min"], have["s"])
rho_abh,p_abh = spearmanr(have["birth_time_min"], have["h"])
P(f"  (all) birthtime vs s: r={rho_abs:+.3f} p={p_abs:.1e}   birthtime vs h: r={rho_abh:+.3f} p={p_abh:.1e}")
# modality of pioneers
P(f"  pioneer modality counts: {pio_df.modality.value_counts().to_dict()}")

# ---------- (6) gaps -> variable wiring (Witvliet reproducibility across 8 stages) ----------
P("\n================ (6) RANK GAPS vs WIRING VARIABILITY (Witvliet) ================")
try:
    wv = pd.read_csv(f"{D}/connectome/witvliet2020/witvliet_combined_edges.csv")
    wv["stage"] = wv["source_file"].str.extract(r"witvliet_2020_(\d)").astype(float)
    wv["cpre"]=wv["pre"].map(n2c); wv["cpost"]=wv["post"].map(n2c)
    wv=wv.dropna(subset=["cpre","cpost","stage"])
    nst = wv["stage"].nunique()
    # per (cpre,cpost) edge: reproducibility = #stages present / nstages
    ed_rep = wv.groupby(["cpre","cpost"])["stage"].nunique()/nst
    # bin the s-axis into levels; a "gap" level has no pioneer. Ask: do edges whose
    # endpoints sit in pioneer-occupied s-bins reproduce more than edges in gap bins?
    nb=10
    bins=np.linspace(slo,shi,nb+1)
    pio_s = pio_df["s"].values
    occ = set(np.clip(np.digitize(pio_s,bins)-1,0,nb-1))
    gap = set(range(nb))-occ
    P(f"  s binned into {nb} levels; pioneer-occupied={sorted(occ)}  gaps={sorted(gap)}")
    def bin_of(cls):
        v=s_cls.get(cls);
        return int(np.clip(np.digitize([v],bins)[0]-1,0,nb-1)) if v is not None else None
    rows=[]
    for (a,b),rep in ed_rep.items():
        ba,bb=bin_of(a),bin_of(b)
        if ba is None or bb is None: continue
        # edge is "in gap" if either endpoint bin lacks a pioneer
        ingap = (ba in gap) or (bb in gap)
        rows.append((rep, ingap))
    rdf=pd.DataFrame(rows,columns=["rep","ingap"])
    g0=rdf[~rdf.ingap]["rep"]; g1=rdf[rdf.ingap]["rep"]
    P(f"  edges in pioneer-occupied s-bins: n={len(g0)} mean reproducibility={g0.mean():.3f}")
    P(f"  edges touching GAP s-bins       : n={len(g1)} mean reproducibility={g1.mean():.3f}")
    if len(g0)>5 and len(g1)>5:
        u=mannwhitneyu(g0,g1,alternative='two-sided')
        P(f"  MWU p={u.pvalue:.3f}  (lower rep in gap-bins => gaps predict variable wiring)")
except Exception as e:
    P(f"  witvliet step failed: {e}")

# save pioneer table
pio_df.to_csv(f"{OUT}/A4-pioneer-scaffold_pioneers_k12.csv", index=False)
P(f"\nwrote {OUT}/A4-pioneer-scaffold_pioneers_k12.csv")
