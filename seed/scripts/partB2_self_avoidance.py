#!/usr/bin/env python
"""Part B2: targeted self-avoidance / like-repulsion MOLECULAR test.

Target (pre-registered from B1): high-contact-NONsynaptic phenotype.
Question: do self-avoidance / like-repulsion MODULES add beyond the KNOWN
structure (contact-area + same_nt + same_mod + same_family + degree)?

DISCIPLINE: pre-registered module list; identical feature construction for
real & matched-random; matched-random by size+expression+variance; BH; do NOT
call positive unless survives matched-random+BH AND beats same-NT baseline.
"""
import json, re, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
rng = np.random.default_rng(20260701)

ROOT = "../seed"
OUT  = f"{ROOT}/data/processed/self_avoidance_apposition"
REG  = f"{ROOT}/data/processed/regulatory_branch"

# ---------------------------------------------------------------- expression
# class x gene log-CPM developmental pseudobulk; average available wiring
# windows (E2/E3/E4) per class -> maximizes class coverage.
wins = ["dev_allgene_pb_E2_270_390.csv","dev_allgene_pb_E3_390_510.csv","dev_allgene_pb_E4_ge510.csv"]
mats = []
for w in wins:
    d = pd.read_csv(f"{REG}/{w}").set_index("class")
    mats.append(d)
genes_common = sorted(set.intersection(*[set(m.columns) for m in mats]))
# classes differ per window; union classes, mean across available windows
allclasses = sorted(set().union(*[set(m.index) for m in mats]))
X = pd.DataFrame(index=allclasses, columns=genes_common, dtype=float)
for c in allclasses:
    vals = []
    for m in mats:
        if c in m.index:
            vals.append(m.reindex(columns=genes_common).loc[c].values.astype(float))
    X.loc[c] = np.nanmean(np.stack(vals), axis=0)
X = X.astype(float)
print(f"[expr] {X.shape[0]} classes x {X.shape[1]} genes (dev wiring-window mean)")

# gene-level mean & variance across classes (for matched-random)
gene_mean = X.mean(0)
gene_var  = X.var(0)
# also global detect/mean for reference
meanexpr = pd.read_csv(f"{REG}/dev_allgene_meanexpr.csv").set_index("WBGene")

# ---------------------------------------------------------------- symbol->WB
gs = json.load(open(f"{REG}/gene_sets_effector.json"))
sym2wb = {}
for cat, info in gs["categories"].items():
    for wb, g in info["genes"].items():
        sym2wb[g["symbol"]] = wb
sym2wb["nlr-1"] = "WBGene00013196"  # not in effector set; add canonical WBID

def wbs(symlist):
    out = []
    for s in symlist:
        wb = sym2wb.get(s)
        if wb is None:
            # try cdh-* / zig-* fuzzy via effector json already covered
            continue
        if wb in X.columns:
            out.append(wb)
    return sorted(set(out))

cdh_syms  = [s for s in sym2wb if s.startswith("cdh-")]
zig_syms  = [s for s in sym2wb if s.startswith("zig-")]

MODULES_SYM = {
    "igsf":        ["rig-1","rig-3","rig-4","rig-5","rig-6"]+zig_syms+["oig-1","wrk-1","sax-7","lad-2","dig-1","nlr-1"],
    "robo_slit":   ["sax-3","slt-1","eva-1"],
    "netrin":      ["unc-6","unc-40","unc-5"],
    "sema_plexin": ["smp-1","smp-2","plx-1","plx-2","mab-20"],
    "cadherin":    cdh_syms+["hmr-1","fmi-1","casy-1"],
    "nrx_nlg":     ["nrx-1","nlg-1"],
    "syg":         ["syg-1","syg-2"],
}
MODULES_SYM["self_avoid_core"] = MODULES_SYM["igsf"] + MODULES_SYM["robo_slit"]

MODULES = {k: wbs(v) for k, v in MODULES_SYM.items()}
for k in MODULES:
    present = MODULES[k]
    req = [s for s in MODULES_SYM[k] if sym2wb.get(s) in X.columns]
    print(f"[module] {k:16s} {len(present):2d} genes present  (of {len(MODULES_SYM[k])} candidates)")

# ---------------------------------------------------------------- neuron->class
CLASSES = set(X.index)
def toclass(n):
    for pat in [n, re.sub(r"[LR]$","",n), re.sub(r"[DV][LR]$","",n),
                re.sub(r"[DV]$","",n), re.sub(r"[0-9]+$","",n),
                re.sub(r"[0-9]+[LR]?$","",n)]:
        if pat in CLASSES:
            return pat
    return None

# ---------------------------------------------------------------- master
m = pd.read_csv(f"{OUT}/high_contact_residual_master.csv")
m["cls_a_dev"] = m["na"].map(toclass)
m["cls_b_dev"] = m["nb"].map(toclass)

def build_pairset(hc_col):
    d = m[(m[hc_col]==1) & m.cls_a_dev.notna() & m.cls_b_dev.notna()].copy()
    return d

# ---------------------------------------------------------------- module feats
Xz = pd.DataFrame(StandardScaler().fit_transform(X.values), index=X.index, columns=X.columns)

def module_scores(genes):
    """z-scored per-class module score (mean over genes then z across classes)."""
    if len(genes)==0: return None
    s = X[genes].mean(1)
    return (s - s.mean())/(s.std()+1e-9)

def add_module_features(d, genes, prefix):
    ms = module_scores(genes)
    za = ms.reindex(d.cls_a_dev).values
    zb = ms.reindex(d.cls_b_dev).values
    both_high = za*zb                 # suppressive-homophily: high in BOTH
    minscore  = np.minimum(za, zb)    # AND gate
    absdiff   = np.abs(za-zb)
    # cosine over module-gene profiles
    Va = X.loc[d.cls_a_dev.values, genes].values
    Vb = X.loc[d.cls_b_dev.values, genes].values
    num = (Va*Vb).sum(1)
    den = (np.linalg.norm(Va,axis=1)*np.linalg.norm(Vb,axis=1))+1e-9
    cos = num/den
    return pd.DataFrame({f"{prefix}_bothhigh":both_high, f"{prefix}_min":minscore,
                         f"{prefix}_absdiff":absdiff, f"{prefix}_cos":cos}, index=d.index)

# ---------------------------------------------------------------- baseline X
def baseline_matrix(d):
    b = pd.DataFrame(index=d.index)
    b["logc"]      = d["logc_adult"].values
    b["same_nt"]   = d["same_nt"].values
    b["same_mod"]  = d["same_mod"].values
    b["same_fam"]  = d["same_family"].values
    b["logdeg_sum"]= np.log1p(d["deg_a"].values)+np.log1p(d["deg_b"].values)
    b["logdeg_dif"]= np.abs(np.log1p(d["deg_a"].values)-np.log1p(d["deg_b"].values))
    return b

# ---------------------------------------------------------------- neuron-block CV
def neuron_block_folds(d, K=5, seed=0):
    r = np.random.default_rng(seed)
    neurons = sorted(set(d.na)|set(d.nb))
    r.shuffle(neurons)
    blk = {n:i%K for i,n in enumerate(neurons)}
    folds = []
    for k in range(K):
        test_n = {n for n,b in blk.items() if b==k}
        is_test = d.na.isin(test_n) & d.nb.isin(test_n)     # both endpoints in block
        is_train= (~d.na.isin(test_n)) & (~d.nb.isin(test_n))# neither endpoint in block
        folds.append((d.index[is_train].values, d.index[is_test].values))
    return folds

def oof_auc(Xdf, y, folds):
    oof = np.full(len(y), np.nan)
    idxmap = {ix:i for i,ix in enumerate(Xdf.index)}
    for tr, te in folds:
        if len(te)==0 or len(tr)<20: continue
        ytr = y.loc[tr]
        if ytr.nunique()<2: continue
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xdf.loc[tr].values)
        Xte = sc.transform(Xdf.loc[te].values)
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, ytr.values)
        p = clf.predict_proba(Xte)[:,1]
        for ix,pv in zip(te,p): oof[idxmap[ix]] = pv
    mask = ~np.isnan(oof)
    if mask.sum()<20 or len(np.unique(y.values[mask]))<2: return np.nan, mask.sum()
    return roc_auc_score(y.values[mask], oof[mask]), mask.sum()

# ---------------------------------------------------------------- matched-random
def matched_random_genes(real_genes, R, seed):
    """draw R gene sets same size, matched per-gene by (mean,var) quantile bin."""
    r = np.random.default_rng(seed)
    # bin all genes by mean & var deciles
    gm = gene_mean; gv = gene_var
    mbin = pd.qcut(gm.rank(method="first"), 10, labels=False)
    vbin = pd.qcut(gv.rank(method="first"), 10, labels=False)
    key = mbin.astype(str)+"_"+vbin.astype(str)
    pools = {}
    for g,k in key.items(): pools.setdefault(k, []).append(g)
    draws = []
    real_set = set(real_genes)
    for _ in range(R):
        pick = []
        for g in real_genes:
            k = key[g]; pool=[x for x in pools[k] if x not in real_set]
            pick.append(r.choice(pool) if pool else r.choice(list(gm.index)))
        draws.append(sorted(set(pick)))
    return draws

# ================================================================ RUN
results = []
nulls   = []
R = 60
for hc_col in ["hc_top20","hc_top10"]:
    d = build_pairset(hc_col)
    y = d["nonsyn"].astype(int)
    y.index = d.index
    folds = neuron_block_folds(d, K=5, seed=42)
    base = baseline_matrix(d)
    auc_base, n_base = oof_auc(base, y, folds)
    print(f"\n=== {hc_col}: n_usable={len(d)} nonsyn={y.sum()} | baseline OOF AUC={auc_base:.4f} (n_scored={n_base})")
    for mod, genes in MODULES.items():
        if len(genes)==0: continue
        mf = add_module_features(d, genes, mod)
        full = pd.concat([base, mf], axis=1)
        auc_full, _ = oof_auc(full, y, folds)
        dauc = auc_full - auc_base
        # matched-random null
        null_dauc = []
        for j,rg in enumerate(matched_random_genes(genes, R, seed=1000+hash(mod)%1000)):
            rmf = add_module_features(d, rg, "rnd")
            rfull = pd.concat([base, rmf], axis=1)
            a,_ = oof_auc(rfull, y, folds)
            if not np.isnan(a): null_dauc.append(a-auc_base)
        null_dauc = np.array(null_dauc)
        p_emp = (1+np.sum(null_dauc >= dauc))/(1+len(null_dauc))
        null_mean = null_dauc.mean() if len(null_dauc) else np.nan
        null_p95  = np.percentile(null_dauc,95) if len(null_dauc) else np.nan
        results.append(dict(hc=hc_col, module=mod, n_genes=len(genes),
                            auc_base=auc_base, auc_full=auc_full, delta_auc=dauc,
                            null_mean_dauc=null_mean, null_p95_dauc=null_p95,
                            n_null=len(null_dauc), p_emp=p_emp))
        for v in null_dauc:
            nulls.append(dict(hc=hc_col, module=mod, delta_auc=v))
        print(f"  {mod:16s} nG={len(genes):2d} dAUC={dauc:+.4f} null_mean={null_mean:+.4f} p95={null_p95:+.4f} p_emp={p_emp:.3f}")

res = pd.DataFrame(results)
# BH within each hc set
def bh(p):
    p=np.asarray(p); n=len(p); o=np.argsort(p); rk=np.empty(n,int); rk[o]=np.arange(1,n+1)
    q=p*n/rk
    # enforce monotone
    qs=q[o]; qs=np.minimum.accumulate(qs[::-1])[::-1]; out=np.empty(n); out[o]=qs
    return np.clip(out,0,1)
res["q_bh"]=np.nan
for hc in res.hc.unique():
    mask=res.hc==hc
    res.loc[mask,"q_bh"]=bh(res.loc[mask,"p_emp"].values)
res.to_csv(f"{OUT}/partB2_self_avoidance_module_results.csv", index=False)
pd.DataFrame(nulls).to_csv(f"{OUT}/partB2_self_avoidance_nulls.csv", index=False)
print("\n[written] module_results + nulls")
print(res.to_string(index=False))
