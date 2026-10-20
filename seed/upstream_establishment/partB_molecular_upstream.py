#!/usr/bin/env python3
"""Part B (molecular identity/expression) -> PRE-ADULT establishment targets.

Question: do MOLECULAR layers (CeNGEN expression, effector modules, ncRNA, miRNA)
predict PRE-ADULT establishment (T_first_L1, T_L1_repro, T_preadult_persist)
BEYOND Part A's baseline (contact-topology + birth + lineage + NT),
on the SAME neuron-disjoint CV folds?

Controls: matched-random-gene-set nulls (size+expression matched, >=20 draws);
neuron-disjoint CV; NEVER use adult_strong_core* as a predictor.
"""
import csv, json, os, math, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

ROOT   = os.path.expanduser("../seed")
CE     = "../data/celegans/data"
OUT    = f"{ROOT}/data/processed/upstream_early_establishment"
DOCDIR = f"{ROOT}/docs/upstream_early_establishment"
os.makedirs(OUT, exist_ok=True); os.makedirs(DOCDIR, exist_ok=True)

MASTER = f"{OUT}/upstream_master_table.csv"
EXPR   = f"{CE}/expression/cengen/thresholded/021821_medium_threshold2.csv"   # 128 classes, TPM(thresholded)
NCRNA  = f"{CE}/expression/cengen/bulk_ncrna_zenodo/Long_form_ncRNA_table_040822.tsv"
GSETS  = f"{ROOT}/data/processed/regulatory_branch/gene_sets_effector.json"

RNG_SEEDS = list(range(5))       # CV shuffle seeds (identical folds for base vs base+mol)
NULL_DRAWS = 25                  # matched-random gene-set draws (>=20 required)

# ---------------------------------------------------------------- load master
rows = list(csv.DictReader(open(MASTER)))
def fnum(x):
    try: return float(x)
    except: return np.nan
for r in rows:
    for k in ('pre_birth','post_birth','lin_dist','contact_area','contact_cos','contact_cn'):
        r[k] = fnum(r[k])
    r['same_founder'] = int(float(r['same_founder'])) if r['same_founder'] not in ('','nan') else 0

# ---------------------------------------------------------------- expression
ex = pd.read_csv(EXPR, index_col=0)
meta_cols = ['gene_name','Wormbase_ID']
classes = [c for c in ex.columns if c not in meta_cols]
wb2row = {}
for idx, wbid in ex['Wormbase_ID'].astype(str).items():
    wb2row.setdefault(wbid, idx)
# class-by-gene log-expression matrix E: rows=class, cols=WBGene
Emat = np.log1p(ex[classes].T.values.astype(float))          # (nclass, ngene)
gene_ids = ex['Wormbase_ID'].astype(str).values              # column order == Emat columns
gidx = {g:i for i,g in enumerate(gene_ids)}
cidx = {c:i for i,c in enumerate(classes)}
# add AWC pseudo-class = mean(AWC_ON, AWC_OFF)
if 'AWC_ON' in cidx and 'AWC_OFF' in cidx:
    awc = (Emat[cidx['AWC_ON']] + Emat[cidx['AWC_OFF']]) / 2.0
    Emat = np.vstack([Emat, awc]); classes = classes + ['AWC']; cidx['AWC'] = len(classes)-1

Cset = set(classes)
def mapc(n):
    for base in ('IL2','RMD','RME'):
        if n.startswith(base):
            suf = n[len(base):]
            if suf in ('DL','DR','VL','VR','D','V'): return base+'_DV' if base+'_DV' in Cset else None
            if suf in ('L','R'):                     return base+'_LR' if base+'_LR' in Cset else None
    if n.startswith('AWC'): return 'AWC'
    cands = [n]
    if n[-1] in 'LR': cands.append(n[:-1])
    if n[-2:] in ('DL','DR','VL','VR'): cands += [n[:-2], n[:-2]+n[-2]]
    for c in cands:
        if c in Cset: return c
    return None

# ---------------------------------------------------------------- effector gene sets
gs = json.load(open(GSETS))
CATS = list(gs['categories'].keys())
cat_genes = {cat: [g for g in v['genes'].keys() if g in gidx] for cat, v in gs['categories'].items()}
all_eff = sorted(set(g for gg in cat_genes.values() for g in gg))
all_eff_cols = [gidx[g] for g in all_eff]

# module score per class per category = mean log-expr over category genes
def module_scores_for(cols_by_cat):
    M = {}
    for c in classes:
        ci = cidx[c]; row = Emat[ci]
        M[c] = {cat: (row[cols].mean() if len(cols) else 0.0) for cat, cols in cols_by_cat.items()}
    return M
cat_cols = {cat: np.array([gidx[g] for g in gg]) for cat, gg in cat_genes.items()}
MODSCORE = module_scores_for(cat_cols)

def cat_cosine(cp, cq, cols):
    a = Emat[cidx[cp]][cols]; b = Emat[cidx[cq]][cols]
    d = np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/d) if d>0 else 0.0

# ---------------------------------------------------------------- restrict to molecularly-covered rows
for r in rows:
    r['cp'] = mapc(r['pre']); r['cq'] = mapc(r['post'])
cov = [r for r in rows if r['cp'] and r['cq']]
print(f"master rows: {len(rows)}  molecular-covered: {len(cov)}")
pairs = [(r['pre'], r['post']) for r in cov]

# ---------------------------------------------------------------- baseline (Part A) feature blocks (from master cols)
def Xcontact():
    return np.array([[r['contact_area'], r['contact_cos'], r['contact_cn']] for r in cov])
def Xbirth():
    out = []
    for r in cov:
        pb, qb = r['pre_birth'], r['post_birth']
        diff = abs(pb-qb) if not (np.isnan(pb) or np.isnan(qb)) else 0.0
        out.append([np.nanmin([pb,qb]), np.nanmax([pb,qb]), diff,
                    r['lin_dist'] if not np.isnan(r['lin_dist']) else 20.0, r['same_founder']])
    return np.nan_to_num(np.array(out))
def Xnt():
    ks = sorted(set((r['pre_nt'], r['post_nt']) for r in cov)); ki = {k:i for i,k in enumerate(ks)}
    Mx = np.zeros((len(cov), len(ks)))
    for i,r in enumerate(cov): Mx[i, ki[(r['pre_nt'], r['post_nt'])]] = 1
    return Mx
Xc, Xb, Xn = Xcontact(), Xbirth(), Xnt()
BASELINE = np.hstack([Xc, Xb, Xn])
print(f"baseline dims: contact={Xc.shape[1]} birth={Xb.shape[1]} NT={Xn.shape[1]} -> {BASELINE.shape[1]}")

# ---------------------------------------------------------------- molecular feature block
def Xmol_from_modscore(MOD, cat_cols_map, eff_cols):
    """Pairwise molecular features from a module-score dict + category cols (for cosines)."""
    feats = []
    for r in cov:
        cp, cq = r['cp'], r['cq']
        row = []
        for cat in CATS:
            sp, sq = MOD[cp][cat], MOD[cq][cat]
            row += [sp, sq, abs(sp-sq), sp*sq, cat_cosine(cp, cq, cat_cols_map[cat])]
        # global effector + full-transcriptome cosine
        row.append(cat_cosine(cp, cq, np.array(eff_cols)))
        a = Emat[cidx[cp]]; b = Emat[cidx[cq]]; d = np.linalg.norm(a)*np.linalg.norm(b)
        row.append(float(a@b/d) if d>0 else 0.0)
        # complementarity: source guidance x target receptor ; adhesion homophily
        row.append(MOD[cp]['axon_guidance'] * MOD[cq]['nt_fate_receptor'])
        row.append(MOD[cp]['adhesion_recognition'] * MOD[cq]['adhesion_recognition'])
        feats.append(row)
    return np.array(feats)
Xmol = Xmol_from_modscore(MODSCORE, cat_cols, all_eff_cols)
print(f"molecular dims: {Xmol.shape[1]}")

# ---------------------------------------------------------------- neuron-disjoint CV (identical folds by seed)
def fold_assign(seed):
    neu = sorted(set(a for a,b in pairs) | set(b for a,b in pairs))
    rng = np.random.default_rng(seed); f = np.array_split(rng.permutation(neu), 5)
    return {n:i for i,ff in enumerate(f) for n in ff}

def ndcv(X, y, seed, model='gb'):
    nf = fold_assign(seed); pred = np.full(len(y), np.nan)
    for k in range(5):
        te = np.array([nf[a]==k and nf[b]==k for a,b in pairs])
        tr = np.array([nf[a]!=k and nf[b]!=k for a,b in pairs])
        if te.sum()<10 or y[tr].sum()<5 or y[te].sum()<3 or len(set(y[te]))<2: continue
        if model=='gb':
            clf = make_pipeline(StandardScaler(with_mean=False), GradientBoostingClassifier(random_state=seed))
        else:
            clf = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(X[tr], y[tr]); pred[te] = clf.predict_proba(X[te])[:,1]
    m = ~np.isnan(pred)
    return roc_auc_score(y[m], pred[m]) if (m.sum()>15 and len(set(y[m]))>1) else np.nan

def mseeds(X, y, model='gb'):
    v = [ndcv(X, y, s, model) for s in RNG_SEEDS]
    return np.nanmean(v), np.nanstd(v)

# ---------------------------------------------------------------- B1-B3 main tests
TARGETS = [("first_synapse=L1","T_first_L1"),
           ("L1_reproducible(>=2/4)","T_L1_repro"),
           ("preadult_persistent(>=2grp)","T_preadult_persist")]
main_res = []
print("\n===== B1-B3: baseline vs baseline+molecular (neuron-disjoint AUC, GradientBoosting) =====")
for tname, tkey in TARGETS:
    y = np.array([int(r[tkey]) for r in cov])
    if y.sum() < 15:
        print(f"  {tname}: too few positives ({y.sum()}) - skip"); continue
    b_m, b_s   = mseeds(BASELINE, y)
    bm_m, bm_s = mseeds(np.hstack([BASELINE, Xmol]), y)
    mo_m, mo_s = mseeds(Xmol, y)   # molecular alone
    add = bm_m - b_m
    print(f"\n  --- {tname} (pos={y.sum()}, rate={y.mean():.2f}) ---")
    print(f"    baseline(A)          AUC={b_m:.3f}±{b_s:.3f}")
    print(f"    molecular ALONE      AUC={mo_m:.3f}±{mo_s:.3f}")
    print(f"    baseline+molecular   AUC={bm_m:.3f}±{bm_s:.3f}")
    print(f"    >> molecular ADD     {add:+.3f}")
    main_res.append(dict(target=tname, key=tkey, pos=int(y.sum()), rate=round(y.mean(),3),
                         auc_baseline=round(b_m,4), sd_baseline=round(b_s,4),
                         auc_mol_alone=round(mo_m,4),
                         auc_base_plus_mol=round(bm_m,4), sd_base_plus_mol=round(bm_s,4),
                         mol_ADD=round(add,4)))

# ---------------------------------------------------------------- B4 module nulls (matched random gene sets)
# For each category: real module ADD vs distribution of ADDs from size+expr-matched random gene sets.
# Module feature block for a single category = [sp, sq, |diff|, product, cosine] (5 feats), added to baseline.
gene_mean_expr = Emat.mean(axis=0)   # mean log-expr per gene across classes (for matching)
# expression-decile bins for matched sampling
valid_gene_cols = np.where(gene_mean_expr > 0)[0]
deciles = np.quantile(gene_mean_expr[valid_gene_cols], np.linspace(0,1,11))
def gene_bin(col):
    return int(np.clip(np.searchsorted(deciles, gene_mean_expr[col], side='right')-1, 0, 9))
col2bin = {c:gene_bin(c) for c in valid_gene_cols}
bin2cols = {}
for c in valid_gene_cols: bin2cols.setdefault(col2bin[c], []).append(c)

def single_cat_features(cols):
    cols = np.array(cols)
    ms = {}
    for c in classes:
        ms[c] = Emat[cidx[c]][cols].mean() if len(cols) else 0.0
    feats = []
    for r in cov:
        cp, cq = r['cp'], r['cq']
        sp, sq = ms[cp], ms[cq]
        feats.append([sp, sq, abs(sp-sq), sp*sq, cat_cosine(cp, cq, cols)])
    return np.array(feats)

def add_of_block(block, y, base_m, model='lr'):
    m,_ = mseeds(np.hstack([BASELINE, block]), y, model=model)
    return m - base_m

null_res = []
print("\n===== B4: effector module ADD vs matched-random gene-set null (LogisticRegression) =====", flush=True)
for tname, tkey in TARGETS:
    y = np.array([int(r[tkey]) for r in cov])
    if y.sum() < 15: continue
    base_m,_ = mseeds(BASELINE, y, model='lr')   # LR baseline for internally-consistent ADD
    print(f"\n  --- {tname} (LR baseline AUC={base_m:.3f}) ---", flush=True)
    for cat in CATS:
        real_cols = cat_cols[cat]
        real_add = add_of_block(single_cat_features(real_cols), y, base_m, model='lr')
        # matched random draws: same size, same expression-bin composition
        real_bins = [col2bin.get(c) for c in real_cols if c in col2bin]
        from collections import Counter
        bincount = Counter(real_bins)
        null_adds = []
        rngm = np.random.default_rng(1234)
        for d in range(NULL_DRAWS):
            draw = []
            for b, cnt in bincount.items():
                pool = bin2cols.get(b, valid_gene_cols)
                draw += list(rngm.choice(pool, size=min(cnt, len(pool)), replace=False))
            null_adds.append(add_of_block(single_cat_features(np.array(draw)), y, base_m, model='lr'))
        null_adds = np.array(null_adds)
        p = float((null_adds >= real_add).mean())
        z = (real_add - null_adds.mean()) / (null_adds.std() + 1e-9)
        beats = "YES" if (real_add > 0 and p < 0.05) else "no"
        print(f"    {cat:24s} real_ADD={real_add:+.4f}  null={null_adds.mean():+.4f}±{null_adds.std():.4f}  "
              f"z={z:+.2f} p={p:.3f}  beats_null={beats}", flush=True)
        null_res.append(dict(target=tname, category=cat, n_genes=len(real_cols),
                             real_ADD=round(real_add,4), null_mean=round(float(null_adds.mean()),4),
                             null_sd=round(float(null_adds.std()),4), z=round(z,3), p_ge=round(p,4),
                             beats_null=beats))

# ---------------------------------------------------------------- write outputs
with open(f"{OUT}/partB_molecular_results.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(main_res[0].keys())); w.writeheader()
    for r in main_res: w.writerow(r)
with open(f"{OUT}/partB_module_nulls.csv","w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(null_res[0].keys())); w.writeheader()
    for r in null_res: w.writerow(r)
print(f"\nwrote {OUT}/partB_molecular_results.csv and partB_module_nulls.csv")

# save a small json summary for the doc step
json.dump(dict(n_master=len(rows), n_cov=len(cov), baseline_dims=int(BASELINE.shape[1]),
               mol_dims=int(Xmol.shape[1]), main=main_res, nulls=null_res),
          open(f"{OUT}/partB_summary.json","w"), indent=2)
print("DONE")
