#!/usr/bin/env python
"""
Part B2 — Candidate localization SANITY (plausibility only).

Class-level module expression ALREADY FAILED the strict matched-random+BH bar
in the self-avoidance branch. This script is therefore NOT predictive: it only
asks the weak, honest question of whether the Sema/Plexin candidate genes
(plx-1, plx-2, mab-20, smp-1, smp-2) are even *expressed* in the head/neck
motor families that carry the suppressive high-contact-same-NT residual
(SMB, SIA, SIB, SAA, SMD, RME) at the wiring window, and how their expression
compares to random genes. Sparse/focal expression is EXPECTED for a genuine
subcellular-tiling gene and is exactly why class-level bulk cannot resolve it.

DO NOT read any mechanism into these numbers.
"""
import json, numpy as np, pandas as pd
from pathlib import Path

REG = Path("data/processed/regulatory_branch")
OUT = Path("data/processed/subcellular_tiling"); OUT.mkdir(parents=True, exist_ok=True)

WINS = {"E2_270_390":"dev_allgene_pb_E2_270_390.csv",
        "E3_390_510":"dev_allgene_pb_E3_390_510.csv",
        "E4_ge510":  "dev_allgene_pb_E4_ge510.csv"}

# candidate genes
SEMA = {"plx-1":"WBGene00004047","plx-2":"WBGene00004048","mab-20":"WBGene00003111",
        "smp-1":"WBGene00004889","smp-2":"WBGene00004890"}
# a few comparison synaptic-specificity genes present in the ranking
COMPARE = {"syg-1":"WBGene00006365","syg-2":"WBGene00007750","sax-3":"WBGene00004729",
           "unc-6":"WBGene00006746","unc-40":"WBGene00006776","unc-5":"WBGene00006745",
           "nrx-1":None,"nlg-1":None,"cdh-4":"WBGene00000396","fmi-1":"WBGene00001475",
           "sax-7":"WBGene00004732","lad-2":"WBGene00002243","nlr-1":"WBGene00013196"}

HEAD_MOTOR = ["SMB","SIA","SIB","SAA","SMD","RME","RMD"]  # target families (dev-matrix subset)

mats = {}
for k,f in WINS.items():
    mats[k] = pd.read_csv(REG/f).set_index("class")

allgenes = sorted(set.intersection(*[set(m.columns) for m in mats.values()]))
allclasses = sorted(set().union(*[set(m.index) for m in mats.values()]))

# mean across available windows (per class)
def meanmat():
    X = pd.DataFrame(index=allclasses, columns=allgenes, dtype=float)
    for c in allclasses:
        vals=[m.reindex(columns=allgenes).loc[c].values.astype(float)
              for m in mats.values() if c in m.index]
        X.loc[c]=np.nanmean(np.stack(vals),axis=0)
    return X.astype(float)
Xmean = meanmat()

# ---------- expression lookup table (gene x class x window) ----------
rows=[]
genes = {**SEMA, **COMPARE}
for sym,wb in genes.items():
    if wb is None or wb not in allgenes:
        rows.append(dict(gene=sym,wb=wb,note="absent_from_dev_matrix"))
        continue
    glob_mean = float(Xmean[wb].mean()); glob_max=float(Xmean[wb].max())
    detect = float((Xmean[wb]>0).mean())
    # global percentile of this gene's mean expression among all genes
    pct = float((Xmean.mean(0) < glob_mean).mean())
    for cls in HEAD_MOTOR:
        rec=dict(gene=sym,wb=wb,cls=cls,candidate=sym in SEMA,
                 global_mean_logcpm=round(glob_mean,4),
                 global_max_logcpm=round(glob_max,4),
                 detect_rate_across_classes=round(detect,3),
                 global_mean_percentile=round(pct,3))
        if cls in Xmean.index:
            rec["in_dev_matrix"]=True
            for k in WINS:
                rec[f"expr_{k}"]= round(float(mats[k].loc[cls,wb]),4) if (cls in mats[k].index and wb in mats[k].columns) else np.nan
            rec["expr_mean"]=round(float(Xmean.loc[cls,wb]),4)
            # rank of this class among all classes for this gene
            rec["class_rank_pct_for_gene"]=round(float((Xmean[wb]<Xmean.loc[cls,wb]).mean()),3)
        else:
            rec["in_dev_matrix"]=False
        rows.append(rec)
exp = pd.DataFrame(rows)
exp.to_csv(OUT/"partB2_candidate_expression_lookup.csv",index=False)

# ---------- plausibility scores ----------
# random baseline: 2000 random genes, their mean expression in head-motor classes present
present_hm=[c for c in HEAD_MOTOR if c in Xmean.index]
rng=np.random.default_rng(0)
rand_hm_mean = Xmean.loc[present_hm].mean(0)          # per-gene mean over head-motor classes
def hm_percentile(wb):
    v = float(Xmean.loc[present_hm, wb].mean())
    return float((rand_hm_mean < v).mean()), v

pl=[]
for sym,wb in genes.items():
    d=dict(gene=sym,wb=wb,candidate=sym in SEMA)
    if wb is None or wb not in allgenes:
        d["status"]="absent"; pl.append(d); continue
    hm_pct,hm_val = hm_percentile(wb)
    detect=float((Xmean[wb]>0).mean())
    glob_mean=float(Xmean[wb].mean())
    d.update(headmotor_mean_logcpm=round(hm_val,4),
             headmotor_expr_percentile_vs_allgenes=round(hm_pct,3),
             detect_rate=round(detect,3),
             global_mean_logcpm=round(glob_mean,4),
             expressed_in_any_headmotor=bool(hm_val>0.05),
             # sparse-focal flag: low bulk + detectable = consistent-with (NOT proof of) focal tiling gene
             sparse_focal_consistent=bool(0<glob_mean<1.0 and detect<0.7))
    pl.append(d)
pls=pd.DataFrame(pl)
pls.to_csv(OUT/"partB2_candidate_plausibility_scores.csv",index=False)

print("HEAD-MOTOR classes present in dev matrix:", present_hm)
print("\n== candidate sema/plexin (plausibility) ==")
print(pls[pls.candidate].to_string(index=False))
print("\n== comparison genes ==")
print(pls[~pls.candidate].to_string(index=False))
print(f"\nwrote {OUT}/partB2_candidate_expression_lookup.csv  ({len(exp)} rows)")
print(f"wrote {OUT}/partB2_candidate_plausibility_scores.csv  ({len(pls)} rows)")
