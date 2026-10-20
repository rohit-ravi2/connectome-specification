"""Feasibility: terminal-selector HOMEOBOX code -> effector-battery structure.
(1) homeobox gene list + CeNGEN coverage. (2) DECISIVE: does homeobox expression predict the
effector repertoire BETTER than a random same-size non-homeobox gene set? (the GRN-branch killer
control — all-TFs failed it). If homeobox >> random -> feasible/promising; if ~= -> well-posed but
likely negative like GRN.
"""
import json, warnings; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
RNG=np.random.default_rng(0)
CE="../data/celegans/data/expression/cengen/thresholded/021821_liberal_threshold1.csv"
PANEL="../data/celegans/data/processed/grn_branch/panel_biophysical_targets.json"

cg=pd.read_csv(CE)
classes=list(cg.columns[3:]); X=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values
gname=cg["gene_name"].astype(str).str.lower().values; nC=len(classes)

# ---- (1) homeobox set = ceh-* + canonical non-ceh homeobox genes (Reilly 2020 ~102) ----
NONCEH=["unc-4","unc-30","unc-39","unc-42","unc-62","unc-86","lin-11","lin-39","lim-4","lim-6","lim-7",
        "mec-3","mab-5","egl-5","php-3","nob-1","vab-7","vab-15","ttx-1","ttx-3","cog-1","alr-1","tab-1",
        "dsc-1","pha-2","dve-1","pros-1","zfh-2","zag-1","hmbx-1","psa-3","eyg-1","ceh-note"]
hb_rows=[i for i in range(len(gname)) if gname[i].startswith("ceh-") or gname[i] in NONCEH]
HB=X[hb_rows]  # nHB x classes
print(f"(1) homeobox genes in CeNGEN: {len(hb_rows)} (ceh-* {sum(gname[i].startswith('ceh-') for i in hb_rows)} + named {sum(gname[i] in NONCEH for i in hb_rows)})")
# expressed variably?
hb_var=((HB>0).sum(1)>=3)&((HB>0).sum(1)<=nC-3)
print(f"    homeobox genes variably expressed (3..{nC-3} classes): {int(hb_var.sum())}")
HBz=np.log1p(HB.T); HBz=(HBz-HBz.mean(0))/(HBz.std(0)+1e-9)  # classes x nHB

# ---- effector panel ----
panel=json.loads(Path(PANEL).read_text())
eff=set(g for fam in panel.values() for g in fam["genes"])
eff_rows=[i for i in range(len(gname)) if gname[i] in {g.lower() for g in eff}]
Ybin=(X[eff_rows]>0).astype(int)  # nEff x classes
print(f"    effector panel genes in CeNGEN: {len(eff_rows)}")

# ---- non-homeobox non-effector pool for random control ----
hb_set=set(hb_rows); eff_set=set(eff_rows)
pool=[i for i in range(len(gname)) if i not in hb_set and i not in eff_set and 10<=(X[i]>0).sum()<=nC-10]
print(f"    random-control gene pool (non-HB, non-effector, variable): {len(pool)}")

# ---- (2) decisive: homeobox vs random-same-size -> effector repertoire ----
skf=StratifiedKFold(5,shuffle=True,random_state=0)
def per_gene_auc(feat_classes):  # feat_classes: classes x nfeat
    aucs=[]
    for gi in range(len(eff_rows)):
        y=Ybin[gi]
        if y.sum()<6 or (nC-y.sum())<6: continue
        try:
            p=cross_val_predict(LogisticRegression(max_iter=1500,C=0.3,class_weight="balanced"),feat_classes,y,cv=skf,method="predict_proba")[:,1]
            aucs.append(roc_auc_score(y,p))
        except: pass
    return np.array(aucs)
nHB=HBz.shape[1]
a_hb=per_gene_auc(HBz)
rand=[]
for s in range(3):
    pick=np.random.default_rng(s).choice(pool,size=min(nHB,len(pool)),replace=False)
    Z=np.log1p(X[pick].T); Z=(Z-Z.mean(0))/(Z.std(0)+1e-9)
    rand.append(per_gene_auc(Z).mean())
rand=np.array(rand)
print("\n"+"="*76)
print(f"(2) homeobox ({nHB} genes)   -> effector repertoire: mean per-gene AUC = {a_hb.mean():.4f} (n={len(a_hb)})")
print(f"    random {nHB} non-HB genes -> effector repertoire: mean AUC = {rand.mean():.4f} ± {rand.std():.4f}")
print(f"    Δ homeobox - random = {a_hb.mean()-rand.mean():+.4f}")
print("="*76)
if a_hb.mean()-rand.mean()>0.03:
    print("FEASIBLE & PROMISING — homeobox carries EFFECTOR-SPECIFIC info beyond generic identity; run the full test")
elif a_hb.mean()-rand.mean()>0.01:
    print("MARGINAL — small homeobox-specific edge; full test worth running but expect a modest result")
else:
    print("WELL-POSED BUT LIKELY NEGATIVE — homeobox ~= random genes (same GRN-branch outcome: it's cell identity, not a recoverable regulatory code)")
