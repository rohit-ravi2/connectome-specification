"""GRN branch — Phase 1 (target panels) + Phase 2 (CeNGEN alignment) + PRECONDITION GATE.

Precondition (must pass before any GRN work): does TF EXPRESSION predict held-out neuron-class
channel/receptor REPERTOIRE above expression-breadth / prevalence baselines and a shuffled-TF null?
If TF expression carries no gene-specific repertoire signal, no GRN built on TF activity can help
-> branch dies here.  (Uses TF EXPRESSION only; CelEsT GRN edges enter in the NEXT phase.)

Target = binary expressed(channel/receptor gene g, neuron class c) in CeNGEN.
Held-out regimes: (I) held-out NEURON CLASSES (5-fold over classes), (II) held-out GENES.
Baselines: gene prevalence; class expression-breadth. Null: shuffled TF expression across classes.
"""
import json, warnings; warnings.simplefilter("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score
RNG=np.random.default_rng(0)
CE="../data/celegans/data/expression/cengen/thresholded/021821_liberal_threshold1.csv"
TFLIST="../grn/CelEsT-app/www/fullset_TFs_BM_with_MOR_for_TS2.txt"
OUT=Path("../data/celegans/data/processed/grn_branch"); OUT.mkdir(parents=True,exist_ok=True)

# ---------------- Phase 1: biophysical target panels (public-name families) ----------------
PANELS={
 "K_channel":["twk-","kcnl-","egl-2","egl-36","egl-23","slo-1","slo-2","shk-1","shl-1","kqt-","exp-2","unc-103","unc-58","sup-9","kvs-","mps-1","irk-","ncs-"],
 "Ca_channel":["egl-19","unc-2","cca-1","unc-36","ccb-","ccle-"],
 "Na_leak_NALCN":["nca-1","nca-2","unc-77","unc-79","unc-80"],
 "Cl_channel":["clh-","best-","anoh-"],
 "TRP":["trp-","trpa-1","osm-9","ocr-","gtl-","pkd-2","cup-5"],
 "DEG_ENaC":["mec-4","mec-10","deg-1","unc-8","del-","asic-","acd-","egas-","flr-1","unc-105"],
 "CNG":["tax-2","tax-4","cng-","che-6"],
 "LGIC_receptor":["acr-","unc-29","unc-38","unc-63","lev-1","lev-8","deg-3","des-2","eat-2","glr-","nmr-","avr-","glc-","unc-49","gab-1","lgc-","ggr-","exp-1","mod-1","pbo-5","pbo-6"],
 "GPCR_metabotropic_receptor":["mgl-","gar-","dop-","ser-","tyra-","octr-","npr-","frpr-","dmsr-","ckr-","ntr-","aex-2","egl-6","gnrr-","seb-","trhr-","pdfr-1","nmur-"],
 "innexin_gap":["inx-","unc-7","unc-9","eat-5","che-7"],
 "NT_transporter_machinery":["unc-17","cha-1","cho-1","eat-4","unc-47","unc-46","cat-1","dat-1","mod-5","snf-11","tph-1","cat-2","tbh-1","tdc-1","bas-1","unc-25"],
}
cg=pd.read_csv(CE)
gene_name=cg["gene_name"].astype(str); gene_lc=gene_name.str.lower()
wbid=cg["Wormbase_ID"].astype(str)
classes=list(cg.columns[3:]); X=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values  # genes x classes
def match(prefixes):
    hit=set()
    for k,g in enumerate(gene_lc):
        for p in prefixes:
            if g==p or g.startswith(p): hit.add(k); break
    return sorted(hit)
panel_idx={}; panel_json={}
for fam,pre in PANELS.items():
    idxs=match(pre); panel_idx[fam]=idxs
    panel_json[fam]={"prefixes":pre,"genes":[gene_name[i] for i in idxs],"n":len(idxs)}
target_idx=sorted(set(i for v in panel_idx.values() for i in v))
(OUT/"panel_biophysical_targets.json").write_text(json.dumps(panel_json,indent=1))
print("PHASE 1 — target panels (CeNGEN public-name families):")
for fam,idxs in panel_idx.items(): print(f"  {fam:28s} {len(idxs):3d} genes")
print(f"  TOTAL distinct target genes: {len(target_idx)}  | saved {OUT/'panel_biophysical_targets.json'}")

# ---------------- Phase 2: alignment (class × TF-expr, class × target binary) ----------------
tfdf=pd.read_csv(TFLIST,sep="\t"); tf_wb=set(tfdf["wormbase_gene"].astype(str))
tf_idx=[k for k in range(len(wbid)) if wbid[k] in tf_wb]
# exclude any target genes that are themselves TFs from the TF feature set is unnecessary (targets are channels)
Xln=np.log1p(X)  # genes x classes
TFexpr=Xln[tf_idx].T                      # classes x nTF
TFexpr=(TFexpr-TFexpr.mean(0))/(TFexpr.std(0)+1e-9)
Ybin=(X[target_idx]>0).astype(int).T      # classes x nTarget  (expressed/not)
Tgt_names=[gene_name[i] for i in target_idx]
fam_of={gene_name[i]:fam for fam,idxs in panel_idx.items() for i in idxs}
nC=len(classes)
print(f"\nPHASE 2 — alignment: {nC} neuron classes | TFs in CeNGEN {len(tf_idx)}/{len(tf_wb)} | target genes {len(target_idx)}")
print(f"  target expression: overall expressed-rate {Ybin.mean():.3f} | per-class breadth mean {Ybin.sum(1).mean():.1f}/{len(target_idx)}")
np.savez(OUT/"aligned.npz", TFexpr=TFexpr, Ybin=Ybin, classes=classes, tgt_names=Tgt_names)

# ---------------- PRECONDITION GATE ----------------
breadth=Ybin.sum(1).astype(float)         # class breadth
prevalence=Ybin.mean(0)                   # per-gene prevalence
def per_gene_auc(predfn, genes):
    aucs=[]
    for gi in genes:
        y=Ybin[:,gi]
        if y.sum()<6 or (nC-y.sum())<6: continue
        p=predfn(gi,y)
        if p is None: continue
        aucs.append(roc_auc_score(y,p))
    return np.array(aucs)
testable=[gi for gi in range(len(target_idx)) if 6<=Ybin[:,gi].sum()<=nC-6]
print(f"\nPRECONDITION GATE — per-gene AUC across neuron classes (held-out CLASSES, 5-fold); {len(testable)} testable genes")
skf=StratifiedKFold(5,shuffle=True,random_state=0)
def tf_pred(gi,y):
    try: return cross_val_predict(LogisticRegression(max_iter=2000,C=0.2,class_weight="balanced"),TFexpr,y,cv=skf,method="predict_proba")[:,1]
    except Exception: return None
def breadth_pred(gi,y):
    # class breadth as sole predictor (broad classes express more of everything)
    return breadth
def tf_null_pred_factory(perm):
    TFp=TFexpr[perm]
    def f(gi,y):
        try: return cross_val_predict(LogisticRegression(max_iter=2000,C=0.2,class_weight="balanced"),TFp,y,cv=skf,method="predict_proba")[:,1]
        except Exception: return None
    return f
a_tf=per_gene_auc(tf_pred,testable)
a_breadth=per_gene_auc(breadth_pred,testable)
# shuffled-TF null (3 perms for speed)
null_means=[]
for s in range(3):
    perm=RNG.permutation(nC)
    null_means.append(np.nanmean(per_gene_auc(tf_null_pred_factory(perm),testable)))
null_means=np.array(null_means)
print(f"  TF-expression model : mean per-gene AUC = {a_tf.mean():.4f}  (median {np.median(a_tf):.3f}, n={len(a_tf)})")
print(f"  class-breadth base  : mean per-gene AUC = {a_breadth.mean():.4f}")
print(f"  shuffled-TF null    : mean per-gene AUC = {null_means.mean():.4f} ± {null_means.std():.4f}")
print(f"  Δ TF over breadth = {a_tf.mean()-a_breadth.mean():+.4f} | Δ TF over shuffled-null = {a_tf.mean()-null_means.mean():+.4f}")

# per-family breakdown
print("\n  per-family TF-model mean AUC:")
for fam in PANELS:
    gg=[gi for gi in testable if fam_of.get(Tgt_names[gi])==fam]
    if len(gg)<3: continue
    fa=per_gene_auc(tf_pred,gg)
    print(f"    {fam:28s} {fa.mean():.3f}  (n={len(fa)})")

# ---------------- Regime II: held-out GENES (per-neuron-class repertoire AUC) ----------------
print("\nPRECONDITION GATE II — per-CLASS repertoire AUC (does TF profile pick which genes a class expresses?)")
# leave-one-gene-family-out style: predict each gene from TF via the per-gene models above, then per class rank
P=np.full_like(Ybin,np.nan,dtype=float)
for gi in testable:
    y=Ybin[:,gi]; P[:,gi]=tf_pred(gi,y)
valid=np.array(testable)
class_aucs=[]; class_base=[]
for c in range(nC):
    yc=Ybin[c,valid]; pc=P[c,valid]
    if yc.sum()<3 or (len(valid)-yc.sum())<3: continue
    class_aucs.append(roc_auc_score(yc,pc))
    class_base.append(roc_auc_score(yc,prevalence[valid]))   # global prevalence ranking
print(f"  per-class repertoire AUC: TF-model {np.mean(class_aucs):.4f} vs prevalence-ranking {np.mean(class_base):.4f} "
      f"(Δ={np.mean(class_aucs)-np.mean(class_base):+.4f}, n={len(class_aucs)} classes)")

print("\n"+"="*80)
alive=(a_tf.mean()-a_breadth.mean()>0.02) and (a_tf.mean()-null_means.mean()>0.03) and (a_tf.mean()>0.6)
print("PRECONDITION VERDICT:", "TF-EXPRESSION CARRIES REPERTOIRE SIGNAL -> proceed to GRN gate" if alive
      else "WEAK/NO TF-expression repertoire signal -> GRN unlikely to help; branch at risk")
print("="*80)
