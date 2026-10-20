#!/usr/bin/env python
"""B2.3 suppressive-homophily direction, B2.4 head-motor specificity,
B2.5 dev-vs-adult, + candidate gene ranking."""
import json, re, warnings
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

ROOT="../seed"; OUT=f"{ROOT}/data/processed/self_avoidance_apposition"; REG=f"{ROOT}/data/processed/regulatory_branch"

wins=["dev_allgene_pb_E2_270_390.csv","dev_allgene_pb_E3_390_510.csv","dev_allgene_pb_E4_ge510.csv"]
mats=[pd.read_csv(f"{REG}/{w}").set_index("class") for w in wins]
genes_common=sorted(set.intersection(*[set(m.columns) for m in mats]))
allclasses=sorted(set().union(*[set(m.index) for m in mats]))
X=pd.DataFrame(index=allclasses,columns=genes_common,dtype=float)
for c in allclasses:
    vals=[m.reindex(columns=genes_common).loc[c].values.astype(float) for m in mats if c in m.index]
    X.loc[c]=np.nanmean(np.stack(vals),axis=0)
X=X.astype(float)

gs=json.load(open(f"{REG}/gene_sets_effector.json")); sym2wb={}
for cat,info in gs["categories"].items():
    for wb,g in info["genes"].items(): sym2wb[g["symbol"]]=wb
sym2wb["nlr-1"]="WBGene00013196"
wb2sym={v:k for k,v in sym2wb.items()}

cdh=[s for s in sym2wb if s.startswith("cdh-")]; zig=[s for s in sym2wb if s.startswith("zig-")]
MOD_SYM={"igsf":["rig-1","rig-3","rig-4","rig-5","rig-6"]+zig+["oig-1","wrk-1","sax-7","lad-2","dig-1","nlr-1"],
 "robo_slit":["sax-3","slt-1","eva-1"],"netrin":["unc-6","unc-40","unc-5"],
 "sema_plexin":["smp-1","smp-2","plx-1","plx-2","mab-20"],"cadherin":cdh+["hmr-1","fmi-1","casy-1"],
 "nrx_nlg":["nrx-1","nlg-1"],"syg":["syg-1","syg-2"]}
MOD_SYM["self_avoid_core"]=MOD_SYM["igsf"]+MOD_SYM["robo_slit"]
def wbs(sl): return sorted({sym2wb[s] for s in sl if sym2wb.get(s) in X.columns})
MODULES={k:wbs(v) for k,v in MOD_SYM.items()}
ALL_CAND=sorted({sym2wb[s] for s in set().union(*MOD_SYM.values()) if sym2wb.get(s) in X.columns})

CLASSES=set(X.index)
def toclass(n):
    for pat in [n,re.sub(r"[LR]$","",n),re.sub(r"[DV][LR]$","",n),re.sub(r"[DV]$","",n),re.sub(r"[0-9]+$","",n),re.sub(r"[0-9]+[LR]?$","",n)]:
        if pat in CLASSES: return pat
    return None
m=pd.read_csv(f"{OUT}/high_contact_residual_master.csv")
m["ca"]=m.na.map(toclass); m["cb"]=m.nb.map(toclass)

def mscore(genes):
    s=X[genes].mean(1); return (s-s.mean())/(s.std()+1e-9)

# ---------------- B2.3 suppressive homophily direction ----------------
b23=[]
for hc in ["hc_top20","hc_top10"]:
    d=m[(m[hc]==1)&m.ca.notna()&m.cb.notna()].copy(); y=d.nonsyn.astype(int).values
    for mod,genes in MODULES.items():
        ms=mscore(genes); za=ms.reindex(d.ca).values; zb=ms.reindex(d.cb).values
        feats={"bothhigh":za*zb,"min":np.minimum(za,zb),"absdiff":np.abs(za-zb)}
        Va=X.loc[d.ca.values,genes].values; Vb=X.loc[d.cb.values,genes].values
        cos=(Va*Vb).sum(1)/((np.linalg.norm(Va,1 if False else 2,axis=1)*np.linalg.norm(Vb,2,axis=1))+1e-9)
        feats["cos"]=cos
        for fn,fv in feats.items():
            g1=fv[y==1]; g0=fv[y==0]
            try: U,p=mannwhitneyu(g1,g0,alternative="two-sided")
            except Exception: p=np.nan
            auc=roc_auc_score(y,fv) if len(np.unique(y))>1 else np.nan
            b23.append(dict(hc=hc,module=mod,feature=fn,mean_nonsyn=np.nanmean(g1),
                mean_syn=np.nanmean(g0),auc_feature=auc,mwu_p=p,
                direction=("similarity-enriched" if np.nanmean(g1)>np.nanmean(g0) else "complementarity-enriched")))
B23=pd.DataFrame(b23); B23.to_csv(f"{OUT}/partB2_suppressive_homophily.csv",index=False)
print("== B2.3 suppressive homophily (self_avoid_core & igsf, bothhigh/min) ==")
print(B23[(B23.module.isin(["self_avoid_core","igsf","robo_slit"]))&(B23.feature.isin(["bothhigh","min"]))].to_string(index=False))

# ---------------- B2.4 head-motor family specificity ----------------
HEADMOTOR=[c for c in ["SMB","SIA","SIB","SMD","RME","SAA","RMD","RIM","RIV"] if c in CLASSES]
print(f"\n== B2.4 head-motor classes present: {HEADMOTOR} ==")
rest=[c for c in X.index if c not in HEADMOTOR]
b24=[]
for wb in ALL_CAND:
    hm=X.loc[HEADMOTOR,wb].values; ot=X.loc[rest,wb].values
    lfc=np.mean(hm)-np.mean(ot)
    try: t,p=ttest_ind(hm,ot,equal_var=False)
    except Exception: t,p=np.nan,np.nan
    b24.append(dict(gene=wb2sym.get(wb,wb),wb=wb,mean_headmotor=np.mean(hm),
        mean_other=np.mean(ot),logfc=lfc,t=t,p=p))
B24=pd.DataFrame(b24).sort_values("logfc",ascending=False)
print(B24.head(12).to_string(index=False))

# ---------------- B2.5 dev-vs-adult module signal ----------------
# adult per-gene CeNGEN not on disk; use adult_module_scores adhesion col as coarse proxy
adult=pd.read_csv(f"{REG}/adult_module_scores.csv").set_index("class")
b25=[]
# per-window module score corr with the wiring-window mean (are candidate modules
# even expressed / stable across dev windows?)
for mod,genes in MODULES.items():
    row={"module":mod}
    for w,label in zip(mats,["E2","E3","E4"]):
        g=[x for x in genes if x in w.columns]
        if len(g)==0: row[label+"_mean"]=np.nan; continue
        row[label+"_mean"]=w[g].mean(1).mean()
    # adult adhesion module (only for adhesion-type)
    row["adult_adhesion_mod_mean"]=adult["adhesion_recognition"].mean()
    b25.append(row)
B25=pd.DataFrame(b25); B25.to_csv(f"{OUT}/partB2_dev_vs_adult.csv",index=False)
print("\n== B2.5 module mean expr across dev windows ==")
print(B25.to_string(index=False))

# ---------------- candidate gene ranking (combined) ----------------
# per-gene univariate: does gene's bothhigh feature associate with nonsyn (hc_top20)?
d=m[(m.hc_top20==1)&m.ca.notna()&m.cb.notna()].copy(); y=d.nonsyn.astype(int).values
rankrows=[]
for wb in ALL_CAND:
    s=X[wb]; z=(s-s.mean())/(s.std()+1e-9)
    za=z.reindex(d.ca).values; zb=z.reindex(d.cb).values
    bh=za*zb; mn=np.minimum(za,zb)
    auc_bh=roc_auc_score(y,bh) if len(np.unique(y))>1 else np.nan
    auc_mn=roc_auc_score(y,mn) if len(np.unique(y))>1 else np.nan
    hm_lfc=B24.set_index("wb").loc[wb,"logfc"]
    rankrows.append(dict(gene=wb2sym.get(wb,wb),wb=wb,
        module=[k for k,v in MODULES.items() if wb in v and k!="self_avoid_core"],
        auc_bothhigh=auc_bh,auc_min=auc_mn,headmotor_logfc=hm_lfc))
RANK=pd.DataFrame(rankrows)
RANK["max_auc"]=RANK[["auc_bothhigh","auc_min"]].max(1)
RANK=RANK.sort_values("max_auc",ascending=False)
RANK.to_csv(f"{OUT}/partB2_candidate_gene_ranking.csv",index=False)
print("\n== candidate gene ranking (top 15 by max univariate bothhigh/min AUC for nonsyn, hc_top20) ==")
print(RANK.head(15).to_string(index=False))
print("\nNOTE: AUC~0.5 = no association. Random baseline AUC spread expected +-0.05.")
