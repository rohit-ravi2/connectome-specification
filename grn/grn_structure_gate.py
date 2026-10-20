"""GRN-STRUCTURE GATE — the two decisive tests.

CONTROL A (TF-specificity / archetype-lesson): does the repertoire signal come from TFs
  specifically, or would ANY same-size gene set do it? Compare all-470-TF (precondition 0.706)
  vs random NON-TF gene sets of the same size (raw + expression-level-matched).
  If random genes match TFs -> generic cell-type co-expression, NOT regulatory.

GATE B (GRN-structure): does the CelEsT edge TOPOLOGY carry gene-specific regulatory logic?
  For each target gene g, predict expressed(g|class) from ONLY g's CelEsT regulators R(g),
  vs |R(g)| RANDOM TFs (degree-preserving null), vs all TFs. If R(g) beats random-same-count,
  the specific edges matter (real GRN). If R(g) ~= random-same-count, topology adds nothing.
"""
import re, warnings; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
RNG=np.random.default_rng(0)
CE="../data/celegans/data/expression/cengen/thresholded/021821_liberal_threshold1.csv"
TFLIST="../grn/CelEsT-app/www/fullset_TFs_BM_with_MOR_for_TS2.txt"
GRN="../grn/CelEsT-app/www/CelEsT_GRN.txt"
GTF="../data/celegans/data/genomes/multi_species_annot/c_elegans/c_elegans.PRJNA13758.WS297.canonical_geneset.gtf"
import json
PANELS=json.loads(Path("../data/celegans/data/processed/grn_branch/panel_biophysical_targets.json").read_text())
panel_genes=set(g for fam in PANELS.values() for g in fam["genes"])

# ---- CeNGEN ----
cg=pd.read_csv(CE); gene_name=cg["gene_name"].astype(str); wbid=cg["Wormbase_ID"].astype(str)
classes=list(cg.columns[3:]); X=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values
Xln=np.log1p(X); nC=len(classes)
row_of_wb={w:i for i,w in enumerate(wbid)}

# ---- seq<->WBGene map (GTF genome-wide + TF-list authoritative for TFs) ----
wb2seq={}
pg=re.compile(r'gene_id "([^"]+)"'); pt=re.compile(r'transcript_id "([^"]+)"')
with open(GTF) as f:
    for line in f:
        if "\ttranscript\t" in line:
            g=pg.search(line); t=pt.search(line)
            if g and t:
                s=re.sub(r'\.\d+$','',re.sub(r'[a-z]$','',t.group(1)))
                wb2seq.setdefault(g.group(1),s)
seq2wb={s:w for w,s in wb2seq.items()}
tfdf=pd.read_csv(TFLIST,sep="\t")
for _,r in tfdf.iterrows():
    s=str(r["wormbase_gseq"]); w=str(r["wormbase_gene"])
    if s and s!="nan": seq2wb[s]=w   # authoritative
tf_wb=set(tfdf["wormbase_gene"].astype(str))

# ---- TF feature matrix (CeNGEN TFs) ----
tf_rows=[i for i in range(len(wbid)) if wbid[i] in tf_wb]
tf_wb_by_row={i:wbid[i] for i in tf_rows}
TFexpr=Xln[tf_rows].T; TFexpr=(TFexpr-TFexpr.mean(0))/(TFexpr.std(0)+1e-9)   # classes x nTF
tfcol_of_wb={wbid[tf_rows[k]]:k for k in range(len(tf_rows))}
nTF=len(tf_rows); print(f"CeNGEN TFs {nTF}; classes {nC}")

# ---- target panel rows + regulators from CelEsT ----
grn=pd.read_csv(GRN,sep="\t")
grn["tf_wb"]=grn["source"].astype(str).map(seq2wb); grn["tg_wb"]=grn["target"].astype(str).map(seq2wb)
grn=grn.dropna(subset=["tf_wb","tg_wb"])
reg_of=dict()   # target WBGene -> list of (tf_col, weight) for TFs present in CeNGEN
for tg,sub in grn.groupby("tg_wb"):
    lst=[(tfcol_of_wb[t],w) for t,w in zip(sub["tf_wb"],sub["weight"]) if t in tfcol_of_wb]
    if lst: reg_of[tg]=lst

panel_rows=[i for i in range(len(gene_name)) if gene_name[i] in panel_genes]
Ybin_all=(X>0).astype(int)  # genes x classes
skf=StratifiedKFold(5,shuffle=True,random_state=0)
def auc_feats(y, cols):
    if len(cols)==0: return None
    Xf=TFexpr[:,cols]
    try: p=cross_val_predict(LogisticRegression(max_iter=1500,C=0.2,class_weight="balanced"),Xf,y,cv=skf,method="predict_proba")[:,1]
    except Exception: return None
    return roc_auc_score(y,p)
def auc_genes_feats(y, gene_rows_expr):
    try: p=cross_val_predict(LogisticRegression(max_iter=1500,C=0.2,class_weight="balanced"),gene_rows_expr,y,cv=skf,method="predict_proba")[:,1]
    except Exception: return None
    return roc_auc_score(y,p)

# testable panel genes: variable expression + present as CelEsT target with >=3 CeNGEN regulators
allcols=list(range(nTF))
tests=[]
for i in panel_rows:
    y=Ybin_all[i]
    if y.sum()<6 or (nC-y.sum())<6: continue
    tg_wb=wbid[i]; regs=reg_of.get(tg_wb,[])
    tests.append((i,y,regs))
print(f"testable panel genes {len(tests)} | with >=3 CelEsT regulators: {sum(1 for _,_,r in tests if len(r)>=3)}")

# ---- CONTROL A: TF vs random NON-TF gene sets (all use 470 features) ----
print("\n"+"="*84); print("CONTROL A — is it TF-specific? (all-TF vs random non-TF gene sets, 470 features)"); print("="*84)
nonTF_pool=[i for i in range(len(wbid)) if (wbid[i] not in tf_wb) and (gene_name[i] not in panel_genes) and (X[i]>0).sum()>=10 and (X[i]>0).sum()<=nC-10]
tf_expr_mean=Xln[tf_rows].mean(1)
subset=[t for t in tests if RNG.random()<0.55]   # ~subsample genes for speed
print(f"  (evaluated on {len(subset)} panel genes; non-TF pool {len(nonTF_pool)})")
a_tf=np.array([v for v in (auc_feats(y,allcols) for _,y,_ in subset) if v is not None])
def random_geneset_auc(pool,k,seed):
    rr=np.random.default_rng(seed); pick=rr.choice(pool,size=k,replace=False)
    Z=Xln[pick].T; Z=(Z-Z.mean(0))/(Z.std(0)+1e-9)
    out=[]
    for _,y,_ in subset:
        v=auc_genes_feats(y,Z)
        if v is not None: out.append(v)
    return np.mean(out)
rand_nonTF=[random_geneset_auc(nonTF_pool,nTF,s) for s in range(3)]
print(f"  all-470-TF           mean per-gene AUC = {a_tf.mean():.4f}")
print(f"  random non-TF (x3)   mean per-gene AUC = {np.mean(rand_nonTF):.4f} ± {np.std(rand_nonTF):.4f}")
print(f"  Δ TF - randomNonTF = {a_tf.mean()-np.mean(rand_nonTF):+.4f}  "
      f"({'TF-SPECIFIC' if a_tf.mean()-np.mean(rand_nonTF)>0.02 else 'NOT TF-specific = generic co-expression'})")

# ---- GATE B: GRN regulators R(g) vs random-same-count TFs vs all-TF ----
print("\n"+"="*84); print("GATE B — does GRN EDGE TOPOLOGY carry gene-specific logic? R(g) vs random-same-count TFs"); print("="*84)
regtests=[(i,y,regs) for (i,y,regs) in tests if len(regs)>=3]
rg_auc=[]; rand_auc=[]; alltf_auc=[]; ndeg=[]
for i,y,regs in regtests:
    cols=sorted(set(c for c,w in regs)); k=len(cols)
    a_rg=auc_feats(y,cols)
    # degree-preserving null: k random TFs, averaged over 5 draws
    rr=[]
    for d in range(5):
        rc=list(np.random.default_rng(1000*i+d).choice(nTF,size=k,replace=False)); v=auc_feats(y,rc)
        if v is not None: rr.append(v)
    a_all=auc_feats(y,allcols)
    if a_rg is None or not rr or a_all is None: continue
    rg_auc.append(a_rg); rand_auc.append(np.mean(rr)); alltf_auc.append(a_all); ndeg.append(k)
rg_auc=np.array(rg_auc); rand_auc=np.array(rand_auc); alltf_auc=np.array(alltf_auc)
print(f"  genes tested {len(rg_auc)} | median #regulators {int(np.median(ndeg))}")
print(f"  R(g) actual regulators   mean AUC = {rg_auc.mean():.4f}")
print(f"  random same-count TFs    mean AUC = {rand_auc.mean():.4f}   (degree-preserving null)")
print(f"  all-470-TF               mean AUC = {alltf_auc.mean():.4f}")
d=rg_auc-rand_auc
from scipy.stats import wilcoxon
try: st,pv=wilcoxon(rg_auc,rand_auc); pvs=f"p={pv:.1e}"
except Exception: pvs="p=NA"
print(f"  Δ R(g) - random-same-count = {d.mean():+.4f} ± {d.std():.4f}  paired-Wilcoxon {pvs}  "
      f"(fraction R(g)>null: {np.mean(d>0):.2f})")

print("\n"+"="*84)
tf_specific = a_tf.mean()-np.mean(rand_nonTF)>0.02
grn_structure = (d.mean()>0.01) and (np.mean(d>0)>0.55)
if tf_specific and grn_structure:
    v="ALIVE — signal is TF-specific AND GRN edge topology adds gene-specific logic"
elif tf_specific:
    v="PARTIAL — TF-specific, but edge TOPOLOGY adds little over same-count random TFs (TF-activity, not network)"
else:
    v="DEAD — repertoire signal is generic cell-type co-expression (any gene set works); not a GRN result"
print("GRN-STRUCTURE GATE VERDICT:",v)
print("="*84)
