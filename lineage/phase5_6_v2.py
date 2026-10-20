"""Phase 5+6 RE-RUN with proper timing = early/late born (Table S4, 201 early / 78 late).
Fixes the bad depth proxy (r=0.14 w/ embryo.time). Tests whether lineage + REAL birth-timing
predicts developmental arrival (Phase 5) and contact opportunity (Phase 6).
"""
import warnings, re; warnings.simplefilter("ignore")
import numpy as np, pandas as pd, openpyxl
from collections import defaultdict
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
RNG=np.random.default_rng(0); D="../data/celegans/data"

lf=pd.read_csv(D+"/processed/developmental_rules/lineage_features.csv")
paths={r.neuron:r.lineage_path.split("/") for r in lf.itertuples()}
founder={r.neuron:r.founder for r in lf.itertuples()}
bt=pd.read_csv(D+"/processed/developmental_rules/birthtime_earlylate.csv")
born={r.neuron:(1 if r.born=="early" else 0) for r in bt.itertuples()}   # 1=early(embryonic)
def td(a,b):
    pa,pb=paths[a],paths[b]; k=0
    while k<len(pa) and k<len(pb) and pa[k]==pb[k]: k+=1
    return (len(pa)-k)+(len(pb)-k)

# ---- sanity: does born-category track modality? (late=ventral-cord motor) ----
master=pd.read_csv(D+"/processed/developmental_rules/neuron_development_master.csv").set_index("neuron")
common=[n for n in paths if n in born]
mod={n:master.loc[n,"modality"] for n in common if n in master.index}
lm=[n for n in common if n in mod]
mot=np.array([1 if mod[n]=="motor" else 0 for n in lm]); bn=np.array([born[n] for n in lm])
print(f"timing sanity: late-born motor-rate {mot[bn==0].mean():.2f} vs early-born {mot[bn==1].mean():.2f} (late-born enriched for motor as expected)")

# ---- contact matrix ----
wb=openpyxl.load_workbook(D+"/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx",read_only=True,data_only=True)
ws=wb['adult nerve ring neighbors']; data=list(ws.iter_rows(values_only=True)); hdr=[h for h in data[0][1:]]
ring=[n for n in hdr if n in paths and n in born]
A=defaultdict(float)
for row in data[1:]:
    rn=row[0]
    if rn not in paths: continue
    for j,val in enumerate(row[1:]):
        cn=hdr[j]
        if cn in paths and val not in (None,'') and rn!=cn:
            try: A[(min(rn,cn),max(rn,cn))]=max(A[(min(rn,cn),max(rn,cn))],float(val))
            except: pass
cdeg=defaultdict(int)
for (a,b),v in A.items():
    if v>0: cdeg[a]+=1; cdeg[b]+=1

skf=StratifiedKFold(5,shuffle=True,random_state=0)
def auc(X,y): return roc_auc_score(y,cross_val_predict(LogisticRegression(max_iter=2000,class_weight="balanced"),X,y,cv=skf,method="predict_proba")[:,1])

# ================= PHASE 6 — contact opportunity from lineage + born-timing =================
print("\n"+"="*84); print("PHASE 6 — contact from lineage + real timing (early/late born)"); print("="*84)
pairs=[(ring[i],ring[j]) for i in range(len(ring)) for j in range(i+1,len(ring))]
y=np.array([1 if A[(min(a,b),max(a,b))]>0 else 0 for a,b in pairs])
TD=np.array([td(a,b) for a,b in pairs],float)
bothearly=np.array([1.0 if born[a]==1 and born[b]==1 else 0.0 for a,b in pairs])
diffborn=np.array([abs(born[a]-born[b]) for a,b in pairs],float)
sf=np.array([1.0 if founder[a]==founder[b] else 0.0 for a,b in pairs])
degp=np.array([np.log1p(cdeg[a])+np.log1p(cdeg[b]) for a,b in pairs])
X_lt=np.column_stack([TD,bothearly,diffborn,sf])
print(f"  pairs {len(y)} contact-rate {y.mean():.3f}")
print(f"  AUC lineage+timing {auc(X_lt,y):.4f} | degree-only {auc(degp.reshape(-1,1),y):.4f} | "
      f"lin+timing+degree {auc(np.column_stack([X_lt,degp]),y):.4f}")
print(f"  same-founder contact-rate {y[sf==1].mean():.3f} vs diff-founder {y[sf==0].mean():.3f} | "
      f"both-early {y[bothearly==1].mean():.3f} vs has-late {y[bothearly==0].mean():.3f}")

# ================= PHASE 5 — developmental arrival from lineage + born-timing =================
print("\n"+"="*84); print("PHASE 5 — developmental ARRIVAL from lineage + real timing"); print("="*84)
pres=defaultdict(set)
for r in pd.read_csv(D+"/connectome/witvliet2020/witvliet_combined_edges.csv").itertuples():
    ds=int(re.search(r'witvliet_2020_(\d)',r.source_file).group(1))
    if r.pre in paths and r.post in paths: pres[(min(r.pre,r.post),max(r.pre,r.post))].add(ds)
cand=[(a,b) for (a,b),v in A.items() if v>0 and 7 in pres.get((a,b),set()) and 8 in pres.get((a,b),set()) and a in born and b in born]
dev=np.array([sum(1 for d in (1,2,3,4,5,6) if d in pres.get(p,set())) for p in cand])   # L1-L3 arrival count
inL1=np.array([1 if any(d in pres.get(p,set()) for d in (1,2,3,4)) else 0 for p in cand])  # present in L1
has_late=np.array([0 if (born[a]==1 and born[b]==1) else 1 for a,b in cand])
TDc=np.array([td(a,b) for a,b in cand],float)
print(f"  reproducible pairs {len(cand)} | present-in-L1 rate {inL1.mean():.3f}")
# does born-timing predict arrival? both-early pairs present in L1 vs has-late
print(f"  present-in-L1: both-early pairs {inL1[has_late==0].mean():.3f} vs has-late-born pair {inL1[has_late==1].mean():.3f}")
print(f"  dev-arrival (L1-L3 count): both-early {dev[has_late==0].mean():.2f} vs has-late {dev[has_late==1].mean():.2f} "
      f"(MannWhitney p={mannwhitneyu(dev[has_late==0],dev[has_late==1]).pvalue:.1e})")
Xa=np.column_stack([has_late,TDc])
print(f"  AUC(lineage+born -> present-in-L1) = {auc(Xa,inL1):.4f} | born-only {auc(has_late.reshape(-1,1),inL1):.4f}")

print("\n"+"="*84)
print("PHASE 5/6 (real timing) SUMMARY: see numbers above — key = does born-timing predict arrival + contact")
print("="*84)
