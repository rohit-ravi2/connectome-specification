"""Phase 5 (developmental-arrival) + Phase 6 (contact-geometry) prediction from lineage/timing.
Uses real Sulston tree (lineage_features.csv) -> tree-distance + depth(timing proxy) + founder.
KEY QUESTIONS: (6) does lineage/timing predict CONTACT OPPORTUNITY (before synapse selection)?
(5) does lineage/timing predict developmental ARRIVAL (which absorbed the NT-pairwise signal)?
Controls: permute lineage labels; test lineage-distance BEYOND degree.
"""
import json, warnings, re; warnings.simplefilter("ignore")
import numpy as np, pandas as pd, openpyxl
from collections import defaultdict
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
RNG=np.random.default_rng(0)
D="../data/celegans/data"
CONTACT=D+"/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx"
EDGES=D+"/connectome/witvliet2020/witvliet_combined_edges.csv"

# ---- lineage features + tree distance ----
lf=pd.read_csv(D+"/processed/developmental_rules/lineage_features.csv")
paths={r.neuron:r.lineage_path.split("/") for r in lf.itertuples()}
depth={r.neuron:r.depth for r in lf.itertuples()}
founder={r.neuron:r.founder for r in lf.itertuples()}
def treedist(a,b):
    pa,pb=paths[a],paths[b]; k=0
    while k<len(pa) and k<len(pb) and pa[k]==pb[k]: k+=1
    return (len(pa)-k)+(len(pb)-k)

# ---- validate depth as timing proxy vs Packer embryo.time ----
pk=pd.read_csv(D+"/expression/scrna_seq_other/GSE126954/GSE126954_cell_annotation.csv",usecols=["cell.subtype","embryo.time"])
pk=pk[pk["cell.subtype"].notna()]; bt=pk.groupby("cell.subtype")["embryo.time"].median().to_dict()
def bclass(n):
    for s in "LRDV":
        if n.endswith(s): return n[:-1]
    return n
cov=[(depth[n],bt.get(n,bt.get(bclass(n)))) for n in paths if bt.get(n,bt.get(bclass(n))) is not None]
cov=[(d,t) for d,t in cov if t==t]
if cov:
    dd,tt=zip(*cov); rho,_=spearmanr(dd,tt)
    print(f"timing proxy: Spearman(lineage depth, Packer embryo.time) = {rho:.3f} (n={len(cov)}) -> depth used as birth-time proxy")

# ---- contact matrix ----
wb=openpyxl.load_workbook(CONTACT,read_only=True,data_only=True); ws=wb['adult nerve ring neighbors']
data=list(ws.iter_rows(values_only=True)); hdr=[h for h in data[0][1:]]
ring=[n for n in hdr if n in paths]           # nerve-ring neurons with lineage
A=defaultdict(float)
for row in data[1:]:
    rn=row[0]
    if rn not in paths: continue
    for j,val in enumerate(row[1:]):
        cn=hdr[j]
        if cn in paths and val not in (None,'') and rn!=cn:
            try: A[(min(rn,cn),max(rn,cn))]=max(A[(min(rn,cn),max(rn,cn))],float(val))
            except: pass
cdeg=defaultdict(int); carea=defaultdict(float)
for (a,b),v in A.items():
    if v>0: cdeg[a]+=1;cdeg[b]+=1;carea[a]+=v;carea[b]+=v
print(f"nerve-ring neurons (lineage-covered) {len(ring)} | contacting pairs {sum(1 for v in A.values() if v>0)}")

# ================= PHASE 6 — does lineage/timing predict CONTACT OPPORTUNITY? =================
print("\n"+"="*84); print("PHASE 6 — contact opportunity from lineage/timing"); print("="*84)
# pairwise: all pairs among ring neurons; target = contact>0
pairs=[(ring[i],ring[j]) for i in range(len(ring)) for j in range(i+1,len(ring))]
y=np.array([1 if A[(min(a,b),max(a,b))]>0 else 0 for a,b in pairs])
td=np.array([treedist(a,b) for a,b in pairs],float)
dd=np.array([abs(depth[a]-depth[b]) for a,b in pairs],float)
sf=np.array([1.0 if founder[a]==founder[b] else 0.0 for a,b in pairs])
degprod=np.array([np.log1p(cdeg[a])+np.log1p(cdeg[b]) for a,b in pairs])
skf=StratifiedKFold(5,shuffle=True,random_state=0)
def auc(X):
    p=cross_val_predict(LogisticRegression(max_iter=2000,class_weight="balanced"),X,y,cv=skf,method="predict_proba")[:,1]
    return roc_auc_score(y,p)
print(f"  pairs {len(y)} | contact rate {y.mean():.3f}")
X_deg=degprod.reshape(-1,1); X_lin=np.column_stack([td,dd,sf]); X_both=np.column_stack([td,dd,sf,degprod])
a_lin=auc(X_lin); a_deg=auc(X_deg); a_both=auc(X_both)
# lineage-distance alone spearman + permutation
rho_c,_=spearmanr(-td,y)   # closer lineage (smaller td) -> more contact?
perm=[]
for _ in range(15):
    pl=RNG.permutation(len(ring)); pm={ring[i]:ring[pl[i]] for i in range(len(ring))}
    tdp=np.array([treedist(pm[a],pm[b]) for a,b in pairs],float)
    perm.append(roc_auc_score(y,cross_val_predict(LogisticRegression(max_iter=1500,class_weight="balanced"),
                 np.column_stack([tdp,dd,sf]),y,cv=skf,method="predict_proba")[:,1]))
perm=np.array(perm)
print(f"  AUC lineage/timing {a_lin:.4f} | degree-only {a_deg:.4f} | lineage+degree {a_both:.4f} (Δlin over deg {a_both-a_deg:+.4f})")
print(f"  lineage perm-null AUC {perm.mean():.4f}±{perm.std():.4f} z={(a_lin-perm.mean())/(perm.std()+1e-9):+.1f} | Spearman(-treedist,contact)={rho_c:+.3f}")
# neuron-level: contact degree from founder/depth
ndf=pd.DataFrame({"n":ring,"founder":[founder[n] for n in ring],"depth":[depth[n] for n in ring],
                  "cdeg":[cdeg[n] for n in ring],"carea":[carea[n] for n in ring]})
from sklearn.preprocessing import OneHotEncoder
Xn=np.column_stack([OneHotEncoder(sparse_output=False,handle_unknown='ignore').fit_transform(ndf[["founder"]]),ndf["depth"]])
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict as cvp
pdeg=cvp(Ridge(),Xn,ndf["cdeg"],cv=5); parea=cvp(Ridge(),Xn,ndf["carea"],cv=5)
print(f"  neuron-level: Spearman(pred,contact-degree)={spearmanr(pdeg,ndf['cdeg'])[0]:+.3f} | contact-area={spearmanr(parea,ndf['carea'])[0]:+.3f} (from founder+depth)")

# ================= PHASE 5 — does lineage/timing predict developmental ARRIVAL? =================
print("\n"+"="*84); print("PHASE 5 — developmental arrival from lineage/timing"); print("="*84)
# per-pair developmental arrival = # of datasets 1-6 (L1-L3) the edge is present; reproducible adult=both 7,8
pres=defaultdict(set)
for r in pd.read_csv(EDGES).itertuples():
    ds=int(re.search(r'witvliet_2020_(\d)',r.source_file).group(1))
    pr=(min(r.pre,r.post),max(r.pre,r.post))
    if r.pre in paths and r.post in paths: pres[pr].add(ds)
# candidate = contacting pairs present in adult (reproducible core)
cand=[(a,b) for (a,b),v in A.items() if v>0 and (7 in pres.get((a,b),set()) and 8 in pres.get((a,b),set()))]
if len(cand)>50:
    dev=np.array([sum(1 for d in (1,2,3,4,5,6) if d in pres.get(p,set())) for p in cand])
    early=(dev>=1).astype(int)   # present in >=1 early (L1-L3) dataset = early-arriving
    tdc=np.array([treedist(a,b) for a,b in cand],float)
    ddc=np.array([abs(depth[a]-depth[b]) for a,b in cand],float)
    mnc=np.array([min(depth[a],depth[b]) for a,b in cand],float)  # earlier-born proxy
    print(f"  reproducible contacting pairs {len(cand)} | early-arriving rate {early.mean():.3f} | dev-arrival range {dev.min()}-{dev.max()}")
    Xa=np.column_stack([tdc,ddc,mnc])
    pa=cross_val_predict(LogisticRegression(max_iter=2000,class_weight="balanced"),Xa,early,cv=skf,method="predict_proba")[:,1]
    print(f"  AUC(lineage+timing -> early arrival) = {roc_auc_score(early,pa):.4f}")
    print(f"  Spearman(min-depth, dev-arrival)={spearmanr(mnc,dev)[0]:+.3f} | Spearman(treedist, dev-arrival)={spearmanr(tdc,dev)[0]:+.3f}")
else:
    print("  too few reproducible pairs with lineage coverage")

print("\n"+"="*84)
p6_ok = (a_lin>0.6) and ((a_lin-perm.mean())/(perm.std()+1e-9)>3)
print("PHASE 6 (contact opportunity):", "lineage/timing DOES predict contact opportunity" if p6_ok else "lineage/timing does NOT robustly predict contact")
print("="*84)
