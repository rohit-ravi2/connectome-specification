"""Phase-4 sanity RE-TEST with the CANONICAL Sulston lineage tree (jtauber cells.json).
Fixes the broken relatedness matrix. Build per-neuron lineage paths -> founder/depth/branch
tokens + real tree-distance. Checks: (a) homolog recovery (relatedness matrix gave 9% - the tree
must fix this), (b) does lineage predict NT/modality now? tree-distance kNN + lineage-feature
classifier, vs prevalence baseline + label permutation + leave-one-out.
"""
import json, warnings; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
RNG=np.random.default_rng(0)
D="../data/celegans/data"

# ---- parse lineage tree -> per-leaf path ----
tree=json.loads(open("../lineage/jtauber_lineage/cells.json").read())
paths={}   # leaf name -> list of ancestor node-names (root..leaf)
def walk(node, anc):
    nm=node["name"]; cur=anc+[nm]
    ch=node.get("children")
    if ch:
        for c in ch: walk(c,cur)
    else:
        paths[nm]=cur
walk(tree,[])
print(f"lineage tree leaves: {len(paths)}")

# ---- master neurons (from Phase 1) ----
master=pd.read_csv(D+"/processed/developmental_rules/neuron_development_master.csv")
neurons=[n for n in master.neuron if n in paths]
print(f"master neurons {len(master)} | with lineage path in tree: {len(neurons)}")
m=master.set_index("neuron").loc[neurons]
nt=m.nt.values; mod=m.modality.values

# ---- lineage features ----
def founder(p):
    # first node after root that is a named founder token (AB/MS/E/C/D/P...) = the leading-alpha of a path name
    for nm in p[1:]:
        import re; mm=re.match(r'^([A-Z]+[0-9]?)',nm)
        if mm: return mm.group(1)
    return "?"
def decisions(leaf):
    # the a/p/l/r/d/v decision string of the terminal path name
    nm=leaf
    import re; mm=re.match(r'^([A-Za-z]+?)([adplrv]+)$',nm)
    return mm.group(2) if mm else ""
fnd=np.array([founder(paths[n]) for n in neurons],dtype=object)
depth=np.array([len(paths[n]) for n in neurons])
dec=[decisions(n) for n in neurons]
# tree distance via common-prefix
def treedist(a,b):
    pa,pb=paths[a],paths[b]; k=0
    while k<len(pa) and k<len(pb) and pa[k]==pb[k]: k+=1
    return (len(pa)-k)+(len(pb)-k)
Nn=len(neurons); TD=np.zeros((Nn,Nn))
for i in range(Nn):
    for j in range(Nn): TD[i,j]=treedist(neurons[i],neurons[j])

# ---- (a) HOMOLOG RECOVERY (the decisive feature-quality check) ----
idx={n:i for i,n in enumerate(neurons)}
pairs=[(n,n[:-1]+'R') for n in neurons if n.endswith('L') and n[:-1]+'R' in idx]
rank=[]
for l,r in pairs:
    i,j=idx[l],idx[r]; d=TD[i]; d[i]=1e9
    rank.append(int(np.sum(d<TD[i,j])))
print(f"\nHOMOLOG RECOVERY (tree): {len(pairs)} L/R pairs | median #neurons closer than homolog {np.median(rank):.0f}/{Nn} "
      f"| homolog in top-3 for {np.mean(np.array(rank)<3):.2f}  (relatedness matrix was 0.09 in top-5)")

# ---- (b) does lineage predict NT / modality? ----
def knn_loo(labels,K=5):
    lab=np.array(labels); pred=np.empty(Nn,dtype=object)
    for i in range(Nn):
        d=TD[i].copy(); d[i]=1e9; order=np.argsort(d)[:K]
        votes={}
        for j in order: votes[lab[j]]=votes.get(lab[j],0)+1
        pred[i]=max(votes,key=votes.get)
    return pred
def feat_cv(labels, mask):
    # lineage-feature classifier: founder onehot + depth + first/last decision tokens
    lab=np.array(labels)
    fe=OneHotEncoder(sparse_output=False,handle_unknown='ignore').fit_transform(fnd.reshape(-1,1))
    dtok=np.array([[ord(t[0])-96 if t else 0, ord(t[-1])-96 if t else 0, len(t)] for t in dec])
    Xf=np.column_stack([fe,depth,dtok])
    y=lab[mask]; Xm=Xf[mask]
    if len(np.unique(y))<2: return None
    skf=StratifiedKFold(5,shuffle=True,random_state=0)
    p=cross_val_predict(LogisticRegression(max_iter=2000,C=1.0,class_weight="balanced"),Xm,y,cv=skf)
    return f1_score(y,p,average="macro"),accuracy_score(y,p)
def perm_f1(labels,mask,fn,reps=20):
    out=[]
    for _ in range(reps):
        pl=RNG.permutation(np.array(labels))
        if fn=="knn":
            pr=knn_loo(pl); out.append(f1_score(pl[mask],pr[mask],average="macro"))
        else:
            r=feat_cv(pl,mask); out.append(r[0] if r else np.nan)
    return np.array(out)

print("\n"+"="*84); print("PHASE-4 RE-TEST — real Sulston lineage tree predicts NT / modality?"); print("="*84)
for name,lab,mask in [("NT",nt,nt!="Unknown"),("modality",mod,np.ones(Nn,bool))]:
    pr=knn_loo(lab); kf1=f1_score(np.array(lab)[mask],pr[mask],average="macro"); kacc=accuracy_score(np.array(lab)[mask],pr[mask])
    kperm=perm_f1(lab,mask,"knn")
    cf=feat_cv(lab,mask); cperm=perm_f1(lab,mask,"feat")
    from collections import Counter
    base=max(Counter(np.array(lab)[mask]).values())/mask.sum()
    print(f"  {name:9s} n={int(mask.sum()):3d} base-acc={base:.3f}")
    print(f"    tree-kNN     macroF1={kf1:.3f} acc={kacc:.3f} | perm {kperm.mean():.3f}±{kperm.std():.3f} z={(kf1-kperm.mean())/(kperm.std()+1e-9):+.1f}")
    print(f"    feat-classif macroF1={cf[0]:.3f} acc={cf[1]:.3f} | perm {np.nanmean(cperm):.3f}±{np.nanstd(cperm):.3f} z={(cf[0]-np.nanmean(cperm))/(np.nanstd(cperm)+1e-9):+.1f}")

print("\n"+"="*84)
prnt=knn_loo(nt); f1nt=f1_score(nt[nt!='Unknown'],prnt[nt!='Unknown'],average='macro'); pnt=perm_f1(nt,nt!='Unknown',"knn")
ok=f1nt>pnt.mean()+3*pnt.std()
print("PHASE-4 RE-TEST VERDICT:", "PASS — real lineage predicts NT above null; feature FIXED -> proceed to Phase 5/6"
      if ok else "FAIL — even the real Sulston tree does not predict NT (=> lineage does not cluster fate; biology, not data)")
print("="*84)
# save lineage features for downstream phases
pd.DataFrame({"neuron":neurons,"founder":fnd,"depth":depth,"decisions":dec,
             "lineage_path":["/".join(paths[n]) for n in neurons]}).to_csv(
             D+"/processed/developmental_rules/lineage_features.csv",index=False)
print("saved lineage_features.csv (",len(neurons),"neurons )")
