#!/usr/bin/env python3
"""Phase 6 (Round 3) -- OR-RESCUE endpoint: among CONTACTING nerve-ring pairs, does
DEVELOPMENTAL effector expression add to GEOMETRY + NT (+topology) for predicting the
REPRODUCIBLE chemical synapse (present in both adults 7&8)?

This is the branch's core hypothesis: developmentally-timed recognition/adhesion programs
select synaptic partners among physical-contact opportunities.

Baselines (spec Phase 6): geometry -> +NT -> +dev effector. Question is the ADD of dev over
geometry+NT, under honest multi-seed class-disjoint CV, vs a random expression-matched gene null.
"""
import openpyxl, csv, re, os, json, math
import numpy as np
from collections import defaultdict, Counter
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, log_loss

RB=os.path.expanduser("../seed/data/processed/regulatory_branch")
DATA="../data/celegans/data"
CONTACT=f"{DATA}/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx"
EDGES=f"{DATA}/connectome/witvliet2020/witvliet_combined_edges.csv"
NPZ="../data/tier5/connectome.npz"
CATS=["adhesion_recognition","axon_guidance","nt_fate_receptor","synaptogenesis","terminal_selector_identity"]
WINS=["E2_270_390","E3_390_510","E4_ge510"]
NSEED=6; NDRAW=25

# ---- dev module (class -> 15-dim) from per-window all-gene pseudobulk (for effector + random) ----
allpb={}; allgenes=None; allidx=None
for w in WINS:
    with open(f"{RB}/dev_allgene_pb_{w}.csv") as f:
        rdr=csv.reader(f); hdr=next(rdr)
        if allgenes is None: allgenes=hdr[1:]; allidx={g:i for i,g in enumerate(allgenes)}
        allpb[w]={r[0]:np.array([float(x) for x in r[1:]]) for r in rdr}
gs=json.load(open(f"{RB}/gene_sets_effector.json"))
cat_genes={c:list(gs["categories"][c]["genes"]) for c in CATS}
gmean={}
for r in csv.DictReader(open(f"{RB}/dev_allgene_meanexpr.csv")): gmean[r["WBGene"]]=float(r["mean_logcpm"])
def module_from(genes_by_cat, cls):
    out=[]
    for w in WINS:
        src=allpb[w].get(cls)
        if src is None and w=="E2_270_390": src=allpb["E3_390_510"].get(cls)
        if src is None: return None
        for c in CATS:
            idx=[allidx[g] for g in genes_by_cat[c] if g in allidx]
            out.append(float(np.mean(src[idx])) if idx else 0.0)
    return np.array(out)
DEVSET=set(allpb["E3_390_510"])|set(allpb["E4_ge510"])
eff_mod={c:module_from(cat_genes,c) for c in DEVSET if module_from(cat_genes,c) is not None}

# ---- NT per neuron ----
d=np.load(NPZ,allow_pickle=True); names=[str(x) for x in d["names"]]; ntp=[str(x) for x in d["nt_primary"]]
def canon(nm,tgt):
    n=nm
    while True:
        if n in tgt: return n
        if len(n)>2 and n[-1] in "LRDV": n=n[:-1]
        elif re.search(r'\d$',n): n=re.sub(r'\d+$','',n)
        else: return None
def ntn(x):
    if "ACh" in x or "Acetyl" in x: return "ACh"
    if "Glu" in x: return "Glu"
    if "GABA" in x: return "GABA"
    if "Dopamine" in x: return "DA"
    if any(k in x for k in("Serotonin","5HT")): return "5HT"
    if "Octopamine" in x: return "OA"
    if "Tyramine" in x: return "TA"
    return "Unk"
nt_of={nm:ntn(nt) for nm,nt in zip(names,ntp)}

# ---- contact area matrix (full 182) for geometry ----
wb=openpyxl.load_workbook(CONTACT,read_only=True,data_only=True); ws=wb['adult nerve ring neighbors']
rows=list(ws.iter_rows(values_only=True)); alln=[h for h in rows[0][1:]]; idx={n:i for i,n in enumerate(alln)}; M=len(alln)
A=np.zeros((M,M))
for row in rows[1:]:
    rn=row[0]
    if rn is None or rn not in idx: continue
    for j,val in enumerate(row[1:]):
        if val in (None,''): continue
        try: A[idx[rn],j]=float(val)
        except: pass
A=np.maximum(A,A.T); deg_area=A.sum(1); deg_num=(A>0).sum(1)

# ---- reproducible synapse (both adults 7,8) ----
chem=defaultdict(set); elec=defaultdict(set)
for r in csv.DictReader(open(EDGES)):
    ds=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
    if ds not in (7,8): continue
    pre,post,typ=r['pre'],r['post'],r['type']
    (elec if typ=='electrical' else chem)[((min(pre,post),max(pre,post)) if typ=='electrical' else (pre,post))].add(ds)
repro=set()
for e,dss in chem.items():
    if len(dss)==2: repro.add((min(e),max(e)))
for e,dss in elec.items():
    if len(dss)==2: repro.add(e)

# ---- contacting pairs, both mapped to dev class ----
pairs=[]; y=[]; Xg=[]; ntpair=[]
for i in range(M):
    for j in range(i+1,M):
        if A[i,j]<=0: continue
        a,b=alln[i],alln[j]
        ca,cb=canon(a,DEVSET),canon(b,DEVSET)
        if ca not in eff_mod or cb not in eff_mod or ca==cb: continue
        vi,vj=A[i],A[j]
        cos=float(vi@vj/(np.linalg.norm(vi)*np.linalg.norm(vj)+1e-9))
        shared=int(((A[i]>0)&(A[j]>0)).sum())
        Xg.append([math.log1p(A[i,j]),cos,shared,
                   math.log1p(min(deg_area[i],deg_area[j])),math.log1p(max(deg_area[i],deg_area[j])),
                   min(deg_num[i],deg_num[j]),max(deg_num[i],deg_num[j])])
        pairs.append((a,b,ca,cb)); y.append(1 if (a,b) in repro or (b,a) in repro else 0)
        ntpair.append(tuple(sorted((nt_of.get(a,"Unk"),nt_of.get(b,"Unk")))))
y=np.array(y); Xg=np.array(Xg)
# NT one-hot
ntk=sorted(set(ntpair)); nti={k:i for i,k in enumerate(ntk)}
Xnt=np.zeros((len(pairs),len(ntk)))
for r,k in enumerate(ntpair): Xnt[r,nti[k]]=1
def sym(a,b): return np.concatenate([a+b,np.abs(a-b),a*b])
Xdev=np.array([sym(eff_mod[ca],eff_mod[cb]) for _,_,ca,cb in pairs])
print(f"contacting mapped pairs {len(pairs)} | reproducible {y.sum()} ({100*y.mean():.1f}%) | classes {len(set(c for _,_,ca,cb in pairs for c in (ca,cb)))}")

# ---- CV helpers ----
def edge_cv(X,seed=0):
    clf=make_pipeline(StandardScaler(with_mean=False),GradientBoostingClassifier(random_state=seed))
    p=cross_val_predict(clf,X,y,cv=StratifiedKFold(5,shuffle=True,random_state=seed),method="predict_proba")[:,1]
    p=np.clip(p,1e-6,1-1e-6); return roc_auc_score(y,p), log_loss(y,p)
def cdcv(X,seed):
    classes=sorted(set(c for _,_,ca,cb in pairs for c in (ca,cb)))
    rng=np.random.default_rng(seed); folds=np.array_split(rng.permutation(classes),5)
    cf={c:i for i,f in enumerate(folds) for c in f}; preds=np.full(len(y),np.nan)
    ca_=[p[2] for p in pairs]; cb_=[p[3] for p in pairs]
    for k in range(5):
        te=np.array([cf[ca_[r]]==k and cf[cb_[r]]==k for r in range(len(pairs))])
        tr=np.array([cf[ca_[r]]!=k and cf[cb_[r]]!=k for r in range(len(pairs))])
        if te.sum()<10 or y[tr].sum()<5 or y[te].sum()<3 or len(set(y[te]))<2: continue
        clf=make_pipeline(StandardScaler(with_mean=False),GradientBoostingClassifier(random_state=seed))
        clf.fit(X[tr],y[tr]); preds[te]=clf.predict_proba(X[te])[:,1]
    m=~np.isnan(preds)
    return roc_auc_score(y[m],preds[m]) if (m.sum()>20 and len(set(y[m]))>1) else np.nan
def cd_mean(X): return np.array([cdcv(X,s) for s in range(NSEED)])
def hs(*a): return np.hstack(a)

print("\n============ REPRODUCIBLE-SYNAPSE MODEL LADDER ============")
ladder={"geometry":Xg,"geometry+NT":hs(Xg,Xnt),"geometry+dev":hs(Xg,Xdev),
        "geometry+NT+dev":hs(Xg,Xnt,Xdev)}
res={}
print(f"{'model':20s} {'edgeCV AUC':>11s} {'edge logloss':>13s} {'classCV AUC(6-seed)':>20s}")
for nm,X in ladder.items():
    ae,le=edge_cv(X); cd=cd_mean(X); res[nm]=(ae,le,np.nanmean(cd),np.nanstd(cd))
    print(f"{nm:20s} {ae:>11.4f} {le:>13.4f}   {np.nanmean(cd):>10.4f} ± {np.nanstd(cd):.4f}")

# ---- ADD of dev over geometry+NT (edge + class-disjoint) ----
dadd_edge=res["geometry+NT+dev"][0]-res["geometry+NT"][0]
dadd_cd  =res["geometry+NT+dev"][2]-res["geometry+NT"][2]
print(f"\nDEV ADD over geometry+NT:  edgeCV ΔAUC={dadd_edge:+.4f} | class-disjoint ΔAUC={dadd_cd:+.4f}")

# ---- random expr-matched gene ADD null (class-disjoint) ----
level=sorted((g for g in allgenes if g in gmean), key=lambda g:gmean[g]); lv=np.array([gmean[g] for g in level])
base_cd=res["geometry+NT"][2]
def rand_mod(seed):
    rng=np.random.default_rng(7000+seed); rnd={}
    for c in CATS:
        picks=[]
        for g in cat_genes[c]:
            if g in gmean:
                j=int(np.abs(lv-(gmean[g]+rng.normal(0,0.05))).argmin()); picks.append(level[j])
        rnd[c]=picks
    return {cl:module_from(rnd,cl) for cl in DEVSET if module_from(rnd,cl) is not None}
rand_add=[]
for dd in range(NDRAW):
    mod=rand_mod(dd)
    if all(ca in mod and cb in mod for _,_,ca,cb in pairs):
        Xr=np.array([sym(mod[ca],mod[cb]) for _,_,ca,cb in pairs])
        rand_add.append(np.nanmean(cd_mean(hs(Xg,Xnt,Xr)))-base_cd)
rand_add=np.array(rand_add)
print(f"\nRANDOM gene ADD over geometry+NT (class-disjoint ΔAUC, {len(rand_add)} draws):")
print(f"  null Δ mean={rand_add.mean():+.4f} 95pct={np.percentile(rand_add,95):+.4f} max={rand_add.max():+.4f}")
print(f"  DEV effector Δ={dadd_cd:+.4f} | P(random ADD >= dev ADD)={(1+(rand_add>=dadd_cd).sum())/(1+len(rand_add)):.3f}")
print("\nGATE (Phase 6): dev keeps branch alive iff dev ADD>0 robustly AND dev ADD > random-gene ADD.")
