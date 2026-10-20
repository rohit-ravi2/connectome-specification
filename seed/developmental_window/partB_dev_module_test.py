#!/usr/bin/env python3
"""Part B (DECISIVE): does DEVELOPMENTAL-WINDOW expression (Packer embryo E2/E3/E4) predict
pre-adult establishment beyond contact+birth+lineage+NT, surviving the STRICT matched-random + BH
control that overturned the adult-expression positive last branch?

Anti-mistake rules (the prior weak-null failure):
  * random modules use IDENTICAL pairwise feature construction as real modules
  * random genes matched for SIZE + expression LEVEL + VARIANCE (mean-decile x var-decile strata)
  * BH correction across all module x target x window tests
  * neuron-disjoint CV; NO adult features; adult only as a labeled comparator
"""
import csv, re, os, json, math
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from scipy.stats import false_discovery_control

ROOT=os.path.expanduser("../seed")
RB=f"{ROOT}/data/processed/regulatory_branch"
OUT=f"{ROOT}/data/processed/developmental_window_expression"
MASTER=f"{ROOT}/data/processed/upstream_early_establishment/upstream_master_table.csv"
GENESETS=f"{RB}/gene_sets_effector.json"
WINDOWS={"E2":f"{RB}/dev_allgene_pb_E2_270_390.csv","E3":f"{RB}/dev_allgene_pb_E3_390_510.csv","E4":f"{RB}/dev_allgene_pb_E4_ge510.csv"}
RNG=np.random.default_rng(0)
NDRAW=60

# ---- dev expression matrices (class x gene log-CPM) per window ----
def load_pb(path):
    with open(path) as f:
        rdr=csv.reader(f); hdr=next(rdr); genes=hdr[1:]
        cls=[]; M=[]
        for r in rdr: cls.append(r[0]); M.append([float(x) for x in r[1:]])
    return {c:i for i,c in enumerate(cls)}, genes, np.array(M)
devw={}  # window -> (class_idx, genes, matrix, gene_idx, mean, var, strata)
for w,p in WINDOWS.items():
    cidx,genes,M=load_pb(p); gi={g:i for i,g in enumerate(genes)}
    mean=M.mean(0); var=M.var(0)
    md=np.clip((np.argsort(np.argsort(mean))*10//len(mean)),0,9)
    vd=np.clip((np.argsort(np.argsort(var))*10//len(var)),0,9)
    strat=md*10+vd
    strata=defaultdict(list)
    for i,s in enumerate(strat): strata[s].append(i)
    devw[w]=dict(cidx=cidx,genes=genes,M=M,gi=gi,strat=strat,strata=strata)
allclasses=set.intersection(*[set(devw[w]['cidx']) for w in devw])
print(f"dev windows loaded; classes present in all windows: {len(allclasses)}")

# ---- modules ----
gs=json.load(open(GENESETS))
modules={cat:list(gs["categories"][cat]["genes_by_symbol"].values()) if "genes_by_symbol" in gs["categories"][cat] else list(gs["categories"][cat]["genes"].keys()) for cat in gs["categories"]}
# genes are WBGene ids; map to indices per window
def mod_gene_idx(genelist,w):
    gi=devw[w]['gi']; return [gi[g] for g in genelist if g in gi]

# ---- master table -> pairs mapped to dev class ----
def canon(nm,tgt):
    n=nm
    while True:
        if n in tgt: return n
        if len(n)>2 and n[-1] in "LRDV": n=n[:-1]
        elif re.search(r'\d$',n): n=re.sub(r'\d+$','',n)
        else: return None
mt=list(csv.DictReader(open(MASTER)))
recs=[]
for r in mt:
    cs=canon(r['pre'],allclasses); ct=canon(r['post'],allclasses)
    if cs and ct:
        r['_cs']=cs; r['_ct']=ct; recs.append(r)
pairs=[(r['pre'],r['post']) for r in recs]
print(f"pairs with both neurons -> dev class: {len(recs)}")

# ---- baseline features (contact+birth+lineage+NT) ----
def baseline_X():
    ntk=sorted(set((r['pre_nt'],r['post_nt']) for r in recs)); ki={k:i for i,k in enumerate(ntk)}
    X=[]
    for r in recs:
        pb=float(r['pre_birth']) if r['pre_birth'] not in('','nan') else np.nan
        qb=float(r['post_birth']) if r['post_birth'] not in('','nan') else np.nan
        nt=[0]*len(ntk); nt[ki[(r['pre_nt'],r['post_nt'])]]=1
        X.append([float(r['contact_area']),float(r['contact_cos']),float(r['contact_cn']),
                  np.nanmin([pb,qb]),np.nanmax([pb,qb]),abs(pb-qb) if not(np.isnan(pb) or np.isnan(qb)) else 0,
                  float(r['lin_dist']) if r['lin_dist'] not in('','nan') else 20, int(float(r['same_founder'])) ]+nt)
    return np.nan_to_num(np.array(X))
Xbase=baseline_X()

# ---- pairwise module features (IDENTICAL for real & random) ----
def module_feats(gene_idx,w):
    M=devw[w]['M']; cidx=devw[w]['cidx']
    if len(gene_idx)==0: return None
    sub=M[:,gene_idx]                      # class x k
    score=sub.mean(1)                      # per-class module score
    norm=np.linalg.norm(sub,axis=1)+1e-9
    F=[]
    for r in recs:
        si,ti=cidx[r['_cs']],cidx[r['_ct']]
        ms,mt_=score[si],score[ti]
        cos=float(sub[si]@sub[ti]/(norm[si]*norm[ti]))
        F.append([ms,mt_,abs(ms-mt_),ms*mt_,cos])
    return np.array(F)

def ndcv_auc(X,y,seed=0):
    neu=sorted(set(a for a,b in pairs)|set(b for a,b in pairs))
    rng=np.random.default_rng(seed); f=np.array_split(rng.permutation(neu),5); nf={n:i for i,ff in enumerate(f) for n in ff}
    pred=np.full(len(y),np.nan)
    for k in range(5):
        te=np.array([nf[a]==k and nf[b]==k for a,b in pairs]); tr=np.array([nf[a]!=k and nf[b]!=k for a,b in pairs])
        if te.sum()<8 or y[tr].sum()<4 or y[te].sum()<3 or len(set(y[te]))<2: continue
        clf=make_pipeline(StandardScaler(with_mean=False),LogisticRegression(max_iter=2000,C=1.0))
        clf.fit(X[tr],y[tr]); pred[te]=clf.predict_proba(X[te])[:,1]
    m=~np.isnan(pred)
    return roc_auc_score(y[m],pred[m]) if (m.sum()>15 and len(set(y[m]))>1) else np.nan
def base_auc(y): return np.nanmean([ndcv_auc(Xbase,y,s) for s in range(3)])
def add_auc(modF,y,base):
    return np.nanmean([ndcv_auc(np.hstack([Xbase,modF]),y,s) for s in range(3)])-base

def matched_random_idx(real_idx,w):
    """random gene set: same size, matched mean-decile x var-decile strata composition."""
    strat=devw[w]['strat']; strata=devw[w]['strata']
    comp=defaultdict(int)
    for i in real_idx: comp[strat[i]]+=1
    pick=[]
    for s,cnt in comp.items():
        pool=strata[s];
        if len(pool)>=cnt: pick+=list(RNG.choice(pool,cnt,replace=False))
        else: pick+=list(RNG.choice(pool,cnt,replace=True))
    return pick

TARGETS={"T_L1_repro":"L1_reproducible","T_preadult_persist":"preadult_persistent","T_first_L1":"first_synapse_L1"}
results=[]; pvals=[]
print("\n===== PART B: developmental-window modules vs matched-random (BH) =====")
for tkey,tname in TARGETS.items():
    y=np.array([int(r[tkey]) for r in recs])
    if y.sum()<15: continue
    b=base_auc(y)
    print(f"\n--- target {tname} (pos={y.sum()}/{len(y)}, base AUC={b:.3f}) ---")
    for w in ["E2","E3","E4"]:
        for mod,genes in modules.items():
            gi=mod_gene_idx(genes,w)
            if len(gi)<3: continue
            modF=module_feats(gi,w); real_add=add_auc(modF,y,b)
            nulls=[]
            for d in range(NDRAW):
                ridx=matched_random_idx(gi,w); rF=module_feats(ridx,w); nulls.append(add_auc(rF,y,b))
            nulls=np.array(nulls); z=(real_add-nulls.mean())/(nulls.std()+1e-9); pemp=(1+(nulls>=real_add).sum())/(1+len(nulls))
            results.append((tname,w,mod,len(gi),round(b,4),round(real_add,4),round(float(nulls.mean()),4),round(float(nulls.std()),4),round(float(z),3),round(pemp,4)))
            pvals.append(pemp)
            print(f"   {w} {mod:26s} n={len(gi):3d} add={real_add:+.4f} null={nulls.mean():+.4f} z={z:+.2f} p={pemp:.3f}")
# BH correction
pbh=false_discovery_control(np.array(pvals),method='bh') if pvals else []
with open(f"{OUT}/partB_dev_module_results.csv","w",newline="") as f:
    wtr=csv.writer(f); wtr.writerow(["target","window","module","n_genes","base_auc","real_add","null_mean","null_sd","z_vs_matched","p_emp","p_BH"])
    for row,pb in zip(results,pbh): wtr.writerow(list(row)+[round(float(pb),4)])
nsurv=sum(1 for pb in pbh if pb<0.05)
print(f"\n===== {nsurv}/{len(results)} module tests survive BH<0.05 =====")
if nsurv>0:
    for row,pb in zip(results,pbh):
        if pb<0.05: print(f"   SURVIVES: {row[0]} {row[1]} {row[2]} add={row[5]} z={row[8]} p_BH={pb:.4f}")
print("Part B done ->",OUT)
