#!/usr/bin/env python3
"""Phase 5 DECISIVE control -- is the developmental contact-presence signal EFFECTOR-SPECIFIC,
or generic developmental transcriptome?

Compares, under repeated CLASS-DISJOINT CV (no class leakage), the 15-dim (E2/E3/E4 x 5) module
features built from:
  EFFECTOR  : the frozen effector gene sets
  RANDOM    : gene sets matched PER CATEGORY by size and expression level (N draws -> null)
  ADULT     : adult effector module scores (5-dim)
  ANATOMY   : modality + birth-time
Target: pairwise contact presence among nerve-ring neurons mapped to a dev class.

Claim ladder:
  dev > anatomy & dev > adult (Phase 5 base)  -> developmental expression carries contact info
  dev-EFFECTOR > RANDOM-matched null          -> the info is EFFECTOR-SPECIFIC (strong)
  dev-EFFECTOR ~ RANDOM                        -> generic developmental/lineage signal (weak)
"""
import openpyxl, csv, re, os, json
import numpy as np
from collections import defaultdict, Counter
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

RB=os.path.expanduser("../seed/data/processed/regulatory_branch")
DATA="../data/celegans/data"
CONTACT=f"{DATA}/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx"
NPZ="../data/tier5/connectome.npz"
PROF=os.path.expanduser("../data/celegans/New Notebooks/data_derived/nb28_neuron_profile.csv")
CATS=["adhesion_recognition","axon_guidance","nt_fate_receptor","synaptogenesis","terminal_selector_identity"]
WINS=["E2_270_390","E3_390_510","E4_ge510"]
NDRAW=30; NSEED=6

# ---- effector gene sets + per-gene expr level ----
gs=json.load(open(f"{RB}/gene_sets_effector.json"))
cat_genes={c:list(gs["categories"][c]["genes"]) for c in CATS}
gmean={}
for r in csv.DictReader(open(f"{RB}/dev_allgene_meanexpr.csv")): gmean[r["WBGene"]]=float(r["mean_logcpm"])

# ---- per-window all-gene pseudobulk (class x gene) ----
allpb={}; allgenes=None; allidx=None
for w in WINS:
    with open(f"{RB}/dev_allgene_pb_{w}.csv") as f:
        rdr=csv.reader(f); hdr=next(rdr)
        if allgenes is None: allgenes=hdr[1:]; allidx={g:i for i,g in enumerate(allgenes)}
        allpb[w]={r[0]:np.array([float(x) for x in r[1:]]) for r in rdr}
def module_from_genes(genelist_by_cat, cls):
    """15-dim module vector for a class from given per-category gene lists (E2 imputed by E3)."""
    vecs=[]
    for w in WINS:
        src = allpb[w].get(cls)
        if src is None and w=="E2_270_390": src=allpb["E3_390_510"].get(cls)
        if src is None: return None
        for c in CATS:
            idx=[allidx[g] for g in genelist_by_cat[c] if g in allidx]
            vecs.append(float(np.mean(src[idx])) if idx else 0.0)
    return np.array(vecs)

# effector module per class
dev_classes=set(allpb["E3_390_510"]) | set(allpb["E4_ge510"])
eff_mod={}
for c in dev_classes:
    v=module_from_genes(cat_genes,c)
    if v is not None: eff_mod[c]=v

# adult module
adult_ms={}
for r in csv.DictReader(open(f"{RB}/adult_module_scores.csv")):
    adult_ms[r["class"]]=np.array([float(r[c]) for c in CATS],float)

# ---- labels: modality + birthtime ----
d=np.load(NPZ,allow_pickle=True); names=[str(x) for x in d["names"]]
prof={r["neuron"]:r for r in csv.DictReader(open(PROF))}
MOTOR={'VA','VB','VD','DA','DB','DD','AS','VC','RMD','RME','RIM','RMF','RMH','RIV','SMD','SMB','URA'}
def canon(nm,tgt):
    n=nm
    while True:
        if n in tgt: return n
        if len(n)>2 and n[-1] in "LRDV": n=n[:-1]
        elif re.search(r'\d$',n): n=re.sub(r'\d+$','',n)
        else: return None
def modality(nm):
    r=prof.get(nm)
    if r:
        if r.get("is_sensory")in("1","1.0","True"): return "sensory"
        if r.get("is_command")in("1","1.0","True"): return "command"
    base=nm[:-1] if nm[-1:] in "LR" else nm
    return "motor" if (base in MOTOR or nm in MOTOR) else "inter"
DEVSET=set(eff_mod)
cls_mod_v=defaultdict(list)
for nm in names:
    c=canon(nm,DEVSET)
    if c: cls_mod_v[c].append(modality(nm))
cls_mod={c:Counter(v).most_common(1)[0][0] for c,v in cls_mod_v.items()}
NON=("parent","grandparent","Neuroblast","possibly","_and_"); first_t=defaultdict(lambda:1e9)
for row in csv.DictReader(open(f"{DATA}/expression/scrna_seq_other/GSE126954/GSE126954_cell_annotation.csv")):
    lab=None
    for f_ in (row['plot.cell.type'],row['cell.subtype']):
        if f_ and f_!='NA' and re.match(r'^[A-Z]{2,4}\d?$',f_) and not any(t in f_ for t in NON): lab=f_; break
    if lab in DEVSET:
        try: t=float(row['embryo.time']); first_t[lab]=min(first_t[lab],t)
        except: pass
birth={k:v for k,v in first_t.items() if v<1e9}

# ---- contact pairs ----
wb=openpyxl.load_workbook(CONTACT,read_only=True,data_only=True); ws=wb['adult nerve ring neighbors']
rows=list(ws.iter_rows(values_only=True)); hdr=[h for h in rows[0][1:]]
areaset=set()
for row in rows[1:]:
    rn=row[0]
    if rn is None: continue
    for j,val in enumerate(row[1:]):
        if val in (None,''): continue
        cn=hdr[j]
        try: a=float(val)
        except: continue
        if a>0 and rn!=cn: areaset.add((min(rn,cn),max(rn,cn)))
neurons=[n for n in hdr if n and canon(n,DEVSET) in eff_mod and canon(n,DEVSET) in adult_ms and canon(n,DEVSET) in birth]
ncls={n:canon(n,DEVSET) for n in neurons}
pairs=[]; y=[]
for i in range(len(neurons)):
    for j in range(i+1,len(neurons)):
        a,b=neurons[i],neurons[j]
        if ncls[a]==ncls[b]: continue
        pairs.append((a,b)); y.append(1 if (min(a,b),max(a,b)) in areaset else 0)
y=np.array(y)
print(f"pairs {len(pairs)} contacts {y.sum()} ({100*y.mean():.1f}%) | classes {len(set(ncls.values()))}")

# ---- pair feature builders ----
def sym(a,b): return np.concatenate([a+b,np.abs(a-b),a*b])
def feat_from_classmod(classmod):
    return np.array([sym(classmod[ncls[a]],classmod[ncls[b]]) for a,b in pairs])
def feat_anatomy():
    keys=sorted(set(tuple(sorted((cls_mod.get(ncls[a],"?"),cls_mod.get(ncls[b],"?")))) for a,b in pairs))
    ki={k:i for i,k in enumerate(keys)}
    M=[]
    for a,b in pairs:
        row=np.zeros(len(keys)+3); row[ki[tuple(sorted((cls_mod.get(ncls[a],"?"),cls_mod.get(ncls[b],"?")))) ]]=1
        ba,bb=birth[ncls[a]],birth[ncls[b]]; row[-3:]= [min(ba,bb),max(ba,bb),abs(ba-bb)]
        M.append(row)
    return np.array(M)

# ---- repeated class-disjoint CV ----
def cdcv(X,y,seed):
    classes=sorted(set(c for p in pairs for c in (ncls[p[0]],ncls[p[1]])))
    rng=np.random.default_rng(seed); order=rng.permutation(classes)
    folds=np.array_split(order,5); cf={c:i for i,f in enumerate(folds) for c in f}
    preds=np.full(len(y),np.nan)
    for k in range(5):
        te=np.array([cf[ncls[a]]==k and cf[ncls[b]]==k for a,b in pairs])
        tr=np.array([cf[ncls[a]]!=k and cf[ncls[b]]!=k for a,b in pairs])
        if te.sum()<10 or y[tr].sum()<5 or y[te].sum()<3 or len(set(y[te]))<2: continue
        clf=make_pipeline(StandardScaler(with_mean=False),GradientBoostingClassifier(random_state=seed))
        clf.fit(X[tr],y[tr]); preds[te]=clf.predict_proba(X[te])[:,1]
    m=~np.isnan(preds)
    return roc_auc_score(y[m],preds[m]) if (m.sum()>20 and len(set(y[m]))>1) else np.nan
def repeated(X):
    return np.array([cdcv(X,y,s) for s in range(NSEED)])

Xeff=feat_from_classmod(eff_mod)
Xad=feat_from_classmod({c:adult_ms[c] for c in DEVSET if c in adult_ms})
Xan=feat_anatomy()
eff=repeated(Xeff); ad=repeated(Xad); an=repeated(Xan)
print(f"\nCLASS-DISJOINT CV AUC (mean±sd over {NSEED} seeds):")
print(f"  ANATOMY (mod+birth) : {np.nanmean(an):.4f} ± {np.nanstd(an):.4f}")
print(f"  ADULT effector mod  : {np.nanmean(ad):.4f} ± {np.nanstd(ad):.4f}")
print(f"  DEV  effector mod   : {np.nanmean(eff):.4f} ± {np.nanstd(eff):.4f}")

# ---- random matched-gene null ----
level_sorted=sorted((g for g in allgenes if g in gmean), key=lambda g: gmean[g])
lv=np.array([gmean[g] for g in level_sorted])
def matched_random(seed):
    rng=np.random.default_rng(9000+seed); rnd={}
    for c in CATS:
        picks=[]
        for g in cat_genes[c]:
            if g not in gmean: continue
            j=int(np.abs(lv-(gmean[g]+rng.normal(0,0.05))).argmin()); picks.append(level_sorted[j])
        rnd[c]=picks
    mod={}
    for cl in DEVSET:
        v=module_from_genes(rnd,cl)
        if v is not None: mod[cl]=v
    return mod
rand_auc=[]
for dd in range(NDRAW):
    mod=matched_random(dd)
    if all(c in mod for c in DEVSET):
        Xr=feat_from_classmod(mod); rand_auc.append(np.nanmean(repeated(Xr)))
rand_auc=np.array(rand_auc)
effm=np.nanmean(eff)
print(f"\nRANDOM matched-gene modules (class-disjoint CV, {len(rand_auc)} draws):")
print(f"  null AUC mean={rand_auc.mean():.4f}  95pct={np.percentile(rand_auc,95):.4f}  max={rand_auc.max():.4f}")
print(f"  DEV effector AUC={effm:.4f}  |  P(random >= effector)={(1+(rand_auc>=effm).sum())/(1+len(rand_auc)):.3f}")
print("\nVERDICT: effector-specific iff DEV effector clearly > random null (low P) AND > adult/anatomy.")
