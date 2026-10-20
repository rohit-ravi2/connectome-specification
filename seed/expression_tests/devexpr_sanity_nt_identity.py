#!/usr/bin/env python3
"""Phase 4 (Round 3) -- SANITY GATE: do developmental effector features recover known
neuron biology (NT class, modality) better than chance / random gene sets / adult?

If dev effector features CANNOT beat random-matched gene sets and chance on NT/modality,
the mapping/features are broken -> debug BEFORE touching wiring (Phases 5-6).

Feature sets compared (all StandardScaled inside CV):
  DEV_eff    : developmental effector expression (510 genes, wiring window E3)
  ADULT_eff  : adult CeNGEN effector expression (510 genes)
  DEV_module : developmental module scores (5 dims)
  RAND_dev   : random gene sets from all-gene dev background, matched by expression level (null)

Targets (per class, majority over member neurons):
  NT        : ACh / Glu / GABA / monoamine   (Unknown dropped)
  modality  : sensory / command / inter / motor

Metrics: stratified CV macro-F1, balanced accuracy, log-loss; label-permutation null.
"""
import csv, os, json, re
import numpy as np
from collections import defaultdict, Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, balanced_accuracy_score, log_loss

RB=os.path.expanduser("../seed/data/processed/regulatory_branch")
DATA="../data/celegans/data"
NPZ="../data/tier5/connectome.npz"
PROF=os.path.expanduser("../data/celegans/New Notebooks/data_derived/nb28_neuron_profile.csv")
RNG=np.random.default_rng(0)
MOTOR={'VA','VB','VD','DA','DB','DD','AS','VC','RMD','RME','RIM','RMF','RMH','RIV','SMD','SMB','URA'}

def canon_to(name, target):
    n=name
    while True:
        if n in target: return n
        if len(n)>2 and n[-1] in "LRDV": n=n[:-1]
        elif re.search(r'\d$',n): n=re.sub(r'\d+$','',n)
        else: return None

# ---- load dev features (wiring) ----
def load_wide(path):
    with open(path) as f:
        rdr=csv.reader(f); hdr=next(rdr); cols=hdr[1:]
        idx={}; X=[]
        for r in rdr:
            idx[r[0]]=len(X); X.append([float(x) for x in r[1:]])
    return np.array(X), cols, idx
Xdev, dev_genes, dev_idx = load_wide(f"{RB}/developmental_effector_features_by_class.csv")
Xad,  ad_genes,  ad_idx  = load_wide(f"{RB}/adult_effector_features_by_class.csv")
dev_classes=list(dev_idx.keys())
print(f"DEV {Xdev.shape} | ADULT {Xad.shape}")

# dev module scores (wiring window)
mod_rows={}
for r in csv.DictReader(open(f"{RB}/developmental_module_scores.csv")):
    if r["window"]=="E3_390_510": mod_rows[r["class"]]=[float(r[c]) for c in
        ["adhesion_recognition","axon_guidance","nt_fate_receptor","synaptogenesis","terminal_selector_identity"]]

# all-gene dev background (wiring) for random matched sets
Xbg, bg_genes, bg_idx = load_wide(f"{RB}/dev_allgene_pseudobulk_wiring.csv")
bg_gcol={g:i for i,g in enumerate(bg_genes)}
gmean={};
for r in csv.DictReader(open(f"{RB}/dev_allgene_meanexpr.csv")): gmean[r["WBGene"]]=float(r["mean_logcpm_allemb"])

# ---- labels per class ----
d=np.load(NPZ,allow_pickle=True); names=[str(x) for x in d["names"]]; ntp=[str(x) for x in d["nt_primary"]]
def nt_norm(x):
    if "ACh" in x or "Acetyl" in x: return "ACh"
    if "Glu" in x: return "Glu"
    if "GABA" in x: return "GABA"
    if any(k in x for k in ("Dopamine","DA","Serotonin","5HT","Octopamine","OA","Tyramine","TA")): return "monoamine"
    return None
# per class majority NT
cls_nt_votes=defaultdict(list)
for nm,nt in zip(names,ntp):
    c=canon_to(nm, set(dev_classes))
    v=nt_norm(nt)
    if c and v: cls_nt_votes[c].append(v)
cls_nt={c:Counter(v).most_common(1)[0][0] for c,v in cls_nt_votes.items()}

# modality from nb28
prof={r["neuron"]:r for r in csv.DictReader(open(PROF))}
def modality(nm):
    r=prof.get(nm)
    if r:
        if r.get("is_sensory")in("1","1.0","True"): return "sensory"
        if r.get("is_command")in("1","1.0","True"): return "command"
    base=canon_to(nm,MOTOR) or (nm[:-1] if nm[-1:] in "LR" else nm)
    if base in MOTOR or nm in MOTOR: return "motor"
    return "inter"
cls_mod_votes=defaultdict(list)
for nm in names:
    c=canon_to(nm,set(dev_classes))
    if c: cls_mod_votes[c].append(modality(nm))
cls_mod={c:Counter(v).most_common(1)[0][0] for c,v in cls_mod_votes.items()}

print("NT dist:", Counter(cls_nt.values()))
print("modality dist:", Counter(cls_mod.values()))

# ---- evaluation ----
def evaluate(X, y, groups, nfold=5, nperm=300, seed=0):
    """macro-F1, balanced-acc, log-loss via stratified CV; permutation null on macro-F1."""
    y=np.array(y)
    # drop classes with <2 members for stratification
    keep=np.array([True]*len(y))
    cnt=Counter(y)
    keep=np.array([cnt[v]>=2 for v in y])
    Xk=X[keep]; yk=y[keep]
    k=min(nfold, min(Counter(yk).values()))
    if k<2 or len(set(yk))<2: return None
    clf=make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    cv=StratifiedKFold(k, shuffle=True, random_state=seed)
    yp=cross_val_predict(clf, Xk, yk, cv=cv)
    pp=cross_val_predict(clf, Xk, yk, cv=cv, method="predict_proba")
    labs=sorted(set(yk))
    f1=f1_score(yk,yp,average="macro"); ba=balanced_accuracy_score(yk,yp)
    try: ll=log_loss(yk,pp,labels=labs)
    except Exception: ll=float("nan")
    # permutation null on macro-F1
    null=[]
    for i in range(nperm):
        ys=RNG.permutation(yk)
        ypn=cross_val_predict(clf, Xk, ys, cv=StratifiedKFold(k,shuffle=True,random_state=1000+i))
        null.append(f1_score(ys,ypn,average="macro"))
    null=np.array(null); p=(1+ (null>=f1).sum())/(1+len(null))
    return dict(n=len(yk), k_classes=len(labs), macroF1=f1, balacc=ba, logloss=ll,
               null_mean=float(null.mean()), null_p=float(p))

def matched_random_null(target_dict, size, ndraw=40, seed=0):
    """null macro-F1 distribution from random expression-level-matched gene sets (dev bg)."""
    rng=np.random.default_rng(seed)
    # bins by mean expr
    genes=[g for g in bg_genes if g in gmean]
    order=np.array(sorted(genes, key=lambda g: gmean[g]))         # genes sorted by expr level
    order_lv=np.array([gmean[g] for g in order])
    eff_levels=np.array([gmean[g] for g in dev_genes if g in gmean])
    res=[]
    classes=[c for c in dev_classes if c in target_dict and c in bg_idx]
    y=[target_dict[c] for c in classes]
    rows=[bg_idx[c] for c in classes]
    for d_ in range(ndraw):
        # nearest-expression-level match to each effector gene (jittered), unique picks
        picks=[]
        for lv in eff_levels:
            j=int(np.abs(order_lv-(lv+rng.normal(0,0.05))).argmin()); picks.append(order[j])
        cols=[bg_gcol[g] for g in picks]
        Xr=Xbg[np.ix_(rows, cols)]
        r=evaluate(Xr, y, None, nperm=0, seed=7)
        if r: res.append(r["macroF1"])
    return np.array(res)

def run_target(name, tdict):
    print(f"\n================= TARGET: {name} =================")
    classes=[c for c in dev_classes if c in tdict]
    y=[tdict[c] for c in classes]
    print(f"  classes with label & dev feats: {len(classes)}  dist={Counter(y)}")
    # feature sets
    Xdev_=np.array([Xdev[dev_idx[c]] for c in classes])
    Xmod_=np.array([mod_rows[c] for c in classes if c in mod_rows])
    mod_classes=[c for c in classes if c in mod_rows]
    ymod=[tdict[c] for c in mod_classes]
    # adult (only classes present in adult)
    ad_classes=[c for c in classes if c in ad_idx]
    Xad_=np.array([Xad[ad_idx[c]] for c in ad_classes]); yad=[tdict[c] for c in ad_classes]
    for label,(Xf,yf) in {"DEV_eff":(Xdev_,y),"DEV_module":(Xmod_,ymod),
                            "ADULT_eff":(Xad_,yad)}.items():
        if len(Xf)==0: continue
        r=evaluate(np.asarray(Xf,float), yf, None)
        if r: print(f"  {label:11s} n={r['n']:2d} K={r['k_classes']} | macroF1={r['macroF1']:.3f} "
                     f"balacc={r['balacc']:.3f} logloss={r['logloss']:.3f} | null~{r['null_mean']:.3f} p={r['null_p']:.3f}")
        else:  print(f"  {label:11s} -- insufficient")
    # random matched gene-set null vs DEV_eff
    rnull=matched_random_null(tdict, size=len(dev_genes), ndraw=30, seed=3)
    if len(rnull):
        dev_r=evaluate(Xdev_, y, None, nperm=0)
        beat=(rnull>=dev_r["macroF1"]).mean()
        print(f"  RAND_dev matched null macroF1: mean={rnull.mean():.3f} 95pct={np.percentile(rnull,95):.3f} "
              f"| DEV_eff={dev_r['macroF1']:.3f}  P(rand>=dev)={beat:.3f}")

run_target("NT", cls_nt)
run_target("modality", cls_mod)
print("\nGATE: DEV_eff must beat chance (null_p<0.05) AND random matched sets (P(rand>=dev) low) on >=1 target.")
