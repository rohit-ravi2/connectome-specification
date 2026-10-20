#!/usr/bin/env python3
"""Phase 5 (Round 3) -- does DEVELOPMENTAL effector expression predict CONTACT GEOMETRY
beyond neuron class(modality)/birth-time/adult expression?

Contact geometry sits UPSTREAM of the saturated synapse ceiling (Phase 0), so this is the
branch's most biologically plausible positive: developmental guidance/adhesion expression
shaping physical opportunity.

TARGET (primary): pairwise CONTACT PRESENCE among nerve-ring neurons mapped to a dev class.
Features are CLASS-level multi-window module scores (E2/E3/E4 x 5 categories = 15 dims),
combined symmetrically per pair.  Model ladder + confound controls + TWO CV schemes:
  - edge-CV (StratifiedKFold over pairs)   -- optimistic (class leakage across folds)
  - class-disjoint CV (GroupKFold on class) -- honest screen; must survive to advance

Decision (spec Phase 5): dev expression predicting contact BEYOND class/lineage/adult AND
beating random expression-matched gene sets = meaningful ("dev effector expression contributes
to physical opportunity formation").
"""
import openpyxl, csv, re, os, math
import numpy as np
from collections import defaultdict, Counter
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, log_loss

RB=os.path.expanduser("../seed/data/processed/regulatory_branch")
DATA="../data/celegans/data"
CONTACT=f"{DATA}/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx"
NPZ="../data/tier5/connectome.npz"
PROF=os.path.expanduser("../data/celegans/New Notebooks/data_derived/nb28_neuron_profile.csv")
CATS=["adhesion_recognition","axon_guidance","nt_fate_receptor","synaptogenesis","terminal_selector_identity"]
WINS=["E2_270_390","E3_390_510","E4_ge510"]
RNG=np.random.default_rng(0)

# ---------- dev multi-window module features (class -> 15-dim) ----------
dev_ms=defaultdict(dict)   # class -> window -> [5 cats]
for r in csv.DictReader(open(f"{RB}/developmental_module_scores.csv")):
    dev_ms[r["class"]][r["window"]]=[float(r[c]) for c in CATS]
def dev_vec(c):
    if c not in dev_ms: return None
    # require E3&E4; impute missing E2 with E3
    d=dev_ms[c]
    if "E3_390_510" not in d or "E4_ge510" not in d: return None
    e2=d.get("E2_270_390", d["E3_390_510"])
    return np.array(e2+d["E3_390_510"]+d["E4_ge510"], float)   # 15 dims
# adult module (class -> 5-dim)
adult_ms={}
for r in csv.DictReader(open(f"{RB}/adult_module_scores.csv")):
    adult_ms[r["class"]]=np.array([float(r[c]) for c in CATS],float)

# ---------- class -> modality, birth-time proxy ----------
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
DEVSET=set(dev_ms)
def modality(nm):
    r=prof.get(nm)
    if r:
        if r.get("is_sensory")in("1","1.0","True"): return "sensory"
        if r.get("is_command")in("1","1.0","True"): return "command"
    base=nm[:-1] if nm[-1:] in "LR" else nm
    if base in MOTOR or nm in MOTOR: return "motor"
    return "inter"
cls_mod_votes=defaultdict(list)
for nm in names:
    c=canon(nm,DEVSET)
    if c: cls_mod_votes[c].append(modality(nm))
cls_mod={c:Counter(v).most_common(1)[0][0] for c,v in cls_mod_votes.items()}
# birth-time proxy = first embryo.time class identity appears
NON=("parent","grandparent","Neuroblast","possibly","_and_")
first_t=defaultdict(lambda:1e9)
for row in csv.DictReader(open(f"{DATA}/expression/scrna_seq_other/GSE126954/GSE126954_cell_annotation.csv")):
    lab=None
    for f in (row['plot.cell.type'],row['cell.subtype']):
        if f and f!='NA' and re.match(r'^[A-Z]{2,4}\d?$',f) and not any(t in f for t in NON): lab=f; break
    if lab in DEVSET:
        try: t=float(row['embryo.time']); first_t[lab]=min(first_t[lab],t)
        except: pass
birth={k:v for k,v in first_t.items() if v<1e9}

# ---------- contact matrix -> neuron pairs, mapped to dev class ----------
wb=openpyxl.load_workbook(CONTACT,read_only=True,data_only=True); ws=wb['adult nerve ring neighbors']
rows=list(ws.iter_rows(values_only=True)); hdr=[h for h in rows[0][1:]]
area={}
for row in rows[1:]:
    rn=row[0]
    if rn is None: continue
    for j,val in enumerate(row[1:]):
        if val in (None,''): continue
        cn=hdr[j]
        try: a=float(val)
        except: continue
        if a>0 and rn!=cn:
            k=(min(rn,cn),max(rn,cn)); area[k]=max(area.get(k,0),a)
neurons=[n for n in hdr if n and canon(n,DEVSET)]
ncls={n:canon(n,DEVSET) for n in neurons}
print(f"mapped nerve-ring neurons: {len(neurons)} -> {len(set(ncls.values()))} dev classes")

# build all pairs among mapped neurons; label = contact present
pairs=[]; y=[]
for i in range(len(neurons)):
    for j in range(i+1,len(neurons)):
        a,b=neurons[i],neurons[j]
        if ncls[a]==ncls[b]: continue   # skip within-class (identical features)
        pairs.append((a,b)); k=(min(a,b),max(a,b)); y.append(1 if k in area else 0)
y=np.array(y); print(f"pairs: {len(pairs)} | contacts: {y.sum()} ({100*y.mean():.1f}%)")

# ---------- feature builders ----------
def sym(a,b): return np.concatenate([a+b, np.abs(a-b), a*b])
def cross_terms(a,b):  # guidance(0-indexed cat order: adh,guid,nt,syn,term) x adhesion/receptor
    # use E4 window slice (last 5) for maturity + E3 (middle 5)
    def cat(v,w,ci): return v[w*5+ci]
    out=[]
    for w in (1,2):  # E3,E4
        gA,gB=cat(a,w,1),cat(b,w,1)          # guidance
        adA,adB=cat(a,w,0),cat(b,w,0)        # adhesion
        rA,rB=cat(a,w,2),cat(b,w,2)          # nt receptor
        out += [gA*adB+gB*adA, gA*rB+gB*rA]  # guidance x adhesion, guidance x receptor
    return np.array(out)

def build(feat_kind):
    X=[]
    for a,b in pairs:
        ca,cb=ncls[a],ncls[b]
        if feat_kind=="dev":
            va,vb=dev_vec(ca),dev_vec(cb)
            X.append(np.concatenate([sym(va,vb), cross_terms(va,vb)]))
        elif feat_kind=="adult":
            va,vb=adult_ms.get(ca),adult_ms.get(cb)
            X.append(sym(va,vb))
        elif feat_kind=="modality":
            ma,mb=cls_mod.get(ca,"?"),cls_mod.get(cb,"?")
            key=tuple(sorted((ma,mb)))
            X.append(key)
        elif feat_kind=="birth":
            ba,bb=birth.get(ca,np.nan),birth.get(cb,np.nan)
            X.append([np.nanmin([ba,bb]),np.nanmax([ba,bb]),abs(ba-bb) if not(np.isnan(ba) or np.isnan(bb)) else 0.0, (ba+bb)/2 if not(np.isnan(ba) or np.isnan(bb)) else 0.0])
    if feat_kind=="modality":
        keys=sorted(set(X)); ki={k:i for i,k in enumerate(keys)}
        M=np.zeros((len(X),len(keys)))
        for r,k in enumerate(X): M[r,ki[k]]=1
        return M
    return np.array(X,float)

# validity mask: all feature kinds available
ok=np.array([dev_vec(ncls[a]) is not None and dev_vec(ncls[b]) is not None
             and ncls[a] in adult_ms and ncls[b] in adult_ms
             and ncls[a] in birth and ncls[b] in birth for a,b in pairs])
pairs=[p for p,k in zip(pairs,ok) if k]; y=y[ok]
print(f"pairs with full features: {len(pairs)} | contacts {y.sum()} ({100*y.mean():.1f}%)")
grp=np.array([hash(ncls[a])^hash(ncls[b]) for a,b in pairs])  # not used; group by class below

# ---------- evaluation: edge-CV + class-disjoint CV ----------
def edge_cv(X,y,seed=0):
    clf=make_pipeline(StandardScaler(with_mean=False), GradientBoostingClassifier(random_state=seed))
    cv=StratifiedKFold(5,shuffle=True,random_state=seed)
    p=cross_val_predict(clf,X,y,cv=cv,method="predict_proba")[:,1]
    p=np.clip(p,1e-6,1-1e-6)
    return roc_auc_score(y,p), log_loss(y,p)

def classdisjoint_cv(X,y,pairs,ncls,nsplit=5,seed=0):
    classes=sorted(set(c for p in pairs for c in (ncls[p[0]],ncls[p[1]])))
    rng=np.random.default_rng(seed); rng.shuffle(classes)
    folds=np.array_split(classes,nsplit); cf={c:i for i,f in enumerate(folds) for c in f}
    aucs=[]; preds=np.full(len(y),np.nan)
    for k in range(nsplit):
        test=np.array([cf[ncls[a]]==k and cf[ncls[b]]==k for a,b in pairs])
        train=np.array([cf[ncls[a]]!=k and cf[ncls[b]]!=k for a,b in pairs])
        if test.sum()<10 or y[train].sum()<5 or y[test].sum()<3: continue
        clf=make_pipeline(StandardScaler(with_mean=False),GradientBoostingClassifier(random_state=seed))
        clf.fit(X[train],y[train]); preds[test]=clf.predict_proba(X[test])[:,1]
    m=~np.isnan(preds)
    if m.sum()<20 or len(set(y[m]))<2: return float("nan"),int(m.sum())
    return roc_auc_score(y[m],preds[m]), int(m.sum())

Xdev=build("dev"); Xad=build("adult"); Xmod=build("modality"); Xbt=build("birth")
def hstack(*a): return np.hstack(a)
LAD={
 "0 base-rate":            None,
 "1 modality":             Xmod,
 "2 birthtime":            Xbt,
 "3 modality+birthtime":   hstack(Xmod,Xbt),
 "4 adult-expr":           Xad,
 "5 dev-expr":             Xdev,
 "6 modality+bt+adult":    hstack(Xmod,Xbt,Xad),
 "7 modality+bt+dev":      hstack(Xmod,Xbt,Xdev),
}
print("\n================ CONTACT-PRESENCE MODEL LADDER ================")
print(f"{'model':26s} {'edgeCV AUC':>11s} {'edge logloss':>13s} {'classCV AUC':>12s}")
base=y.mean()
for name,X in LAD.items():
    if X is None:
        print(f"{name:26s} {'-':>11s} {'-':>13s} {'-':>12s}  (rate {base:.3f})"); continue
    a_e,ll_e=edge_cv(X,y); a_c,ncov=classdisjoint_cv(X,y,pairs,ncls)
    print(f"{name:26s} {a_e:>11.4f} {ll_e:>13.4f} {a_c:>12.4f}")

# ---------- controls on dev-expr ----------
print("\n================ CONTROLS (dev-expr) ================")
# permute dev features across classes
cls_list=sorted(set(ncls.values()))
perm_auc=[]
for s in range(20):
    rng=np.random.default_rng(100+s)
    shuf=dict(zip(cls_list, rng.permutation(cls_list)))
    ncls_sh={n:shuf[c] for n,c in ncls.items()}
    Xs=[]
    for a,b in pairs:
        va,vb=dev_vec(ncls_sh[a]),dev_vec(ncls_sh[b]); Xs.append(np.concatenate([sym(va,vb),cross_terms(va,vb)]))
    perm_auc.append(edge_cv(np.array(Xs),y,seed=s)[0])
perm_auc=np.array(perm_auc)
dev_e=edge_cv(Xdev,y)[0]
print(f"dev-expr edgeCV AUC={dev_e:.4f} | class-permuted null mean={perm_auc.mean():.4f} "
      f"95pct={np.percentile(perm_auc,95):.4f} | p={(1+(perm_auc>=dev_e).sum())/(1+len(perm_auc)):.3f}")
print("\nGATE: dev-expr must (a) beat modality+birthtime (adds AUC), (b) survive class-disjoint CV,")
print("      (c) beat class-permuted null. Random-matched-gene control -> devexpr_null_controls.py (Phase 8).")
