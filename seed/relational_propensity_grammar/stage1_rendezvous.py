#!/usr/bin/env python3
"""STAGE 1 (one-object canalization gradient): does EARLY competence-gated RENDEZVOUS predict canalization
(edge reproducibility) BEYOND static contact magnitude? Dev-INDEPENDENT predictors only: contact-area
trajectory (c_L1..c_adult, geometry) + birth timing (competence proxy). NOT synapse history (circular).
Core test: contact-matched partial effect of early rendezvous. Make-or-break: does timing move the
dispersion ratio beyond static contact?"""
import csv, re, os, math
import numpy as np
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

C="../data/celegans/data"
OUT=os.path.expanduser("../seed/data/processed/relational_propensity_grammar")
MASTER=os.path.expanduser("../seed/data/processed/contact_decomposition/contact_decomposition_master.csv")
EDGES=f"{C}/connectome/witvliet2020/witvliet_combined_edges.csv"

# reproducibility target from the two adults
ae=defaultdict(set)
for r in csv.DictReader(open(EDGES)):
    if r['type']=='electrical': continue
    ae[int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))].add((r['pre'],r['post']))
W7,W8=ae[7],ae[8]

rows=list(csv.DictReader(open(MASTER)))
def f(r,k,d=0.0):
    try: return float(r[k])
    except: return d
# build arrays; the master is UNDIRECTED pairs (na,nb). reproducibility = present in both adults (either direction)
def present(a,b,S): return ((a,b) in S) or ((b,a) in S)
recs=[]
for r in rows:
    a,b=r['na'],r['nb']
    if f(r,'c_adult')<=0: continue      # contact-positive
    in7=present(a,b,W7); in8=present(a,b,W8)
    adult_present = in7 or in8
    recs.append(dict(a=a,b=b,
        c_L1=f(r,'c_L1'),c_L2=f(r,'c_L2'),c_L3=f(r,'c_L3'),c_adult=f(r,'c_adult'),
        first_contact_stage={'L1':0,'L2':1,'L3':2,'adult':3}.get(r.get('first_contact_stage','L1'),0),
        contact_persist=f(r,'contact_persist'),
        pre_birth=f(r,'pre_birth',np.nan),post_birth=f(r,'post_birth',np.nan),
        both_preL1=f(r,'both_born_preL1'),
        strong_core=f(r,'adult_strong_core1'),
        repro=1 if (in7 and in8) else 0,      # canalized: present in BOTH adults
        adult_present=1 if adult_present else 0,
        in7=1 if in7 else 0,in8=1 if in8 else 0))
print(f"contact-positive undirected pairs: {len(recs)} | adult-present: {sum(x['adult_present'] for x in recs)} | reproducible(both): {sum(x['repro'] for x in recs)}")

neu=sorted(set(x['a'] for x in recs)|set(x['b'] for x in recs))
rng=np.random.default_rng(0); fold=np.array_split(rng.permutation(neu),5); nf={n:i for i,ff in enumerate(fold) for n in ff}
def birthmax(x):
    v=[x['pre_birth'],x['post_birth']]; v=[z for z in v if not math.isnan(z)]
    return max(v) if v else 400.0
# feature sets (ALL dev-independent: contact geometry + birth timing)
def feat_static(x): return [math.log1p(x['c_adult'])]
def feat_timing(x): return [math.log1p(x['c_L1']),math.log1p(x['c_L2']),math.log1p(x['c_L3']),
                            x['first_contact_stage'],x['contact_persist'],x['both_preL1'],birthmax(x)/1000.0]
def rendezvous_integral(x):
    # competence-gated early contact: sum of early-stage contact when both plausibly competent (born pre-L1)
    g=1.0 if x['both_preL1']>=0.5 else 0.5
    return g*(x['c_L1']*3+x['c_L2']*2+x['c_L3']*1+x['c_adult']*0.5)  # weight EARLY contact more

def ndcv(Xf,y):
    y=np.array(y); p=np.full(len(y),np.nan); pr=[(nf[x['a']],nf[x['b']]) for x in recs]
    for k in range(5):
        te=np.array([fa==k and fb==k for fa,fb in pr]); tr=np.array([fa!=k and fb!=k for fa,fb in pr])
        if te.sum()<10 or y[tr].sum()<5: continue
        clf=make_pipeline(StandardScaler(with_mean=False),GradientBoostingClassifier(random_state=0)); clf.fit(Xf[tr],y[tr]); p[te]=clf.predict_proba(Xf[te])[:,1]
    return p

y_repro=np.array([x['repro'] for x in recs])
y_adult=np.array([x['adult_present'] for x in recs])
Xstat=np.array([feat_static(x) for x in recs])
Xtime=np.array([feat_static(x)+feat_timing(x) for x in recs])

# ===== TEST A: predict CANALIZATION (reproducible=both adults). Does timing beat static contact? =====
p_s=ndcv(Xstat,y_repro); p_t=ndcv(Xtime,y_repro); m=(~np.isnan(p_s))&(~np.isnan(p_t))
print("\n=== TEST A: predict reproducible/canalized edge (present in BOTH adults) ===")
print(f"  static contact (c_adult)        AUC={roc_auc_score(y_repro[m],p_s[m]):.3f}")
print(f"  + rendezvous timing             AUC={roc_auc_score(y_repro[m],p_t[m]):.3f}  (d={roc_auc_score(y_repro[m],p_t[m])-roc_auc_score(y_repro[m],p_s[m]):+.3f})")

# ===== TEST B (CORE): contact-MATCHED partial effect of early rendezvous =====
ca=np.array([x['c_adult'] for x in recs]); R=np.array([rendezvous_integral(x) for x in recs])
fcs=np.array([x['first_contact_stage'] for x in recs]); cL1=np.array([x['c_L1'] for x in recs])
q=np.quantile(ca,[0.25,0.5,0.75]); qz=np.digitize(ca,q)
print("\n=== TEST B (CORE): within adult-contact-area quartiles, does EARLY contact-onset predict canalization? ===")
print("  (isolates rendezvous TIMING from contact MAGNITUDE)")
for qi in range(4):
    s=qz==qi
    if s.sum()<20: continue
    early=s&(fcs==0)  # contact present already at L1
    late=s&(fcs>0)    # contact appears later
    if early.sum()>=8 and late.sum()>=8:
        print(f"  c_adult Q{qi+1} (n={s.sum():4d}): repro | early-contact(L1)={y_repro[early].mean():.3f} (n={early.sum():3d})  vs later-contact={y_repro[late].mean():.3f} (n={late.sum():3d})")
# pooled partial: logistic repro ~ log c_adult + first_contact_stage (does onset add beyond magnitude?)
from scipy.stats import spearmanr
Xb=np.column_stack([np.log1p(ca),fcs,np.log1p(cL1)]); Xbs=StandardScaler().fit_transform(Xb)
lr=LogisticRegression(max_iter=2000).fit(Xbs,y_repro);
print(f"  pooled logistic repro ~ [log c_adult, first_contact_stage, log c_L1] standardized coefs:")
for nm,cf in zip(["log c_adult(magnitude)","first_contact_stage(later=+)","log c_L1(early mag)"],lr.coef_[0]):
    print(f"     {nm:32s} beta={cf:+.3f}  ({'promotes' if cf>0 else 'suppresses'} canalization)")

# ===== TEST C (make-or-break): does timing move the DISPERSION RATIO beyond static contact? =====
# propensity for a single adult; dispersion = |in7-in8| / 2p(1-p). Predictors dev-independent.
y7=np.array([x['in7'] for x in recs]); y8=np.array([x['in8'] for x in recs])
def ndcv_y(Xf,y):
    y=np.array(y); p=np.full(len(y),np.nan); pr=[(nf[x['a']],nf[x['b']]) for x in recs]
    for k in range(5):
        te=np.array([fa==k and fb==k for fa,fb in pr]); tr=np.array([fa!=k and fb!=k for fa,fb in pr])
        if te.sum()<10 or y[tr].sum()<5: continue
        clf=make_pipeline(StandardScaler(with_mean=False),GradientBoostingClassifier(random_state=0)); clf.fit(Xf[tr],y[tr]); p[te]=clf.predict_proba(Xf[te])[:,1]
    return p
def disp(Xf):
    p=np.nanmean(np.vstack([ndcv_y(Xf,y7),ndcv_y(Xf,y8)]),axis=0); mm=~np.isnan(p)
    obs=np.abs(y7[mm]-y8[mm]).sum(); pred=(2*p[mm]*(1-p[mm])).sum(); return obs/pred, roc_auc_score(((y7|y8)>0).astype(int)[mm],p[mm])
ds,as_=disp(Xstat); dt,at_=disp(Xtime)
print("\n=== TEST C (make-or-break): dispersion ratio, dev-independent predictors ===")
print(f"  static contact (c_adult)   dispersion={ds:.3f}  AUC={as_:.3f}")
print(f"  + rendezvous timing        dispersion={dt:.3f}  AUC={at_:.3f}   (toward 1 by {dt-ds:+.3f})")
print("  (frame: dynamic timing closes MORE under-dispersion than static endpoint => rendezvous carries reproducibility)")

with open(f"{OUT}/stage1_rendezvous.csv","w",newline="") as fo:
    w=csv.writer(fo); w.writerow(["test","static","+timing"])
    w.writerow(["canalization_AUC",round(roc_auc_score(y_repro[m],p_s[m]),4),round(roc_auc_score(y_repro[m],p_t[m]),4)])
    w.writerow(["dispersion_ratio",round(ds,4),round(dt,4)])
    w.writerow(["dispersion_AUC",round(as_,4),round(at_,4)])
    w.writerow(["logit_first_contact_stage_beta","",round(float(lr.coef_[0][1]),4)])
print("\nStage 1 done ->",OUT)
