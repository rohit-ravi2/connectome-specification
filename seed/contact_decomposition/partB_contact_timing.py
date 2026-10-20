#!/usr/bin/env python3
"""Part B (CRUX): developmental contact timing/persistence + ELIGIBILITY-RESTRICTED test.
Attacks the birth-time definitional confound: among pairs BOTH born pre-L1 AND contacting at L1,
does anything predict which form an L1 synapse? And does contact TIMING/PERSISTENCE explain synapse
establishment beyond adult contact area?"""
import csv, os, math
import numpy as np
from collections import Counter, defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

OUT=os.path.expanduser("../seed/data/processed/contact_decomposition")
rows=list(csv.DictReader(open(f"{OUT}/contact_decomposition_master.csv")))
def fl(x):
    try: return float(x)
    except: return np.nan
for r in rows:
    for k in ['c_L1','c_L2','c_L3','c_adult','logc_L1','logc_adult','frac_src','frac_tgt','frac_src_L1',
              'contact_persist','pre_birth','post_birth']:
        r[k]=fl(r[k])
    for k in ['T_first_L1','T_preadult_persist','adult_strong_core1','adult_strong_core2','both_born_preL1','contact_no_syn']:
        r[k]=int(r[k])

def ndcv(X,y,pairs,seed=0):
    neu=sorted(set(a for a,b in pairs)|set(b for a,b in pairs))
    rng=np.random.default_rng(seed); f=np.array_split(rng.permutation(neu),5); nf={n:i for i,ff in enumerate(f) for n in ff}
    pred=np.full(len(y),np.nan)
    for k in range(5):
        te=np.array([nf[a]==k and nf[b]==k for a,b in pairs]); tr=np.array([nf[a]!=k and nf[b]!=k for a,b in pairs])
        if te.sum()<8 or y[tr].sum()<4 or y[te].sum()<3 or len(set(y[te]))<2: continue
        clf=make_pipeline(StandardScaler(with_mean=False),GradientBoostingClassifier(random_state=seed)); clf.fit(X[tr],y[tr]); pred[te]=clf.predict_proba(X[te])[:,1]
    m=~np.isnan(pred); return roc_auc_score(y[m],pred[m]) if (m.sum()>15 and len(set(y[m]))>1) else np.nan
def m3(X,y,pairs): return np.nanmean([ndcv(X,y,pairs,s) for s in range(3)])

# ============ B1: contact->synapse ordering (already: synapses never precede contact) ============
print("=== B1: first-contact vs first-synapse (delay) ===")
withsyn=[r for r in rows if r['first_synapse_stage']!='none' and r['first_contact_stage']!='none']
order={'L1':0,'L2':1,'L3':2,'adult':3}
delays=[order[r['first_synapse_stage']]-order[r['first_contact_stage']] for r in withsyn]
print(f"  pairs with both contact+synapse: {len(withsyn)}")
print(f"  synapse-before-contact (delay<0): {sum(1 for d in delays if d<0)} (should be ~0)")
print(f"  same-stage (delay=0): {sum(1 for d in delays if d==0)} | later (delay>0): {sum(1 for d in delays if d>0)}")

# ============ B3: ELIGIBILITY-RESTRICTED test (the crux) ============
print("\n=== B3: eligibility-restricted (both born pre-L1 AND contact at L1) ===")
elig=[r for r in rows if r['both_born_preL1']==1 and r['c_L1']>0]
pairs=[(r['na'],r['nb']) for r in elig]
y=np.array([r['T_first_L1'] for r in elig])
print(f"  eligible L1 contacts: {len(elig)} | of which form L1 synapse: {y.sum()} ({100*y.mean():.1f}%)")
def feat(keys,rr): return np.array([[r[k] for k in keys] for r in rr])
def birthdiff(rr): return np.array([[abs(r['pre_birth']-r['post_birth'])] for r in rr])
Xarea=feat(['logc_L1'],elig)
Xareabudget=feat(['logc_L1','frac_src_L1','contact_persist'],elig)
Xbd=birthdiff(elig)
res={}
for lab,X in [('L1-contact-area',Xarea),('birthtime-diff-ALONE',Xbd),
              ('L1-area+budget+persist',Xareabudget),
              ('L1-area + birthdiff',np.hstack([Xarea,Xbd])),
              ('L1-area+budget+persist + birthdiff',np.hstack([Xareabudget,Xbd]))]:
    a=m3(X,y,pairs); res[lab]=a; print(f"    {lab:36s} neuron-disjoint AUC={a:.3f}")
print(f"  >> ADD of birthtime-diff over L1-contact-area: {res['L1-area + birthdiff']-res['L1-contact-area']:+.3f}")
print(f"  >> ADD of birthtime-diff over L1-area+budget+persist: {res['L1-area+budget+persist + birthdiff']-res['L1-area+budget+persist']:+.3f}")
print("  (if birthdiff adds ~0 among ELIGIBLE contacts -> prior birth-time effect was DEFINITIONAL/eligibility)")

# ============ B2: does contact PERSISTENCE explain synapse persistence / adult core beyond adult area? ============
print("\n=== B2: contact persistence vs adult contact area -> adult strong-core ===")
sub=[r for r in rows if r['c_adult']>0]   # pairs contacting in adult
p2=[(r['na'],r['nb']) for r in sub]; y2=np.array([r['adult_strong_core1'] for r in sub])
Xaa=feat(['logc_adult'],sub); Xpers=feat(['contact_persist'],sub); Xboth=feat(['logc_adult','contact_persist','c_L1_count' if 'c_L1_count' in sub[0] else 'contact_persist'],sub)
# fix c_L1_count numeric
for r in sub: r['c_L1_count']=fl(r['c_L1_count'])
Xboth=np.array([[r['logc_adult'],r['contact_persist'],r['c_L1_count']] for r in sub])
for lab,X in [('adult-contact-area',Xaa),('contact-persistence-alone',Xpers),('area+persistence+L1count',Xboth)]:
    print(f"    {lab:30s} neuron-disjoint AUC={m3(X,y2,p2):.3f}  (target adult strong-core, N={len(sub)}, pos={y2.sum()})")

# ============ B3b: conversion rate by L1 contact reproducibility ============
print("\n=== B3b: L1-synapse conversion by L1-contact reproducibility (count across 4 L1 animals) ===")
for r in rows: r['c_L1_count']=fl(r['c_L1_count'])
byc=defaultdict(lambda:[0,0])
for r in elig:
    c=int(r['c_L1_count']); byc[c][0]+=r['T_first_L1']; byc[c][1]+=1
for c in sorted(byc):
    a,b=byc[c]
    if b>=10: print(f"    contact in {c}/4 L1 animals: {a}/{b} = {100*a/b:.1f}% form L1 synapse")

with open(f"{OUT}/partB_eligible_contact_results.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["test","feature_set","neuron_disjoint_AUC"])
    for lab,a in res.items(): w.writerow(["eligibility_restricted_T_first_L1",lab,round(a,4)])
print("\nPart B done ->",OUT)
