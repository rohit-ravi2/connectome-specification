#!/usr/bin/env python3
"""FUNNEL / TEMPORAL-CANALIZATION test. Is the early-window canalization a GLOBAL developmental-clock
contraction (low-dim, nested, coherent developmental trajectory = funnel) or EDGE-SPECIFIC determinism
(independent per-edge sharpness, random deviations)?

Signature of a contractive canalized trajectory: NESTED (Guttman) structure across the 8-animal
developmental series — edges added in a canalized order, animals differ by developmental maturity, few
off-schedule deviations. Decisive quantity: what FRACTION of animal-to-animal disagreements is explained
by a single developmental-maturity axis (nested) vs idiosyncratic (anti-nested)?
Controls: null (prevalence-preserving random assignment) + neuron-stable subset (nesting not from neuron birth)."""
import csv, re, os
import numpy as np
from collections import defaultdict

C="../data/celegans/data"
OUT=os.path.expanduser("../seed/data/processed/relational_propensity_grammar")
EDGES=f"{C}/connectome/witvliet2020/witvliet_combined_edges.csv"

ae=defaultdict(set)
for r in csv.DictReader(open(EDGES)):
    if r['type']=='electrical': continue
    ae[int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))].add((r['pre'],r['post']))
# developmental age order (Witvliet 2021 approx hours): d1=0 d2=5 d3=8 d4=16 (L1) d5=23(L2) d6=27(L3) d7=45 d8=45(adult)
animals=[1,2,3,4,5,6,7,8]; age=[0,5,8,16,23,27,45,45]
# neurons present in each animal (appear in any edge)
present_neurons={a:set(x for e in ae[a] for x in e) for a in animals}
stable_neurons=set.intersection(*[present_neurons[a] for a in animals])  # present in ALL 8 -> birth-independent
print(f"neurons present in all 8 animals (stable): {len(stable_neurons)}")

def build_matrix(edge_filter):
    active=sorted(set(e for a in animals for e in ae[a] if edge_filter(e)))
    B=np.array([[1 if e in ae[a] else 0 for e in active] for a in animals])  # 8 x P
    return B, active

def guttman(B, age_rank):
    """Coefficient of reproducibility against a FIXED animal age ordering.
    Edge with prevalence k predicted present in the k most-mature animals; errors = deviations."""
    order=np.argsort(age_rank)  # youngest->oldest indices
    Bo=B[order]                 # rows youngest->oldest
    n,P=Bo.shape; errors=0; disagree_cells=0; nested=0; antinested=0
    for j in range(P):
        col=Bo[:,j]; k=int(col.sum())
        if k==0 or k==n: continue
        pred=np.zeros(n,int); pred[n-k:]=1     # present in k oldest
        errors+=int((col!=pred).sum())
        # disagreement decomposition: is the variation nested (young lacks, old has) or anti-nested?
        # count monotonicity: for each younger<older pair, nested if not (younger has & older lacks)
        for i in range(n):
            for ii in range(i+1,n):
                if col[i]!=col[ii]:
                    disagree_cells+=1
                    if col[i]<=col[ii]: nested+=1      # younger<=older (edge added with age) = nested
                    else: antinested+=1
    CR=1-errors/(n*P)
    frac_nested=nested/(nested+antinested) if (nested+antinested)>0 else np.nan
    return CR, frac_nested, P

def null_CR(B, age_rank, nb=300):
    n,P=B.shape; rng=np.random.default_rng(0); crs=[]
    for s in range(nb):
        Bn=np.zeros_like(B)
        for j in range(P):
            k=int(B[:,j].sum())
            idx=np.random.default_rng(s*1000+j).choice(n,k,replace=False); Bn[idx,j]=1  # random animals, preserve prevalence
        crs.append(guttman(Bn,age_rank)[0])
    return float(np.mean(crs)), float(np.std(crs))

age_rank=np.array(age)
print("\n=== FUNNEL TEST: is developmental variation NESTED (funnel) or idiosyncratic (edge-specific)? ===")
for label,filt in [("ALL edges", lambda e: True),
                    ("neuron-STABLE subset (birth-independent)", lambda e: e[0] in stable_neurons and e[1] in stable_neurons)]:
    B,active=build_matrix(filt)
    CR,fn,P=guttman(B,age_rank); nCR,nsd=null_CR(B,age_rank)
    z=(CR-nCR)/(nsd+1e-9)
    print(f"\n  [{label}]  animals=8  variable-edges P={P}")
    print(f"    Guttman CR (real) = {CR:.3f}   null(prevalence-preserving) = {nCR:.3f}±{nsd:.3f}   z={z:.1f}")
    print(f"    fraction of disagreements that are AGE-NESTED (funnel) = {fn:.3f}  (0.5=random, ->1=pure developmental clock)")

# within-L1 contraction (4 L1 animals, ages 0-16h)
print("\n=== within-L1 contraction (4 L1 animals, ages 0/5/8/16h) ===")
L1a=[1,2,3,4];
active=sorted(set(e for a in L1a for e in ae[a] if e[0] in stable_neurons and e[1] in stable_neurons))
BL1=np.array([[1 if e in ae[a] else 0 for e in active] for a in L1a])
CRl,fnl,Pl=guttman(BL1,np.array([0,5,8,16]))
print(f"  L1 Guttman CR={CRl:.3f}  fraction age-nested={fnl:.3f}  (P={Pl})  => do younger L1 nest within older L1?")

# convergence to adult target (recall of adult-consensus edges by age)
target=ae[7]&ae[8]
print("\n=== convergence to adult target (recall of adult-consensus edges by developmental age) ===")
for a,ag in zip(animals,age):
    rec=len(ae[a]&target)/len(target) if target else 0
    extra=len(ae[a]-target)/max(1,len(ae[a]))
    print(f"  d{a} (age~{ag:2d}h): recall of adult-core={rec:.3f}  non-target fraction={extra:.3f}")
with open(f"{OUT}/stage2_funnel.csv","w",newline="") as fo:
    w=csv.writer(fo); w.writerow(["subset","CR_real","CR_null","z","frac_age_nested"])
    for label,filt in [("all",lambda e:True),("stable",lambda e:e[0] in stable_neurons and e[1] in stable_neurons)]:
        B,_=build_matrix(filt); CR,fn,_=guttman(B,age_rank); nCR,nsd=null_CR(B,age_rank)
        w.writerow([label,round(CR,4),round(nCR,4),round((CR-nCR)/(nsd+1e-9),2),round(fn,4)])
print("\nStage 2 funnel test done ->",OUT)
