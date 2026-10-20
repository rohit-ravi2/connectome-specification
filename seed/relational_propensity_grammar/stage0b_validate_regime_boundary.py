#!/usr/bin/env python3
"""Validate the regime boundary NON-CIRCULARLY before Stage 1. Question: are LATE-formed edges genuinely
OVER-dispersed (a separate stochastic regime) or just AT independence (uncanalized, one-object gradient)?
Calibration-FREE test: pairwise disagreement among independent adults CANNOT exceed 0.5 under ANY
independent-draw model at any propensity. If a clean late stratum's disagreement > 0.5 (CI clears it) =>
genuine over-dispersion. Else collapse to one-object gradient.
De-circularized: formation time labeled ONLY from dev datasets 1-6; reproducibility measured on W7,W8,+Cook."""
import csv, re, os
import numpy as np, openpyxl
from collections import defaultdict

C="../data/celegans/data"
OUT=os.path.expanduser("../seed/data/processed/relational_propensity_grammar")
EDGES=f"{C}/connectome/witvliet2020/witvliet_combined_edges.csv"
CONTACT=f"{C}/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx"
COOK=os.path.expanduser("../seed/data/external/mechanistic_frontier/cook2019_adult/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx")

animal_edges=defaultdict(set)
for r in csv.DictReader(open(EDGES)):
    if r['type']=='electrical': continue
    a=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1)); animal_edges[a].add((r['pre'],r['post']))
L1set=animal_edges[1]|animal_edges[2]|animal_edges[3]|animal_edges[4]; L2set=animal_edges[5]; L3set=animal_edges[6]
W7,W8=animal_edges[7],animal_edges[8]
wb=openpyxl.load_workbook(CONTACT,read_only=True,data_only=True); ws=wb['adult nerve ring neighbors']
rows=list(ws.iter_rows(values_only=True)); alln=[h for h in rows[0][1:]]; ix={n:i for i,n in enumerate(alln) if n}
M=len(alln); A=np.zeros((M,M))
for row in rows[1:]:
    if row[0] in ix:
        for j,v in enumerate(row[1:]):
            if v not in (None,''):
                try:A[ix[row[0]],j]=float(v)
                except:pass
A=np.maximum(A,A.T)
pairs=[(alln[i],alln[j]) for i in range(M) for j in range(M) if i!=j and alln[i] and alln[j] and A[i,j]>0]

# ---- Cook HERMAPHRODITE chemical (independent adult) ----
def load_cook_herm():
    wbk=openpyxl.load_workbook(COOK,read_only=True,data_only=True)
    sh=[s for s in wbk.sheetnames if 'herm' in s.lower() and 'chem' in s.lower()]
    print("Cook sheets:",wbk.sheetnames,"| using:",sh[0] if sh else None)
    ws2=wbk[sh[0]]; rr=list(ws2.iter_rows(values_only=True))
    hdr=[str(x) if x is not None else '' for x in rr[0][1:]]; E=set()
    for row in rr[1:]:
        pre=str(row[0]) if row[0] else ''
        if pre not in ix: continue
        for j,v in enumerate(row[1:]):
            if j<len(hdr) and hdr[j] in ix and v not in (None,'',0):
                try:
                    if float(v)>0: E.add((pre,hdr[j]))
                except:pass
    return E
cook=load_cook_herm(); print(f"Cook herm chem directed edges (nerve-ring): {len(cook)}")

def first_stage(a,b):
    if (a,b) in L1set: return 'L1'
    if (a,b) in L2set: return 'L2'
    if (a,b) in L3set: return 'L3'
    if (a,b) in W7 or (a,b) in W8: return 'adult'
    return 'never'
fstage=np.array([first_stage(a,b) for a,b in pairs])

# presence across 3 independent adults
pres=np.array([[1 if (a,b) in W7 else 0,1 if (a,b) in W8 else 0,1 if (a,b) in cook else 0] for a,b in pairs])
k=pres.sum(1)  # 0..3 adults with the edge
# pairwise disagreement among the 3 adults = fraction of the 3 pairs that differ = k(3-k)/3  (0,2/3,2/3,0 for k=0,1,2,3)
pair_disagree=k*(3-k)/3.0
# W7/W8 only disagreement (calibration-free)
d78=np.abs(pres[:,0]-pres[:,1]).astype(float)

def boot_mean(vals,nb=2000):
    if len(vals)<10: return (np.nan,np.nan,np.nan)
    rs=[np.random.default_rng(s).choice(vals,len(vals),replace=True).mean() for s in range(nb)]
    return (float(np.mean(vals)),float(np.percentile(rs,5)),float(np.percentile(rs,95)))

print("\n=== CALIBRATION-FREE regime test (label from dev datasets 1-6; hard ceiling 0.5 under independence) ===")
print(f"{'stratum':8s} {'n':>4s} | {'W7/W8 disagree':>22s} | {'3-adult pairwise disagree':>26s} | over-disp?")
for st in ['L1','L2','L3','adult']:
    s=fstage==st; n=int(s.sum())
    m78,lo78,hi78=boot_mean(d78[s]); m3,lo3,hi3=boot_mean(pair_disagree[s])
    # over-dispersion (genuine separate regime) requires disagreement CI to CLEAR 0.5
    over = (lo78>0.5) or (lo3>0.5)
    circ = " (CIRCULAR label)" if st=='adult' else ""
    print(f"{st:8s} {n:4d} | {m78:.3f} [{lo78:.3f},{hi78:.3f}] | {m3:.3f} [{lo3:.3f},{hi3:.3f}] | {'YES' if over else 'no'}{circ}")
print("  hard ceiling: independent-draw disagreement <= 0.5 ALWAYS. >0.5 (CI clears) = genuine over-dispersion.")

# distribution of k among late clean (L3) vs early (L1): over-dispersion => excess of k=1,2 (split), under => excess k=0 or 3
print("\n=== presence-count distribution across 3 adults (k = #adults with edge) ===")
for st in ['L1','L2','L3','adult']:
    s=fstage==st
    if s.sum()<10: continue
    dist=[int((k[s]==kk).sum()) for kk in range(4)]
    frac=[d/s.sum() for d in dist]
    print(f"  {st:6s} n={int(s.sum()):4d}  k=0:{frac[0]:.2f} k=1:{frac[1]:.2f} k=2:{frac[2]:.2f} k=3:{frac[3]:.2f}  (canalized=extreme k0/k3; stochastic=k1/k2)")
with open(f"{OUT}/stage0b_regime_validation.csv","w",newline="") as fo:
    w=csv.writer(fo); w.writerow(["stratum","n","w78_disagree","w78_lo","w78_hi","adult3_disagree","a3_lo","a3_hi","over_dispersed"])
    for st in ['L1','L2','L3','adult']:
        s=fstage==st; m78,lo78,hi78=boot_mean(d78[s]); m3,lo3,hi3=boot_mean(pair_disagree[s])
        w.writerow([st,int(s.sum()),round(m78,3),round(lo78,3),round(hi78,3),round(m3,3),round(lo3,3),round(hi3,3),(lo78>0.5 or lo3>0.5)])
print("\nregime validation done ->",OUT)
