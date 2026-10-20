"""A5 — Are NON-REPRODUCIBLE synapses at the CONTACT-AREA THRESHOLD?
Hypothesis: variable synapses (present some animals, absent others) sit at INTERMEDIATE
contact area (geometry ambiguous); reproducibly-present at HIGH area; reproducibly-absent
(but contacting) at LOW area — margin classifier with boundary noise.

Works at INDIVIDUAL-NEURON level (contact matrix + Witvliet edges are both neuron-level).
Reproducibility = fraction of Witvliet datasets (animals) in which a directed chemical
synapse exists, AMONG physically contacting ordered pairs.
Confound noted: datasets 1..8 are a DEVELOPMENTAL series (grow 775->2186 synapses), so
'absent in early, present in late' can be developmental ADDITION not stochastic variability.
We therefore report 3 replicate sets: L1(1-4), LATE(5-8), ALL(1-8).
"""
import warnings,sys,csv; warnings.simplefilter("ignore")
import numpy as np, pandas as pd, openpyxl
from collections import Counter, defaultdict
from scipy.stats import mannwhitneyu, spearmanr
def P(*a): print(*a); sys.stdout.flush()
D="../data/celegans/data"
OUT="../accounting/principles/"

# ---------- CONTACT AREA (adult + L4 nerve ring neighbors, undirected) ----------
def load_contact(sheet):
    df=pd.read_excel(f"{D}/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx",sheet,header=None)
    names=[str(x).strip() for x in df.iloc[0,1:].tolist()]
    M=df.iloc[1:,1:].apply(pd.to_numeric,errors="coerce").values
    row_names=[str(x).strip() for x in df.iloc[1:,0].tolist()]
    area={}
    for r,rn in enumerate(row_names):
        for c,cn in enumerate(names):
            v=M[r,c]
            if pd.notna(v) and v>0:
                key=frozenset((rn,cn))
                if len(key)==2: area[key]=max(area.get(key,0),v)
    return area,set(names)|set(row_names)
area_adult,neur_adult=load_contact('adult nerve ring neighbors')
area_l4,neur_l4=load_contact('L4 nerve ring neighbors')
P("contact pairs adult:",len(area_adult),"L4:",len(area_l4),"neurons adult:",len(neur_adult))

# ---------- WITVLIET directed chemical presence per dataset ----------
# presence[(pre,post)] = set of dataset indices (1..8) with chemical synapse
pres=defaultdict(set); syncount=defaultdict(dict)
with open(f"{D}/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']!='chemical': continue
        ds=int(r['source_file'].split('_')[2].split('.')[0])
        a,b=r['pre'],r['post']
        pres[(a,b)].add(ds)
        syncount[(a,b)][ds]=int(r['synapses'])
wit_neurons=set()
for (a,b) in pres: wit_neurons.add(a); wit_neurons.add(b)
P("witvliet directed chem pairs (any ds):",len(pres),"neurons:",len(wit_neurons))
P("neurons in BOTH contact-adult and witvliet:",len(neur_adult & wit_neurons))

SETS={'L1_1-4':[1,2,3,4],'LATE_5-8':[5,6,7,8],'ALL_1-8':[1,2,3,4,5,6,7,8]}

def analyze(dslist,area,neur,tag):
    P("\n"+"="*70); P("REPLICATE SET",tag,"datasets",dslist,"area-sheet neurons",len(neur))
    nds=len(dslist)
    # universe = contacting ordered pairs where both neurons are in witvliet universe AND contact matrix
    valid=neur & wit_neurons
    rows=[]
    for key,ar in area.items():
        u,v=tuple(key)
        if u not in valid or v not in valid: continue
        for (i,j) in [(u,v),(v,u)]:
            k=sum(1 for ds in pres.get((i,j),()) if ds in dslist)
            rows.append((i,j,ar,k))
    dfp=pd.DataFrame(rows,columns=['pre','post','area','k'])
    P("contacting ordered pairs analyzed:",len(dfp))
    # bins
    def binf(k):
        if k==0: return 'absent'
        if k==nds: return 'repro_present'
        return 'variable'
    dfp['bin']=dfp['k'].apply(binf)
    P("bin counts:",dict(dfp['bin'].value_counts()))
    P("k distribution:",dict(dfp['k'].value_counts().sort_index()))
    P("\nCONTACT AREA by reproducibility bin (log10 scale where useful):")
    for b in ['absent','variable','repro_present']:
        s=dfp[dfp['bin']==b]['area']
        if len(s):
            P(f"  {b:14s} n={len(s):5d}  median={s.median():8.0f}  mean={s.mean():8.0f}  "
              f"q25={s.quantile(.25):7.0f} q75={s.quantile(.75):7.0f}  "
              f"logmean={np.log10(s+1).mean():.3f}")
    # area vs k (monotone?)
    P("\nMedian contact area by exact k (n_animals present):")
    for k in range(nds+1):
        s=dfp[dfp['k']==k]['area']
        if len(s): P(f"  k={k}  n={len(s):5d}  median_area={s.median():8.0f}  mean={s.mean():8.0f}")
    rho,pv=spearmanr(dfp['k'],dfp['area'])
    P(f"Spearman(k, area) = {rho:.3f} (p={pv:.1e})")
    # test: variable vs repro_present ; variable vs absent
    for a_,b_ in [('variable','repro_present'),('variable','absent'),('absent','repro_present')]:
        sa=dfp[dfp['bin']==a_]['area']; sb=dfp[dfp['bin']==b_]['area']
        if len(sa)>5 and len(sb)>5:
            u,p=mannwhitneyu(sa,sb)
            P(f"  MWU {a_}({sa.median():.0f}) vs {b_}({sb.median():.0f}): p={p:.2e}")
    return dfp

dfs={}
for tag,ds in SETS.items():
    dfs[tag]=analyze(ds,area_adult,neur_adult,tag+" [ADULT-area]")
# L4 area for LATE set (closer developmental match)
analyze(SETS['LATE_5-8'],area_l4,neur_l4,"LATE_5-8 [L4-area]")

# save the ALL df for downstream structure analysis
dfs['ALL_1-8'].to_csv(OUT+"A5_pairs_all.csv",index=False)
dfs['LATE_5-8'].to_csv(OUT+"A5_pairs_late.csv",index=False)
P("\nsaved A5_pairs_all.csv, A5_pairs_late.csv")
