"""A5 part 2 — dose-response curve, threshold width, and STRUCTURE in the variable zone.
Q1: is presence-probability a graded logistic in area (margin) or a sharp step?
Q2: excess dispersion at the margin beyond the global dose-response?
Q3: at MATCHED area, do variable pairs differ (weaker synapse count; identity/trophic)?
Q4: neuron-class / NT / trophic structure of the variable bin.
"""
import warnings,sys,csv; warnings.simplefilter("ignore")
import numpy as np, pandas as pd, openpyxl
from collections import Counter,defaultdict
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.linear_model import LogisticRegression
def P(*a): print(*a); sys.stdout.flush()
D="../data/celegans/data"; OUT="../accounting/principles/"

# ---- rebuild contact area (adult) + witvliet presence + synapse counts ----
df=pd.read_excel(f"{D}/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx",'adult nerve ring neighbors',header=None)
names=[str(x).strip() for x in df.iloc[0,1:].tolist()]; row_names=[str(x).strip() for x in df.iloc[1:,0].tolist()]
M=df.iloc[1:,1:].apply(pd.to_numeric,errors="coerce").values
area={}
for r,rn in enumerate(row_names):
    for c,cn in enumerate(names):
        v=M[r,c]
        if pd.notna(v) and v>0:
            key=frozenset((rn,cn))
            if len(key)==2: area[key]=max(area.get(key,0),v)

DS=[1,2,3,4,5,6,7,8]; nds=8
pres=defaultdict(set); syn=defaultdict(dict)
with open(f"{D}/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']!='chemical': continue
        ds=int(r['source_file'].split('_')[2].split('.')[0])
        pres[(r['pre'],r['post'])].add(ds); syn[(r['pre'],r['post'])][ds]=int(r['synapses'])
wit_neurons=set(a for a,b in pres)|set(b for a,b in pres)
valid=(set(names)|set(row_names)) & wit_neurons

rows=[]
for key,ar in area.items():
    u,v=tuple(key)
    if u not in valid or v not in valid: continue
    for (i,j) in [(u,v),(v,u)]:
        ks=[ds for ds in pres.get((i,j),()) if ds in DS]
        k=len(ks)
        meansyn=np.mean([syn[(i,j)][ds] for ds in ks]) if ks else 0.0
        rows.append((i,j,ar,k,meansyn))
dfp=pd.DataFrame(rows,columns=['pre','post','area','k','meansyn'])
dfp['p']=dfp['k']/nds

# ---- Q1: dose-response P(present per animal) vs area (log bins) ----
P("="*70); P("Q1  DOSE-RESPONSE: P(synapse present in a random animal) vs contact area")
dfp['la']=np.log10(dfp['area']+1)
bins=np.quantile(dfp['la'],np.linspace(0,1,13))
dfp['ab']=np.digitize(dfp['la'],bins[1:-1])
for b in sorted(dfp['ab'].unique()):
    s=dfp[dfp['ab']==b]
    P(f"  area[{10**s['la'].min():7.0f}-{10**s['la'].max():7.0f}] n={len(s):5d}  "
      f"P(present)={s['p'].mean():.3f}  frac_variable(0<k<8)={((s['k']>0)&(s['k']<8)).mean():.3f}  "
      f"frac_reproPresent={ (s['k']==8).mean():.3f}")
# logistic fit area->per-animal presence (expand each pair into nds bernoulli)
X=np.repeat(dfp['la'].values,nds).reshape(-1,1)
yv=np.concatenate([[1]*k+[0]*(nds-k) for k in dfp['k'].values])
lr=LogisticRegression().fit(X,yv)
b0,b1=lr.intercept_[0],lr.coef_[0][0]
half=-b0/b1
P(f"logistic per-animal-presence ~ log10(area): P=0.5 at area={10**half:.0f}, slope(log)={b1:.2f}")
P(f"  area for P=0.25 -> {10**((np.log(1/3)-b0)/b1):.0f} ; P=0.75 -> {10**((np.log(3)-b0)/b1):.0f}  (transition width)")

# ---- Q2: EXCESS dispersion at margin? compare observed disagreement to Bernoulli(p) ----
P("\n"+"="*70); P("Q2  Is variability concentrated at intermediate area beyond mechanical p(1-p)?")
P("  Among CONTACTING pairs, mean p(1-p) [max at p=.5] by area bin, and observed among-animal var:")
for b in sorted(dfp['ab'].unique()):
    s=dfp[dfp['ab']==b]
    exp_var=(s['p']*(1-s['p'])).mean()
    P(f"  area[{10**s['la'].min():7.0f}-{10**s['la'].max():7.0f}]  meanP={s['p'].mean():.3f}  E[p(1-p)]={exp_var:.3f}")

# ---- Q3: at MATCHED area, are variable pairs weaker (fewer synapses when present)? ----
P("\n"+"="*70); P("Q3  Matched-area: do variable pairs differ from reproducible-present?")
sub=dfp[dfp['k']>0].copy()
sub['bin']=np.where(sub['k']==nds,'repro_present','variable')
# match on area quantile bin
for b in sorted(sub['ab'].unique()):
    s=sub[sub['ab']==b]
    va=s[s['bin']=='variable']; pr=s[s['bin']=='repro_present']
    if len(va)>8 and len(pr)>8:
        P(f"  areaBin{b}: meansyn variable={va['meansyn'].mean():.2f}(n{len(va)}) vs present={pr['meansyn'].mean():.2f}(n{len(pr)})")
rho,pv=spearmanr(sub['k'],sub['meansyn'])
P(f"  Spearman(k, mean synapse count | present) = {rho:.3f} p={pv:.1e}  (are more-reproducible edges also stronger?)")

# ---- Q4: structure of the variable zone: NT + presyn identity ----
P("\n"+"="*70); P("Q4  Structure in the VARIABLE bin (0<k<8)")
var=dfp[(dfp['k']>0)&(dfp['k']<nds)]
P("  top presynaptic neurons among variable pairs:")
P("   ", dict(Counter(var['pre']).most_common(12)))
P("  top postsynaptic neurons among variable pairs:")
P("   ", dict(Counter(var['post']).most_common(12)))
# NT class via CeNGEN? use simple neuron-class NT map from cengen not available; use presyn base letters
# proxy: are variable edges dominated by specific cell-type prefixes?
def pref(n):
    import re; m=re.match(r'[A-Z]+',n); return m.group() if m else n
P("  presyn cell-class prefix freq (variable):", dict(Counter(var['pre'].map(pref)).most_common(10)))
P("  presyn cell-class prefix freq (repro_present):", dict(Counter(dfp[dfp['k']==nds]['pre'].map(pref)).most_common(10)))
dfp.to_csv(OUT+"A5_doseresponse_pairs.csv",index=False)
P("\nsaved A5_doseresponse_pairs.csv")
