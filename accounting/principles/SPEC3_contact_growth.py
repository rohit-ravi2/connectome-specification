"""SPEC 3 — is the developmental threshold shift PHYSICAL (contact areas grow) or BIOLOGICAL (criterion rises)?
DATA LIMIT: direct contact area exists only at L4 + adult (WormWiring nerve-ring). Witvliet synapse series is
L1..adult with NO contact area + NO L4. So the L1->adult contact-growth test is NOT directly runnable; we run:
  Part A: L4->adult contact-area growth (same pairs, both sheets) — does contact grow at all? overall + stable core.
  Part C: per-Witvliet-stage (1-8) synapse-formation threshold using FIXED adult area (the accumulation trajectory).
  Part B: variable-zone contact area across stage-sets (fixed adult area).
Report numbers + implications; flag what needs L1 contact area (absent).
"""
import warnings,sys,csv; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from collections import defaultdict
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
D="../data/celegans/data"
def load_contact(sheet):
    df=pd.read_excel(f"{D}/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx",sheet,header=None)
    names=[str(x).strip() for x in df.iloc[0,1:].tolist()]
    Mv=df.iloc[1:,1:].apply(pd.to_numeric,errors="coerce").values
    rns=[str(x).strip() for x in df.iloc[1:,0].tolist()]
    area={}
    for r,rnm in enumerate(rns):
        for c,cn in enumerate(names):
            v=Mv[r,c]
            if pd.notna(v) and v>0: area[frozenset((rnm,cn))]=max(area.get(frozenset((rnm,cn)),0),v) if rnm!=cn else area.get(frozenset((rnm,cn)),0)
    return {k:v for k,v in area.items() if len(k)==2}
adult=load_contact('adult nerve ring neighbors'); l4=load_contact('L4 nerve ring neighbors')
P(f"SPEC3 — contact pairs: adult {len(adult)}, L4 {len(l4)}")
# ---- Part A: L4 -> adult contact-area growth (same undirected pairs) ----
common=set(adult)&set(l4)
dfA=pd.DataFrame([(tuple(k)[0],tuple(k)[1],l4[k],adult[k]) for k in common],columns=['a','b','area_L4','area_adult'])
dfA['dlog']=np.log10(dfA.area_adult.clip(lower=1))-np.log10(dfA.area_L4.clip(lower=1))
dfA['ratio']=dfA.area_adult/dfA.area_L4.clip(lower=1)
P("\n=== PART A: L4 -> adult contact-area change (same pairs, n=%d) ==="%len(dfA))
P(f"  median area L4={dfA.area_L4.median():.0f}  adult={dfA.area_adult.median():.0f}")
P(f"  Δlog10(area) adult-L4: median={dfA.dlog.median():+.3f}  mean={dfA.dlog.mean():+.3f}  q25={dfA.dlog.quantile(.25):+.3f} q75={dfA.dlog.quantile(.75):+.3f}")
P(f"  fraction of pairs that GREW (adult>L4): {(dfA.dlog>0).mean():.3f}  | median ratio adult/L4 = {dfA.ratio.median():.2f}x")
# stable-synapse subset: Witvliet-persistent (>=6/8 stages)
syn=defaultdict(dict)
with open(f"{D}/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']!='chemical': continue
        ds=int(r['source_file'].split('_')[2].split('.')[0]); syn[(r['pre'],r['post'])][ds]=int(r['synapses'])
def kcount(a,b,dss):
    sc=syn.get((a,b),{}); sc2=syn.get((b,a),{}); return sum(1 for d in dss if sc.get(d,0)>0 or sc2.get(d,0)>0)
dfA['kall']=[kcount(a,b,range(1,9)) for a,b in zip(dfA.a,dfA.b)]
stab=dfA[dfA.kall>=6]
P(f"  STABLE core (Witvliet >=6/8 synapse, n={len(stab)}): Δlog10 median={stab.dlog.median():+.3f}  frac grew={(stab.dlog>0).mean():.3f}  ratio={stab.ratio.median():.2f}x")
# ---- Part C: per-stage synapse-formation threshold (FIXED adult area) ----
P("\n=== PART C: per-Witvliet-stage present|absent contact-area threshold (FIXED adult area) ===")
stages={1:'L1',2:'L1',3:'L1',4:'L1',5:'L2',6:'L3',7:'ad',8:'ad'}
def thr(ds):
    y=[];x=[]
    for k,ar in adult.items():
        a,b=tuple(k); pres=1 if (syn.get((a,b),{}).get(ds,0)>0 or syn.get((b,a),{}).get(ds,0)>0) else 0
        y.append(pres); x.append(ar)
    y=np.array(y);x=np.array(x)
    if y.sum()<5: return None
    auc=roc_auc_score(y,x)
    cuts=np.quantile(x,np.linspace(.3,.97,40)); best=None
    for t in cuts:
        j=(x[y==1]>t).mean()-(x[y==0]>t).mean()
        if best is None or j>best[1]: best=(t,j)
    return best[0],auc,int(y.sum())
P(f"  {'ds':>3} {'stage':>5} {'nSyn':>6} {'thr(present|absent)':>20} {'AUC':>6}")
traj=[]
for ds in range(1,9):
    r=thr(ds)
    if r: P(f"  {ds:>3} {stages[ds]:>5} {r[2]:>6} {r[0]:>20.0f} {r[1]:>6.3f}"); traj.append((ds,r[0],r[2]))
tv=np.array([t[1] for t in traj]); nv=np.array([t[2] for t in traj])
P(f"  threshold trajectory (area): {[f'{t[1]:.0f}' for t in traj]}")
P(f"  Spearman(stage, threshold)={np.corrcoef(range(len(tv)),np.log(tv))[0,1]:+.2f} | Spearman(nSyn, threshold-log) = {np.corrcoef(np.log(nv),np.log(tv))[0,1]:+.2f}")
# ---- Part B: variable-zone contact area across stage-sets (fixed adult area) ----
P("\n=== PART B: variable-zone median contact area across stage-sets (fixed adult area) ===")
def variable_area(dss):
    nds=len(dss); vals=[]
    for k,ar in adult.items():
        a,b=tuple(k); kk=kcount(a,b,dss)
        if 0<kk<nds: vals.append(ar)
    return np.median(vals),len(vals)
for tag,dss in [('L1(1-4)',[1,2,3,4]),('LATE(5-8)',[5,6,7,8]),('ALL(1-8)',list(range(1,9)))]:
    m,n=variable_area(dss); P(f"  {tag}: variable-zone median adult-area = {m:.0f}  (n={n})")
P("\nIMPLICATION NOTES (no verdict):")
P("- Part A tells whether contact areas physically grow L4->adult (the only window with 2 area measurements).")
P("- Part C threshold used FIXED adult area, so its trajectory is synapse-ACCUMULATION vs final geometry, NOT area growth.")
P("- The L1->adult physical test needs L1 contact area, which is NOT on disk (WormWiring/Brittin = L4+adult only).")
