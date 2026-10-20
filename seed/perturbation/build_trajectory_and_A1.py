#!/usr/bin/env python3
"""Phase 1 master object + Part A1 Test 1 (DECISIVE, leakage-safe).
Builds dev_edge_trajectory_table.csv from Witvliet 8-animal developmental connectome, then tests
whether PRE-ADULT-ONLY establishment/persistence predicts adult strong-core BEYOND contact topology
+ NT, under neuron-disjoint CV. No adult features in the clean predictor set.
"""
import csv, re, math, json, os
import numpy as np, openpyxl
from collections import defaultdict, Counter
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

C="../data/celegans/data"
EXT=os.path.expanduser("../seed/data/external/mechanistic_frontier")
OUT=os.path.expanduser("../seed/data/processed/mechanistic_perturbation")
EDGES=f"{C}/connectome/witvliet2020/witvliet_combined_edges.csv"
CONTACT=f"{C}/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx"
NPZ="../data/tier5/connectome.npz"
PROF=os.path.expanduser("../data/celegans/New Notebooks/data_derived/nb28_neuron_profile.csv")
BIRTH=f"{EXT}/neuron_birth_times_sulston/neuron_birth_times.csv"
CLASSIF=f"{EXT}/witvliet2021_developmental/tables/connection_classifications.csv"
stage={1:'L1',2:'L1',3:'L1',4:'L1',5:'L2',6:'L3',7:'adult',8:'adult'}

# ---- per-animal directed chemical synapse counts ----
syn=defaultdict(dict)   # (pre,post) -> {animal: synapses}
for r in csv.DictReader(open(EDGES)):
    if r['type']=='electrical': continue
    a=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
    syn[(r['pre'],r['post'])][a]=int(r['synapses'])
pairs=sorted(syn)
# ---- contact area + topology ----
wb=openpyxl.load_workbook(CONTACT,read_only=True,data_only=True); ws=wb['adult nerve ring neighbors']
rows=list(ws.iter_rows(values_only=True)); alln=[h for h in rows[0][1:]]; ix={n:i for i,n in enumerate(alln) if n}
M=len(alln); A=np.zeros((M,M))
for row in rows[1:]:
    rn=row[0]
    if rn in ix:
        for j,v in enumerate(row[1:]):
            if v not in (None,''):
                try:A[ix[rn],j]=float(v)
                except:pass
A=np.maximum(A,A.T); B=(A>0).astype(float); deg_a=A.sum(1); deg_n=B.sum(1)
def contact_feats(p,q):
    if p not in ix or q not in ix: return None
    i,j=ix[p],ix[q]
    if A[i,j]<=0: return None
    cos=float(A[i]@A[j]/(np.linalg.norm(A[i])*np.linalg.norm(A[j])+1e-9))
    cn=int(((B[i]>0)&(B[j]>0)).sum())
    return [math.log1p(A[i,j]),cos,cn,math.log1p(min(deg_a[i],deg_a[j])),math.log1p(max(deg_a[i],deg_a[j])),
            min(deg_n[i],deg_n[j]),max(deg_n[i],deg_n[j])]
# ---- NT, modality, birth, classification ----
d=np.load(NPZ,allow_pickle=True); nm=[str(x) for x in d["names"]]; ntp=[str(x) for x in d["nt_primary"]]
def ntn(x):
    for k,v in [("Acetyl","ACh"),("Glu","Glu"),("GABA","GABA"),("Dopamine","mono"),("Serotonin","mono"),("Octopamine","mono"),("Tyramine","mono")]:
        if k in x: return v
    return "Unk"
NT={a:ntn(b) for a,b in zip(nm,ntp)}
prof={r["neuron"]:r for r in csv.DictReader(open(PROF))}
MOTOR={'VA','VB','VD','DA','DB','DD','AS','VC','RMD','RME','RIM','RMF','RMH','RIV','SMD','SMB','URA'}
def modality(n):
    r=prof.get(n)
    if r:
        if r.get("is_sensory")in("1","1.0","True"): return "sensory"
        if r.get("is_command")in("1","1.0","True"): return "command"
    base=n[:-1] if n[-1:] in "LR" else n
    return "motor" if (base in MOTOR or n in MOTOR) else "inter"
birth={r['neuron']:float(r['birth_time_min']) for r in csv.DictReader(open(BIRTH))}
classif={}
if os.path.exists(CLASSIF):
    for r in csv.DictReader(open(CLASSIF)):
        classif[(r['pre'],r['post'])]=r.get('classification','')

# ---- build trajectory table ----
def in_stage(cnts, animals): return [cnts.get(a,0) for a in animals]
recs=[]
for (p,q) in pairs:
    cnts=syn[(p,q)]
    L1=in_stage(cnts,[1,2,3,4]); L2=cnts.get(5,0); L3=cnts.get(6,0); a7=cnts.get(7,0); a8=cnts.get(8,0)
    L1_count=sum(1 for x in L1 if x>0)          # 0-4 reproducibility at L1
    L1_maxsyn=max(L1) if L1 else 0
    n_preadult=int(L1_count>0)+int(L2>0)+int(L3>0)   # 0-3 pre-adult stage-groups
    first_pre = 'L1' if L1_count>0 else ('L2' if L2>0 else ('L3' if L3>0 else 'none'))
    preadult_maxsyn=max(L1_maxsyn,L2,L3)
    # targets (adult)
    sc1=int(a7>0 and a8>0)
    sc2=int(a7>=2 and a8>=2)
    sc4=int(a7>=4 and a8>=4)
    variable=int((a7>0) != (a8>0))              # exactly one adult
    cf=contact_feats(p,q)
    recs.append(dict(pre=p,post=q,pre_class=p,post_class=q,
        pre_mod=modality(p),post_mod=modality(q),pre_nt=NT.get(p,'Unk'),post_nt=NT.get(q,'Unk'),
        pre_birth=birth.get(p,np.nan),post_birth=birth.get(q,np.nan),
        contact=1 if cf else 0, contact_area=cf[0] if cf else 0.0,
        L1_count=L1_count,L2_present=int(L2>0),L3_present=int(L3>0),
        n_preadult_groups=n_preadult,first_preadult_stage=first_pre,preadult_maxsyn=preadult_maxsyn,
        # FULL (descriptive only): includes adult
        full_n_groups=n_preadult+int(a7>0 or a8>0),
        adult7=int(a7>0),adult8=int(a8>0),adult_syn_min=min(a7,a8),
        strong_core1=sc1,strong_core2=sc2,strong_core4=sc4,variable=variable,
        classification=classif.get((p,q),''),
        _cf=cf))
# write table (drop _cf)
cols=[k for k in recs[0] if k!='_cf']
with open(f"{OUT}/dev_edge_trajectory_table.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
    for r in recs: w.writerow({k:r[k] for k in cols})
print(f"trajectory table: {len(recs)} directed chemical pairs -> {OUT}/dev_edge_trajectory_table.csv")
print(f"  with contact: {sum(r['contact'] for r in recs)} | strong_core1: {sum(r['strong_core1'] for r in recs)}")

# ============ PART A1 TEST 1 (CLEAN, leakage-safe) ============
# restrict to CONTACTING pairs (so contact baseline is defined); both endpoints -> neuron-disjoint CV
sub=[r for r in recs if r['_cf'] is not None]
pairs_sub=[(r['pre'],r['post']) for r in sub]
print(f"\n===== A1 TEST 1: pre-adult persistence -> adult strong-core (contacting pairs N={len(sub)}) =====")
# feature blocks
def Xcontact(): return np.array([r['_cf'] for r in sub])
def Xnt():
    ks=sorted(set((r['pre_nt'],r['post_nt']) for r in sub)); ki={k:i for i,k in enumerate(ks)}
    Mx=np.zeros((len(sub),len(ks)))
    for i,r in enumerate(sub): Mx[i,ki[(r['pre_nt'],r['post_nt'])]]=1
    return Mx
def Xpreadult():  # CLEAN: pre-adult ONLY, no adult info
    fp={'L1':0,'L2':1,'L3':2,'none':3}
    return np.array([[r['L1_count'],r['L2_present'],r['L3_present'],r['n_preadult_groups'],
                      fp[r['first_preadult_stage']],math.log1p(r['preadult_maxsyn'])] for r in sub])
Xc=Xcontact(); Xn=Xnt(); Xp=Xpreadult()
def hs(*a): return np.hstack(a)
def ndcv(X,y,seed=0):
    neu=sorted(set(a for a,b in pairs_sub)|set(b for a,b in pairs_sub))
    rng=np.random.default_rng(seed); f=np.array_split(rng.permutation(neu),5); nf={n:i for i,ff in enumerate(f) for n in ff}
    pred=np.full(len(y),np.nan)
    for k in range(5):
        te=np.array([nf[a]==k and nf[b]==k for a,b in pairs_sub]); tr=np.array([nf[a]!=k and nf[b]!=k for a,b in pairs_sub])
        if te.sum()<10 or y[tr].sum()<5 or y[te].sum()<3 or len(set(y[te]))<2: continue
        clf=make_pipeline(StandardScaler(with_mean=False),GradientBoostingClassifier(random_state=seed))
        clf.fit(X[tr],y[tr]); pred[te]=clf.predict_proba(X[te])[:,1]
    m=~np.isnan(pred)
    return (roc_auc_score(y[m],pred[m]), average_precision_score(y[m],pred[m])) if (m.sum()>15 and len(set(y[m]))>1) else (np.nan,np.nan)
def nd_multi(X,y,ns=5):
    rs=[ndcv(X,y,s) for s in range(ns)]; return np.nanmean([a for a,_ in rs]), np.nanstd([a for a,_ in rs]), np.nanmean([b for _,b in rs])
for tname,tkey in [("strong_core>=1","strong_core1"),("strong_core>=2","strong_core2"),("strong_core>=4","strong_core4"),("variable/halo","variable")]:
    y=np.array([r[tkey] for r in sub])
    if y.sum()<15: print(f"  [{tname}] too few positives"); continue
    print(f"\n  --- target {tname} (pos={y.sum()}, rate={y.mean():.2f}) ---")
    for label,X in [("contact-topology",Xc),("contact+NT",hs(Xc,Xn)),("PRE-ADULT only",Xp),
                    ("contact+preadult",hs(Xc,Xp)),("contact+NT+preadult",hs(Xc,Xn,Xp))]:
        a,s,ap=nd_multi(X,y); print(f"    {label:22s} neuron-disjoint AUC={a:.3f}±{s:.3f}  AP={ap:.3f}")
    # delta of preadult over contact+NT
    base=nd_multi(hs(Xc,Xn),y)[0]; full=nd_multi(hs(Xc,Xn,Xp),y)[0]
    print(f"    >> ADD of pre-adult over contact+NT: dAUC={full-base:+.3f}")
    # stage-shuffle null: permute pre-adult features across edges (break edge-persistence link)
    nulls=[]
    for sd in range(15):
        rng=np.random.default_rng(1000+sd); Xps=Xp[rng.permutation(len(Xp))]
        nulls.append(nd_multi(hs(Xc,Xn,Xps),y,ns=2)[0])
    nulls=np.array(nulls); p=(1+(nulls>=full).sum())/(1+len(nulls))
    print(f"    >> stage-shuffle null add: mean full-with-shuffled={nulls.mean():.3f} | real={full:.3f} | p={p:.3f}")
print("\nA1 Test1 done.")
