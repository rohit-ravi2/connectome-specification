#!/usr/bin/env python3
"""Phase 1 (upstream master table) + Part A (contact/birth/lineage/NT baseline).
TARGETS are PRE-ADULT ONLY (first-synapse timing, L1 reproducibility, pre-adult persistence).
Adult labels carried for downstream validation only; NEVER used as predictors.
"""
import csv, re, math, json, os
import numpy as np, openpyxl
from collections import defaultdict, Counter
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT=os.path.expanduser("../seed")
C="../data/celegans/data"
OUT=f"{ROOT}/data/processed/upstream_early_establishment"
TRAJ=f"{ROOT}/data/processed/mechanistic_perturbation/dev_edge_trajectory_table.csv"
CONTACT=f"{C}/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx"
LIN=os.path.expanduser("../lineage/jtauber_lineage/cells.json")

# ---- lineage paths (for lineage distance) ----
tree=json.load(open(LIN)); path={}
def walk(n,anc):
    p=anc+[n["name"]]; path[n["name"]]=p
    for c in n.get("children",[]): walk(c,p)
walk(tree,[])
def lin_dist(a,b):
    if a not in path or b not in path: return np.nan,np.nan
    pa,pb=path[a],path[b]; L=0
    for x,y in zip(pa,pb):
        if x==y: L+=1
        else: break
    dist=(len(pa)-L)+(len(pb)-L)
    same_founder=1 if (len(pa)>1 and len(pb)>1 and pa[1]==pb[1]) else 0
    return dist,same_founder
# ---- contact topology ----
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
A=np.maximum(A,A.T); Bnb=(A>0).astype(float); deg_a=A.sum(1); deg_n=Bnb.sum(1)
def ctopo(p,q):
    if p not in ix or q not in ix or A[ix[p],ix[q]]<=0: return None
    i,j=ix[p],ix[q]
    cos=float(A[i]@A[j]/(np.linalg.norm(A[i])*np.linalg.norm(A[j])+1e-9))
    cn=int(((Bnb[i]>0)&(Bnb[j]>0)).sum())
    return [math.log1p(A[i,j]),cos,cn,math.log1p(min(deg_a[i],deg_a[j])),math.log1p(max(deg_a[i],deg_a[j])),min(deg_n[i],deg_n[j]),max(deg_n[i],deg_n[j])]

# ---- load trajectory, build master with targets + Part A features ----
tr=list(csv.DictReader(open(TRAJ)))
recs=[]
for r in tr:
    if r['contact']!='1': continue           # restrict to contacting candidate pairs
    p,q=r['pre'],r['post']; ct=ctopo(p,q)
    if ct is None: continue
    L1c=int(r['L1_count']); npg=int(r['n_preadult_groups']); fps=r['first_preadult_stage']
    pb=float(r['pre_birth']) if r['pre_birth'] not in ('','nan') else np.nan
    qb=float(r['post_birth']) if r['post_birth'] not in ('','nan') else np.nan
    ld,sf=lin_dist(p,q)
    recs.append(dict(pre=p,post=q,pre_nt=r['pre_nt'],post_nt=r['post_nt'],pre_mod=r['pre_mod'],post_mod=r['post_mod'],
        ct=ct, pre_birth=pb,post_birth=qb, lin_dist=ld, same_founder=sf,
        # TARGETS (pre-adult only)
        T_L1_repro=int(L1c>=2), T_preadult_persist=int(npg>=2), T_first_L1=int(fps=='L1'),
        T_preadult_strength=math.log1p(int(r['preadult_maxsyn'])),
        T_first_stage=fps if fps!='none' else ('adult' if (r['adult7']=='1' or r['adult8']=='1') else 'never'),
        # downstream validation only
        adult_strong_core1=int(r['strong_core1']),adult_strong_core2=int(r['strong_core2'])))
print(f"contacting candidate pairs: {len(recs)}")
# write master table
cols=['pre','post','pre_nt','post_nt','pre_mod','post_mod','pre_birth','post_birth','lin_dist','same_founder',
      'T_L1_repro','T_preadult_persist','T_first_L1','T_preadult_strength','T_first_stage','adult_strong_core1','adult_strong_core2']
with open(f"{OUT}/upstream_master_table.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(cols+['contact_area','contact_cos','contact_cn'])
    for r in recs: w.writerow([r[c] for c in cols]+[r['ct'][0],r['ct'][1],r['ct'][2]])
print("target prevalences:")
for t in ['T_L1_repro','T_preadult_persist','T_first_L1']:
    y=[r[t] for r in recs]; print(f"  {t}: {sum(y)}/{len(y)} = {np.mean(y):.2f}")

# ---- Part A feature blocks ----
pairs=[(r['pre'],r['post']) for r in recs]
def Xcontact(): return np.array([r['ct'] for r in recs])
def Xbirth():
    out=[]
    for r in recs:
        pb,qb=r['pre_birth'],r['post_birth']
        out.append([np.nanmin([pb,qb]),np.nanmax([pb,qb]),abs(pb-qb) if not(np.isnan(pb) or np.isnan(qb)) else 0,
                    r['lin_dist'] if not np.isnan(r['lin_dist']) else 20, r['same_founder']])
    return np.nan_to_num(np.array(out))
def Xnt():
    ks=sorted(set((r['pre_nt'],r['post_nt']) for r in recs)); ki={k:i for i,k in enumerate(ks)}
    Mx=np.zeros((len(recs),len(ks)))
    for i,r in enumerate(recs): Mx[i,ki[(r['pre_nt'],r['post_nt'])]]=1
    return Mx
Xc,Xb,Xn=Xcontact(),Xbirth(),Xnt()
def hs(*a): return np.hstack(a)
def ndcv(X,y,seed=0):
    neu=sorted(set(a for a,b in pairs)|set(b for a,b in pairs))
    rng=np.random.default_rng(seed); f=np.array_split(rng.permutation(neu),5); nf={n:i for i,ff in enumerate(f) for n in ff}
    pred=np.full(len(y),np.nan)
    for k in range(5):
        te=np.array([nf[a]==k and nf[b]==k for a,b in pairs]); tr=np.array([nf[a]!=k and nf[b]!=k for a,b in pairs])
        if te.sum()<10 or y[tr].sum()<5 or y[te].sum()<3 or len(set(y[te]))<2: continue
        clf=make_pipeline(StandardScaler(with_mean=False),GradientBoostingClassifier(random_state=seed)); clf.fit(X[tr],y[tr]); pred[te]=clf.predict_proba(X[te])[:,1]
    m=~np.isnan(pred)
    return roc_auc_score(y[m],pred[m]) if (m.sum()>15 and len(set(y[m]))>1) else np.nan
def m5(X,y): return np.nanmean([ndcv(X,y,s) for s in range(5)]), np.nanstd([ndcv(X,y,s) for s in range(5)])

print("\n===== PART A: contact/birth/lineage/NT -> PRE-ADULT establishment (neuron-disjoint AUC) =====")
res=[]
for tname,tkey in [("L1_reproducible(>=2/4)","T_L1_repro"),("preadult_persistent(>=2grp)","T_preadult_persist"),("first_synapse=L1","T_first_L1")]:
    y=np.array([r[tkey] for r in recs])
    if y.sum()<15: continue
    print(f"\n  --- target {tname} (pos={y.sum()}, rate={y.mean():.2f}) ---")
    for label,X in [("contact-topology",Xc),("contact+birth+lineage",hs(Xc,Xb)),("contact+NT",hs(Xc,Xn)),
                    ("contact+birth+lineage+NT",hs(Xc,Xb,Xn))]:
        a,s=m5(X,y); res.append((tname,label,a,s)); print(f"    {label:26s} AUC={a:.3f}±{s:.3f}")
    # does birth/lineage add over contact?
    base=m5(Xc,y)[0]; full=m5(hs(Xc,Xb,Xn),y)[0]
    print(f"    >> ADD of birth+lineage+NT over contact: {full-base:+.3f}")
    # random-label null
    nulls=[]
    for sd in range(15):
        rng=np.random.default_rng(sd); ys=rng.permutation(y); nulls.append(ndcv(Xc,ys,sd))
    nulls=np.array(nulls); print(f"    >> label-shuffle null (contact): mean={np.nanmean(nulls):.3f} (real contact={base:.3f})")
with open(f"{OUT}/partA_baseline_results.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["target","feature_set","neuron_disjoint_AUC","sd"])
    for t,l,a,s in res: w.writerow([t,l,round(a,4),round(s,4)])
print("\nPart A done ->",OUT)
