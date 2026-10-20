import sys, csv, re, numpy as np
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); m0=np.load(f"{SC}/mode0.npz")
I,J,y7,y8,base,neurons=z['I'],z['J'],z['y7'],z['y8'],z['base'],list(z['neurons']); idx={n:i for i,n in enumerate(neurons)}
M=len(neurons); N=len(I); yrep=((y7==1)&(y8==1)).astype(int); agree=m0['agree']; area=base[:,0]
# per-dataset chemical presence (PRE-ADULT stages only: L1=ds1-4, L2=ds5, L3=ds6)
dp=defaultdict(set)
with open("../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']=='chemical':
            d=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
            if r['pre'] in idx and r['post'] in idx: dp[(idx[r['pre']],idx[r['post']])].add(d)
def preadult_stage(k):  # earliest PRE-ADULT stage the edge appears (1=L1,2=L2,3=L3), 0 if never pre-adult
    s=dp[(I[k],J[k])]
    if s&{1,2,3,4}: return 1
    if 5 in s: return 2
    if 6 in s: return 3
    return 0
fstage=np.array([preadult_stage(k) for k in range(N)])
L1frac=np.array([len(dp[(I[k],J[k])]&{1,2,3,4})/4.0 for k in range(N)])
L1syn=(L1frac>=0.5).astype(float)  # persistence
GL1=np.zeros((M,M))
for (a,b),s in dp.items():
    if s&{1,2,3,4}: GL1[a,b]=1
Cff=GL1@GL1; closure=np.log1p(Cff[I,J])
# WITHIN-NEURON FORMATION ORDER: rank of edge among its source's (and target's) pre-adult edges by (stage, -L1frac)
def ranks(byrole):
    key=defaultdict(list)  # node -> list of (k, sortkey)
    for k in range(N):
        if fstage[k]>0: key[byrole[k]].append((k,(fstage[k],-L1frac[k])))
    rk=np.full(N,np.nan)
    for node,lst in key.items():
        lst.sort(key=lambda x:x[1])
        for r,(k,_) in enumerate(lst): rk[k]=r/max(len(lst)-1,1)  # 0=first-formed .. 1=last-formed
    return rk
src_rank=ranks(I); tgt_rank=ranks(J)
# TARGET: prunable/ambiguous = edges with PRE-ADULT presence, agree animals; predict RETAINED(yrep=1) vs pruned(0)
prun=(fstage>0)&agree; yt=yrep[prun]
P(f"[FORMATION-SEQUENCE TEST] prunable set (pre-adult present & agree) n={prun.sum()} retained-rate={yt.mean():.3f}")
BASE=np.column_stack([area,closure,L1syn,L1frac])  # contact+closure+PERSISTENCE (persistence fully in)
ORD=np.column_stack([np.nan_to_num(src_rank,nan=0.5),np.nan_to_num(tgt_rank,nan=0.5)])
nf={n:i%5 for i,n in enumerate(sorted(set([neurons[i] for i in I]+[neurons[j] for j in J])))}
def nho(X):
    ii=np.where(prun)[0]; pred=np.full(len(ii),np.nan)
    for f in range(5):
        te=[t for t,k in enumerate(ii) if nf[neurons[I[k]]]==f and nf[neurons[J[k]]]==f]; tr=[t for t,k in enumerate(ii) if nf[neurons[I[k]]]!=f and nf[neurons[J[k]]]!=f]
        if len(te)<8 or len(set(yt[tr]))<2: continue
        pred[np.array(te)]=GradientBoostingClassifier(random_state=0).fit(X[ii][tr],yt[tr]).predict_proba(X[ii][te])[:,1]
    m=~np.isnan(pred); return roc_auc_score(yt[m],pred[m])
a0=nho(BASE); a1=nho(np.hstack([BASE,ORD]))
P(f"  contact+closure+persistence AUC={a0:.3f} -> +within-neuron ORDER AUC={a1:.3f} (Δ{a1-a0:+.3f})")
# permutation: shuffle formation-order WITHIN each neuron
rng=np.random.default_rng(0); nl=[]
for s in range(8):
    sp=src_rank.copy(); tp=tgt_rank.copy()
    for node in set(I):
        m=(I==node)&(fstage>0); v=sp[m]; sp[m]=rng.permutation(v) if len(v)>1 else v
    for node in set(J):
        m=(J==node)&(fstage>0); v=tp[m]; tp[m]=rng.permutation(v) if len(v)>1 else v
    nl.append(nho(np.hstack([BASE,np.column_stack([np.nan_to_num(sp,nan=0.5),np.nan_to_num(tp,nan=0.5)])]))-a0)
nl=np.array(nl); P(f"  [PERM within-neuron shuffle] real Δ={a1-a0:+.3f} vs shuffled {nl.mean():+.3f}±{nl.std():.3f} -> {'ORDER REAL (residual is a within-neuron SEQUENTIAL object)' if (a1-a0)>nl.mean()+2*nl.std() else 'ORDER NULL (residual not sequential either)'}")

# ---- DISCIPLINE RE-TEST: restrict to the AMBIGUOUS RESIDUAL (contact+closure+persistence uncertain) ----
cv=StratifiedKFold(5,shuffle=True,random_state=0)
pm=np.full(N,np.nan)
ii=np.where(prun)[0]
pm_prun=cross_val_predict(GradientBoostingClassifier(random_state=0),BASE[ii],yt,cv=cv,method='predict_proba')[:,1]
amb=(pm_prun>0.3)&(pm_prun<0.7)  # ambiguous = model genuinely uncertain
P(f"\n[RE-TEST on AMBIGUOUS residual only, n={amb.sum()} of {prun.sum()} prunable]")
Xii_base=BASE[ii][amb]; Xii_ord=np.hstack([BASE,ORD])[ii][amb]; ya=yt[amb]
def nho2(X,y,grp):
    pred=np.full(len(y),np.nan)
    for f in range(5):
        te=[t for t in range(len(y)) if grp[t]==f]; tr=[t for t in range(len(y)) if grp[t]!=f]
        if len(te)<6 or len(set(y[tr]))<2: continue
        pred[np.array(te)]=GradientBoostingClassifier(random_state=0).fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
    m=~np.isnan(pred); return roc_auc_score(y[m],pred[m]) if len(set(y[~np.isnan(pred)]))>1 else np.nan
grp=np.array([nf[neurons[I[ii[t]]]] if nf[neurons[I[ii[t]]]]==nf[neurons[J[ii[t]]]] else -1 for t in range(len(ii))])[amb]
a0a=nho2(Xii_base,ya,grp); a1a=nho2(Xii_ord,ya,grp)
P(f"  ambiguous: contact+closure+persistence AUC={a0a:.3f} -> +ORDER AUC={a1a:.3f} (Δ{a1a-a0a:+.3f})")
# also: direct correlation of order-residual with outcome-residual (regress out persistence)
from numpy.polynomial import polynomial as _p
def resid(yv,xv): 
    b=np.polyfit(xv,yv,1); return yv-(b[0]*xv+b[1])
sr=np.nan_to_num(src_rank[ii],nan=0.5); pers=L1frac[ii]
or_res=resid(sr,pers); y_res=resid(yt.astype(float),pers)
P(f"  corr(order | persistence-removed, retention | persistence-removed) = {np.corrcoef(or_res,y_res)[0,1]:+.3f}")
P(f"  => VERDICT: within-neuron formation ORDER is {'REAL beyond persistence' if abs(np.corrcoef(or_res,y_res)[0,1])>0.05 and a1a-a0a>0.01 else 'NULL beyond persistence (residual is NOT a within-neuron sequential object at 8-stage resolution)'}")
