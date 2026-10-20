import sys, csv, re, numpy as np
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); m0=np.load(f"{SC}/mode0.npz")
I,J,y7,y8,neurons=z['I'],z['J'],z['y7'],z['y8'],list(z['neurons']); idx={n:i for i,n in enumerate(neurons)}
M=len(neurons); N=len(I); yrep=((y7==1)&(y8==1)).astype(int); L1c=m0['L1c']; agree=m0['agree']
# per-neuron L1 BUDGET = total L1 synaptic WEIGHT (sum synapse counts ds1-4); DEGREE = # L1 partners
budget=np.zeros(M); deg=np.zeros(M); seen=defaultdict(set)
with open("../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']!='chemical': continue
        d=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
        if d in (1,2,3,4) and r['pre'] in idx and r['post'] in idx:
            w=int(r['synapses']); a,b=idx[r['pre']],idx[r['post']]
            budget[a]+=w; budget[b]+=w; seen[a].add(b); seen[b].add(a)
for n in range(M): deg[n]=len(seen[n])
# prunable set: pairs WITH an L1 synapse (robust >=2), restricted to agree; target pruned(adult absent)=1 vs stable=0
prun=(L1c>=2)&agree; yt=(yrep[prun]==0).astype(int)  # 1=pruned, 0=stable
P(f"prunable set (L1>=2 & agree) n={prun.sum()} | pruned={yt.sum()} stable={(yt==0).sum()}")
BUD=np.column_stack([budget[I]+budget[J],np.maximum(budget[I],budget[J]),budget[I],budget[J]])
DEG=np.column_stack([deg[I]+deg[J],np.maximum(deg[I],deg[J]),deg[I],deg[J]])
# over-budget = budget relative to degree (weight per partner)
OB=np.column_stack([budget[I]/np.maximum(deg[I],1)+budget[J]/np.maximum(deg[J],1)])
nf={n:i%5 for i,n in enumerate(sorted(set([neurons[i] for i in I]+[neurons[j] for j in J])))}
def nho(X,mask,y):
    ii=np.where(mask)[0]; pred=np.full(len(ii),np.nan)
    for f in range(5):
        te=[t for t,k in enumerate(ii) if nf[neurons[I[k]]]==f and nf[neurons[J[k]]]==f]; tr=[t for t,k in enumerate(ii) if nf[neurons[I[k]]]!=f and nf[neurons[J[k]]]!=f]
        if len(te)<6 or len(set(y[tr]))<2: continue
        pred[np.array(te)]=GradientBoostingClassifier(random_state=0).fit(X[ii][tr],y[tr]).predict_proba(X[ii][te])[:,1]
    m=~np.isnan(pred); return roc_auc_score(y[m],pred[m]) if m.sum()>10 and len(set(y[~np.isnan(pred)]))>1 else np.nan
aB=nho(BUD,prun,yt); aD=nho(DEG,prun,yt); aOB=nho(OB,prun,yt)
P(f"  budget-only AUC={aB:.3f} | degree-only AUC={aD:.3f} | over-budget(weight/partner)-only AUC={aOB:.3f}")
# HARD GATE: budget must beat DEGREE-MATCHED null (shuffle budget within degree strata)
rng=np.random.default_rng(0); order=np.argsort(deg); strata=np.array_split(order,6); nl=[]
for s in range(30):
    bsh=budget.copy()
    for st in strata:
        pp=rng.permutation(len(st)); bsh[st]=budget[st][pp]
    BUDsh=np.column_stack([bsh[I]+bsh[J],np.maximum(bsh[I],bsh[J]),bsh[I],bsh[J]])
    nl.append(nho(BUDsh,prun,yt))
nl=np.array([x for x in nl if not np.isnan(x)]); P(f"  [HARD GATE degree-matched null] budget AUC={aB:.3f} vs shuffled {nl.mean():.3f}±{nl.std():.3f} z={(aB-nl.mean())/nl.std():.1f} -> {'BUDGET BEATS degree (real pruning signal)' if (aB-nl.mean())>2*nl.std() else 'DEGREE IN A COSTUME (budget=degree relabeled)'}")
