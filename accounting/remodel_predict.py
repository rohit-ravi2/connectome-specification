import sys, csv, re, numpy as np
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); zs=np.load(f"{SC}/subcellular.npz")
I,J,y7,y8,base,neurons=z['I'],z['J'],z['y7'],z['y8'],z['base'],list(z['neurons'])
M=len(neurons); N=len(I); yrep=((y7==1)&(y8==1)).astype(int); idx={n:i for i,n in enumerate(neurons)}
sub=zs['sub']
# per-dataset synapses
dp=defaultdict(set)
with open("../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']!='chemical': continue
        d=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
        if r['pre'] in idx and r['post'] in idx: dp[(idx[r['pre']],idx[r['post']])].add(d)
L1syn=np.array([len(dp[(I[k],J[k])]&{1,2,3,4})>=2 for k in range(N)])
# L1 GRAPH motifs (PRE-ADULT features only; NOT Cook adult)
GL1=np.zeros((M,M))
for (a,b),dss in dp.items():
    if dss&{1,2,3,4}: GL1[a,b]=1
Cin=GL1.T@GL1;Cout=GL1@GL1.T;Cff=GL1@GL1;Cu=((GL1+GL1.T)>0).astype(float);comm=Cu@Cu
# NT
t5=np.load("../data/tier5/connectome.npz",allow_pickle=True)
t5nt={str(n):str(v) for n,v in zip(t5["names"],t5["nt_primary"])}
def bcl(n): return n[:-1] if n[-1] in 'LR' else n
nt=[t5nt.get(neurons[i],t5nt.get(bcl(neurons[i]),"U")) for i in range(M)]
sament=np.array([nt[I[k]]==nt[J[k]] for k in range(N)]).astype(float)
# PRE-ADULT feature set: contact(base 0-4) + L1-graph motifs + subcellular + same-NT  (NO adult connectome)
PRE=np.column_stack([base[:,:5], np.log1p(Cin[I,J]),np.log1p(Cout[I,J]),np.log1p(Cff[I,J]),np.log1p(comm[I,J]),
                     sub[:,0],sub[:,1],sub[:,2],sub[:,5],sub[:,6], sament])
nf={n:i%5 for i,n in enumerate(sorted(set([neurons[i] for i in I]+[neurons[j] for j in J])))}
def nho(mask,y):
    ii=np.where(mask)[0]; pred=np.full(len(ii),np.nan)
    for f in range(5):
        te=[t for t,k in enumerate(ii) if nf[neurons[I[k]]]==f and nf[neurons[J[k]]]==f]
        tr=[t for t,k in enumerate(ii) if nf[neurons[I[k]]]!=f and nf[neurons[J[k]]]!=f]
        if len(te)<10 or len(set(y[tr]))<2: continue
        pred[np.array(te)]=GradientBoostingClassifier(random_state=0).fit(PRE[ii][tr],y[tr]).predict_proba(PRE[ii][te])[:,1]
    m=~np.isnan(pred); return roc_auc_score(y[m],pred[m]),m.sum()
# PRUNING: among L1-synapse pairs, predict stable(1) vs pruned(0) from PRE-ADULT structure
yp=yrep[L1syn]; ap,np_=nho(L1syn,yp)
P(f"[PRUNING rule] among {L1syn.sum()} L1-synapse pairs ({(1-yp.mean())*100:.0f}% pruned): predict stable-vs-pruned from PRE-ADULT structure AUC={ap:.3f} (n_eval={np_})")
# ADDITION: among non-L1 contacting pairs, predict added(1) vs stays-absent(0)
addmask=~L1syn; ya=yrep[addmask]; aa,na=nho(addmask,ya)
P(f"[ADDITION rule] among {addmask.sum()} non-L1 contacting pairs ({ya.mean()*100:.0f}% added): predict added-vs-absent from PRE-ADULT structure AUC={aa:.3f} (n_eval={na})")
P(f"\n(0.5=irreducible/no local rule; high=discoverable local rewrite rule)")
