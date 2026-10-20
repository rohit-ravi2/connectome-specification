import sys, csv, re, numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); m0=np.load(f"{SC}/mode0.npz")
I,J,y7,y8,base,neurons=z['I'],z['J'],z['y7'],z['y8'],z['base'],list(z['neurons']); idx={n:i for i,n in enumerate(neurons)}
M=len(neurons); N=len(I); yrep=((y7==1)&(y8==1)).astype(int); agree=m0['agree']; area=base[:,0]
GL1=np.zeros((M,M))
with open("../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']=='chemical':
            d=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
            if d in (1,2,3,4) and r['pre'] in idx and r['post'] in idx: GL1[idx[r['pre']],idx[r['post']]]=1
t5=np.load("../data/tier5/connectome.npz",allow_pickle=True); t5nt={str(x):str(v) for x,v in zip(t5["names"],t5["nt_primary"])}
def bcl(x): return x[:-1] if x[-1] in 'LR' else x
NT=[t5nt.get(neurons[i],t5nt.get(bcl(neurons[i]),"U")) for i in range(M)]; nts=sorted(set(NT))
# ROLE/NICHE vector per neuron (PRE-ADULT: NT + L1 in/out partner profile + contact profile) — NOT adult wiring
NToh=np.array([[NT[i]==c for c in nts] for i in range(M)],float)
Lout=GL1; Lin=GL1.T  # L1 partner profiles
role=np.hstack([NToh, Lout, Lin]); role=(role-role.mean(0))/(role.std(0)+1e-9)
from sklearn.decomposition import PCA
roleP=PCA(12,random_state=0).fit_transform(np.nan_to_num(role))
lab=KMeans(20,n_init=5,random_state=0).fit_predict(roleP)  # role-orbits from pre-adult
sameNT=np.array([NT[I[k]]==NT[J[k]] for k in range(N)]).astype(float)
samerole=np.array([lab[I[k]]==lab[J[k]] for k in range(N)]).astype(float)
# avoidance among HIGH-CONTACT pairs (top quartile), restricted to agree
hc=(area>np.quantile(area,0.75))&agree
P(f"[ORBIT/NICHE avoidance among high-contact agree pairs, n={hc.sum()}]")
P(f"  synapse rate: same-role={yrep[hc&(samerole==1)].mean():.3f} vs diff-role={yrep[hc&(samerole==0)].mean():.3f} | same-NT={yrep[hc&(sameNT==1)].mean():.3f} vs diff-NT={yrep[hc&(sameNT==0)].mean():.3f}")
# does same-role add BEYOND same-NT? (predict yrep on high-contact pairs)
cv=StratifiedKFold(5,shuffle=True,random_state=0); yh=yrep[hc]
def auc(X): return roc_auc_score(yh,cross_val_predict(LogisticRegression(max_iter=1000),X,yh,cv=cv,method='predict_proba')[:,1])
a_c=auc(area[hc][:,None]); a_nt=auc(np.column_stack([area[hc],sameNT[hc]])); a_r=auc(np.column_stack([area[hc],sameNT[hc],samerole[hc]]))
P(f"  predict synapse | contact={a_c:.3f} +sameNT={a_nt:.3f} +sameROLE={a_r:.3f}  (role beyond NT: Δ{a_r-a_nt:+.3f})")
# permutation: shuffle role labels
rng=np.random.default_rng(0); nl=[]
for s in range(10):
    lp=lab[rng.permutation(M)]; sr=np.array([lp[I[k]]==lp[J[k]] for k in range(N)]).astype(float)
    nl.append(auc(np.column_stack([area[hc],sameNT[hc],sr[hc]]))-a_nt)
nl=np.array(nl); P(f"  [PERM role] real Δ={a_r-a_nt:+.3f} vs shuffled {nl.mean():+.3f}±{nl.std():.3f} -> {'ROLE adds beyond NT' if (a_r-a_nt)>nl.mean()+2*nl.std() else 'role subsumed by NT (no orbit signal beyond identity)'}")
