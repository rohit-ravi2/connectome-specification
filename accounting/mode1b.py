import sys, csv, re, numpy as np
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); m0=np.load(f"{SC}/mode0.npz"); zs=np.load(f"{SC}/subcellular.npz"); cd=np.load(f"{SC}/contactdyn_raw.npz",allow_pickle=True)
I,J,y7,y8,base,neurons=z['I'],z['J'],z['y7'],z['y8'],z['base'],list(z['neurons']); idx={n:i for i,n in enumerate(neurons)}
M=len(neurons); N=len(I); yrep=((y7==1)&(y8==1)).astype(int); L1c=m0['L1c']; agree=m0['agree']; sub=zs['sub']
udk=cd['UD_keys']; udv=cd['UD_vals']; L1md={}
for r in range(len(udk)): a,b=int(udk[r][0]),int(udk[r][1]); L1md[(min(a,b),max(a,b))]=udv[r][0]
L1elig=np.array([ (lambda v:(not np.isnan(v)) and v<2000)(L1md.get((min(I[k],J[k]),max(I[k],J[k])),np.nan)) for k in range(N)])
dp=defaultdict(set)
with open("../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']=='chemical':
            d=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
            if d in (1,2,3,4) and r['pre'] in idx and r['post'] in idx: dp[(idx[r['pre']],idx[r['post']])].add(d)
def build_triad(GL1):
    Cu=((GL1+GL1.T)>0).astype(float); return np.column_stack([np.log1p((Cu@Cu)[I,J]),np.log1p((GL1.T@GL1)[I,J]),np.log1p((GL1@GL1.T)[I,J]),np.log1p((GL1@GL1)[I,J])])
GL1=np.zeros((M,M))
for (a,b) in dp: GL1[a,b]=1
t5=np.load("../data/tier5/connectome.npz",allow_pickle=True); t5nt={str(n):str(v) for n,v in zip(t5["names"],t5["nt_primary"])}
def bcl(n): return n[:-1] if n[-1] in 'LR' else n
nt=[t5nt.get(neurons[i],t5nt.get(bcl(neurons[i]),"U")) for i in range(M)]; sament=np.array([nt[I[k]]==nt[J[k]] for k in range(N)]).astype(float)
l1deg=((GL1+GL1.T)>0).sum(1)
NONTRI=np.column_stack([base[:,0],base[:,1],base[:,2],base[:,3],base[:,4],l1deg[I],l1deg[J],sub[:,0],sub[:,1],sub[:,2],sub[:,5],sub[:,6],sament])  # contact+degree+subcell+NT, NO triadic
gate=(L1c==0)&L1elig&agree; yg=yrep[gate]
nf={n:i%5 for i,n in enumerate(sorted(set([neurons[i] for i in I]+[neurons[j] for j in J])))}
def nho(X):
    ii=np.where(gate)[0]; pred=np.full(len(ii),np.nan)
    for f in range(5):
        te=[t for t,k in enumerate(ii) if nf[neurons[I[k]]]==f and nf[neurons[J[k]]]==f]; tr=[t for t,k in enumerate(ii) if nf[neurons[I[k]]]!=f and nf[neurons[J[k]]]!=f]
        if len(te)<8 or len(set(yg[tr]))<2: continue
        pred[np.array(te)]=GradientBoostingClassifier(random_state=0).fit(X[ii][tr],yg[tr]).predict_proba(X[ii][te])[:,1]
    m=~np.isnan(pred); return roc_auc_score(yg[m],pred[m])
a0=nho(NONTRI); areal=nho(np.column_stack([NONTRI,build_triad(GL1)]))
P(f"[incremental] contact+degree+subcell+NT (no triadic) AUC={a0:.3f} -> +L1-TRIADIC-closure AUC={areal:.3f} (Δ{areal-a0:+.3f})")
rng=np.random.default_rng(0); nl=[]
for s in range(6):
    p=rng.permutation(M); nl.append(nho(np.column_stack([NONTRI,build_triad(GL1[np.ix_(p,p)])]))-a0)
nl=np.array(nl); P(f"[PERM: shuffle L1-graph node labels] real Δ={areal-a0:+.3f} vs shuffled {nl.mean():+.3f}±{nl.std():.3f} -> {'REAL local developmental closure' if (areal-a0)>nl.mean()+2*nl.std() else 'NOT beating null (=degree/autocorr)'}")
