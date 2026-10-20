import sys, csv, re, numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); m0=np.load(f"{SC}/mode0.npz")
I,J,y7,y8,base,neurons=z['I'],z['J'],z['y7'],z['y8'],z['base'],list(z['neurons']); idx={n:i for i,n in enumerate(neurons)}
M=len(neurons); N=len(I); yrep=((y7==1)&(y8==1)).astype(int)
GL1=np.zeros((M,M))
with open("../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']=='chemical':
            d=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
            if d in (1,2,3,4) and r['pre'] in idx and r['post'] in idx: GL1[idx[r['pre']],idx[r['post']]]=1
Cu=((GL1+GL1.T)>0).astype(float); Cff=GL1@GL1; comm=Cu@Cu; Cin=GL1.T@GL1; Cout=GL1@GL1.T; l1deg=Cu.sum(1)
t5=np.load("../data/tier5/connectome.npz",allow_pickle=True); t5nt={str(n):str(v) for n,v in zip(t5["names"],t5["nt_primary"])}
def bcl(n): return n[:-1] if n[-1] in 'LR' else n
nt=[t5nt.get(neurons[i],t5nt.get(bcl(neurons[i]),"U")) for i in range(M)]; sament=np.array([nt[I[k]]==nt[J[k]] for k in range(N)]).astype(float)
H=np.column_stack([base[:,0],GL1[I,J],GL1[J,I],np.log1p(Cff[I,J]),np.log1p(comm[I,J]),np.log1p(Cin[I,J]),np.log1p(Cout[I,J]),l1deg[I],l1deg[J],sament])
# process propensity p from pre-adult history (cross-validated, F1-clean)
cv=StratifiedKFold(5,shuffle=True,random_state=0)
p=np.clip(cross_val_predict(DecisionTreeClassifier(min_samples_leaf=30,random_state=0),H,yrep,cv=cv,method='predict_proba')[:,1],1e-4,1-1e-4)
disagree=(y7!=y8).astype(int); pv=2*p*(1-p)
P(f"[FINGERPRINT: mean structure]")
P(f"  edge rate: real adult {yrep.mean():.3f} | process E[p] {p.mean():.3f}  -> {'MATCH' if abs(yrep.mean()-p.mean())<0.02 else 'off'}")
# forward-generate: sample 2 synthetic animals per pair ~ Bernoulli(p); compute dispersion of GENERATED data
rng=np.random.default_rng(0); dg=[]
for _ in range(200):
    a=(rng.random(N)<p).astype(int); b=(rng.random(N)<p).astype(int)
    dg.append((a!=b).mean()/pv.mean())
dg=np.array(dg)
disp_real=disagree.mean()/pv.mean()
P(f"\n[FINGERPRINT: animal-to-animal variability — the crux]")
P(f"  dispersion ratio: REAL={disp_real:.3f} | PROCESS-generated ensemble={dg.mean():.3f}±{dg.std():.3f}")
P(f"  -> process (independent Bernoulli sampling) gives ~1.0 by construction; REAL is {disp_real:.2f}")
P(f"  GAP: real is {'MORE deterministic' if disp_real<dg.mean()-2*dg.std() else 'consistent'} than the process predicts")
P(f"     under-dispersion {1-disp_real:.2f} = per-pair determinism the {6}-state process does NOT generate (= the white per-edge residual)")
# add/prune split reproduction (among eligible)
addcand=(GL1[I,J]==0); prcand=(GL1[I,J]==1)
P(f"\n[FINGERPRINT: add/prune]")
P(f"  ADD: real added-rate among L1-absent contacting = {yrep[addcand].mean():.3f} | process E[p|L1-absent] = {p[addcand].mean():.3f}")
P(f"  PRUNE: real retained-rate among L1-present = {yrep[prcand].mean():.3f} | process E[p|L1-present] = {p[prcand].mean():.3f}")
# is same-NT needed to hit the fingerprint? compare process WITH vs WITHOUT identity on the mean structure
pno=np.clip(cross_val_predict(DecisionTreeClassifier(min_samples_leaf=30,random_state=0),H[:,:9],yrep,cv=cv,method='predict_proba')[:,1],1e-4,1-1e-4)
# same-NT high-contact avoidance signature: among high-contact L1-absent same-NT pairs, real add-rate vs process
hc=(base[:,0]>np.quantile(base[:,0],0.75))&addcand
snt=hc&(sament==1); dnt=hc&(sament==0)
P(f"\n[IDENTITY signature] high-contact L1-absent: real add-rate same-NT={yrep[snt].mean():.3f} vs diff-NT={yrep[dnt].mean():.3f}")
P(f"  process WITH identity captures it (p same-NT={p[snt].mean():.3f} vs diff-NT={p[dnt].mean():.3f}); WITHOUT identity p same-NT={pno[snt].mean():.3f} vs diff-NT={pno[dnt].mean():.3f}")
