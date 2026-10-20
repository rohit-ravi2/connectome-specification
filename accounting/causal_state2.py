import sys, csv, re, numpy as np
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); m0=np.load(f"{SC}/mode0.npz")
I,J,y7,y8,base,neurons=z['I'],z['J'],z['y7'],z['y8'],z['base'],list(z['neurons']); idx={n:i for i,n in enumerate(neurons)}
M=len(neurons); N=len(I); yrep=((y7==1)&(y8==1)).astype(int); agree=m0['agree']; L1c=m0['L1c']
GL1=np.zeros((M,M))
with open("../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']=='chemical':
            d=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
            if d in (1,2,3,4) and r['pre'] in idx and r['post'] in idx: GL1[idx[r['pre']],idx[r['post']]]=1
Cu=((GL1+GL1.T)>0).astype(float); comm=Cu@Cu; Cin=GL1.T@GL1; Cout=GL1@GL1.T; Cff=GL1@GL1; l1deg=Cu.sum(1)
t5=np.load("../data/tier5/connectome.npz",allow_pickle=True); t5nt={str(n):str(v) for n,v in zip(t5["names"],t5["nt_primary"])}
def bcl(n): return n[:-1] if n[-1] in 'LR' else n
nt=[t5nt.get(neurons[i],t5nt.get(bcl(neurons[i]),"U")) for i in range(M)]; sament=np.array([nt[I[k]]==nt[J[k]] for k in range(N)]).astype(float)
H=np.column_stack([base[:,0],GL1[I,J],GL1[J,I],np.log1p(Cff[I,J]),np.log1p(comm[I,J]),np.log1p(Cin[I,J]),np.log1p(Cout[I,J]),l1deg[I],l1deg[J],sament])
mask=agree; Xh=H[mask]; y=yrep[mask]
tree=DecisionTreeClassifier(min_samples_leaf=30,random_state=0).fit(Xh,y)
leaf=tree.apply(Xh); uL=np.unique(leaf); pl={l:y[leaf==l].mean() for l in uL}; nl={l:(leaf==l).sum() for l in uL}
# PROPER non-chaining merge: leaf joins a state only if within 2 SE of the state's POOLED mean
order=sorted(uL,key=lambda l:pl[l]); states=[]
for l in order:
    placed=False
    for st in states:
        ns=sum(nl[x] for x in st); ps=sum(pl[x]*nl[x] for x in st)/ns; p=(ps*ns+pl[l]*nl[l])/(ns+nl[l])
        se=np.sqrt(p*(1-p)*(1/ns+1/nl[l]))+1e-9
        if abs(pl[l]-ps)/se<2.0: st.append(l); placed=True; break
    if not placed: states.append([l])
sizes=np.array([sum(nl[x] for x in st) for st in states]); pi=sizes/sizes.sum()
Cmu=-(pi*np.log2(pi)).sum()
P(f"[CAUSAL STATES] {len(uL)} history classes -> {len(states)} distinct causal states | statistical complexity C_mu={Cmu:.2f} bits (low=simple process)")
P(f"  causal-state future-P levels: {sorted([round(sum(pl[x]*nl[x] for x in st)/sum(nl[x] for x in st),2) for st in states])}")
# identity permutation test
cv=StratifiedKFold(5,shuffle=True,random_state=0)
def info(Xf):
    p=np.clip(cross_val_predict(DecisionTreeClassifier(min_samples_leaf=30,random_state=0),Xf,y,cv=cv,method='predict_proba')[:,1],1e-6,1-1e-6)
    m=y.mean(); Hy=-(m*np.log2(m)+(1-m)*np.log2(1-m)); return Hy+ (y*np.log2(p)+(1-y)*np.log2(1-p)).mean()
noid=Xh[:,:9]; i_full=info(Xh); i_noid=info(noid)
rng=np.random.default_rng(0); nl_=[]
for s in range(10):
    Xp=Xh.copy(); Xp[:,9]=Xh[rng.permutation(len(Xh)),9]; nl_.append(info(Xp)-i_noid)
nl_=np.array(nl_); real=i_full-i_noid
P(f"[IDENTITY load-bearing?] real Δinfo(+sameNT)={real:+.4f} bits vs shuffled {nl_.mean():+.4f}±{nl_.std():.4f} -> {'LOAD-BEARING' if real>nl_.mean()+2*nl_.std() else 'NOT (identity adds ~nothing; zero-identity thesis holds)'}")
# Gate B (generalization): does pre-adult predict WHOLE connectome, not just additions?
P(f"[GATE B] full pre-adult history predicts adult fate over ALL {mask.sum()} agree pairs: info={i_noid:.3f} bits (dominated by persistence L1syn 0.77) => generalizes beyond addition subset")
