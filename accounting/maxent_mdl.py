import sys, csv, re, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); m0=np.load(f"{SC}/mode0.npz")
I,J,y7,y8,base,neurons=z['I'],z['J'],z['y7'],z['y8'],z['base'],list(z['neurons']); idx={n:i for i,n in enumerate(neurons)}
M=len(neurons); N=len(I); yrep=((y7==1)&(y8==1)).astype(int); agree=m0['agree']; L1c=m0['L1c']; area=base[:,0]
GL1=np.zeros((M,M)); bud=np.zeros(M)
with open("../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']=='chemical':
            d=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
            if d in (1,2,3,4) and r['pre'] in idx and r['post'] in idx:
                GL1[idx[r['pre']],idx[r['post']]]=1; w=int(r['synapses']); bud[idx[r['pre']]]+=w; bud[idx[r['post']]]+=w
Cff=GL1@GL1; l1deg=((GL1+GL1.T)>0).sum(1)
t5=np.load("../data/tier5/connectome.npz",allow_pickle=True); t5nt={str(n):str(v) for n,v in zip(t5["names"],t5["nt_primary"])}
def bcl(n): return n[:-1] if n[-1] in 'LR' else n
nt=[t5nt.get(neurons[i],t5nt.get(bcl(neurons[i]),"U")) for i in range(M)]; sament=np.array([nt[I[k]]==nt[J[k]] for k in range(N)]).astype(float)
L1syn=GL1[I,J]
cols={'rate':np.ones(N), 'contact':area, 'L1persist':L1syn, 'L1closure':np.log1p(Cff[I,J]),
      'degree':l1deg[I]+l1deg[J], 'budget':bud[I]+bud[J], 'sameNT':sament}
stages=[['rate'],['rate','contact'],['rate','contact','L1persist'],['rate','contact','L1persist','L1closure'],
        ['rate','contact','L1persist','L1closure','degree'],['rate','contact','L1persist','L1closure','degree','budget'],
        ['rate','contact','L1persist','L1closure','degree','budget','sameNT']]
mask=agree; y=yrep[mask]; cv=StratifiedKFold(5,shuffle=True,random_state=0)
def bits(feats):
    X=np.column_stack([cols[c][mask] for c in feats])
    if len(feats)==1: p=np.full(len(y),y.mean())
    else: p=np.clip(cross_val_predict(LogisticRegression(max_iter=2000),X,y,cv=cv,method='predict_proba')[:,1],1e-6,1-1e-6)
    err=-(y*np.log2(p)+(1-y)*np.log2(1-p)).sum()
    mdl=len(feats)*0.5*np.log2(len(y))  # BIC model DL
    return err, err+mdl
P(f"MaxEnt/MDL staged constraints (n={mask.sum()} agree pairs, base rate {y.mean():.3f}):")
P(f"  {'constraints added':52s} {'error bits':>10s} {'total MDL':>10s} {'Δbits':>8s}")
prev=None
for st in stages:
    e,t=bits(st); d=(prev-e) if prev else 0
    P(f"  {'+'.join(st):52s} {e:10.0f} {t:10.0f} {d:+8.0f}")
    prev=e
P("\n(necessary constraint = large Δbits; saturation = added constraint gives ~0; sameNT Δ = identity's bits)")
