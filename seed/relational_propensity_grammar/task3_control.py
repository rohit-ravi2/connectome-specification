import sys,numpy as np,pandas as pd,itertools
from sklearn.ensemble import GradientBoostingClassifier
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/feats.npz",allow_pickle=True)
I,J=z['I'],z['J']; y7,y8,yrep=z['y7'],z['y8'],z['yrep']
CONTACT,HO,TIMING=z['CONTACT'],z['HO'],z['TIMING']; neurons=list(z['neurons']); N=len(I)
HN=list(z['HO_NAMES'])
# HO_pure = transitivity/assortativity ONLY (drop direct-edge-leaking Katz_ij/ji, RDPG_ij/ji)
pure_idx=[k for k,n in enumerate(HN) if n not in ('RDPG16_ij','RDPG16_ji','Katz_ij','Katz_ji')]
HOp=HO[:,pure_idx]
P("HO_pure features:",[HN[k] for k in pure_idx])
blocks={'C':CONTACT,'H':HOp,'T':TIMING}
endpts=sorted(set([neurons[int(I[k])] for k in range(N)]+[neurons[int(J[k])] for k in range(N)]))
rng=np.random.RandomState(0); perm=rng.permutation(len(endpts)); nf={endpts[perm[i]]:i%5 for i in range(len(endpts))}
NI=np.array([nf[neurons[int(I[k])]] for k in range(N)]); NJ=np.array([nf[neurons[int(J[k])]] for k in range(N)])
covered=np.zeros(N,bool)
for f in range(5): covered|=((NI==f)&(NJ==f))
def hp(X,y):
    pred=np.full(N,np.nan)
    for f in range(5):
        te=(NI==f)&(NJ==f); tr=(NI!=f)&(NJ!=f)
        if te.sum()==0: continue
        pred[te]=GradientBoostingClassifier(random_state=0).fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
    return pred
def bss(X):
    p=hp(X,yrep); m=covered; pc=np.clip(p[m],1e-4,1-1e-4); yc=yrep[m]; base=yc.mean()
    return 1-np.mean((pc-yc)**2)/np.mean((base-yc)**2)
R={}
for r in range(1,4):
    for c in itertools.combinations('CHT',r):
        s=''.join(c); R[s]=bss(np.hstack([blocks[b] for b in s])); P(f"  {s:4s} BSS={R[s]:+.4f}")
U={'C':R['CHT']-R['HT'],'H':R['CHT']-R['CT'],'T':R['CHT']-R['CH']}
tot=R['CHT']
P("\n=== CONTROL: HO = pure transitivity only (no direct-edge leak) ===")
P(f"full BSS={tot:+.4f}")
P(f"  unique_CONTACT      ={U['C']:+.4f} ({100*U['C']/tot:+.1f}%)")
P(f"  unique_HO_transitivity={U['H']:+.4f} ({100*U['H']/tot:+.1f}%)")
P(f"  unique_TIMING       ={U['T']:+.4f} ({100*U['T']/tot:+.1f}%)")
