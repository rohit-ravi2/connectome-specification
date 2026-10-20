import numpy as np
from scipy.stats import fisher_exact
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True)
I,J=z['I'],z['J']; y7,y8=z['y7'],z['y8']; yrep=((y7==1)&(y8==1)).astype(int); N=len(I)
r1=np.load(f"{SC}/resid2.npz",allow_pickle=True)['r1']
raw=np.load(f"{SC}/contactdyn_raw.npz",allow_pickle=True)
keys=[tuple(k) for k in raw['UD_keys']]; vals=raw['UD_vals']
UD={keys[i]:vals[i] for i in range(len(keys))}
colmed=np.nanmedian(vals,0)
def stagevec(k):
    a,b=int(I[k]),int(J[k]); d=UD.get((min(a,b),max(a,b)),None)
    if d is None: return None
    di=d.copy()
    for s in range(4):
        if np.isnan(di[s]): di[s]=colmed[s]
    return di
FP=r1<-0.3; WC=yrep==1
T=500.0
def trans_mask(mask):
    out=[]
    for k in np.where(mask)[0]:
        v=stagevec(k)
        if v is None: continue
        inc=v<T; out.append(bool((inc[0] or inc[1]) and not inc[3]))
    return np.array(out)
tfp=trans_mask(FP); twc=trans_mask(WC)
a,b=tfp.sum(),(~tfp).sum(); c,d=twc.sum(),(~twc).sum()
OR,p=fisher_exact([[a,b],[c,d]],alternative='greater')
print(f"T={T}: FP transient {a}/{len(tfp)} ({a/len(tfp):.3f})  WC {c}/{len(twc)} ({c/len(twc):.3f})")
print(f"Fisher OR={OR:.2f} p={p:.3g}")
