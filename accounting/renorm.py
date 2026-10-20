import sys, numpy as np
from sklearn.model_selection import KFold
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); m0=np.load(f"{SC}/mode0.npz")
I,J,y7,y8,neurons=z['I'],z['J'],z['y7'],z['y8'],list(z['neurons']); M=len(neurons); N=len(I); yrep=((y7==1)&(y8==1)).astype(int); agree=m0['agree']
mask=agree; Iy,Jy,y=I[mask],J[mask],yrep[mask]; n=len(y)
t5=np.load("../data/tier5/connectome.npz",allow_pickle=True); t5nt={str(x):str(v) for x,v in zip(t5["names"],t5["nt_primary"])}
def bcl(x): return x[:-1] if x[-1] in 'LR' else x
NT=[t5nt.get(neurons[i],t5nt.get(bcl(neurons[i]),"U")) for i in range(M)]
# groupings (coarse-grainings)
G_ind={i:i for i in range(M)}
cls=sorted(set(bcl(neurons[i]) for i in range(M))); G_cls={i:cls.index(bcl(neurons[i])) for i in range(M)}
nts=sorted(set(NT)); G_nt={i:nts.index(NT[i]) for i in range(M)}
G_one={i:0 for i in range(M)}
kf=KFold(5,shuffle=True,random_state=0)
def mdl(gmap,nb):
    bi=np.array([gmap[i] for i in Iy]); bj=np.array([gmap[j] for j in Jy])
    err=0.0
    for tr,te in kf.split(np.arange(n)):
        rate=np.full((nb,nb),y[tr].mean()); cnt=np.zeros((nb,nb)); s=np.zeros((nb,nb))
        np.add.at(s,(bi[tr],bj[tr]),y[tr]); np.add.at(cnt,(bi[tr],bj[tr]),1)
        m=cnt>0; rate[m]=(s[m]+y[tr].mean())/(cnt[m]+1)  # smoothed block rate
        p=np.clip(rate[bi[te],bj[te]],1e-6,1-1e-6)
        err+=-(y[te]*np.log2(p)+(1-y[te])*np.log2(1-p)).sum()
    occ=len(set(zip(bi.tolist(),bj.tolist())))  # occupied block-pairs = params
    return err, err+occ*0.5*np.log2(n), occ
P(f"[RENORMALIZATION: block-model MDL across scales, n={n} agree pairs]")
P(f"  {'scale':18s} {'#blocks':>8s} {'err bits':>10s} {'+modelDL=MDL':>13s} {'#params':>8s}")
for lab,g,nb in [("individual",G_ind,M),("L/R class",G_cls,len(cls)),("NT class",G_nt,len(nts)),("global rate",G_one,1)]:
    e,t,occ=mdl(g,nb); P(f"  {lab:18s} {nb:8d} {e:10.0f} {t:13.0f} {occ:8d}")
P("\n(min MDL scale = natural description scale; individual should win on err but lose on params if specification is coarse)")
