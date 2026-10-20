import sys, json, numpy as np
from scipy.spatial import cKDTree
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
SK="../seed/data/external/mechanistic_frontier/witvliet2021_developmental/data/skeletons"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True)
I,J,neurons=z['I'],z['J'],list(z['neurons']); idx={n:i for i,n in enumerate(neurons)}; M=len(neurons); N=len(I)
und=sorted(set((min(I[k],J[k]),max(I[k],J[k])) for k in range(N)))
T=2000.0  # nm, contact-zone threshold (centerline)
def load(dsn):
    sk=json.load(open(f"{SK}/Dataset{dsn}_skeletons.json")); D={}
    for n,v in sk.items():
        if n in idx and v.get('coords'):
            keys=list(v['coords']); C=np.array([v['coords'][k] for k in keys],float)
            dr=np.array([v['dist_to_root'][k] for k in keys],float); L=max(v.get('length',1),1)
            D[n]=(C,dr,L,cKDTree(C))
    return D
def pair_feats(D,a,b):
    if neurons[a] not in D or neurons[b] not in D: return None
    Ca,dra,La,Ta=D[neurons[a]]; Cb,drb,Lb,Tb=D[neurons[b]]
    da,_=Tb.query(Ca); db,_=Ta.query(Cb)  # dist from each a-node to nearest b-node, and vice versa
    ia=np.argmin(da); ib=np.argmin(db); dmin=da[ia]
    along_a=dra[ia]/La; along_b=drb[ib]/Lb
    ma=da<T; mb=db<T
    cfrac_a=ma.mean(); cfrac_b=mb.mean()
    spread_a=(dra[ma].std()/La) if ma.sum()>1 else 0.0; spread_b=(drb[mb].std()/Lb) if mb.sum()>1 else 0.0
    pos=(Ca[ia]+Cb[ib])/2
    return np.array([along_a,along_b,dmin,cfrac_a,cfrac_b,spread_a,spread_b,pos[0],pos[1],pos[2]])
D7=load(7); P(f"adult7 skeletons: {len(D7)} neurons"); D8=load(8); P(f"adult8 skeletons: {len(D8)}")
UF={}; miss=0
for (a,b) in und:
    f7=pair_feats(D7,a,b); f8=pair_feats(D8,a,b)
    fs=[f for f in (f7,f8) if f is not None]
    if fs: UF[(a,b)]=np.mean(fs,0)
    else: miss+=1
P(f"undirected pairs with subcellular feats: {len(UF)}/{len(und)} (miss {miss})")
# build directed rows: along_src/along_tgt oriented; symmetric feats shared
med=np.median(np.array(list(UF.values())),0)
sub=np.zeros((N,10))
for k in range(N):
    a,b=I[k],J[k]; key=(min(a,b),max(a,b)); f=UF.get(key,med).copy()
    if a>b:  # swap along_a<->along_b, cfrac, spread to keep src=I orientation
        f=f.copy(); f[[0,1]]=f[[1,0]]; f[[3,4]]=f[[4,3]]; f[[5,6]]=f[[6,5]]
    sub[k]=f
# standardize position cols
sub[:,7:10]=(sub[:,7:10]-sub[:,7:10].mean(0))/(sub[:,7:10].std(0)+1e-9)
np.savez(f"{SC}/subcellular.npz",sub=sub)
P("saved subcellular.npz",sub.shape,"| feature means:",sub.mean(0).round(2))
