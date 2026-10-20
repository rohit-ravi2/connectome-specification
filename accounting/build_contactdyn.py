import sys, json, numpy as np
from scipy.spatial import cKDTree
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
SK="../seed/data/external/mechanistic_frontier/witvliet2021_developmental/data/skeletons"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True)
I,J,neurons=z['I'],z['J'],list(z['neurons']); idx={n:i for i,n in enumerate(neurons)}; N=len(I)
und=sorted(set((min(int(I[k]),int(J[k])),max(int(I[k]),int(J[k]))) for k in range(N)))
P(f"N={N} undirected={len(und)}")

def load(dsn):
    sk=json.load(open(f"{SK}/Dataset{dsn}_skeletons.json")); D={}
    for n,v in sk.items():
        if n in idx and v.get('coords'):
            C=np.array([v['coords'][k] for k in v['coords']],float)
            D[n]=cKDTree(C) if len(C)>0 else None
    return D

def mindist(D,a,b):
    na,nb=neurons[a],neurons[b]
    if na not in D or nb not in D or D[na] is None or D[nb] is None: return np.nan
    Ta,Tb=D[na],D[nb]
    da,_=Tb.query(Ta.data)  # each a-node to nearest b-node
    return float(da.min())

# stages: L1=avg(1-4), L2=5, L3=6, adult=avg(7-8)
stage_ds={'L1':[1,2,3,4],'L2':[5],'L3':[6],'adult':[7,8]}
stages=['L1','L2','L3','adult']
# accumulate per-dataset mindist, then average within stage
pair_ds={}  # (a,b) -> {dsn: mindist}
for dsn in range(1,9):
    D=load(dsn); P(f"ds{dsn}: {len(D)} skeletons")
    for (a,b) in und:
        pair_ds.setdefault((a,b),{})[dsn]=mindist(D,a,b)
    del D
    P(f"ds{dsn} done")

# per-stage min-distance per undirected pair
UD={}  # (a,b) -> array of 4 stage mindists (nm), NaN if missing
for (a,b) in und:
    row=[]
    for s in stages:
        vals=[pair_ds[(a,b)][d] for d in stage_ds[s] if not np.isnan(pair_ds[(a,b)][d])]
        row.append(np.mean(vals) if vals else np.nan)
    UD[(a,b)]=np.array(row,float)

np.savez(f"{SC}/contactdyn_raw.npz",
         und=np.array(und), stages=np.array(stages,dtype=object),
         UD_keys=np.array(list(UD.keys())), UD_vals=np.array(list(UD.values())))
P("saved contactdyn_raw.npz", np.array(list(UD.values())).shape)
