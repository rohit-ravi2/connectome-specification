import numpy as np
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True)
I,J,neurons=z['I'],z['J'],list(z['neurons']); N=len(I)
raw=np.load(f"{SC}/contactdyn_raw.npz",allow_pickle=True)
keys=[tuple(k) for k in raw['UD_keys']]; vals=raw['UD_vals']  # (5030,4) stage mindists nm
UD={keys[i]:vals[i] for i in range(len(keys))}
T=2000.0
stages=['L1','L2','L3','adult']

# column medians (ignoring NaN) for imputation of missing distances
colmed=np.nanmedian(vals,0)
print("stage NaN fraction:", np.isnan(vals).mean(0).round(3), "colmed(nm):",colmed.round(0))

def feats_undirected(d):
    # d = 4 stage mindists (nm), possibly NaN
    dd=d.copy()
    obs=~np.isnan(dd)
    di=dd.copy()
    for s in range(4):
        if np.isnan(di[s]): di[s]=colmed[s]
    inc=(di<T).astype(float)  # in-contact per stage (after imputation)
    inc_obs=np.where(obs,(dd<T),False)  # true contact only where observed
    ncontact=inc.sum()
    # first stage in contact
    first=next((s for s in range(4) if inc[s]>0),4)
    # transient: in contact at L1 or L2 but NOT adult
    early_contact=(inc[0]>0)or(inc[1]>0)
    transient=float(early_contact and inc[3]==0)
    late_emerge=float((not early_contact) and inc[3]>0)
    # slope of min-distance across stages (nm per stage); positive = retreating
    x=np.arange(4); slope=np.polyfit(x,di,1)[0]
    # adult vs L1 delta
    delta_a_l1=di[3]-di[0]
    return np.array([di[0],di[1],di[2],di[3], inc[0],inc[1],inc[2],inc[3],
                     ncontact, float(first), transient, late_emerge, slope, delta_a_l1]), inc_obs

# undirected feature dict
UF={}; OBS={}
for k in keys:
    f,o=feats_undirected(UD[k]); UF[k]=f; OBS[k]=o
med=np.median(np.array(list(UF.values())),0)
nf=len(med)

DYN=np.zeros((N,nf))
for k in range(N):
    a,b=int(I[k]),int(J[k]); key=(min(a,b),max(a,b))
    DYN[k]=UF.get(key,med)

# standardize continuous columns (distances, slope, delta); keep binaries/counts as-is but scale
cont=[0,1,2,3,8,9,12,13]
DYN[:,cont]=(DYN[:,cont]-DYN[:,cont].mean(0))/(DYN[:,cont].std(0)+1e-9)

fnames=['d_L1','d_L2','d_L3','d_adult','c_L1','c_L2','c_L3','c_adult',
        'ncontact','first','transient','late_emerge','slope','delta_a_l1']
np.savez(f"{SC}/contactdyn.npz",DYN=DYN,fnames=np.array(fnames,dtype=object))

# also save RAW per-pair-directed for the FP/FN test (unstandardized transient & slope)
rawdir=np.zeros((N,3))  # transient, slope(nm/stage), ncontact_observed
for k in range(N):
    a,b=int(I[k]),int(J[k]); key=(min(a,b),max(a,b))
    d=UD.get(key,None)
    if d is None:
        rawdir[k]=[np.nan,np.nan,np.nan]; continue
    di=d.copy()
    for s in range(4):
        if np.isnan(di[s]): di[s]=colmed[s]
    inc=(di<T)
    early=(inc[0])or(inc[1])
    trans=float(early and not inc[3])
    slope=np.polyfit(np.arange(4),di,1)[0]
    obs=(~np.isnan(d)).sum() and ((d<T)&(~np.isnan(d))).sum()
    rawdir[k]=[trans,slope,((d<T)&(~np.isnan(d))).sum()]
np.savez(f"{SC}/contactdyn_dir.npz",rawdir=rawdir)
print("saved contactdyn.npz",DYN.shape,"| feat means:",DYN.mean(0).round(2))
print("transient frac overall:", np.nanmean(rawdir[:,0]).round(3))
