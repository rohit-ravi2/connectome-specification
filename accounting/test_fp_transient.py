import numpy as np
from scipy.stats import mannwhitneyu
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True)
I,J=z['I'],z['J']; y7,y8=z['y7'],z['y8']; yrep=((y7==1)&(y8==1)).astype(int); N=len(I)
r1=np.load(f"{SC}/resid2.npz",allow_pickle=True)['r1']
raw=np.load(f"{SC}/contactdyn_raw.npz",allow_pickle=True)
keys=[tuple(k) for k in raw['UD_keys']]; vals=raw['UD_vals']
UD={keys[i]:vals[i] for i in range(len(keys))}
colmed=np.nanmedian(vals,0)

# per directed pair: stage mindists (imputed), slope, adult-L1 delta, transient@thr
def stagevec(k):
    a,b=int(I[k]),int(J[k]); d=UD.get((min(a,b),max(a,b)),None)
    if d is None: return None
    di=d.copy()
    for s in range(4):
        if np.isnan(di[s]): di[s]=colmed[s]
    return di
FP=r1<-0.3; WC=yrep==1
print(f"FP n={FP.sum()} wired-core n={WC.sum()}")

def gather(mask):
    S=[];
    for k in np.where(mask)[0]:
        v=stagevec(k)
        if v is not None: S.append(v)
    return np.array(S)
Sfp=gather(FP); Swc=gather(WC)
print(f"usable FP={len(Sfp)} WC={len(Swc)}")

def stats(S):
    slope=np.array([np.polyfit(np.arange(4),s,1)[0] for s in S])
    delta=S[:,3]-S[:,0]  # adult - L1 (positive=retreating)
    return slope,delta
sfp,dfp=stats(Sfp); swc,dwc=stats(Swc)
print(f"slope(nm/stage): FP median={np.median(sfp):.1f} WC median={np.median(swc):.1f}")
u,p=mannwhitneyu(sfp,swc,alternative='greater')  # FP retreat MORE?
# rank-biserial effect size
rb=1-2*u/(len(sfp)*len(swc));
print(f"  MWU FP>WC slope: U={u:.0f} p={p:.3g} rank-biserial={-rb:+.3f}")
print(f"adult-L1 delta(nm): FP median={np.median(dfp):.1f} WC median={np.median(dwc):.1f}")
u2,p2=mannwhitneyu(dfp,dwc,alternative='greater'); rb2=1-2*u2/(len(dfp)*len(dwc))
print(f"  MWU FP>WC delta: U={u2:.0f} p={p2:.3g} rank-biserial={-rb2:+.3f}")

# transient indicator across thresholds
for T in [500,1000,2000,3000]:
    def transient(S):
        inc=S<T; early=inc[:,0]|inc[:,1]; return (early&~inc[:,3]).mean()
    tf=transient(Sfp); tw=transient(Swc)
    # fisher-ish: fraction transient
    print(f"  T={T}nm transient-frac FP={tf:.3f} WC={tw:.3f}  ratio={tf/(tw+1e-9):.2f}")

# also: are FP just SHORTER-distance (tighter) at all stages? (tests co-fasciculation = stable)
print(f"mean mindist all-stage: FP={Sfp.mean():.0f}nm WC={Swc.mean():.0f}nm")
print(f"adult mindist: FP={np.median(Sfp[:,3]):.0f}nm WC={np.median(Swc[:,3]):.0f}nm")
