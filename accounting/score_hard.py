import numpy as np
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True)
y7,y8=z['y7'],z['y8']; yrep=((y7==1)&(y8==1)).astype(int)
r1=np.load(f"{SC}/resid2.npz",allow_pickle=True)['r1']
h=np.load(f"{SC}/holdout_preds.npz",allow_pickle=True)
p0,p1,cov=h['p0'],h['p1'],h['cov']
hard=np.abs(r1)>0.3; FP=r1<-0.3; FN=r1>0.3
def improv(mask):
    m=mask&cov
    if m.sum()==0: return 0,0
    better=(np.abs(yrep-p1)[m]<np.abs(yrep-p0)[m]).mean()
    return better,m.sum()
for name,mask in [('hard-968',hard),('FP-390',FP),('FN-578',FN)]:
    b,n=improv(mask)
    print(f"{name}: covered={n}  frac improved={b:.3f}")
# mean abs-error change on covered hard
m=hard&cov
print(f"hard MAE p0={np.abs(yrep-p0)[m].mean():.3f} p1={np.abs(yrep-p1)[m].mean():.3f}")
