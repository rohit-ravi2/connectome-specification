import openpyxl, csv, re, math, numpy as np
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
D="../data/celegans/data/connectome"
# adult contact
wb=openpyxl.load_workbook(f"{D}/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx",read_only=True,data_only=True)
data=list(wb['adult nerve ring neighbors'].iter_rows(values_only=True)); neurons=[h for h in data[0][1:] if isinstance(h,str)]; idx={n:i for i,n in enumerate(neurons)}; M=len(neurons)
A=np.zeros((M,M))
for row in data[1:]:
    if row[0] in idx:
        for j,v in enumerate(row[1:]):
            if v not in (None,''):
                try: A[idx[row[0]],j]=float(v)
                except: pass
A=np.maximum(A,A.T); cdeg=(A>0).sum(1)
# chemical directed presence per dataset
dchem=defaultdict(set)
with open(f"{D}/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']=='chemical':
            ds=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1)); dchem[(r['pre'],r['post'])].add(ds)
# directed contacting pairs + features
I,J,feat,pres=[],[],[],[]
for i in range(M):
    for j in range(M):
        if i==j or A[i,j]<=0: continue
        s=dchem.get((neurons[i],neurons[j]),set())
        feat.append([math.log1p(A[i,j]),int(((A[i]>0)&(A[j]>0)).sum()),
                     float(A[i]@A[j]/(np.linalg.norm(A[i])*np.linalg.norm(A[j])+1e-9)),int(cdeg[i]),int(cdeg[j])])
        pres.append(s)
X=np.array(feat); N=len(X)
cv=StratifiedKFold(5,shuffle=True,random_state=0)
def ratio(anims):
    n=len(anims)
    ys=[np.array([1 if a in s else 0 for s in pres]) for a in anims]
    # fit p pooled over these animals
    yp=np.concatenate(ys); Xp=np.vstack([X]*n)
    p=np.clip(cross_val_predict(GradientBoostingClassifier(random_state=0),Xp,yp,cv=cv,method='predict_proba')[:,1][:N],1e-4,1-1e-4)
    k=np.sum(ys,axis=0)  # count present across animals
    # observed avg pairwise disagreement per edge = k(n-k)/C(n,2); predicted = 2p(1-p)
    npairs=n*(n-1)/2
    obs=(k*(n-k)/npairs).mean(); pred=(2*p*(1-p)).mean()
    return obs,pred,obs/pred, np.mean([y.mean() for y in ys])
print("DISPERSION across replicate sets (ratio<1 = under-dispersed = sharper propensity than features)")
print(f"  {'replicate set':22s} {'edge rate':>9} {'obs_disagree':>13} {'pred 2p(1-p)':>13} {'ratio':>6}")
for lab,an in [("adults (7,8)",[7,8]),("L1 animals (1,2,3,4)",[1,2,3,4]),("early L1 (1,2,3)",[1,2,3])]:
    o,pr,rt,er=ratio(an); print(f"  {lab:22s} {er:9.3f} {o:13.3f} {pr:13.3f} {rt:6.2f}")
print("\n  interpretation:")
print("  - if L1 ratio<1 too -> under-dispersion is stage-general (real sharp propensity, not adult-pipeline artifact)")
print("  - L1 spans 0-16h so developmental change ADDS variance (pushes ratio UP); surviving under-dispersion is strong")
