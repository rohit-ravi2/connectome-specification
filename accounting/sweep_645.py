import openpyxl, csv, re, math, json, numpy as np
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
D="../data/celegans/data/connectome"
SK="../seed/data/external/mechanistic_frontier/witvliet2021_developmental/data/skeletons"
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
# chemical + electrical edges by dataset
dchem=defaultdict(set); delec=np.zeros((M,M))
with open(f"{D}/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        ds=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
        if r['pre'] in idx and r['post'] in idx:
            if r['type']=='chemical': dchem[(r['pre'],r['post'])].add(ds)
            elif r['type']=='electrical' and ds in (7,8):
                delec[idx[r['pre']],idx[r['post']]]=1; delec[idx[r['post']],idx[r['pre']]]=1
# Cook chemical (for higher-order + RDPG)
def isneu(x): return bool(isinstance(x,str) and re.fullmatch(r"[A-Z]{2,4}[0-9]?[LR]?[0-9]?",str(x).strip()) and str(x).strip() in idx)
wc=openpyxl.load_workbook(f"{D}/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
cd=list(wc['hermaphrodite chemical'].iter_rows(values_only=True))
hr=max(range(6),key=lambda r:sum(isneu(x) for x in cd[r])); colmap={c:cd[hr][c].strip() for c in range(len(cd[hr])) if isneu(cd[hr][c])}
rhc=max(range(6),key=lambda c:sum(isneu(cd[r][c]) for r in range(hr+1,len(cd)) if c<len(cd[r])))
C=np.zeros((M,M))
for r in range(hr+1,len(cd)):
    if rhc<len(cd[r]) and isneu(cd[r][rhc]):
        pi=idx[cd[r][rhc].strip()]
        for c,post in colmap.items():
            if c<len(cd[r]) and cd[r][c] not in (None,''):
                try: C[pi,idx[post]]=1.0 if float(cd[r][c])>=1 else 0.0
                except: pass
Cin=C.T@C;Cout=C@C.T;Cff=C@C;Cu=((C+C.T)>0).astype(float);comm=Cu@Cu
U,s,Vt=np.linalg.svd(C); d=16; RD=(U[:,:d]*np.sqrt(s[:d]))@(np.sqrt(s[:d])[:,None]*Vt[:d])
# 3D centroids from skeletons (avg over adults 7,8)
cent=defaultdict(list)
for dsn in (7,8):
    sk=json.load(open(f"{SK}/Dataset{dsn}_skeletons.json"))
    for n,v in sk.items():
        if n in idx and 'coords' in v and v['coords']:
            pts=np.array(list(v['coords'].values()),float); cent[n].append(pts.mean(0))
centroid={n:np.mean(v,0) for n,v in cent.items() if v}
missing=[n for n in neurons if n not in centroid]
# contact-graph closeness centrality (global position, non-circular)
Ab=(A>0).astype(float)
import scipy.sparse.csgraph as csg
from scipy.sparse import csr_matrix
dist=csg.shortest_path(csr_matrix(Ab),directed=False,unweighted=True)
dist[np.isinf(dist)]=dist[~np.isinf(dist)].max()+1
closeness=1.0/(dist.mean(1)+1e-9)
# electrical higher-order
Ein=delec.T@delec; Ecomm=((delec+delec.T)>0).astype(float); Ecomm=Ecomm@Ecomm
# ---- build pair table ----
I,J,y7,y8,base,wire,cross,poly=[],[],[],[],[],[],[],[]
for i in range(M):
    for j in range(M):
        if i==j or A[i,j]<=0: continue
        st=dchem.get((neurons[i],neurons[j]),set())
        I.append(i);J.append(j);y7.append(1 if 7 in st else 0);y8.append(1 if 8 in st else 0)
        # best baseline: contact + Cook HO + RDPG-16
        base.append([math.log1p(A[i,j]),int(((A[i]>0)&(A[j]>0)).sum()),float(A[i]@A[j]/(np.linalg.norm(A[i])*np.linalg.norm(A[j])+1e-9)),int(cdeg[i]),int(cdeg[j]),
                     np.log1p(Cin[i,j]),np.log1p(Cout[i,j]),np.log1p(Cff[i,j]),np.log1p(comm[i,j]),RD[i,j],RD[j,i]])
        # #6 wiring economy
        dd=np.linalg.norm(centroid[neurons[i]]-centroid[neurons[j]]) if (neurons[i] in centroid and neurons[j] in centroid) else np.nan
        wire.append([dd,closeness[i],closeness[j],closeness[i]*closeness[j]])
        # #4 cross-modal (gap junction)
        cross.append([delec[i,j],np.log1p(Ein[i,j]),np.log1p(Ecomm[i,j])])
        # #5 polyadic / co-innervation: shared pre (converge) and shared post beyond pairwise, plus max co-target overlap
        poly.append([np.log1p((C[:,i]*C[:,j]).sum()),np.log1p((C[i]*C[j]).sum())])
I=np.array(I);J=np.array(J);y7=np.array(y7);y8=np.array(y8);N=len(I)
base=np.array(base);wire=np.array(wire);cross=np.array(cross);poly=np.array(poly)
# impute missing distance with median
md=np.nanmedian(wire[:,0]); wire[np.isnan(wire[:,0]),0]=md
disagree=(y7!=y8).astype(int);yrep=((y7==1)&(y8==1)).astype(int);cv=StratifiedKFold(5,shuffle=True,random_state=0)
nf={n:i%5 for i,n in enumerate(sorted(set([neurons[i] for i in I]+[neurons[j] for j in J])))}
def nho(X,y=yrep):
    pred=np.full(N,np.nan)
    for f in range(5):
        te=[k for k in range(N) if nf[neurons[I[k]]]==f and nf[neurons[J[k]]]==f];tr=[k for k in range(N) if nf[neurons[I[k]]]!=f and nf[neurons[J[k]]]!=f]
        if len(te)<20: continue
        pred[np.array(te)]=GradientBoostingClassifier(random_state=0).fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
    m=~np.isnan(pred);return roc_auc_score(y[m],pred[m])
def disp(X):
    y2=np.concatenate([y7,y8]);X2=np.vstack([X,X]);p=np.clip(cross_val_predict(GradientBoostingClassifier(random_state=0),X2,y2,cv=cv,method='predict_proba')[:,1][:N],1e-4,1-1e-4)
    return disagree.mean()/(2*p*(1-p)).mean()
b_disp,b_auc=disp(base),nho(base)
print(f"BEST BASELINE (contact + Cook-HO + RDPG16): disp={b_disp:.2f} AUC={b_auc:.3f}  [n_missing_coords={len(missing)}]")
print("\n#6 GLOBAL OPTIMIZATION / WIRING ECONOMY (3D distance + contact-graph closeness)")
# wiring-economy ALONE (does a global cost predict wiring at all?) then ON TOP of best
print(f"  wiring-economy features ALONE:        disp={disp(wire):.2f} AUC={nho(wire):.3f}")
print(f"  best + wiring-economy:                disp={disp(np.hstack([base,wire])):.2f} AUC={nho(np.hstack([base,wire])):.3f}  (Δauc={nho(np.hstack([base,wire]))-b_auc:+.3f})")
print("\n#4 CROSS-MODAL CO-SPECIFICATION (gap-junction network predicts chemical edges?)")
print(f"  gap-junction features ALONE:          disp={disp(cross):.2f} AUC={nho(cross):.3f}")
print(f"  best + gap-junction:                  disp={disp(np.hstack([base,cross])):.2f} AUC={nho(np.hstack([base,cross])):.3f}  (Δauc={nho(np.hstack([base,cross]))-b_auc:+.3f})")
print("\n#5 POLYADIC / CO-INNERVATION (shared pre/post convergence beyond pairwise)")
print(f"  best + polyad co-innervation:         disp={disp(np.hstack([base,poly])):.2f} AUC={nho(np.hstack([base,poly])):.3f}  (Δauc={nho(np.hstack([base,poly]))-b_auc:+.3f})")
print("\n  ALL THREE stacked on best:            disp={:.2f} AUC={:.3f}".format(disp(np.hstack([base,wire,cross,poly])),nho(np.hstack([base,wire,cross,poly]))))
