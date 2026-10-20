"""AUDIT part 3 — close it out: (9) is the small NT residual Unknown-driven / one-group?
inspect coefficients + drop-Unknown. (5) directedness: source vs target NT."""
import openpyxl, csv, re, math, numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy import sparse
DIR="../data/celegans/data/connectome"; T5="../data/tier5/connectome.npz"
wb=openpyxl.load_workbook(f"{DIR}/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx",read_only=True,data_only=True); ws=wb['adult nerve ring neighbors']
data=list(ws.iter_rows(values_only=True)); neurons=[h for h in data[0][1:]]; idx={n:i for i,n in enumerate(neurons) if isinstance(n,str)}; M=len(neurons)
A=np.zeros((M,M))
for row in data[1:]:
    rn=row[0]
    if rn not in idx: continue
    for j,val in enumerate(row[1:]):
        if val not in (None,''):
            try: A[idx[rn],j]=float(val)
            except: pass
A=np.maximum(A,A.T); deg_area=A.sum(1); deg_num=(A>0).sum(1)
pres=defaultdict(set); dchem=defaultdict(set)
with open(f"{DIR}/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        ds=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
        if ds not in (7,8): continue
        pres[(min(r['pre'],r['post']),max(r['pre'],r['post']))].add(ds)
        if r['type']=='chemical': dchem[(r['pre'],r['post'])].add(ds)
t5=np.load(T5,allow_pickle=True); t5nt={str(n):str(v) for n,v in zip(t5["names"],t5["nt_primary"])}
def bc(n):
    b=str(n)
    for s in("L","R"):
        if b.endswith(s): b=b[:-1]
    return b
nt_of=[t5nt.get(neurons[i], t5nt.get(bc(neurons[i]),"Unknown")) if isinstance(neurons[i],str) else "Unknown" for i in range(M)]

def geomfeat(i,j):
    vi,vj=A[i],A[j]; cos=float(vi@vj/(np.linalg.norm(vi)*np.linalg.norm(vj)+1e-9)); shared=int(((A[i]>0)&(A[j]>0)).sum())
    return [math.log1p(A[i,j]),cos,shared,math.log1p(min(deg_area[i],deg_area[j])),math.log1p(max(deg_area[i],deg_area[j])),min(deg_num[i],deg_num[j]),max(deg_num[i],deg_num[j])]
# undirected universe
I,J,y,geom=[],[],[],[]
for i in range(M):
    for j in range(i+1,M):
        if A[i,j]<=0 or not isinstance(neurons[i],str) or not isinstance(neurons[j],str): continue
        s=pres.get((min(neurons[i],neurons[j]),max(neurons[i],neurons[j])),set())
        I.append(i);J.append(j);y.append(1 if s=={7,8} else 0);geom.append(geomfeat(i,j))
I=np.array(I);J=np.array(J);y=np.array(y,float);G=np.array(geom);Gs=(G-G.mean(0))/(G.std(0)+1e-9);N=len(y)
folds=list(StratifiedKFold(5,shuffle=True,random_state=0).split(np.zeros(N),y))
cats=sorted(set(nt_of));cix={c:k for k,c in enumerate(cats)};K=len(cats)
def ntpair(lab):
    P=sparse.lil_matrix((N,K*K))
    for k in range(N):
        a,b=cix[lab[I[k]]],cix[lab[J[k]]];lo,hi=min(a,b),max(a,b);P[k,lo*K+hi]=1
    return P.tocsr()
Gsp=sparse.csr_matrix(Gs)
au=lambda X: roc_auc_score(y,cross_val_predict(LogisticRegression(max_iter=2000),X,y,cv=folds,method='predict_proba')[:,1])
a_g=au(Gsp); a_gn=au(sparse.hstack([Gsp,ntpair(nt_of)]).tocsr())
print("="*90); print("(9a) DROP-UNKNOWN: does the NT lift survive removing Unknown-NT pairs?"); print("="*90)
mask=np.array([nt_of[I[k]]!="Unknown" and nt_of[J[k]]!="Unknown" for k in range(N)])
print(f"  full: geom {a_g:.4f} -> geom+NT {a_gn:.4f}  lift {a_gn-a_g:+.4f}   (N={N})")
ym=y[mask]; Gm=sparse.csr_matrix(Gs[mask]); NTm=ntpair(nt_of)[mask]
fm=list(StratifiedKFold(5,shuffle=True,random_state=0).split(np.zeros(mask.sum()),ym))
aum=lambda X: roc_auc_score(ym,cross_val_predict(LogisticRegression(max_iter=2000),X,ym,cv=fm,method='predict_proba')[:,1])
print(f"  drop-Unknown: geom {aum(Gm):.4f} -> geom+NT {aum(sparse.hstack([Gm,NTm]).tocsr()):.4f}  "
      f"lift {aum(sparse.hstack([Gm,NTm]).tocsr())-aum(Gm):+.4f}   (N={mask.sum()}, {100*mask.mean():.0f}% of pairs)")
print("\n"+"="*90); print("(9b) NT-PAIR COEFFICIENTS (LR on geom+NT, full fit) — which pairs drive it?"); print("="*90)
lr=LogisticRegression(max_iter=3000).fit(sparse.hstack([Gsp,ntpair(nt_of)]).tocsr(),y)
coef=lr.coef_[0][Gs.shape[1]:]  # NT-pair coefficients
order=np.argsort(coef)
def pairname(idx0):
    lo,hi=divmod(idx0,K); return f"{cats[lo][:14]} x {cats[hi][:14]}"
print("  top +NT-pair (favor connection):")
for o in order[::-1][:6]:
    if coef[o]!=0: print(f"    {coef[o]:+.2f}  {pairname(o)}")
print("  top -NT-pair (disfavor):")
for o in order[:6]:
    if coef[o]!=0: print(f"    {coef[o]:+.2f}  {pairname(o)}")
unk=sum(1 for o in np.argsort(-np.abs(coef))[:10] if 'Unknown' in pairname(o))
print(f"  Unknown-involving among top-10 |coef|: {unk}/10")
print("\n"+"="*90); print("(5) DIRECTEDNESS: source-NT vs target-NT (directed chemical synapses)"); print("="*90)
Id,Jd,yd,gd,sc,tg=[],[],[],[],[],[]
for i in range(M):
    for j in range(M):
        if i==j or A[i,j]<=0 or not isinstance(neurons[i],str) or not isinstance(neurons[j],str): continue
        s=dchem.get((neurons[i],neurons[j]),set())
        Id.append(i);Jd.append(j);yd.append(1 if s=={7,8} else 0);gd.append(geomfeat(i,j));sc.append(cix[nt_of[i]]);tg.append(cix[nt_of[j]])
yd=np.array(yd,float);Gd=np.array(gd);Gd=(Gd-Gd.mean(0))/(Gd.std(0)+1e-9);Nd=len(yd);sc=np.array(sc);tg=np.array(tg)
fd=list(StratifiedKFold(5,shuffle=True,random_state=0).split(np.zeros(Nd),yd))
Gdp=sparse.csr_matrix(Gd)
def oneh(v,Kk):
    P=sparse.lil_matrix((len(v),Kk))
    for k,x in enumerate(v):P[k,x]=1
    return P.tocsr()
def pair_st(a,b):
    P=sparse.lil_matrix((len(a),K*K))
    for k in range(len(a)):P[k,a[k]*K+b[k]]=1
    return P.tocsr()
aud=lambda X: roc_auc_score(yd,cross_val_predict(LogisticRegression(max_iter=2000),X,yd,cv=fd,method='predict_proba')[:,1])
base=aud(Gdp)
print(f"  directed universe Nd={Nd}, reproducible {int(yd.sum())} ({100*yd.mean():.1f}%)")
print(f"  geometry                    {base:.4f}")
print(f"  + source-NT only            {aud(sparse.hstack([Gdp,oneh(sc,K)]).tocsr()):.4f}   (lift {aud(sparse.hstack([Gdp,oneh(sc,K)]).tocsr())-base:+.4f})")
print(f"  + target-NT only            {aud(sparse.hstack([Gdp,oneh(tg,K)]).tocsr()):.4f}   (lift {aud(sparse.hstack([Gdp,oneh(tg,K)]).tocsr())-base:+.4f})")
print(f"  + source+target additive    {aud(sparse.hstack([Gdp,oneh(sc,K),oneh(tg,K)]).tocsr()):.4f}")
print(f"  + source x target pair      {aud(sparse.hstack([Gdp,pair_st(sc,tg)]).tocsr()):.4f}   (lift {aud(sparse.hstack([Gdp,pair_st(sc,tg)]).tocsr())-base:+.4f})")
print("="*90)
