"""DECISIVE: is the post-geometry residual a low-rank source-target COMPATIBILITY code,
or is it just cross-animal NOISE?  Two things the earlier scripts lacked:

  (1) low-rank latent factor fit ON TOP OF geometry:
        logit(i,j) = w.geom(i,j) + a_i + a_j + <U_i,U_j>,  sweep rank d
      -> does pairwise compatibility reduce held-out bits BELOW geometry+identity (2861)?
      (prints ||U|| to confirm the latent actually trains, not a vacuous optimizer.)

  (2) the NOISE CEILING (adult replicate validation, the reframe):
      reproducibility is defined by only 2 animals (adults 7,8). The best ANY mechanistic
      model can do is bounded by how well one animal's wiring predicts the other's.
      If geometry already approaches that ceiling, the 'residual' is animal-to-animal
      NOISE, not a missing address code. Compute cross-replicate AUC/bits and compare.

Metric = held-out log-loss (bits) + AUC. Entropy-accounting note: residual conditional
entropy is NOT 'bits the code must supply' — do not divide by neurons and compare to
log2(classes); that was the earlier error.
"""
import openpyxl, csv, re, math, numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

CONTACT="../data/celegans/data/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx"
EDGES="../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv"
RNG=np.random.default_rng(0)

# ---- contact ----
wb=openpyxl.load_workbook(CONTACT,read_only=True,data_only=True); ws=wb['adult nerve ring neighbors']
data=list(ws.iter_rows(values_only=True)); neurons=[h for h in data[0][1:]]; idx={n:i for i,n in enumerate(neurons)}; M=len(neurons)
A=np.zeros((M,M))
for row in data[1:]:
    rn=row[0]
    if rn is None or rn not in idx: continue
    for j,val in enumerate(row[1:]):
        if val in (None,''): continue
        try: A[idx[rn],j]=float(val)
        except (TypeError,ValueError): pass
A=np.maximum(A,A.T); deg_area=A.sum(1); deg_num=(A>0).sum(1)

# ---- per-adult undirected presence among contacting pairs ----
present=defaultdict(set)   # undirected pair -> set of adult datasets {7,8} present in
with open(EDGES) as f:
    for r in csv.DictReader(f):
        ds=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
        if ds not in (7,8): continue
        pre,post=r['pre'],r['post']; present[(min(pre,post),max(pre,post))].add(ds)

I,J,y,y7,y8,geom=[],[],[],[],[],[]
for i in range(M):
    for j in range(i+1,M):
        if A[i,j]<=0: continue
        pr=(min(neurons[i],neurons[j]),max(neurons[i],neurons[j])); s=present.get(pr,set())
        I.append(i);J.append(j)
        y.append(1 if s=={7,8} else 0)          # reproducible core (both)
        y7.append(1 if 7 in s else 0); y8.append(1 if 8 in s else 0)
        vi,vj=A[i],A[j]; cos=float(vi@vj/(np.linalg.norm(vi)*np.linalg.norm(vj)+1e-9)); shared=int(((A[i]>0)&(A[j]>0)).sum())
        geom.append([math.log1p(A[i,j]),cos,shared,math.log1p(min(deg_area[i],deg_area[j])),
                     math.log1p(max(deg_area[i],deg_area[j])),min(deg_num[i],deg_num[j]),max(deg_num[i],deg_num[j])])
I=np.array(I);J=np.array(J);y=np.array(y,float);y7=np.array(y7);y8=np.array(y8)
G=np.array(geom); G=(G-G.mean(0))/(G.std(0)+1e-9); N=len(y); F=G.shape[1]
pbar=y.mean(); sig=lambda z:1/(1+np.exp(-np.clip(z,-30,30)))
def bits(yy,p): p=np.clip(p,1e-6,1-1e-6); return -(yy*np.log2(p)+(1-yy)*np.log2(1-p)).sum()
base=bits(y,np.full(N,pbar))
print(f"N={N} contacting pairs | reproducible-core {int(y.sum())} ({100*pbar:.1f}%) | base H={base:.0f} bits")

# ======================= (2) NOISE CEILING — adult replicate validation =======================
print("\n"+"="*84); print("NOISE CEILING (adult-7 <-> adult-8 replicate agreement)"); print("="*84)
# how well does one animal's presence predict the other's?
def cross(ya,yb,lab):
    # P(yb=1|ya) two-bin predictor
    p1=yb[ya==1].mean() if (ya==1).any() else 0; p0=yb[ya==0].mean() if (ya==0).any() else 0
    pred=np.where(ya==1,p1,p0);
    return roc_auc_score(yb,pred), bits(yb,pred), p1,p0
a78,b78,p1,p0=cross(y7,y8,"7->8"); a87,b87,_,_=cross(y8,y7,"8->7")
print(f"  P(adult8 edge | adult7 edge)={p1:.3f}   P(adult8 edge | adult7 no-edge)={p0:.3f}")
print(f"  CEILING AUC (one adult predicts the other): 7->8 {a78:.3f} | 8->7 {a87:.3f}  (~max any model can reach)")
# geometry trained to predict a FRESH animal (adult8) using adult7-defined... use geometry to predict y8
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
pg8=cross_val_predict(GradientBoostingClassifier(random_state=0),G,y8,cv=StratifiedKFold(5,shuffle=True,random_state=0),method='predict_proba')[:,1]
print(f"  geometry predicting a single fresh animal (adult8): AUC {roc_auc_score(y8,pg8):.3f}")
print("  => if geometry AUC ~ ceiling AUC, the reproducible-core 'residual' is animal-to-animal NOISE.")

# ======================= (1) geometry + identity + LOW-RANK compatibility =======================
print("\n"+"="*84); print("GEOMETRY + IDENTITY + LOW-RANK COMPATIBILITY (held-out bits, AUC)"); print("="*84)
def fit(tr,te,d,use_id,steps=4000,lr=0.05,lam=1e-2):
    Itr,Jtr,ytr,Gtr=I[tr],J[tr],y[tr],G[tr]
    w=np.zeros(F); b=math.log(ytr.mean()/(1-ytr.mean()))
    a=np.zeros(M); U=RNG.normal(0,0.1,(M,d)) if d>0 else None
    mw=np.zeros(F);vw=np.zeros(F);ma=np.zeros(M);va=np.zeros(M);mb=vb=0.0
    if d>0: mU=np.zeros((M,d));vU=np.zeros((M,d))
    def adam(p,g,m,v,t):
        m=0.9*m+0.1*g; v=0.999*v+0.001*g*g
        return p-lr*(m/(1-0.9**t))/(np.sqrt(v/(1-0.999**t))+1e-8),m,v
    for t in range(1,steps+1):
        lin=b+Gtr@w+(a[Itr]+a[Jtr] if use_id else 0)+((U[Itr]*U[Jtr]).sum(1) if d>0 else 0)
        g=sig(lin)-ytr
        gw=Gtr.T@g/len(g)+lam*w; w,mw,vw=adam(w,gw,mw,vw,t)
        b-=lr*g.mean()
        if use_id:
            ga=np.zeros(M); np.add.at(ga,Itr,g); np.add.at(ga,Jtr,g); ga=ga/len(g)+lam*a
            a,ma,va=adam(a,ga,ma,va,t)
        if d>0:
            gU=np.zeros((M,d)); np.add.at(gU,Itr,g[:,None]*U[Jtr]); np.add.at(gU,Jtr,g[:,None]*U[Itr]); gU=gU/len(g)+lam*U
            U,mU,vU=adam(U,gU,mU,vU,t)
    Ite,Jte=I[te],J[te]
    lin=b+G[te]@w+(a[Ite]+a[Jte] if use_id else 0)+((U[Ite]*U[Jte]).sum(1) if d>0 else 0)
    return sig(lin), (np.abs(U).mean() if d>0 else 0.0)
cv=StratifiedKFold(5,shuffle=True,random_state=0)
def evalmodel(d,use_id,label):
    preds=np.zeros(N); unorm=[]
    for tr,te in cv.split(I,y):
        p,un=fit(tr,te,d,use_id); preds[te]=p; unorm.append(un)
    print(f"  {label:34s} bits={bits(y,preds):7.0f}  AUC={roc_auc_score(y,preds):.4f}"+(f"  <|U|>={np.mean(unorm):.3f}" if d>0 else ""))
    return bits(y,preds),roc_auc_score(y,preds)
bg,ag=evalmodel(0,False,"geometry only")
bgi,agi=evalmodel(0,True,"geometry + identity")
best=bgi
for d in [2,4,8,16]:
    bd,ad=evalmodel(d,True,f"geometry + identity + rank-{d}")
    best=min(best,bd)

print("\n"+"="*84)
lift_bits=bgi-best; lift_frac=lift_bits/base
if lift_bits>=0.03*base:
    v="LOW-RANK COMPATIBILITY REAL — pairwise code compresses the residual beyond geometry+identity"
else:
    v="NO PAIRWISE CODE — low-rank compatibility adds ~nothing over geometry+identity (decision-rule 5)"
print("VERDICT:", v)
print(f"  geometry {bg:.0f}b/AUC{ag:.3f} -> +identity {bgi:.0f}b/AUC{agi:.3f} -> best low-rank {best:.0f}b "
      f"(lift {lift_bits:.0f}b = {100*lift_frac:.1f}% of base)")
print(f"  ceiling AUC (replicate) ~{max(a78,a87):.3f} vs geometry-on-fresh-animal {roc_auc_score(y8,pg8):.3f}")
print("="*84)
