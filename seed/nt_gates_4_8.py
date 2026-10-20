"""GATE 4 (contact-area stratification) + GATE 8 (adult-replicate validation) for NT.
G4: where does NT help? Bin contacting pairs by contact area; a real compatibility selector
    should help most in AMBIGUOUS medium-contact pairs (contact permits but doesn't determine).
G8: cleanest validation — train on ONE adult's edges, predict the OTHER adult's edges.
    Does geometry+NT beat geometry at predicting a genuinely independent animal?
"""
import openpyxl, csv, re, math, numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy import sparse

CONTACT="../data/celegans/data/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx"
EDGES="../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv"
T5="../data/tier5/connectome.npz"

wb=openpyxl.load_workbook(CONTACT,read_only=True,data_only=True); ws=wb['adult nerve ring neighbors']
data=list(ws.iter_rows(values_only=True)); neurons=[h for h in data[0][1:]]; idx={n:i for i,n in enumerate(neurons)}; M=len(neurons)
A=np.zeros((M,M))
for row in data[1:]:
    rn=row[0]
    if rn is None or rn not in idx: continue
    for j,val in enumerate(row[1:]):
        if val in (None,''): continue
        try: A[idx[rn],j]=float(val)
        except: pass
A=np.maximum(A,A.T); deg_area=A.sum(1); deg_num=(A>0).sum(1)
in7=defaultdict(int); in8=defaultdict(int)
with open(EDGES) as f:
    for r in csv.DictReader(f):
        ds=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
        if r['type']!='chemical': continue
        pr=(min(r['pre'],r['post']),max(r['pre'],r['post']))
        if ds==7: in7[pr]+=int(r['synapses'])
        elif ds==8: in8[pr]+=int(r['synapses'])
t5=np.load(T5,allow_pickle=True); t5nt={str(n):str(v) for n,v in zip(t5["names"],t5["nt_primary"])}
def bclass(n):
    b=str(n)
    for s in("L","R"):
        if b.endswith(s): b=b[:-1]
    return b
def normnt(x):
    x=str(x)
    if x.startswith("ACh") or x.startswith("Acetyl"): return "ACh"
    if x.startswith("Glu"): return "Glu"
    for key,lab in [("GABA","GABA"),("Dopamine","DA"),("Serot","5HT"),("Octopamine","OA"),("Tyramine","TA")]:
        if key in x: return lab
    return "Unknown"
nt_of=np.array([normnt(t5nt.get(str(neurons[i]),t5nt.get(bclass(neurons[i]),"Unknown"))) for i in range(M)],dtype=object)
cats=sorted(set(nt_of)); cix={c:k for k,c in enumerate(cats)}; K=len(cats)

UI,UJ,geom,area,s7,s8=[],[],[],[],[],[]
for i in range(M):
    for j in range(i+1,M):
        if A[i,j]<=0: continue
        UI.append(i);UJ.append(j); area.append(A[i,j])
        vi,vj=A[i],A[j]; cos=float(vi@vj/(np.linalg.norm(vi)*np.linalg.norm(vj)+1e-9)); sh=int(((A[i]>0)&(A[j]>0)).sum())
        geom.append([math.log1p(A[i,j]),cos,sh,math.log1p(min(deg_area[i],deg_area[j])),
                     math.log1p(max(deg_area[i],deg_area[j])),min(deg_num[i],deg_num[j]),max(deg_num[i],deg_num[j])])
        pr=(min(str(neurons[i]),str(neurons[j])),max(str(neurons[i]),str(neurons[j])))
        s7.append(in7.get(pr,0)); s8.append(in8.get(pr,0))
UI=np.array(UI);UJ=np.array(UJ); G=np.array(geom); G=(G-G.mean(0))/(G.std(0)+1e-9)
area=np.array(area); s7=np.array(s7); s8=np.array(s8); N=len(UI)
def ntpair(labels=nt_of):
    P=sparse.lil_matrix((N,K*K))
    for k in range(N):
        a,b=cix[labels[UI[k]]],cix[labels[UJ[k]]]; lo,hi=min(a,b),max(a,b); P[k,lo*K+hi]=1
    return P.tocsr()
Gs=sparse.csr_matrix(G); GN=sparse.hstack([Gs,ntpair()]).tocsr()
def bits(yy,p): p=np.clip(p,1e-6,1-1e-6); return float(-(yy*np.log2(p)+(1-yy)*np.log2(1-p)).sum())

# ================= GATE 4 — contact-area stratification =================
print("="*90); print("GATE 4 — where does NT help? (OOF preds, stratified by contact-area quintile)"); print("="*90)
y=((s7>=1)&(s8>=1)).astype(int)
cv=StratifiedKFold(5,shuffle=True,random_state=0)
pg=cross_val_predict(LogisticRegression(max_iter=3000),Gs,y,cv=cv,method='predict_proba')[:,1]
pgn=cross_val_predict(LogisticRegression(max_iter=3000),GN,y,cv=cv,method='predict_proba')[:,1]
order=np.argsort(area); nb=5
print(f"  {'area quintile':>26} | {'n':>4} | {'pos rate':>8} | {'geom AUC':>8} | {'+NT AUC':>8} | {'ΔAUC':>7} | {'Δbits':>6}")
for b in range(nb):
    m=order[b*N//nb:(b+1)*N//nb]
    if y[m].sum()<10 or (y[m]==0).sum()<10:
        print(f"  bin {b}: degenerate"); continue
    ag=roc_auc_score(y[m],pg[m]); an=roc_auc_score(y[m],pgn[m])
    print(f"  [{area[m].min():6.0f}-{area[m].max():7.0f}] area | {len(m):>4} | {y[m].mean():>8.3f} | {ag:>8.4f} | {an:>8.4f} | {an-ag:>+7.4f} | {bits(y[m],pg[m])-bits(y[m],pgn[m]):>+6.0f}")
print("  prediction: a real selector helps most in AMBIGUOUS (mid) contact bins, not just extremes.")

# ================= GATE 8 — adult-replicate validation =================
print("\n"+"="*90); print("GATE 8 — train on one adult, predict the OTHER (cleanest; labels from different animals)"); print("="*90)
def crossrep(train_y, test_y, tag):
    # train geometry / geometry+NT on train_y, eval AUC/bits on test_y (all pairs; no CV needed, different label)
    out={}
    for lab,X in [("geometry",Gs),("geometry+NT",GN)]:
        m=LogisticRegression(max_iter=3000,C=1.0).fit(X,train_y); p=m.predict_proba(X)[:,1]
        out[lab]=(roc_auc_score(test_y,p),bits(test_y,p))
    dA=out["geometry+NT"][0]-out["geometry"][0]
    print(f"  {tag}: geom AUC {out['geometry'][0]:.4f} -> +NT {out['geometry+NT'][0]:.4f}  ΔAUC {dA:+.4f}  "
          f"Δbits {out['geometry'][1]-out['geometry+NT'][1]:+.0f}")
    return dA
y7=(s7>=1).astype(int); y8=(s8>=1).astype(int)
d78=crossrep(y7,y8,"train adult-7 -> test adult-8")
d87=crossrep(y8,y7,"train adult-8 -> test adult-7")
print(f"\n  mean cross-replicate ΔAUC(NT) = {(d78+d87)/2:+.4f}")
print("  => NT improving prediction of an INDEPENDENT animal's wiring is the cleanest validation.")
