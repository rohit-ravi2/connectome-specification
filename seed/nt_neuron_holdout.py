"""GATE 1 (audit) + GATE 2 (neuron-holdout) for the NT-class-pair signal.
Decisive leakage test: does geometry+NT-pair beat geometry when tested on neurons whose
pairs were NEVER in training? If the +0.0147 edge-CV lift collapses under node-holdout,
it was leakage -> CLOSE the branch. Full calibrated metrics (AUC, bits, Brier, ECE) on
matched folds; NT-label permutation under the SAME splits.
"""
import openpyxl, csv, re, math, numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy import sparse

CONTACT="../data/celegans/data/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx"
EDGES="../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv"
T5="../data/tier5/connectome.npz"
RNG=np.random.default_rng(0)

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
present=defaultdict(set)
with open(EDGES) as f:
    for r in csv.DictReader(f):
        ds=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
        if ds not in (7,8): continue
        pre,post=r['pre'],r['post']; present[(min(pre,post),max(pre,post))].add(ds)
t5=np.load(T5,allow_pickle=True); t5nt={str(n):str(v) for n,v in zip(t5["names"],t5["nt_primary"])}
def bclass(n):
    b=str(n)
    for s in("L","R"):
        if b.endswith(s): b=b[:-1]
    return b
nt_of=np.array([t5nt.get(str(neurons[i]),t5nt.get(bclass(neurons[i]),"Unknown")) for i in range(M)],dtype=object)

I,J,y,geom=[],[],[],[]
for i in range(M):
    for j in range(i+1,M):
        if A[i,j]<=0: continue
        pr=(min(str(neurons[i]),str(neurons[j])),max(str(neurons[i]),str(neurons[j])))
        I.append(i);J.append(j); y.append(1 if present.get(pr,set())=={7,8} else 0)
        vi,vj=A[i],A[j]; cos=float(vi@vj/(np.linalg.norm(vi)*np.linalg.norm(vj)+1e-9)); sh=int(((A[i]>0)&(A[j]>0)).sum())
        geom.append([math.log1p(A[i,j]),cos,sh,math.log1p(min(deg_area[i],deg_area[j])),
                     math.log1p(max(deg_area[i],deg_area[j])),min(deg_num[i],deg_num[j]),max(deg_num[i],deg_num[j])])
I=np.array(I);J=np.array(J);y=np.array(y);G=np.array(geom); G=(G-G.mean(0))/(G.std(0)+1e-9); N=len(y)
cats=sorted(set(nt_of)); cix={c:k for k,c in enumerate(cats)}; K=len(cats)
def ntpair(labels):
    P=sparse.lil_matrix((N,K*K))
    for k in range(N):
        a,b=cix[labels[I[k]]],cix[labels[J[k]]]; lo,hi=min(a,b),max(a,b); P[k,lo*K+hi]=1
    return P.tocsr()
Gs=sparse.csr_matrix(G)
def bits(yy,p): p=np.clip(p,1e-6,1-1e-6); return float(-(yy*np.log2(p)+(1-yy)*np.log2(1-p)).sum())
def ece(yy,p,nb=10):
    e=0.0
    for b in range(nb):
        m=(p>=b/nb)&(p<(b+1)/nb)
        if m.sum(): e+=abs(yy[m].mean()-p[m].mean())*m.sum()
    return e/len(yy)

print(f"N contacting pairs={N} | reproducible-core positives={int(y.sum())} ({y.mean():.3f}) | NT classes {K}: {cats}")
cnt={c:int((nt_of==c).sum()) for c in cats}; print("  NT neuron counts:",cnt)

# ================= GATE 1 — edge-CV audit (matched folds, full metrics) =================
print("\n"+"="*88); print("GATE 1 — edge-CV audit (5-fold, matched-fold metrics)"); print("="*88)
cv=StratifiedKFold(5,shuffle=True,random_state=0)
def oof(X):
    return cross_val_predict(LogisticRegression(max_iter=3000,C=1.0),X,y,cv=cv,method='predict_proba')[:,1]
pg=oof(Gs); pgn=oof(sparse.hstack([Gs,ntpair(nt_of)]).tocsr())
for lab,p in [("geometry",pg),("geometry+NT-pair",pgn)]:
    print(f"  {lab:18s} AUC={roc_auc_score(y,p):.4f}  bits={bits(y,p):7.0f}  Brier={brier_score_loss(y,p):.4f}  ECE={ece(y,p):.4f}")
dA=roc_auc_score(y,pgn)-roc_auc_score(y,pg); dB=bits(y,pg)-bits(y,pgn)
perm=[]
for s in range(20):
    pl=nt_of[RNG.permutation(M)]
    perm.append(roc_auc_score(y,oof(sparse.hstack([Gs,ntpair(pl)]).tocsr()))-roc_auc_score(y,pg))
perm=np.array(perm)
print(f"  ΔAUC(NT)= {dA:+.4f} | Δbits(NT)= {dB:+.0f} | perm ΔAUC {perm.mean():+.4f}±{perm.std():.4f}  z={(dA-perm.mean())/(perm.std()+1e-9):+.1f}")

# ================= GATE 2 — NEURON-HOLDOUT (decisive) =================
print("\n"+"="*88); print("GATE 2 — NEURON-HOLDOUT (10 random 80/20 node splits)"); print("="*88)
present_nodes=sorted(set(I)|set(J));
def fit_eval(train_mask,test_mask,Xextra=None):
    Xtr=Gs if Xextra is None else sparse.hstack([Gs,Xextra]).tocsr()
    m=LogisticRegression(max_iter=3000,C=1.0); m.fit(Xtr[train_mask],y[train_mask])
    return m.predict_proba(Xtr[test_mask])[:,1]
def run_regime(name, both):
    # pooled predictions across splits
    yg,pg_all,pgn_all,pperm_all=[],[],[],[]
    dAs=[]; permdAs=[]
    for s in range(10):
        nodes=np.array(present_nodes); RNG.shuffle(nodes)
        test_nodes=set(nodes[:len(nodes)//5]);
        inI=np.array([i in test_nodes for i in I]); inJ=np.array([j in test_nodes for j in J])
        if both:  # BLOCK: both endpoints held out; train both endpoints retained
            test=inI&inJ; train=(~inI)&(~inJ)
        else:     # INCIDENT: >=1 endpoint held out; train neither endpoint held out
            test=inI^inJ; train=(~inI)&(~inJ)
        if test.sum()<20 or y[test].sum()<5 or y[train].sum()<20: continue
        NTf=ntpair(nt_of)
        pg_=fit_eval(train,test); pgn_=fit_eval(train,test,NTf)
        yg.append(y[test]); pg_all.append(pg_); pgn_all.append(pgn_)
        dAs.append(roc_auc_score(y[test],pgn_)-roc_auc_score(y[test],pg_))
        pl=nt_of[RNG.permutation(M)]
        pperm=fit_eval(train,test,ntpair(pl)); permdAs.append(roc_auc_score(y[test],pperm)-roc_auc_score(y[test],pg_))
    yg=np.concatenate(yg); pg_all=np.concatenate(pg_all); pgn_all=np.concatenate(pgn_all)
    print(f"\n  [{name}]  pooled test pairs={len(yg)}  positives={int(yg.sum())} ({yg.mean():.3f})  ({len(dAs)} usable splits)")
    print(f"    geometry        AUC={roc_auc_score(yg,pg_all):.4f}  bits={bits(yg,pg_all):.0f}  Brier={brier_score_loss(yg,pg_all):.4f}")
    print(f"    geometry+NT     AUC={roc_auc_score(yg,pgn_all):.4f}  bits={bits(yg,pgn_all):.0f}  Brier={brier_score_loss(yg,pgn_all):.4f}")
    dAs=np.array(dAs); permdAs=np.array(permdAs)
    print(f"    ΔAUC(NT) per-split = {dAs.mean():+.4f} ± {dAs.std():.4f}  |  permuted-NT ΔAUC = {permdAs.mean():+.4f} ± {permdAs.std():.4f}")
    print(f"    pooled ΔAUC = {roc_auc_score(yg,pgn_all)-roc_auc_score(yg,pg_all):+.4f} | pooled Δbits = {bits(yg,pg_all)-bits(yg,pgn_all):+.0f}")
    real_beats_perm = dAs.mean() > permdAs.mean()+ permdAs.std()
    survives = (dAs.mean()>0.003) and real_beats_perm and (roc_auc_score(yg,pgn_all)>roc_auc_score(yg,pg_all))
    return survives, dAs.mean()
s_inc,d_inc=run_regime("INCIDENT: >=1 endpoint novel", both=False)
s_blk,d_blk=run_regime("BLOCK: both endpoints novel", both=True)

print("\n"+"="*88)
if s_inc or s_blk:
    print(f"GATE 2 VERDICT: NT SURVIVES neuron-holdout (incident Δ={d_inc:+.4f} survive={s_inc} | block Δ={d_blk:+.4f} survive={s_blk})")
    print("  -> generalizing biological compatibility signal; branch stays ALIVE for gates 3-9.")
else:
    print(f"GATE 2 VERDICT: NT COLLAPSES under neuron-holdout (incident Δ={d_inc:+.4f} | block Δ={d_blk:+.4f})")
    print("  -> the edge-CV lift was neuron/edge leakage; CLOSE the NT branch.")
print("="*88)
