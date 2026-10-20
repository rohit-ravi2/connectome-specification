"""BRANCH 28 — is the direction 'address' COMPACT or DIFFUSE? Minimal gene set that reconstructs synapse-direction
prediction at the cross-animal-validated level. Three distinguishable outcomes:
  COMPACT: small SELECTED panel reaches ceiling AND >> random same-size panels (specific genes matter).
  REDUNDANT-DIFFUSE: small panels reach ceiling but SELECTED ~= RANDOM (any genes work; signal smeared everywhere).
  IRREDUCIBLY-DISTRIBUTED: even selected small panels underperform; need hundreds of genes.
Target = DIRECTION (not reconstructing s, which is circular). L1 sparsity path + forward-greedy vs random-panel nulls,
evaluated pair-grouped (Cook) AND cross-animal transfer (Cook->Witvliet, leak-free).
"""
import warnings,sys,openpyxl,csv,re; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, GroupKFold
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
RNG=np.random.default_rng(0); D="../data/celegans/data"
cg=pd.read_csv(f"{D}/expression/cengen/thresholded/021821_liberal_threshold1.csv")
classes=list(cg.columns[3:]); genes_all=cg["gene_name"].astype(str).values
Xf=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values.T>0
br=Xf.mean(0); mask=(br>0.05)&(br<0.95); Xc=Xf[:,mask].astype(float); genes=genes_all[mask]; cls_idx={c:k for k,c in enumerate(classes)}
def n2c(n):
    if not isinstance(n,str): return None
    if n in cls_idx: return n
    b=n
    for s in "LR":
        if b.endswith(s): b=b[:-1]; break
    if b in cls_idx: return b
    for s in "DV":
        if b.endswith(s) and b[:-1] in cls_idx: return b[:-1]
    return None
def cook_chem():
    wb=openpyxl.load_workbook(f"{D}/connectome/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
    A=list(wb['hermaphrodite chemical'].iter_rows(values_only=True))
    def looks(n): return isinstance(n,str) and len(n)>=2 and n[0].isalpha()
    hr=max(range(len(A)),key=lambda r:sum(looks(x) for x in A[r])); cols=[(str(x).strip() if x is not None else None) for x in A[hr]]
    hc=max(range(min(6,len(A[0]))),key=lambda c:sum(looks(A[r][c]) for r in range(len(A))))
    ed=set()
    for r in range(hr+1,len(A)):
        cpre=n2c(A[r][hc])
        if not cpre: continue
        for c in range(hc+1,len(A[r])):
            cpost=n2c(cols[c]); v=A[r][c]
            if cpost and cpost!=cpre and v not in (None,''):
                try:
                    if float(v)>0: ed.add((cpre,cpost))
                except: pass
    return ed
chem=cook_chem(); asym=[(a,b) for (a,b) in chem if (b,a) not in chem]
# Witvliet asym (cross-animal, leak-free)
from collections import Counter
wEd=Counter()
with open(f"{D}/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']=='chemical':
            a=n2c(r['pre']); b=n2c(r['post'])
            if a and b and a!=b: wEd[(a,b)]+=1
wset=set(wEd); wasym=[(a,b) for (a,b) in wset if (b,a) not in wset and a in cls_idx and b in cls_idx]
# antisym design (both orientations), grouped by pair
def design(pairs):
    Fp=[];yp=[];gp=[]
    for k,(a,b) in enumerate(pairs):
        d=Xc[cls_idx[a]]-Xc[cls_idx[b]]; Fp+=[d,-d]; yp+=[1,0]; gp+=[k,k]
    return np.array(Fp),np.array(yp),np.array(gp)
Fc,yc,gc=design(asym); Fw,yw,gw=design(wasym)
P(f"BRANCH 28 — Cook {len(asym)} asym pairs, Witvliet {len(wasym)} | {len(genes)} genes")
gk=GroupKFold(5)
def grouped_auc(F,y,g,model): return roc_auc_score(y,cross_val_predict(model,F,y,cv=gk,groups=g,method='predict_proba')[:,1])
def transfer_auc(cols):
    # fit on ALL Cook (restricted genes), score direction on Witvliet
    m=LogisticRegression(max_iter=2000,C=0.5).fit(Fc[:,cols],yc)
    return roc_auc_score(yw,m.decision_function(Fw[:,cols]))
allcols=np.arange(len(genes))
# CEILING (full model)
ceil_cook=grouped_auc(Fc,yc,gc,LogisticRegression(max_iter=1000,C=0.5))
ceil_tr=transfer_auc(allcols)
P(f"\n  CEILING (all {len(genes)} genes): Cook pair-grouped {ceil_cook:.3f} | Cook->Witvliet transfer {ceil_tr:.3f}")
# ---- L1 sparsity path ----
P(f"\n  L1 SPARSITY PATH (selected genes vs size):")
P(f"    {'C':>7} {'#genes':>7} {'Cook-CV':>8} {'Witvliet':>9}")
path=[]
for C in [0.002,0.004,0.007,0.01,0.02,0.04,0.08,0.15,0.3,0.6]:
    l1=LogisticRegression(penalty='l1',solver='liblinear',C=C,max_iter=2000).fit(Fc,yc)
    sel=np.where(np.abs(l1.coef_[0])>1e-8)[0]
    if len(sel)==0: continue
    cvA=grouped_auc(Fc[:,sel],yc,gc,LogisticRegression(penalty='l1',solver='liblinear',C=C,max_iter=2000))
    trA=transfer_auc(sel)
    path.append((C,len(sel),cvA,trA,sel))
    P(f"    {C:>7} {len(sel):>7} {cvA:>8.3f} {trA:>9.3f}")
# ---- random panels at matched sizes ----
sizes=sorted(set(min(len(genes),s) for s in [3,5,8,12,20,40,80,150,400]))
P(f"\n  RANDOM same-size panels (mean±sd over 40) vs L1-selected at matched size:")
P(f"    {'size':>5} {'rand Cook':>10} {'rand Witv':>10} {'L1 Cook':>8} {'L1 Witv':>8}")
# nearest L1 point per size
def nearest_l1(sz):
    if not path: return None
    return min(path,key=lambda p:abs(p[1]-sz))
rand_curve=[]
for sz in sizes:
    rc=[];rt=[]
    for _ in range(40):
        cols=RNG.choice(allcols,sz,replace=False)
        rc.append(grouped_auc(Fc[:,cols],yc,gc,LogisticRegression(max_iter=1000,C=0.5)))
        rt.append(transfer_auc(cols))
    rc=np.array(rc);rt=np.array(rt); rand_curve.append((sz,rc.mean(),rc.std(),rt.mean(),rt.std()))
    nl=nearest_l1(sz); l1c=nl[2] if nl else float('nan'); l1t=nl[3] if nl else float('nan'); l1n=nl[1] if nl else 0
    P(f"    {sz:>5} {rc.mean():>6.3f}±{rc.std():.2f} {rt.mean():>6.3f}±{rt.std():.2f} {l1c:>8.3f} {l1t:>8.3f}  (L1 n={l1n})")
# ---- forward-greedy on a prefiltered pool (top univariate direction genes) ----
P(f"\n  FORWARD-GREEDY (prefiltered top-300 univariate genes; add to maximize Cook-CV):")
uni=np.array([roc_auc_score(yc,Fc[:,j]) for j in range(len(genes))]); uni=np.abs(uni-0.5)
pool=np.argsort(-uni)[:300]
chosen=[]; rem=list(pool)
for step in range(12):
    best=None
    for j in rem[:120]:  # cap candidates per step for speed
        cols=chosen+[j]
        a=grouped_auc(Fc[:,cols],yc,gc,LogisticRegression(max_iter=800,C=0.5))
        if best is None or a>best[1]: best=(j,a)
    chosen.append(best[0]); rem.remove(best[0])
    trg=transfer_auc(np.array(chosen))
    P(f"    +{genes[best[0]]:<14} n={len(chosen):2d} Cook-CV {best[1]:.3f} Witvliet {trg:.3f}")
# report genes at a compact L1 point (~10-20 genes)
comp=[p for p in path if 8<=p[1]<=25]
if comp:
    pt=comp[0]
    P(f"\n  L1 panel at n={pt[1]} genes ({pt[2]:.3f} Cook / {pt[3]:.3f} Witvliet): {', '.join(genes[i] for i in pt[4][:25])}")
P(f"\n  INTERPRETATION:")
P(f"    - fraction of ceiling reached by ~10-20 selected genes, and whether SELECTED >> RANDOM at that size,")
P(f"      decides COMPACT (selected>>random, reaches ceiling) vs REDUNDANT-DIFFUSE (selected~=random, both reach")
P(f"      ceiling) vs IRREDUCIBLY-DISTRIBUTED (small panels underperform ceiling).")
