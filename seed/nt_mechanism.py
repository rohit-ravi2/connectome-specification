"""MECHANISM: (1) which source-NT classes drive the selector, (2) does the residual
source×target pairwise part map to NT->receptor logic (target expresses the cognate receptor)?

Directed chem reproducible target (both adults). Geometry residual r = y - p_geom(OOF).
PART 1: mean geometry-residual by SOURCE-NT class + permutation z; leave-one-class-out AUC.
PART 2a: source×target NT residual matrix (which NT->NT are over/under-represented beyond geometry).
PART 2b: cognate-receptor test -- build target-side cognate receptor expression for the SOURCE's NT
         (ACh->nAChR, Glu->iGluR+GluCl, GABA->GABAR from CeNGEN). Does it (i) add over geometry and
         (ii) ABSORB the src×tgt-NT pairwise lift (=> pairwise part IS receptor logic)?
"""
import openpyxl, csv, re, math, numpy as np, pandas as pd
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy import sparse
RNG=np.random.default_rng(0)
CONTACT="../data/celegans/data/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx"
EDGES="../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv"
T5="../data/tier5/connectome.npz"
CG="../data/celegans/data/expression/cengen/thresholded/021821_liberal_threshold1.csv"

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
chemdir=defaultdict(set)
with open(EDGES) as f:
    for r in csv.DictReader(f):
        ds=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
        if r['type']=='chemical': chemdir[(r['pre'],r['post'])].add(ds)
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

# CeNGEN receptor scores per neuron (target-side)
cg=pd.read_csv(CG); classes=list(cg.columns[3:]); Xt=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values
gname=cg['gene_name'].astype(str).str.lower().values; cls_idx={c:k for k,c in enumerate(classes)}
def n2c(n):
    n=str(n)
    if n in cls_idx: return cls_idx[n]
    b=n
    for s in("L","R"):
        if b.endswith(s): b=b[:-1]
    if b in cls_idx: return cls_idx[b]
    for s in("D","V"):
        if b.endswith(s) and b[:-1] in cls_idx: return cls_idx[b[:-1]]
    return cls_idx.get(b,None)
def recscore(genes):
    rows=[np.where(gname==g)[0] for g in genes]; rows=[r[0] for r in rows if len(r)]
    s=np.log1p(Xt[rows].sum(0)) if rows else np.zeros(len(classes))  # summed then log
    out=np.zeros(M)
    for i in range(M):
        c=n2c(neurons[i]);
        if c is not None: out[i]=s[c]
    return out
nAChR=recscore(["unc-29","unc-38","unc-63","acr-2","acr-3","acr-5","acr-8","acr-12","acr-16","lev-1","lev-8","acr-23","deg-3","des-2"])
gluR =recscore(["glr-1","glr-2","glr-3","glr-4","glr-5","glr-6","glr-7","glr-8","nmr-1","nmr-2","avr-14","avr-15","glc-1","glc-2","glc-3","glc-4"])
gabaR=recscore(["unc-49","lgc-35","lgc-36","lgc-37","lgc-38","gab-1","exp-1"])
COG={"ACh":nAChR,"Glu":gluR,"GABA":gabaR}

# directed chem candidate table
DI,DJ,dg,dy=[],[],[],[]
for i in range(M):
    for j in range(M):
        if i==j or A[i,j]<=0: continue
        DI.append(i);DJ.append(j)
        vi,vj=A[i],A[j]; cos=float(vi@vj/(np.linalg.norm(vi)*np.linalg.norm(vj)+1e-9)); sh=int(((A[i]>0)&(A[j]>0)).sum())
        dg.append([math.log1p(A[i,j]),cos,sh,math.log1p(deg_area[i]),math.log1p(deg_area[j]),deg_num[i],deg_num[j]])
        s=chemdir.get((str(neurons[i]),str(neurons[j])),set()); dy.append(1 if (7 in s and 8 in s) else 0)
DI=np.array(DI);DJ=np.array(DJ); DG=np.array(dg); DG=(DG-DG.mean(0))/(DG.std(0)+1e-9); dy=np.array(dy); Nd=len(DI)
snt=nt_of[DI]; tnt=nt_of[DJ]
cv=StratifiedKFold(5,shuffle=True,random_state=0)
p_geom=cross_val_predict(LogisticRegression(max_iter=3000),sparse.csr_matrix(DG),dy,cv=cv,method='predict_proba')[:,1]
resid=dy-p_geom
def bits(yy,p): p=np.clip(p,1e-6,1-1e-6); return float(-(yy*np.log2(p)+(1-yy)*np.log2(1-p)).sum())

print("="*90); print("PART 1 — which SOURCE-NT class drives the selector (mean geometry-residual, perm z)"); print("="*90)
print(f"  directed reproducible chem {int(dy.sum())}/{Nd} ({dy.mean():.3f}) | overall mean-resid ~0")
print(f"  {'src-NT':>8} | {'n edges':>7} | {'mean resid':>10} | {'perm z':>7}  (resid>0 => wires MORE than contact predicts)")
for c in ["ACh","Glu","GABA","DA","5HT","Unknown"]:
    m=snt==c
    if m.sum()<20: continue
    obs=resid[m].mean()
    null=np.array([resid[RNG.permutation(Nd)][:m.sum()].mean() for _ in range(2000)])
    z=(obs-null.mean())/(null.std()+1e-9)
    print(f"  {c:>8} | {int(m.sum()):>7} | {obs:>+10.4f} | {z:>+7.1f}")
# leave-one-class-out AUC of geometry+source-NT
ntc=sorted(set(nt_of)); ntx={c:k for k,c in enumerate(ntc)}
def srcNT_oh(collapse=None):
    labs=[('other' if (collapse and nt_of[i]==collapse) else nt_of[i]) for i in DI]
    u=sorted(set(labs)); ui={c:k for k,c in enumerate(u)}
    P=sparse.lil_matrix((Nd,len(u)))
    for k,l in enumerate(labs): P[k,ui[l]]=1
    return P.tocsr()
def aucX(X): return roc_auc_score(dy,cross_val_predict(LogisticRegression(max_iter=3000),X.tocsr(),dy,cv=cv,method='predict_proba')[:,1])
full=aucX(sparse.hstack([sparse.csr_matrix(DG),srcNT_oh()]))
print(f"\n  geometry+source-NT AUC={full:.4f}; AUC drop when a class is merged into 'other':")
for c in ["ACh","Glu","GABA"]:
    print(f"    drop removing {c}: {full-aucX(sparse.hstack([sparse.csr_matrix(DG),srcNT_oh(collapse=c)])):+.4f}")

print("\n"+"="*90); print("PART 2a — source×target NT residual matrix (mean geometry-residual, powered classes)"); print("="*90)
big=["ACh","Glu","GABA"]
print(f"  rows=source NT, cols=target NT ; value = mean(y - p_geom)*100 (n in parens)")
hdr="  src\\tgt |"+"".join(f"{c:>12}" for c in big); print(hdr)
for sc in big:
    line=f"  {sc:>7} |"
    for tc in big:
        m=(snt==sc)&(tnt==tc)
        line+= (f"{resid[m].mean()*100:>7.2f}({m.sum():>3})" if m.sum()>=10 else f"{'--':>12}")
    print(line)

print("\n"+"="*90); print("PART 2b — does TARGET COGNATE-RECEPTOR expression (NT->receptor logic) absorb the pairwise NT part?"); print("="*90)
# cognate receptor score per edge = target expresses receptor cognate to SOURCE nt
cog=np.array([COG.get(nt_of[DI[k]],np.zeros(M))[DJ[k]] for k in range(Nd)])
cog=((cog-cog.mean())/(cog.std()+1e-9)).reshape(-1,1)
srcNTf=srcNT_oh(); tgtlabs=[nt_of[j] for j in DJ]; tui={c:k for k,c in enumerate(sorted(set(tgtlabs)))}
tgtNTf=sparse.lil_matrix((Nd,len(tui)));
for k,l in enumerate(tgtlabs): tgtNTf[k,tui[l]]=1
tgtNTf=tgtNTf.tocsr()
pairNT=sparse.lil_matrix((Nd,len(ntx)*len(ntx)))
for k in range(Nd): pairNT[k,ntx[nt_of[DI[k]]]*len(ntx)+ntx[nt_of[DJ[k]]]]=1
pairNT=pairNT.tocsr()
Gs=sparse.csr_matrix(DG); COGf=sparse.csr_matrix(cog)
a_geom=aucX(Gs)
a_cog=aucX(sparse.hstack([Gs,COGf]))
a_add=aucX(sparse.hstack([Gs,srcNTf,tgtNTf]))
a_pair=aucX(sparse.hstack([Gs,pairNT]))
a_add_cog=aucX(sparse.hstack([Gs,srcNTf,tgtNTf,COGf]))
a_pair_cog=aucX(sparse.hstack([Gs,pairNT,COGf]))
print(f"  geometry                         AUC={a_geom:.4f}")
print(f"  geometry + cognate-receptor      AUC={a_cog:.4f}   (Δ over geom = {a_cog-a_geom:+.4f}; is there NT->receptor logic at all?)")
print(f"  geometry + additive src+tgt NT   AUC={a_add:.4f}")
print(f"  geometry + src×tgt NT (pairwise) AUC={a_pair:.4f}   pairwise lift over additive = {a_pair-a_add:+.4f}")
print(f"  geometry + additive NT + cognate AUC={a_add_cog:.4f}")
print(f"  geometry + pairwise NT + cognate AUC={a_pair_cog:.4f}")
print(f"\n  Does cognate-receptor absorb the pairwise part? pairwise-lift-over-additive:")
print(f"    without cognate: {a_pair-a_add:+.4f}   |   after adding cognate to additive: {a_pair_cog-a_add_cog:+.4f}")
print(f"    cognate adds beyond additive-NT: {a_add_cog-a_add:+.4f}")
