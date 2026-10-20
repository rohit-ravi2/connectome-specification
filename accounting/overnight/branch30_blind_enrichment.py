"""BRANCH 30 — (1) BLIND domain enrichment (every InterPro/Pfam term tested vs background, BH-corrected; no keyword
list) on the bootstrap-stable compact direction-address genes. (2) SPECIFIC vs GENERAL: does the compact panel predict
synapse direction (cross-animal Cook->Witvliet) better than random panels drawn from the a-priori SURFACE/membrane
gene pool? If compact ~= random-surface >> random-all -> 'surface genes generally' (address = being-surface). If
compact >> random-surface -> THESE specific surface genes carry it.
"""
import warnings,sys,openpyxl,gzip,csv; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from collections import Counter, defaultdict
from sklearn.linear_model import LogisticRegression
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
def P(*a): print(*a); sys.stdout.flush()
RNG=np.random.default_rng(0); D="../data/celegans/data"
cg=pd.read_csv(f"{D}/expression/cengen/thresholded/021821_liberal_threshold1.csv")
classes=list(cg.columns[3:]); gnames=cg["gene_name"].astype(str).values; wbids=cg["Wormbase_ID"].astype(str).values
Xf=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values.T>0
br=Xf.mean(0); mask=(br>0.05)&(br<0.95); Xc=Xf[:,mask].astype(float); genes=gnames[mask]; wbv=wbids[mask]; cls_idx={c:k for k,c in enumerate(classes)}
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
def cook():
    wb=openpyxl.load_workbook(f"{D}/connectome/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
    A=list(wb['hermaphrodite chemical'].iter_rows(values_only=True))
    def looks(n): return isinstance(n,str) and len(n)>=2 and n[0].isalpha()
    hr=max(range(len(A)),key=lambda r:sum(looks(x) for x in A[r])); cols=[(str(x).strip() if x is not None else None) for x in A[hr]]
    hc=max(range(min(6,len(A[0]))),key=lambda c:sum(looks(A[r][c]) for r in range(len(A))))
    ed=set()
    for r in range(hr+1,len(A)):
        cp=n2c(A[r][hc])
        if not cp: continue
        for c in range(hc+1,len(A[r])):
            q=n2c(cols[c]); v=A[r][c]
            if q and q!=cp and v not in (None,''):
                try:
                    if float(v)>0: ed.add((cp,q))
                except: pass
    return ed
chem=cook(); asym=[(a,b) for (a,b) in chem if (b,a) not in chem]
wEd=set()
with open(f"{D}/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']=='chemical':
            a=n2c(r['pre']); b=n2c(r['post'])
            if a and b and a!=b: wEd.add((a,b))
wasym=[(a,b) for (a,b) in wEd if (b,a) not in wEd and a in cls_idx and b in cls_idx]
def design(pairs):
    F=[];y=[]
    for a,b in pairs:
        d=Xc[cls_idx[a]]-Xc[cls_idx[b]]; F+=[d,-d]; y+=[1,0]
    return np.array(F),np.array(y)
Fc,yc=design(asym); Fw,yw=design(wasym)
from sklearn.metrics import roc_auc_score
def transfer(cols):
    m=LogisticRegression(max_iter=2000,C=0.5).fit(Fc[:,cols],yc); return roc_auc_score(yw,m.decision_function(Fw[:,cols]))
# bootstrap-stable compact set
def l1sel(F,y,C=0.01):
    m=LogisticRegression(penalty='l1',solver='liblinear',C=C,max_iter=2000).fit(F,y); return np.where(np.abs(m.coef_[0])>1e-8)[0]
freq=Counter()
for b in range(40):
    idx=RNG.integers(0,len(asym),len(asym)); Fb,yb=design([asym[i] for i in idx])
    for j in l1sel(Fb,yb): freq[genes[j]]+=1
stable=[g for g,c in freq.items() if c>=12]          # recur in >=30% of bootstraps
comp_idx=np.array([np.where(genes==g)[0][0] for g in stable])
P(f"BRANCH 30 — stable compact set (bootstrap>=12/40): {len(stable)} genes: {', '.join(stable)}")
# domains
dom=defaultdict(set)
with gzip.open(f"{D}/genomes/multi_species_annot/c_elegans/compressed/c_elegans.PRJNA13758.WS297.protein_domains.csv.gz",'rt') as f:
    for line in f:
        p=line.rstrip("\n").split("\t");
        if len(p)>=2:
            for x in p[3:]:
                if '"' in x: dom[p[0]].add(x.split('"')[1])
def terms(g):
    out=set()
    for w in wbv[np.where(genes==g)[0]]: out|=dom.get(w,set())
    return out
# ---- (1) BLIND term enrichment: every term in >=2 compact genes, Fisher vs background ----
comp_terms=Counter()
for g in stable:
    for t in terms(g): comp_terms[t]+=1
bg_term=Counter()
for g in genes:
    for t in terms(g): bg_term[t]+=1
NG=len(genes); NC=len(stable)
rows=[]
for t,ct in comp_terms.items():
    if ct<2: continue
    bg=bg_term[t]
    _,p=fisher_exact([[ct,NC-ct],[bg-ct,NG-NC-(bg-ct)]],alternative='greater')
    rows.append((t,ct,bg,p))
if rows:
    ps=[r[3] for r in rows]; rej,q,_,_=multipletests(ps,method='fdr_bh')
    rows=[(rows[i][0],rows[i][1],rows[i][2],rows[i][3],q[i]) for i in range(len(rows))]
    rows.sort(key=lambda r:r[3])
    P(f"\n  (1) BLIND term enrichment (compact vs all {NG} genes, BH-corrected):")
    for t,ct,bg,p,qq in rows[:14]:
        P(f"    {('*' if qq<0.1 else ' ')} {ct}/{NC} compact ({bg} bg) q={qq:.3f}  {t[:60]}")
# ---- (2) specific vs general: surface pool ----
MEM=["membran","transmembran","receptor","g protein","gpcr","7tm","serpentine","immunoglobulin","signal peptide",
     "egf","cadherin","fibronectin","leucine-rich repeat","adhesion","kinesin","semaphorin","plexin","disintegrin",
     "metallopept","metalloprotein","synaptot","c2 domain","sphingo","ceramide","tlc domain","extracellular","ig-like"]
def is_surf(g):
    s=" ".join(terms(g)).lower(); return any(k in s for k in MEM)
surf_pool=np.array([i for i in range(len(genes)) if is_surf(genes[i])])
nonsurf=np.array([i for i in range(len(genes)) if not is_surf(genes[i])])
P(f"\n  surface/membrane a-priori pool: {len(surf_pool)} genes; compact set surface-fraction {np.mean([is_surf(g) for g in stable]):.2f}")
k=len(stable)
comp_auc=transfer(comp_idx)
def sample_auc(pool,reps=40):
    v=[transfer(RNG.choice(pool,k,replace=False)) for _ in range(reps)]; return np.mean(v),np.std(v)
ms,ss=sample_auc(surf_pool); mn,sn=sample_auc(nonsurf); ma,sa=sample_auc(np.arange(len(genes)))
P(f"\n  (2) cross-animal direction AUC (panel size {k}):")
P(f"    COMPACT stable panel        {comp_auc:.3f}")
P(f"    random from SURFACE pool     {ms:.3f}±{ss:.3f}")
P(f"    random from NON-surface pool {mn:.3f}±{sn:.3f}")
P(f"    random from ALL genes        {ma:.3f}±{sa:.3f}")
zsurf=(comp_auc-ms)/(ss+1e-9); zss=(ms-mn)/np.sqrt(ss**2+sn**2+1e-9)
P(f"\n  compact vs random-surface z={zsurf:+.1f} | surface vs non-surface z={zss:+.1f}")
P(f"\n  READ: if surface>>non-surface, 'being surface' carries direction; if compact>>random-surface too, THESE specific genes add beyond generic surface identity.")
