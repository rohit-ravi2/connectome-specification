"""BRANCH 29 — what do the compact direction-address genes HAVE IN COMMON? (1) the L1 n=11 panel + greedy panel,
(2) BOOTSTRAP-STABILITY (which genes recur when pairs are resampled — L1 with correlated features picks arbitrary
representatives, so the STABLE recurring genes are what matters), (3) GROUND-TRUTH protein domains (InterPro/Pfam
from WS297, not memory) per gene, (4) enrichment test: are compact-code genes enriched for membrane/surface/guidance
domains vs expression-matched random panels? Answers 'what do they share and what could it reveal' rigorously.
"""
import warnings,sys,openpyxl,gzip; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from collections import Counter, defaultdict
from sklearn.linear_model import LogisticRegression
def P(*a): print(*a); sys.stdout.flush()
RNG=np.random.default_rng(0); D="../data/celegans/data"
cg=pd.read_csv(f"{D}/expression/cengen/thresholded/021821_liberal_threshold1.csv")
classes=list(cg.columns[3:]); gnames=cg["gene_name"].astype(str).values; wbids=cg["Wormbase_ID"].astype(str).values
Xf=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values.T>0
br=Xf.mean(0); mask=(br>0.05)&(br<0.95); Xc=Xf[:,mask].astype(float); genes=gnames[mask]; wb=wbids[mask]; cls_idx={c:k for k,c in enumerate(classes)}
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
wbk=openpyxl.load_workbook(f"{D}/connectome/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
A=list(wbk['hermaphrodite chemical'].iter_rows(values_only=True))
def looks(n): return isinstance(n,str) and len(n)>=2 and n[0].isalpha()
hr=max(range(len(A)),key=lambda r:sum(looks(x) for x in A[r])); cols=[(str(x).strip() if x is not None else None) for x in A[hr]]
hc=max(range(min(6,len(A[0]))),key=lambda c:sum(looks(A[r][c]) for r in range(len(A))))
chem=set()
for r in range(hr+1,len(A)):
    cpre=n2c(A[r][hc])
    if not cpre: continue
    for c in range(hc+1,len(A[r])):
        cpost=n2c(cols[c]); v=A[r][c]
        if cpost and cpost!=cpre and v not in (None,''):
            try:
                if float(v)>0: chem.add((cpre,cpost))
            except: pass
asym=[(a,b) for (a,b) in chem if (b,a) not in chem]
def design(pairs):
    F=[];y=[]
    for a,b in pairs:
        d=Xc[cls_idx[a]]-Xc[cls_idx[b]]; F+=[d,-d]; y+=[1,0]
    return np.array(F),np.array(y)
Fc,yc=design(asym)
# L1 at C=0.01 (gave ~11)
def l1sel(F,y,C=0.01):
    m=LogisticRegression(penalty='l1',solver='liblinear',C=C,max_iter=2000).fit(F,y)
    return np.where(np.abs(m.coef_[0])>1e-8)[0]
sel0=l1sel(Fc,yc); P(f"BRANCH 29 — L1 n={len(sel0)} panel: {', '.join(genes[i] for i in sel0)}")
# BOOTSTRAP stability over resampled pairs
P(f"\n  BOOTSTRAP-STABILITY (60 resamples of the {len(asym)} pairs; L1 selection recurrence):")
freq=Counter()
for b in range(60):
    idx=RNG.integers(0,len(asym),len(asym)); prs=[asym[i] for i in idx]
    Fb,yb=design(prs)
    for j in l1sel(Fb,yb): freq[genes[j]]+=1
stable=freq.most_common(20)
for g,c in stable: P(f"    {g:<14} {c}/60")
# protein domains (ground truth)
dom=defaultdict(list)
with gzip.open(f"{D}/genomes/multi_species_annot/c_elegans/compressed/c_elegans.PRJNA13758.WS297.protein_domains.csv.gz",'rt') as f:
    for line in f:
        p=line.rstrip("\n").split("\t")
        if len(p)>=2:
            wbg=p[0]; descs=[x.split('"')[1] for x in p[3:] if '"' in x]
            if descs: dom[wbg].extend(descs)
def domains_for(gene):
    w=set(wb[np.where(genes==gene)[0]]) if gene in genes else set()
    ds=[]
    for x in w: ds+=dom.get(x,[])
    return sorted(set(ds))
panel=list(dict.fromkeys([genes[i] for i in sel0]+[g for g,_ in stable[:12]]))
P(f"\n  PROTEIN DOMAINS (InterPro/Pfam, WS297 ground-truth) for compact/stable genes:")
for g in panel:
    ds=domains_for(g); tag='(uncharacterized/no-domain)' if not ds else '; '.join(ds[:4])
    P(f"    {g:<14} {tag}")
# enrichment: membrane/surface/guidance keyword hits vs random matched-breadth panels
KEY=["membrane","transmembrane","receptor","G protein","GPCR","7TM","immunoglobulin","signal peptide","EGF",
     "cadherin","fibronectin","leucine-rich","adhesion","kinesin","semaphorin","plexin","metallopeptidase",
     "metalloprotease","ADAM","synaptot","C2 domain","sphingo","ceramide","lipid","secret"]
def is_surface(gene):
    ds=" ".join(domains_for(gene)).lower()
    return any(k.lower() in ds for k in KEY)
comp_genes=[g for g,_ in stable[:12]]
comp_hit=np.mean([is_surface(g) for g in comp_genes])
# expression-breadth-matched random background
brv=Xc.mean(0); tgt=np.array([brv[np.where(genes==g)[0][0]] for g in comp_genes])
randfrac=[]
for _ in range(500):
    pick=[]
    for t in tgt:
        cand=np.argsort(np.abs(brv-t))[:40]; pick.append(genes[RNG.choice(cand)])
    randfrac.append(np.mean([is_surface(g) for g in pick]))
z=(comp_hit-np.mean(randfrac))/(np.std(randfrac)+1e-9)
P(f"\n  membrane/surface/guidance-domain fraction: compact {comp_hit:.2f} vs matched-random {np.mean(randfrac):.2f}±{np.std(randfrac):.2f} -> z={z:+.1f}")
chard=np.mean([len(domains_for(g))==0 for g in comp_genes])
P(f"  fraction of compact genes that are UNCHARACTERIZED (no domain): {chard:.2f}")
P(f"\n  READ: stable recurring genes + their shared domain classes above = what the address is 'made of'.")
