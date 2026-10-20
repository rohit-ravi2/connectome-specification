"""BRANCH 9 — DECODE the arrow axis. The presynaptic-identity axis w (logistic on antisym direction feature)
sets which way a synapse points (AUC~0.89). Branch 1 said it's distributed + NT-independent. WHAT molecular
vocabulary carries it? Rank genes by |w|; test which GENE FAMILIES (by WormBase naming: neuropeptides flp/nlp/ins,
GPCR chemoreceptors sr*, innexins inx/unc-7/9, homeobox ceh + known TFs, adhesion/IgSF curated) carry more
arrow-discriminative weight than background (Mann-Whitney family-|w| vs rest). Tells us the KIND of code the arrow
is: transcription-factor identity, surface-recognition, or signaling.
"""
import warnings,sys,openpyxl; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
def P(*a): print(*a); sys.stdout.flush()
D="../data/celegans/data"
cg=pd.read_csv(f"{D}/expression/cengen/thresholded/021821_liberal_threshold1.csv")
classes=list(cg.columns[3:]); Xf=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values.T>0
allgenes=cg["gene_name"].astype(str).str.lower().values
br=Xf.mean(0); mask=(br>0.05)&(br<0.95); Xc=Xf[:,mask].astype(float); genes=allgenes[mask]
cls_idx={c:k for k,c in enumerate(classes)}
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
# Cook chemical directed -> asym pairs -> learn arrow axis w
wb=openpyxl.load_workbook(f"{D}/connectome/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
A=list(wb['hermaphrodite chemical'].iter_rows(values_only=True))
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
F=[];y=[]
for a,b in asym:
    d=Xc[cls_idx[a]]-Xc[cls_idx[b]]; F+=[d,-d]; y+=[1,0]
w=LogisticRegression(max_iter=1000,C=0.5).fit(np.array(F),np.array(y)).coef_[0]
aw=np.abs(w); order=np.argsort(-aw)
P(f"BRANCH 9 — arrow axis over {len(genes)} variable genes | {len(asym)} asymmetric directed pairs")
P("\n  TOP presynaptic-SOURCE genes (w>0, high in the presynaptic partner):")
pos=[i for i in order if w[i]>0][:15]; P("    "+", ".join(f"{genes[i]}({w[i]:+.2f})" for i in pos))
P("  TOP postsynaptic-SINK genes (w<0, high in the postsynaptic partner):")
neg=[i for i in order if w[i]<0][:15]; P("    "+", ".join(f"{genes[i]}({w[i]:+.2f})" for i in neg))
# family membership by naming convention
def fam_prefix(g,prefs): return any(g.startswith(p) for p in prefs)
GPCR=("srh-","str-","sri-","srd-","sre-","srab","srx-","srt-","srg-","srz-","srsx","srw-","srbc","srr-","srb-","sra-","srv-","srj-","srxa","dmsr","npr-","gar-","ser-","dop-","tyra","octr","mgl-","gbb-")
NP=("flp-","nlp-","ins-")
INX=("inx-","unc-7","unc-9","eat-5","che-7")
HOX=("ceh-","unc-4","unc-30","unc-42","unc-86","unc-3","vab-3","vab-7","vab-15","lin-11","mec-3","ttx-1","ttx-3","cog-1","alr-1","egl-5","egl-46","lim-","zag-1","fax-1","die-1","ceh")
ADH=("nrx-1","nlg-1","sax-3","sax-7","cam-1","casy-1","ptp-3","rig-","zig-","oig-","wrk-1","madd-4","unc-40","sax-","dma-1","hpo-30","lron","igcm","syg-1","syg-2","clc-","cdh-","fmi-","him-4","lad-2","ncam")
fams={"GPCR/receptor":GPCR,"neuropeptide":NP,"innexin":INX,"homeobox/TF":HOX,"adhesion/IgSF":ADH}
P("\n  FAMILY ENRICHMENT in arrow-discriminative weight |w| (Mann-Whitney family vs rest):")
base=np.median(aw)
for name,pref in fams.items():
    sel=np.array([fam_prefix(g,pref) for g in genes])
    if sel.sum()<5: P(f"    {name:18s}: n={sel.sum()} (too few)"); continue
    u,p=mannwhitneyu(aw[sel],aw[~sel],alternative='greater')
    P(f"    {name:18s}: n={sel.sum():3d}  median|w| {np.median(aw[sel]):.3f} vs {base:.3f} rest  MWU(greater) p={p:.2e} {'<-- enriched' if p<0.01 else ''}")
P(f"\n  VERDICT: the arrow's molecular vocabulary is read from the family enrichments above — whichever families carry\n  disproportionate |w| define the KIND of presynaptic-identity code (recognition/signaling/TF) that sets synapse polarity.")
