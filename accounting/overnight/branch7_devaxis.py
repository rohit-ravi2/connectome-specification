"""BRANCH 7 — is the functional sensory->motor ARROW-AXIS a DEVELOPMENTAL axis? The synthesis says identity codes
the arrow (presynaptic-ness). Development lays down identity. So: does a neuron's position on the functional axis
(trophic level: low=sensory-source, high=motor-sink; AND the learned presynaptic-identity score w.expr) track
BIRTH TIME / lineage depth / founder? If sensory sources are early-born and motor sinks late-born, the functional
arrow is a developmental-temporal code. Nulls: shuffle the birth labels; compare to node-degree confound.
"""
import warnings,sys,openpyxl; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.linear_model import LogisticRegression
def P(*a): print(*a); sys.stdout.flush()
D="../data/celegans/data"
# expression (variable genes)
cg=pd.read_csv(f"{D}/expression/cengen/thresholded/021821_liberal_threshold1.csv")
classes=list(cg.columns[3:]); Xf=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values.T>0
br=Xf.mean(0); Xc=Xf[:,(br>0.05)&(br<0.95)].astype(float); cls_idx={c:k for k,c in enumerate(classes)}
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
# Cook chemical directed weighted -> trophic level per neuron-class
wb=openpyxl.load_workbook(f"{D}/connectome/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
A=list(wb['hermaphrodite chemical'].iter_rows(values_only=True))
def looks(n): return isinstance(n,str) and len(n)>=2 and n[0].isalpha()
hr=max(range(len(A)),key=lambda r:sum(looks(x) for x in A[r])); cols=[(str(x).strip() if x is not None else None) for x in A[hr]]
hc=max(range(min(6,len(A[0]))),key=lambda c:sum(looks(A[r][c]) for r in range(len(A))))
Ed=[]
for r in range(hr+1,len(A)):
    pre=A[r][hc]; pre=pre.strip() if isinstance(pre,str) else None
    cpre=n2c(pre)
    if not cpre: continue
    for c in range(hc+1,len(A[r])):
        cpost=n2c(cols[c]); v=A[r][c]
        if cpost and cpost!=cpre and v not in (None,''):
            try:
                if float(v)>0: Ed.append((cpre,cpost,float(v)))
            except: pass
neur=sorted({a for a,b,w in Ed}|{b for a,b,w in Ed}); idx={n:i for i,n in enumerate(neur)}; M=len(neur)
W=np.zeros((M,M))
for a,b,w in Ed: W[idx[a],idx[b]]+=w   # W[pre,post]
indeg=W.sum(0); outdeg=W.sum(1); u=indeg+outdeg; v=indeg-outdeg
h=np.linalg.lstsq(np.diag(u)-(W+W.T)+np.ones((M,M))/M, v, rcond=None)[0]; h=h-h.min()  # trophic level (low=source)
troph={neur[i]:h[i] for i in range(M)}
P(f"BRANCH 7 — {M} neuron-classes on directed connectome; trophic range {h.max()-h.min():.2f}")
# presynaptic-identity axis w (arrow code): logistic on antisym direction feature over asym chemical pairs
chem=set((a,b) for a,b,w in Ed)
asym=[(a,b) for (a,b) in chem if (b,a) not in chem]
F=[];y=[]
for a,b in asym:
    d=Xc[cls_idx[a]]-Xc[cls_idx[b]]; F+=[d,-d]; y+=[1,0]
lr=LogisticRegression(max_iter=1000,C=0.5).fit(np.array(F),np.array(y)); w=lr.coef_[0]
presyn={n:float(Xc[cls_idx[n]]@w) for n in neur}   # higher = more presynaptic (source-like)
# developmental data
bt=pd.read_csv(f"{D}/processed/developmental_rules/birthtime_earlylate.csv")
lf=pd.read_csv(f"{D}/processed/developmental_rules/lineage_features.csv")
born={r.neuron:r.born for r in bt.itertuples()}; depth={r.neuron:r.depth for r in lf.itertuples()}; founder={r.neuron:r.founder for r in lf.itertuples()}
def cval(dmap,cn):  # aggregate cell-level dev value to class (mean over L/R/DV members present)
    vs=[dmap[k] for k in dmap if n2c(k)==cn]
    return np.mean(vs) if vs and all(isinstance(x,(int,float,np.integer,np.floating)) for x in vs) else (vs[0] if vs else None)
def cborn(cn):
    vs=[born[k] for k in born if n2c(k)==cn];
    return (sum(x=='early' for x in vs)/len(vs)) if vs else None   # frac early
rows=[]
for n in neur:
    d=cval(depth,n); b=cborn(n)
    rows.append((n,troph[n],presyn[n],d,b,indeg[idx[n]]+outdeg[idx[n]]))
df=pd.DataFrame(rows,columns=['cls','troph','presyn','depth','fracEarly','deg']).dropna(subset=['depth'])
P(f"  matched to lineage depth: {len(df)} classes")
# CORE Q1: does trophic level track lineage depth? (are motor-sinks deeper/later-born?)
rs,ps=spearmanr(df['troph'],df['depth'])
P(f"\n  Q1 trophic level (sensory-low->motor-high) vs LINEAGE DEPTH: Spearman r={rs:+.3f} p={ps:.1e}")
# degree confound: partial via controlling degree (residualize both on degree)
def resid(a,ctrl):
    b=np.polyfit(ctrl,a,1); return a-np.polyval(b,ctrl)
rs_c,ps_c=spearmanr(resid(df['troph'].values,df['deg'].values), resid(df['depth'].values,df['deg'].values))
P(f"     controlling node-degree: partial r={rs_c:+.3f} p={ps_c:.1e}")
# CORE Q2: presynaptic-identity axis vs depth
r2,p2=spearmanr(df['presyn'],df['depth'])
P(f"  Q2 presynaptic-IDENTITY score (arrow code) vs LINEAGE DEPTH: r={r2:+.3f} p={p2:.1e}")
# CORE Q3: born-early vs born-late trophic level (categorical)
dfb=df.dropna(subset=['fracEarly']); early=dfb[dfb.fracEarly>=0.5]['troph']; late=dfb[dfb.fracEarly<0.5]['troph']
if len(early)>3 and len(late)>3:
    u,pu=mannwhitneyu(early,late)
    P(f"  Q3 trophic level: early-born classes {early.mean():.2f} (n{len(early)}) vs late-born {late.mean():.2f} (n{len(late)})  MWU p={pu:.1e}")
# null: shuffle depth labels
rn=[abs(spearmanr(df['troph'],np.random.permutation(df['depth']))[0]) for _ in range(500)]
P(f"\n  NULL (shuffled depth): |r| {np.mean(rn):.3f}±{np.std(rn):.3f}; real |r|={abs(rs):.3f} -> z={ (abs(rs)-np.mean(rn))/np.std(rn):.1f}")
sig = ps<0.01 and abs(rs_c)>0.1
P(f"\n  VERDICT: {'the functional sensory->motor axis IS partly a DEVELOPMENTAL-TIMING axis — trophic level tracks lineage depth beyond degree+shuffle (later-born neurons sit deeper in the feedforward hierarchy)' if sig else 'functional axis is NOT explained by lineage depth (developmental timing is orthogonal to the functional hierarchy)'}.")
