"""BRANCH 11 — anchor the identity-rank to ANATOMY. Branch 10: identity-rank s_n = functional trophic hierarchy
(r=-0.67). Does that axis line up with the classical anatomical SENSORY -> INTERNEURON -> MOTOR categorization?
Use a small HIGH-CONFIDENCE curated set (unambiguous classes only) to avoid mislabeling. Predict: sensory highest
presynaptic-source rank s (lowest trophic h), motor lowest s (highest h), interneurons between. Ordered = the
molecular identity-rank recovers the textbook sensory->motor axis without ever being told the categories.
"""
import warnings,sys,openpyxl; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.stats import kruskal
def P(*a): print(*a); sys.stdout.flush()
D="../data/celegans/data"
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
wb=openpyxl.load_workbook(f"{D}/connectome/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
A=list(wb['hermaphrodite chemical'].iter_rows(values_only=True))
def looks(n): return isinstance(n,str) and len(n)>=2 and n[0].isalpha()
hr=max(range(len(A)),key=lambda r:sum(looks(x) for x in A[r])); cols=[(str(x).strip() if x is not None else None) for x in A[hr]]
hc=max(range(min(6,len(A[0]))),key=lambda c:sum(looks(A[r][c]) for r in range(len(A))))
Ed=[]
for r in range(hr+1,len(A)):
    cpre=n2c(A[r][hc])
    if not cpre: continue
    for c in range(hc+1,len(A[r])):
        cpost=n2c(cols[c]); v=A[r][c]
        if cpost and cpost!=cpre and v not in (None,''):
            try:
                if float(v)>0: Ed.append((cpre,cpost,float(v)))
            except: pass
chem=set((a,b) for a,b,w in Ed); asym=[(a,b) for (a,b) in chem if (b,a) not in chem]
F=[];y=[]
for a,b in asym:
    d=Xc[cls_idx[a]]-Xc[cls_idx[b]]; F+=[d,-d]; y+=[1,0]
w=LogisticRegression(max_iter=1000,C=0.5).fit(np.array(F),np.array(y)).coef_[0]
neur=sorted({a for a,b,w_ in Ed}|{b for a,b,w_ in Ed}); idx={n:i for i,n in enumerate(neur)}; M=len(neur)
s={n:float(Xc[cls_idx[n]]@w) for n in neur}
W=np.zeros((M,M))
for a,b,w_ in Ed: W[idx[a],idx[b]]+=w_
indeg=W.sum(0); outdeg=W.sum(1)
h=np.linalg.lstsq(np.diag(indeg+outdeg)-(W+W.T)+np.ones((M,M))/M, indeg-outdeg, rcond=None)[0]; h=h-h.min()
hh={neur[i]:h[i] for i in range(M)}
# HIGH-CONFIDENCE curated categories (class names; unambiguous only)
SENS=["ASE","ASH","ASI","ASJ","ASK","ASG","ADF","ADL","AWA","AWB","AWC","AFD","ASEL","BAG","URX","AQR","PQR","ADE","ALM","PLM","AVM","PVM","FLP","OLQ","OLL","IL2","CEP","URY"]
INTER=["AVA","AVB","AVD","AVE","PVC","AIA","AIB","AIY","AIZ","RIA","RIB","RIM","RIF","RIG","AVK","AVJ","AVH","PVP","PVQ","RIH","AUA","AIN"]
MOTOR=["VA","VB","VD","DA","DB","DD","AS","RMD","SMD","RME","SMB","RMF","RMH","RIV","RIM","VC","HSN","URA","SAB","SIA","SIB"]
def score_cat(names):
    ss=[s[n] for n in names if n in s]; hs=[hh[n] for n in names if n in hh]; return ss,hs
labels={"sensory":SENS,"interneuron":INTER,"motor":MOTOR}
P(f"BRANCH 11 — identity-rank s and trophic h by anatomical category (high-confidence curated):")
allS=[];grp=[]
for lab,names in labels.items():
    ss,hs=score_cat(names)
    P(f"    {lab:12s} (n={len(ss):2d}): identity-rank s = {np.mean(ss):+.3f}   trophic h = {np.mean(hs):.3f}")
    allS.append(ss); grp+=[lab]*len(ss)
ks,pk=kruskal(*allS)
P(f"\n  Kruskal-Wallis across categories on identity-rank s: H={ks:.1f} p={pk:.1e}")
mS=[np.mean(x) for x in allS]
ordered = mS[0]>mS[1]>mS[2]   # sensory > inter > motor in presynaptic-source rank
P(f"  order sensory>inter>motor in presynaptic-source rank: {'YES' if ordered else 'NO'} ({mS[0]:+.2f} > {mS[1]:+.2f} > {mS[2]:+.2f})")
P(f"\n  VERDICT: {'the molecular identity-rank recovers the textbook SENSORY->INTER->MOTOR axis (ordered, p<0.05) WITHOUT being told categories — geometry=edge, whole-cell identity=a scalar rank=the sensory->motor hierarchy' if ordered and pk<0.05 else 'identity-rank only partly aligns with anatomical category (H p=%.2g, order=%s)'%(pk,ordered)}.")
