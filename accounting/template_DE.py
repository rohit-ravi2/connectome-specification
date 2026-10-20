"""TEMPLATE D (threshold: per-neuron load -> COUNT outcome, not edge) + TEMPLATE E (temporal: formation-order
-> residual). Both gate-passed. D targets node synapse-degree beyond contact-degree (threshold = 'how many');
Cook replication built in. E uses the developmental stage series (not adult CeNGEN); note: E cannot Cook-replicate
(Cook is single-stage) so E FP-claims are unbankable by rule; test FN (reproducible synapses) within Witvliet.
"""
import warnings,sys,csv,re,json,collections; warnings.simplefilter("ignore")
import numpy as np, pandas as pd, openpyxl
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
def P(*a): print(*a); sys.stdout.flush()
RNG=np.random.default_rng(0); SC="../work"; D="../data/celegans/data"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); m0=np.load(f"{SC}/mode0.npz")
I,J,y7,y8,neurons=z['I'],z['J'],z['y7'],z['y8'],list(z['neurons']); N=len(I); M=len(neurons); idx={n:i for i,n in enumerate(neurons)}
yrep=((y7==1)&(y8==1)).astype(int); agree=m0['agree']; FN=m0['FN']; r1=np.load(f"{SC}/resid2.npz")['r1']
cg=pd.read_csv(D+"/expression/cengen/thresholded/021821_liberal_threshold1.csv")
classes=list(cg.columns[3:]); Xt=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values
gname=cg["gene_name"].astype(str).str.lower().values; cls_idx={c:k for k,c in enumerate(classes)}; nCls=len(classes); Xb=(Xt>0).astype(float)
def n2c(n):
    if n in cls_idx: return cls_idx[n]
    b=n
    for s in "LR":
        if b.endswith(s): b=b[:-1]; break
    if b in cls_idx: return cls_idx[b]
    for s in "DV":
        if b.endswith(s) and b[:-1] in cls_idx: return cls_idx[b[:-1]]
    return cls_idx.get(b,None)
cl=[n2c(n) for n in neurons]
Ci=np.array([cl[i] if cl[i] is not None else -1 for i in I]); Cj=np.array([cl[j] if cl[j] is not None else -1 for j in J]); valid=(Ci>=0)&(Cj>=0)

# ===================== TEMPLATE D — THRESHOLD =====================
P("="*82); P("TEMPLATE D — threshold: family load -> node synapse-degree beyond contact-degree"); P("="*82)
# per-class synapse-degree (Witvliet yrep) and contact-degree
def class_deg(mask):
    d=np.zeros(nCls)
    for k in np.where(mask&valid)[0]: d[Ci[k]]+=1; d[Cj[k]]+=1
    return d
syn_deg=class_deg(yrep>0); con_deg=class_deg(np.ones(N,bool))
have=con_deg>0
# residualize synapse-degree on contact-degree
b=np.polyfit(con_deg[have],syn_deg[have],1); sdeg_res=np.zeros(nCls); sdeg_res[have]=syn_deg[have]-np.polyval(b,con_deg[have])
# Cook synapse-degree
wb=openpyxl.load_workbook(f"{D}/connectome/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
A=list(wb['hermaphrodite chemical'].iter_rows(values_only=True)); nset=set(neurons)
def nm(seq): return sum(1 for x in seq if isinstance(x,str) and x.strip() in nset)
hrow=int(np.argmax([nm(r) for r in A])); cols=[(str(x).strip() if x is not None else None) for x in A[hrow]]
hcol=int(np.argmax([nm([A[r][c] for r in range(len(A))]) for c in range(min(6,len(A[0])))]))
cookdeg=np.zeros(nCls)
for r in range(hrow+1,len(A)):
    rn=A[r][hcol]; rn=rn.strip() if isinstance(rn,str) else None
    if rn not in idx: continue
    c0=n2c(rn)
    for c in range(hcol+1,len(A[r])):
        cn=cols[c]; v=A[r][c]
        if cn in idx and v not in (None,''):
            try:
                if float(v)>0 and c0 is not None and n2c(cn) is not None: cookdeg[c0]+=1; cookdeg[n2c(cn)]+=1
            except: pass
bc=np.polyfit(con_deg[have],cookdeg[have],1); cook_res=np.zeros(nCls); cook_res[have]=cookdeg[have]-np.polyval(bc,con_deg[have])
def gp(prefs): return [i for i in range(len(gname)) if any(gname[i].startswith(p) for p in prefs)]
fams={'zig':gp(['zig-']),'IgSF':gp(['oig-','igcm-','igdb-','lron-']),'cadherin':gp(['cdh-']),'innexin':gp(['inx-']),
      'all-receptors(-r/npr/glr)':gp(['npr-','glr-','ser-','dop-','gbb-','lgc-']),'neuropeptide(flp/nlp/ins)':gp(['flp-','nlp-','ins-'])}
fams={k:v for k,v in fams.items() if len(v)>=4}
def zload(load,tgt):
    o=np.corrcoef(load[have],tgt[have])[0,1]; nul=[np.corrcoef(RNG.permutation(load[have]),tgt[have])[0,1] for _ in range(1000)]
    return (o-np.mean(nul))/(np.std(nul)+1e-9)
# calibration
rz=[zload(Xb[np.random.default_rng(s).choice(len(gname),20,replace=False)].sum(0),sdeg_res) for s in range(8)]
P(f"  [calib] random-gene load -> syn-deg-residual max|z|={np.nanmax(np.abs(rz)):.1f} (want<3)")
P(f"  {'family':26s} {'Witv z':>7s} {'Cook z':>7s}")
for k,g in fams.items():
    load=Xb[g].sum(0); zW=zload(load,sdeg_res); zC=zload(load,cook_res)
    tag="REPLICATES" if (zW>2 and zC>2) else ("Cook-fails" if zW>2 else "")
    P(f"  {k:26s} {zW:+7.1f} {zC:+7.1f}  {tag}")

# ===================== TEMPLATE E — TEMPORAL =====================
P("\n"+"="*82); P("TEMPLATE E — temporal: formation-stage -> FN residual, controlling confirmation-count"); P("="*82)
app=collections.defaultdict(set)
with open(f"{D}/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']!='chemical': continue
        mm=re.search(r'witvliet_2020_(\d)',r['source_file'])
        if not mm: continue
        d=int(mm.group(1))
        if r['pre'] in idx and r['post'] in idx:
            a,b2=idx[r['pre']],idx[r['post']]; app[(min(a,b2),max(a,b2))].add(d)
form=np.full(N,np.nan); nobs=np.full(N,np.nan)
for k in range(N):
    key=(min(int(I[k]),int(J[k])),max(int(I[k]),int(J[k])))
    if key in app: form[k]=min(app[key]); nobs[k]=len(app[key])
syn=(yrep==1)&agree&valid&~np.isnan(form)
from sklearn.linear_model import LogisticRegression
Xe=np.column_stack([nobs[syn],form[syn]]); ye=FN[syn].astype(int)
lr=LogisticRegression(max_iter=500).fit(Xe,ye)
# permutation z on the formation coefficient (control nobs by permuting form within nobs-strata)
coef=lr.coef_[0][1]
nul=[]
for _ in range(500):
    fp=form[syn].copy()
    for nb in np.unique(nobs[syn]):
        ii=np.where(nobs[syn]==nb)[0]
        if len(ii)>1: fp[ii]=fp[ii][RNG.permutation(len(ii))]
    nul.append(LogisticRegression(max_iter=500).fit(np.column_stack([nobs[syn],fp]),ye).coef_[0][1])
zE=(coef-np.mean(nul))/(np.std(nul)+1e-9)
P(f"  formation-stage -> FN (logistic, nobs-controlled): coef={coef:+.2f} z={zE:+.1f} (nobs-stratified perm null)")
P(f"  direction: {'earlier-forming synapses are MORE FN' if coef<0 else 'later-forming synapses are MORE FN'}")
P(f"  NOTE: E cannot Cook-replicate (Cook = single-stage adult) -> by the non-negotiable rule this is a")
P(f"        within-Witvliet CANDIDATE only, not banked; the temporal scheme needs a 2nd developmental series to bank.")
P("="*82)
