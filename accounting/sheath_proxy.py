import sys, csv, re, numpy as np
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import fisher_exact
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); m0=np.load(f"{SC}/mode0.npz")
I,J,y7,y8,base,neurons=z['I'],z['J'],z['y7'],z['y8'],z['base'],list(z['neurons']); idx={n:i for i,n in enumerate(neurons)}
M=len(neurons); N=len(I); yrep=((y7==1)&(y8==1)).astype(int); agree=m0['agree']; FP=m0['FP']; area=base[:,0]
# KNOWN sheath-organ membership (textbook C. elegans sensilla; AMsh wraps the amphid set)
def sheath(n):
    b=n[:-1] if n[-1] in 'LR' else n; side=n[-1] if n[-1] in 'LR' else 'X'; bc=b.rstrip('DV')
    amphid={'ADF','ADL','ASE','ASG','ASH','ASI','ASJ','ASK','AWA','AWB','AWC','AUA'}
    if bc in amphid: return f'AMsh_{side}'                 # amphid sheath, left/right
    if bc in {'IL1','IL2'}: return f'ILsh_{b[3:] or side}' # inner labial sheath (6, by quadrant)
    if bc in {'OLL','OLQ'}: return f'OLsh_{b[3:] or side}' # outer labial sheath
    if bc in {'CEP','CEM'}: return f'CEPsh_{b[3:] or side}'# cephalic sheath
    return None
sh=[sheath(n) for n in neurons]
same_sheath=np.array([sh[I[k]] is not None and sh[I[k]]==sh[J[k]] for k in range(N)])
same_amsh=np.array([sh[I[k]] is not None and sh[I[k]].startswith('AMsh') and sh[I[k]]==sh[J[k]] for k in range(N)])
P(f"neurons with a known sheath organ: {sum(x is not None for x in sh)}/{M} | same-sheath pairs {same_sheath.sum()} | same-AMsh(amphid) pairs {same_amsh.sum()}")
hc=(area>np.quantile(area,0.75))&agree
def enr(mask,cls,lab):
    a=(mask&cls).sum();b=(mask&~cls).sum();c=(~mask&cls).sum();d=(~mask&~cls).sum()
    orr,p=fisher_exact([[a,b],[c,d]]); P(f"   {lab}: FP-rate in-group {a/(a+b):.3f} vs out {c/(c+d):.3f}  OR={orr:.2f} p={p:.1e}")
P(f"[SHEATH-COMPARTMENT vs FP hard core, high-contact agree pairs n={hc.sum()}]")
enr(hc&same_amsh, FP[hc&same_amsh] if False else FP, "")  # placeholder
# proper: among high-contact agree pairs, is FP enriched in same-AMsh / same-sheath?
def fp_enr(sel,lab):
    m=hc&sel; base_fp=FP[hc].mean()
    P(f"   same-{lab}: n={m.sum()} FP-rate={FP[m].mean():.3f} vs all-high-contact {base_fp:.3f} vs different-{lab} {FP[hc&~sel].mean():.3f}")
fp_enr(same_amsh,"AMsh(amphid)"); fp_enr(same_sheath,"sheath-organ")
# does same-sheath predict FP BEYOND contact+closure+persistence+sameNT? (neuron-holdout)
GL1=np.zeros((M,M))
with open("../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']=='chemical':
            d=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
            if d in (1,2,3,4) and r['pre'] in idx and r['post'] in idx: GL1[idx[r['pre']],idx[r['post']]]=1
Cff=GL1@GL1
t5=np.load("../data/tier5/connectome.npz",allow_pickle=True); t5nt={str(x):str(v) for x,v in zip(t5["names"],t5["nt_primary"])}
def bcl(n): return n[:-1] if n[-1] in 'LR' else n
nt=[t5nt.get(neurons[i],t5nt.get(bcl(neurons[i]),"U")) for i in range(M)]; sameNT=np.array([nt[I[k]]==nt[J[k]] for k in range(N)]).astype(float)
BASE=np.column_stack([area,np.log1p(Cff[I,J]),GL1[I,J],sameNT])
nf={n:i%5 for i,n in enumerate(sorted(set([neurons[i] for i in I]+[neurons[j] for j in J])))}
yFP=FP.astype(int)
def nho(X,mask):
    ii=np.where(mask)[0]; pred=np.full(len(ii),np.nan)
    for f in range(5):
        te=[t for t,k in enumerate(ii) if nf[neurons[I[k]]]==f and nf[neurons[J[k]]]==f];tr=[t for t,k in enumerate(ii) if nf[neurons[I[k]]]!=f and nf[neurons[J[k]]]!=f]
        if len(te)<8 or len(set(yFP[ii][tr]))<2: continue
        pred[np.array(te)]=GradientBoostingClassifier(random_state=0).fit(X[ii][tr],yFP[ii][tr]).predict_proba(X[ii][te])[:,1]
    m=~np.isnan(pred); return roc_auc_score(yFP[ii][m],pred[m])
SH=np.column_stack([same_sheath.astype(float),same_amsh.astype(float)])
a0=nho(BASE,agree); a1=nho(np.hstack([BASE,SH]),agree)
P(f"\n[predict FP among agree pairs] contact+closure+persist+sameNT AUC={a0:.3f} -> +sheath-comembership AUC={a1:.3f} (Δ{a1-a0:+.3f})")
