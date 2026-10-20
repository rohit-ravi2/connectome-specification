import sys, csv, re, numpy as np
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); m0=np.load(f"{SC}/mode0.npz")
I,J,y7,y8,base,neurons=z['I'],z['J'],z['y7'],z['y8'],z['base'],list(z['neurons']); idx={n:i for i,n in enumerate(neurons)}
M=len(neurons); N=len(I); yrep=((y7==1)&(y8==1)).astype(int); agree=m0['agree']; L1c=m0['L1c']
# pre-adult L1 synapse graph
GL1=np.zeros((M,M))
with open("../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']!='chemical': continue
        d=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
        if d in (1,2,3,4) and r['pre'] in idx and r['post'] in idx: GL1[idx[r['pre']],idx[r['post']]]=1
Cu=((GL1+GL1.T)>0).astype(float); comm=Cu@Cu; Cin=GL1.T@GL1; Cout=GL1@GL1.T; Cff=GL1@GL1; l1deg=Cu.sum(1)
t5=np.load("../data/tier5/connectome.npz",allow_pickle=True); t5nt={str(n):str(v) for n,v in zip(t5["names"],t5["nt_primary"])}
def bcl(n): return n[:-1] if n[-1] in 'LR' else n
nt=[t5nt.get(neurons[i],t5nt.get(bcl(neurons[i]),"U")) for i in range(M)]; sament=np.array([nt[I[k]]==nt[J[k]] for k in range(N)]).astype(float)
# PRE-ADULT history (NO adult structure); identity feature = sameNT (last col, watch if machine uses it)
fn=['contact','L1syn','L1recip','L1Cff','L1comm','L1Cin','L1Cout','L1deg_s','L1deg_t','sameNT']
H=np.column_stack([base[:,0],GL1[I,J],GL1[J,I],np.log1p(Cff[I,J]),np.log1p(comm[I,J]),np.log1p(Cin[I,J]),np.log1p(Cout[I,J]),l1deg[I],l1deg[J],sament])
# target: adult fate on AGREE (deterministic) contacting pairs
mask=agree; Xh=H[mask]; y=yrep[mask]; P(f"causal-state target: {mask.sum()} agree contacting pairs, adult-present rate {y.mean():.3f}")
# ---- GATE: history classes with >=30 samples (estimable futures) ----
tree=DecisionTreeClassifier(min_samples_leaf=30,random_state=0).fit(Xh,y)
leaf=tree.apply(Xh); uL=np.unique(leaf); sizes=np.array([(leaf==l).sum() for l in uL])
P(f"[GATE] history classes (tree leaves, min 30/leaf): {len(uL)} | leaf sizes min/median/max = {sizes.min()}/{int(np.median(sizes))}/{sizes.max()} -> {'PASS (futures estimable)' if sizes.min()>=30 else 'FAIL'}")
# leaf future distributions
pl={l:y[leaf==l].mean() for l in uL}
# ---- MERGE statistically-indistinguishable futures -> causal states ----
def zprop(l1,l2):
    n1,n2=(leaf==l1).sum(),(leaf==l2).sum(); p1,p2=pl[l1],pl[l2]; p=(p1*n1+p2*n2)/(n1+n2)
    se=np.sqrt(p*(1-p)*(1/n1+1/n2))+1e-9; return abs(p1-p2)/se
order=sorted(uL,key=lambda l:pl[l]); states=[[order[0]]]
for l in order[1:]:
    if zprop(states[-1][-1],l)<2.0: states[-1].append(l)
    else: states.append([l])
P(f"[CAUSAL STATES] {len(uL)} history classes MERGE to {len(states)} causal states (distinct future distributions) | future-P range {pl[order[0]]:.3f}..{pl[order[-1]]:.3f}")
# predictive info (cross-validated) full-history vs triadic-only vs identity-added
cv=StratifiedKFold(5,shuffle=True,random_state=0)
def pinfo(Xf):
    p=np.clip(cross_val_predict(DecisionTreeClassifier(min_samples_leaf=30,random_state=0),Xf,y,cv=cv,method='predict_proba')[:,1],1e-6,1-1e-6)
    Hy=-(y.mean()*np.log2(y.mean())+(1-y.mean())*np.log2(1-y.mean()))
    Hcond=-(y*np.log2(p)+(1-y)*np.log2(1-p)).mean(); return Hy-Hcond, roc_auc_score(y,p)
tri=np.column_stack([np.log1p(Cff[I,J])])[mask]; noid=H[mask][:,:9]  # history minus sameNT
i_full,a_full=pinfo(Xh); i_tri,a_tri=pinfo(tri); i_noid,a_noid=pinfo(noid)
P(f"[PREDICTIVE INFO bits/pair] triadic-only={i_tri:.3f}(AUC{a_tri:.3f}) | full-history-NO-identity={i_noid:.3f}(AUC{a_noid:.3f}) | +identity(sameNT)={i_full:.3f}(AUC{a_full:.3f})")
imp=sorted(zip(fn,tree.feature_importances_),key=lambda x:-x[1])
P(f"[does machine use IDENTITY?] tree feature importance: {[(f,round(v,3)) for f,v in imp[:6]]}")
P(f"  sameNT importance = {dict(imp)['sameNT']:.4f} | identity ΔinfoBits = {i_full-i_noid:+.4f} (>0 & sig => identity load-bearing)")
