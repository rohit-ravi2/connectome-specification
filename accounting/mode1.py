import sys, csv, re, numpy as np
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); m0=np.load(f"{SC}/mode0.npz"); zs=np.load(f"{SC}/subcellular.npz")
cd=np.load(f"{SC}/contactdyn_raw.npz",allow_pickle=True)
I,J,y7,y8,base,neurons=z['I'],z['J'],z['y7'],z['y8'],z['base'],list(z['neurons']); idx={n:i for i,n in enumerate(neurons)}
M=len(neurons); N=len(I); yrep=((y7==1)&(y8==1)).astype(int); L1c=m0['L1c']; agree=m0['agree']; sub=zs['sub']
P("contactdyn stages:",list(cd['stages']))
# L1 eligibility: in contact at L1 (stage 0 min-dist < 2000nm, not nan)
udk=cd['UD_keys']; udv=cd['UD_vals']; L1md={}
for r in range(len(udk)):
    a,b=int(udk[r][0]),int(udk[r][1]); L1md[(min(a,b),max(a,b))]=udv[r][0]
L1elig=np.array([ (lambda v: (not np.isnan(v)) and v<2000)(L1md.get((min(I[k],J[k]),max(I[k],J[k])),np.nan)) for k in range(N)])
# L1 synapse graph motifs (pre-adult)
dp=defaultdict(set)
with open("../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']!='chemical': continue
        d=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
        if d in (1,2,3,4) and r['pre'] in idx and r['post'] in idx: dp[(idx[r['pre']],idx[r['post']])].add(d)
GL1=np.zeros((M,M))
for (a,b) in dp: GL1[a,b]=1
Cu=((GL1+GL1.T)>0).astype(float); comm=Cu@Cu; Cin=GL1.T@GL1; Cout=GL1@GL1.T; Cff=GL1@GL1
l1deg=Cu.sum(1)
t5=np.load("../data/tier5/connectome.npz",allow_pickle=True); t5nt={str(n):str(v) for n,v in zip(t5["names"],t5["nt_primary"])}
def bcl(n): return n[:-1] if n[-1] in 'LR' else n
nt=[t5nt.get(neurons[i],t5nt.get(bcl(neurons[i]),"U")) for i in range(M)]; sament=np.array([nt[I[k]]==nt[J[k]] for k in range(N)]).astype(float)
# feature blocks
TRIAD=np.column_stack([np.log1p(comm[I,J]),np.log1p(Cin[I,J]),np.log1p(Cout[I,J]),np.log1p(Cff[I,J])])  # local L1 closure
DEG=np.column_stack([base[:,3],base[:,4],l1deg[I],l1deg[J]])
CON=np.column_stack([base[:,0],base[:,1],base[:,2]])
SUB=sub[:,[0,1,2,5,6]]
ALL=np.column_stack([CON,TRIAD,DEG,SUB,sament])
fnames=['area','n_shared','cos','L1comm','L1Cin','L1Cout','L1Cff','cdeg_s','cdeg_t','L1deg_s','L1deg_t','sub_along_s','sub_along_t','sub_dmin','sub_spread_s','sub_spread_t','sameNT']
nf={n:i%5 for i,n in enumerate(sorted(set([neurons[i] for i in I]+[neurons[j] for j in J])))}
def nho(X,mask,y):
    ii=np.where(mask)[0]; pred=np.full(len(ii),np.nan)
    for f in range(5):
        te=[t for t,k in enumerate(ii) if nf[neurons[I[k]]]==f and nf[neurons[J[k]]]==f]
        tr=[t for t,k in enumerate(ii) if nf[neurons[I[k]]]!=f and nf[neurons[J[k]]]!=f]
        if len(te)<8 or len(set(y[tr]))<2: continue
        pred[np.array(te)]=GradientBoostingClassifier(random_state=0).fit(X[ii][tr],y[tr]).predict_proba(X[ii][te])[:,1]
    m=~np.isnan(pred); return roc_auc_score(y[m],pred[m]),m.sum()
# UNGATED (reproduce prior): among L1c<2 contacting pairs
ung=(L1c<2); a_ung,n_ung=nho(ALL,ung,yrep[ung]); P(f"\n[ungated L1c<2] n={ung.sum()} pos={yrep[ung].mean():.2f} AUC={a_ung:.3f} (prior ~0.816)")
# STRICT ELIGIBILITY GATE: un-wired at L1 (L1c==0) AND L1-eligible (contact at L1) AND adults agree
gate=(L1c==0)&L1elig&agree; yg=yrep[gate]
P(f"[GATED L1c==0 & L1-contact-eligible & agree] n={gate.sum()} added-rate={yg.mean():.2f} eligible-frac={L1elig[L1c==0].mean():.2f}")
a_g,n_g=nho(ALL,gate,yg); P(f"  => full pre-adult AUC={a_g:.3f}  (vs ungated {a_ung:.3f})")
# ablations: triadic vs degree vs contact
for lab,X in [("TRIADIC only(L1 closure)",np.column_stack([TRIAD])),("DEGREE only",DEG),("CONTACT only",CON),("SUBCELL only",SUB)]:
    ag,_=nho(X,gate,yg); P(f"     {lab:26s} AUC={ag:.3f}")
# feature importance on gated full model
ii=np.where(gate)[0]; gbc=GradientBoostingClassifier(random_state=0).fit(ALL[ii],yg)
imp=sorted(zip(fnames,gbc.feature_importances_),key=lambda x:-x[1])[:8]
P("  top features (gated full):", [(f,round(v,3)) for f,v in imp])
