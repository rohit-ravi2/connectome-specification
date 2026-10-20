import sys, csv, re, numpy as np
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); m0=np.load(f"{SC}/mode0.npz"); zs=np.load(f"{SC}/subcellular.npz"); cd=np.load(f"{SC}/contactdyn_raw.npz",allow_pickle=True)
I,J,y7,y8,base,neurons=z['I'],z['J'],z['y7'],z['y8'],z['base'],list(z['neurons']); idx={n:i for i,n in enumerate(neurons)}
M=len(neurons); N=len(I); yrep=((y7==1)&(y8==1)).astype(int); L1c=m0['L1c']; agree=m0['agree']; coinflip=m0['coinflip']; sub=zs['sub']
r1=np.load(f"{SC}/resid2.npz",allow_pickle=True)['r1']
udk=cd['UD_keys']; udv=cd['UD_vals']; L1md={}
for r in range(len(udk)): a,b=int(udk[r][0]),int(udk[r][1]); L1md[(min(a,b),max(a,b))]=udv[r][0]
L1elig=np.array([ (lambda v:(not np.isnan(v)) and v<2000)(L1md.get((min(I[k],J[k]),max(I[k],J[k])),np.nan)) for k in range(N)])
G={s:np.zeros((M,M)) for s in ['L1','L2','L3']}
with open("../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']!='chemical': continue
        d=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
        if r['pre'] in idx and r['post'] in idx:
            s={1:'L1',2:'L1',3:'L1',4:'L1',5:'L2',6:'L3'}.get(d)
            if s: G[s][idx[r['pre']],idx[r['post']]]=1
GL1=G['L1']
def paths(Gm,o):
    Pm=Gm.copy()
    for _ in range(o-1): Pm=Pm@Gm
    return Pm
Cff=GL1@GL1; G3=paths(GL1,3); G4=paths(GL1,4)
Cu=((GL1+GL1.T)>0).astype(float); comm=Cu@Cu; Cin=GL1.T@GL1; Cout=GL1@GL1.T
rho=np.abs(np.linalg.eigvals(GL1)).max(); beta=0.5/max(rho,1e-6); Katz=np.linalg.inv(np.eye(M)-beta*GL1)-np.eye(M)
CffL2=G['L2']@G['L2']; CffL3=G['L3']@G['L3']; l1deg=Cu.sum(1)
t5=np.load("../data/tier5/connectome.npz",allow_pickle=True); t5nt={str(n):str(v) for n,v in zip(t5["names"],t5["nt_primary"])}
def bcl(n): return n[:-1] if n[-1] in 'LR' else n
nt=[t5nt.get(neurons[i],t5nt.get(bcl(neurons[i]),"U")) for i in range(M)]; sament=np.array([nt[I[k]]==nt[J[k]] for k in range(N)]).astype(float)
BASE_NC=np.column_stack([base[:,0],base[:,1],base[:,2],base[:,3],base[:,4],l1deg[I],l1deg[J],sub[:,0],sub[:,1],sub[:,2],sub[:,5],sub[:,6],sament])
ONESTEP=np.column_stack([np.log1p(Cff[I,J]),np.log1p(comm[I,J]),np.log1p(Cin[I,J]),np.log1p(Cout[I,J])])
ITER=np.column_stack([np.log1p(G3[I,J]),np.log1p(G4[I,J]),Katz[I,J],Katz[J,I]])
gate=(L1c==0)&L1elig&agree
nf={n:i%5 for i,n in enumerate(sorted(set([neurons[i] for i in I]+[neurons[j] for j in J])))}
def nho(X,mask,y,ret=False):
    ii=np.where(mask)[0]; pred=np.full(len(ii),np.nan)
    for f in range(5):
        te=[t for t,k in enumerate(ii) if nf[neurons[I[k]]]==f and nf[neurons[J[k]]]==f]; tr=[t for t,k in enumerate(ii) if nf[neurons[I[k]]]!=f and nf[neurons[J[k]]]!=f]
        if len(te)<6 or len(set(y[tr]))<2: continue
        pred[np.array(te)]=GradientBoostingClassifier(random_state=0).fit(X[ii][tr],y[tr]).predict_proba(X[ii][te])[:,1]
    m=~np.isnan(pred); auc=roc_auc_score(y[m],pred[m]) if m.sum()>10 and len(set(y[m]))>1 else np.nan
    return (auc,pred,ii) if ret else auc

# ============ GATE A: closure-fit-as-error-correction for PRUNING ============
P("="*70); P("GATE A — do reproducibly-pruned L1 synapses have LOW PRE-ADULT closure-fit?")
P("Pre-register: pruned edges FAIL the closure rule => LOW closure-fit => closure-fit predicts pruning.")
P("Flip: clears iff closure-fit AUC beats degree-matched(rewire) null by >2σ (same gate that killed budget).")
prun=(L1c>=2)&agree; yt=(yrep[prun]==0).astype(int)
P(f"prunable set (L1>=2 & agree) n={prun.sum()} pruned={yt.sum()} stable={(yt==0).sum()}")
def closurefit(Cf,G3m):  # PRE-ADULT closure support for the pair (2-path + 3-path, both directions)
    return np.column_stack([np.log1p(Cf[I,J]),np.log1p(Cf[J,I]),np.log1p(G3m[I,J]),np.log1p(G3m[J,I])])
CFIT=closurefit(Cff,G3)
aCF=nho(CFIT,prun,yt)
# directed degree-preserving rewire of GL1 -> recompute closure-fit -> null AUC
edges=list(zip(*np.where(GL1>0)))
def rewire(seed):
    rng=np.random.default_rng(seed); E=[list(e) for e in edges]; S=set(map(tuple,edges)); ne=len(E)
    for _ in range(5*ne):
        x,y2=rng.integers(0,ne,2)
        if x==y2: continue
        a,b=E[x]; c,d=E[y2]
        if len({a,b,c,d})<4: continue
        if (a,d) not in S and (c,b) not in S:
            S.discard((a,b)); S.discard((c,d)); S.add((a,d)); S.add((c,b)); E[x]=[a,d]; E[y2]=[c,b]
    Gp=np.zeros((M,M));
    for a,b in S: Gp[a,b]=1
    return Gp
nl=[]
for s in range(20):
    Gp=rewire(s); Cfp=Gp@Gp; G3p=paths(Gp,3); nl.append(nho(closurefit(Cfp,G3p),prun,yt))
nl=np.array([x for x in nl if not np.isnan(x)]); zc=(aCF-nl.mean())/nl.std()
P(f"  closure-fit AUC={aCF:.3f} vs degree-rewire null {nl.mean():.3f}±{nl.std():.3f}  z={zc:.1f}")
# direction check: mean closure-fit pruned vs stable
cf_scalar=np.log1p(Cff[I,J]+Cff[J,I])[prun]
P(f"  mean closure-support: pruned={cf_scalar[yt==1].mean():.2f} vs stable={cf_scalar[yt==0].mean():.2f} (pred: pruned LOWER)")
GATEA = "CLEARS -> UNIFIED single-operator generator (closure governs add & prune)" if zc>2 else "FAILS -> two-operator generator (pruning stochastic/external; 3rd pruning framework down)"
P(f"  GATE A: {GATEA}")

# ============ GATE B: does closure GENERALIZE or only patch the FN-578? ============
P("\n"+"="*70); P("GATE B — closure a generative grammar (whole addition set) or FN-578 patch?")
abase,p0,ii=nho(BASE_NC,gate,yrep[gate],ret=True); aceil,p1,_=nho(np.column_stack([BASE_NC,ONESTEP,ITER]),gate,yrep[gate],ret=True)
apure=nho(np.column_stack([ONESTEP,ITER]),gate,yrep[gate])
P(f"  base={abase:.3f} ceiling={aceil:.3f} | PURE-CLOSURE-only AUC on whole gate={apure:.3f} (>>0.5 => predicts additions broadly)")
yy=yrep[ii]; FNg=(r1[ii]>0.3); add=(yy==1)
imp=(np.abs(yy-p1)<np.abs(yy-p0))
mFNadd=add&FNg; mNONadd=add&(~FNg)
P(f"  additions in gate={add.sum()} | FN-additions={mFNadd.sum()} non-FN-additions={mNONadd.sum()}")
P(f"  closure improves prediction: on FN-additions={imp[mFNadd].mean():.2f} | on NON-FN additions={imp[mNONadd].mean():.2f}")
# pure-closure ranking of non-FN additions vs non-additions (does grammar see the un-surprising adds?)
_,ppure,_=nho(np.column_stack([ONESTEP,ITER]),gate,yrep[gate],ret=True)
nonFN=~FNg
try:
    aucNONFN=roc_auc_score(yy[nonFN],ppure[nonFN]) if len(set(yy[nonFN]))>1 else np.nan
except: aucNONFN=np.nan
P(f"  pure-closure AUC restricted to NON-FN pairs (grammar generalization)={aucNONFN:.3f}")
GATEB = "GENERALIZES -> proceed to generator" if (apure>0.65 and aucNONFN>0.6) else "PATCH-ONLY -> closure is a residual patch, NOT a generative grammar; report & stop"
P(f"  GATE B: {GATEB}")
P("\n"+"="*70); P(f"STAGE-0 GATES: A={'CLEAR' if zc>2 else 'FAIL'} | B={'GENERALIZE' if (apure>0.65 and aucNONFN>0.6) else 'PATCH'}")

# ---- Gate B FIX: nan-filtered subset AUCs for pure closure ----
P("\n"+"="*70); P("GATE B (fixed) — pure-closure generalization, nan-filtered")
_,ppure2,ii2=nho(np.column_stack([ONESTEP,ITER]),gate,yrep[gate],ret=True)
yy2=yrep[ii2]; FN2=(r1[ii2]>0.3); ok=~np.isnan(ppure2)
def sub_auc(mask,label):
    m=mask&ok
    if m.sum()>10 and len(set(yy2[m]))>1:
        P(f"  pure-closure AUC [{label}] = {roc_auc_score(yy2[m],ppure2[m]):.3f}  (n={m.sum()}, adds={int(yy2[m].sum())})")
    else: P(f"  [{label}] n={m.sum()} insufficient")
sub_auc(np.ones(len(yy2),bool),"WHOLE gate")
sub_auc(~FN2,"NON-FN pairs (grammar generalization)")
sub_auc(FN2,"FN pairs (the patch subset)")
# among NON-additions + NON-FN additions only: can closure still rank the un-surprising additions?
nonFN_adds_vs_nonadds = (~FN2)  # non-FN pairs: additions(73) vs non-additions
P("  => if NON-FN AUC >0.6, closure predicts un-surprising additions too = GENERALIZES (grammar); ~0.5 = FN patch")
