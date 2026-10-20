import sys, csv, re, numpy as np
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); m0=np.load(f"{SC}/mode0.npz"); zs=np.load(f"{SC}/subcellular.npz"); cd=np.load(f"{SC}/contactdyn_raw.npz",allow_pickle=True)
I,J,y7,y8,base,neurons=z['I'],z['J'],z['y7'],z['y8'],z['base'],list(z['neurons']); idx={n:i for i,n in enumerate(neurons)}
M=len(neurons); N=len(I); yrep=((y7==1)&(y8==1)).astype(int); L1c=m0['L1c']; agree=m0['agree']; coinflip=m0['coinflip']; sub=zs['sub']
udk=cd['UD_keys']; udv=cd['UD_vals']; L1md={}
for r in range(len(udk)): a,b=int(udk[r][0]),int(udk[r][1]); L1md[(min(a,b),max(a,b))]=udv[r][0]
L1elig=np.array([ (lambda v:(not np.isnan(v)) and v<2000)(L1md.get((min(I[k],J[k]),max(I[k],J[k])),np.nan)) for k in range(N)])
# per-stage synapse graphs
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
    P_=Gm.copy()
    for _ in range(o-1): P_=P_@Gm
    return P_
Cff=GL1@GL1; G3=paths(GL1,3); G4=paths(GL1,4)
Cu=((GL1+GL1.T)>0).astype(float); comm=Cu@Cu; Cin=GL1.T@GL1; Cout=GL1@GL1.T
rho=np.abs(np.linalg.eigvals(GL1)).max(); beta=0.5/max(rho,1e-6); Katz=np.linalg.inv(np.eye(M)-beta*GL1)-np.eye(M)
CffL2=G['L2']@G['L2']; CffL3=G['L3']@G['L3']
l1deg=Cu.sum(1)
t5=np.load("../data/tier5/connectome.npz",allow_pickle=True); t5nt={str(n):str(v) for n,v in zip(t5["names"],t5["nt_primary"])}
def bcl(n): return n[:-1] if n[-1] in 'LR' else n
nt=[t5nt.get(neurons[i],t5nt.get(bcl(neurons[i]),"U")) for i in range(M)]; sament=np.array([nt[I[k]]==nt[J[k]] for k in range(N)]).astype(float)
BASE_NC=np.column_stack([base[:,0],base[:,1],base[:,2],base[:,3],base[:,4],l1deg[I],l1deg[J],sub[:,0],sub[:,1],sub[:,2],sub[:,5],sub[:,6],sament])
ONESTEP=np.column_stack([np.log1p(Cff[I,J]),np.log1p(comm[I,J]),np.log1p(Cin[I,J]),np.log1p(Cout[I,J])])
ITER=np.column_stack([np.log1p(G3[I,J]),np.log1p(G4[I,J]),Katz[I,J],Katz[J,I]])
STAGE=np.column_stack([np.log1p(CffL2[I,J]),np.log1p(CffL3[I,J]),G['L2'][I,J],G['L3'][I,J]])
gate=(L1c==0)&L1elig&agree; yg=yrep[gate]
nf={n:i%5 for i,n in enumerate(sorted(set([neurons[i] for i in I]+[neurons[j] for j in J])))}
def nho(X,mask=gate,y=None,ret=False):
    if y is None: y=yrep[mask]
    ii=np.where(mask)[0]; pred=np.full(len(ii),np.nan)
    for f in range(5):
        te=[t for t,k in enumerate(ii) if nf[neurons[I[k]]]==f and nf[neurons[J[k]]]==f]; tr=[t for t,k in enumerate(ii) if nf[neurons[I[k]]]!=f and nf[neurons[J[k]]]!=f]
        if len(te)<8 or len(set(y[tr]))<2: continue
        pred[np.array(te)]=GradientBoostingClassifier(random_state=0).fit(X[ii][tr],y[tr]).predict_proba(X[ii][te])[:,1]
    m=~np.isnan(pred); return (roc_auc_score(y[m],pred[m]),pred,ii) if ret else roc_auc_score(y[m],pred[m])
P(f"clean addition target: gate n={gate.sum()} added={yg.sum()} ({yg.mean()*100:.1f}%)")
P("\n=== PRIMARY: ceiling ===")
ceil,pc,iic=nho(np.column_stack([BASE_NC,ONESTEP,ITER]),ret=True)
P(f"  BASE_NC+one-step+iterated (ceiling) AUC={ceil:.3f}")
P(f"  BASE_NC alone={nho(BASE_NC):.3f} | +closure(all)={ceil:.3f}")
P("\n=== FORK 1: one-step vs iterated closure ===")
a1=nho(np.column_stack([BASE_NC,ONESTEP])); ai=nho(np.column_stack([BASE_NC,ONESTEP,ITER]))
P(f"  BASE_NC+ONE-STEP(o2) AUC={a1:.3f} | +ITERATED(o3,o4,Katz) AUC={ai:.3f} (Δ{ai-a1:+.3f})")
# perm control on iterated-beyond-onestep (shuffle L1 graph nodes)
rng=np.random.default_rng(0); nl=[]
for s in range(6):
    p=rng.permutation(M); Gp=GL1[np.ix_(p,p)]; G3p=paths(Gp,3);G4p=paths(Gp,4)
    rp=np.abs(np.linalg.eigvals(Gp)).max(); bp=0.5/max(rp,1e-6); Kp=np.linalg.inv(np.eye(M)-bp*Gp)-np.eye(M)
    ITp=np.column_stack([np.log1p(G3p[I,J]),np.log1p(G4p[I,J]),Kp[I,J],Kp[J,I]])
    nl.append(nho(np.column_stack([BASE_NC,ONESTEP,ITp]))-a1)
nl=np.array(nl); P(f"  [PERM iterated] real Δ={ai-a1:+.3f} vs shuffled {nl.mean():+.3f}±{nl.std():.3f} -> {'ITERATION REAL(trajectory)' if (ai-a1)>nl.mean()+2*nl.std() else 'one-step MARKOVIAN (iteration adds ~0)'}")
aS=nho(np.column_stack([BASE_NC,ONESTEP,STAGE])); P(f"  +real-stage L2/L3 closure AUC={aS:.3f} (Δ over one-step {aS-a1:+.3f}) [NOTE: L2/L3 temporally near adult=leak-prone]")
P("\n=== PRIMARY residual characterization ===")
# refit on ALL eligible-un-wired (incl coinflip), see where coinflip preds land
gall=(L1c==0)&L1elig; _,pa,iia=nho(np.column_stack([BASE_NC,ONESTEP,ITER]),mask=gall,y=yrep[gall],ret=True)
st=np.array(['add-agree' if (agree[k] and yrep[k]==1) else ('coinflip' if coinflip[k] else 'noadd-agree') for k in iia])
for lab in ['add-agree','coinflip','noadd-agree']:
    mm=(st==lab)&(~np.isnan(pa)); P(f"  mean rule-prediction [{lab:11s}] = {pa[mm].mean():.3f} (n={mm.sum()})")
P("  (if coinflip sits BETWEEN add & noadd => rule treats them as borderline = they ARE the stochastic residual, not missed structure)")
# residual structure on agree gate: per-neuron concentration of rule errors vs white
res=np.abs(yg[~np.isnan(pc)]-pc[~np.isnan(pc)]); kk=iic[~np.isnan(pc)]
resn=np.array([res[[t for t,k in enumerate(kk) if I[k]==nn or J[k]==nn]].mean() if any(I[k]==nn or J[k]==nn for k in kk) else np.nan for nn in range(M)])
vr=np.nanvar(resn); vn=[]
for _ in range(50):
    rp=res[rng.permutation(len(res))]; vn.append(np.nanvar(np.array([rp[[t for t,k in enumerate(kk) if I[k]==nn or J[k]==nn]].mean() if any(I[k]==nn or J[k]==nn for k in kk) else np.nan for nn in range(M)])))
vn=np.array(vn); P(f"  rule-residual per-neuron concentration: real var={vr:.4f} vs shuffle {vn.mean():.4f}±{vn.std():.4f} z={(vr-vn.mean())/vn.std():.1f} -> {'STRUCTURED (another rule underneath)' if (vr-vn.mean())>2*vn.std() else 'WHITE (rule explains deterministic addition; rest stochastic)'}")
