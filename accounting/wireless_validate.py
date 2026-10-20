import sys, csv, numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); z2=np.load(f"{SC}/resid2.npz")
I,J,y7,y8,base,katz,neurons=z['I'],z['J'],z['y7'],z['y8'],z['base'],z['katz'],list(z['neurons'])
N=len(I); yrep=((y7==1)&(y8==1)).astype(int); idx={n:i for i,n in enumerate(neurons)}; M=len(neurons)
BM=np.hstack([base,katz,z2['MF']]); r1=z2['r1']
rows=list(csv.reader(open("../data/celegans/data/expression/bentley2016/Bentley_et_al_2016_expression.csv")))[1:]
peps=sorted(set(r[2] for r in rows if r[1]=='Neuropeptide')); mons=sorted(set(r[2] for r in rows if r[1]=='Neurotransmitter')); recs=sorted(set(r[2] for r in rows if r[1]=='Receptor'))
ligs=peps+mons; li={l:k for k,l in enumerate(ligs)}; ri={r:k for k,r in enumerate(recs)}
Lvec=np.zeros((M,len(ligs))); Rvec=np.zeros((M,len(recs)))
for nu,rel,ent,*_ in rows:
    if nu in idx:
        if rel in ('Neuropeptide','Neurotransmitter') and ent in li: Lvec[idx[nu],li[ent]]=1
        elif rel=='Receptor' and ent in ri: Rvec[idx[nu],ri[ent]]=1
pair={'Dopamine':['DOP-1','DOP-2','DOP-3','DOP-4'],'Serotonin':['MOD-1','SER-1','SER-4','SER-5','SER-7'],
 'Tyramine':['TYRA-2','TYRA-3','SER-2','LGC-55','LGC-53'],'Octopamine':['OCTR-1','SER-3','SER-6'],
 'PDF-1':['PDFR-1'],'PDF-2':['PDFR-1'],'NLP-12':['CKR-2'],'NTC-1':['NTR-1','NTR-2'],
 'FLP-18':['NPR-1','NPR-4','NPR-5'],'FLP-21':['NPR-1'],'FLP-1':['NPR-4','NPR-11'],'NLP-1':['NPR-11']}
Lm=np.zeros((M,len(recs)))  # ligand-implied receptor reachability per neuron: does neuron express a ligand whose receptor is R
def buildW(Lv,Rv):
    W=np.zeros((M,M))
    for L,Rs in pair.items():
        if L not in li: continue
        s=Lv[:,li[L]]>0
        for R in Rs:
            if R in ri: W+=np.outer(s,Rv[:,ri[R]]>0)
    np.fill_diagonal(W,0); return W
W=buildW(Lvec,Rvec); wedge=W[I,J]; wrev=W[J,I]; allw=((wedge>0)|(wrev>0))
FP=(r1<-0.3)
obs=allw[FP].mean(); base_rate=allw.mean()
# node-permutation null (shuffle which neuron has which repertoire; ligands+receptors move together)
rng=np.random.default_rng(0); nl=[]
for _ in range(300):
    p=rng.permutation(M); Wp=buildW(Lvec[p],Rvec[p]); ap=((Wp[I,J]>0)|(Wp[J,I]>0)); nl.append(ap[FP].mean())
nl=np.array(nl); P(f"[ENRICHMENT] co-fasc-no-synapse wireless rate obs={obs:.3f} (baseline {base_rate:.3f}) vs node-perm null {nl.mean():.3f}±{nl.std():.3f}  z={(obs-nl.mean())/nl.std():.1f}  p={(nl>=obs).mean():.3f}")
# also for surprising CONNECTIONS
FN=(r1>0.3); P(f"[compare] surprising-CONNECTIONS wireless rate={allw[FN].mean():.3f} | wired-core={allw[yrep==1].mean():.3f} | non-core={allw[yrep==0].mean():.3f}")
# perm control on the +0.004 assumption-free lift
nf={n:i%5 for i,n in enumerate(sorted(set([neurons[i] for i in I]+[neurons[j] for j in J])))}
def nho(Xf):
    pred=np.full(N,np.nan)
    for f in range(5):
        te=[k for k in range(N) if nf[neurons[I[k]]]==f and nf[neurons[J[k]]]==f];tr=[k for k in range(N) if nf[neurons[I[k]]]!=f and nf[neurons[J[k]]]!=f]
        pred[np.array(te)]=GradientBoostingClassifier(random_state=0).fit(Xf[tr],yrep[tr]).predict_proba(Xf[te])[:,1]
    m=~np.isnan(pred); return roc_auc_score(yrep[m],pred[m])
WV=np.column_stack([Lvec[I],Rvec[J]]); a0=nho(BM); areal=nho(np.hstack([BM,WV]))
pl=[]
for s in range(4):
    p=rng.permutation(M); pl.append(nho(np.hstack([BM,np.column_stack([Lvec[p][I],Rvec[p][J]])]))-a0)
pl=np.array(pl); P(f"[LIFT] ligand/receptor real Δ={areal-a0:+.3f} vs node-shuffled {pl.mean():+.3f}±{pl.std():.3f} -> {'REAL' if (areal-a0)>pl.mean()+2*pl.std() else 'NOT beating null'}")
