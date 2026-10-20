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
# (1) assumption-free: source ligands + target receptors
WV=np.column_stack([Lvec[I],Rvec[J]])
# (2) explicit high-confidence wireless network
pair={'Dopamine':['DOP-1','DOP-2','DOP-3','DOP-4'],'Serotonin':['MOD-1','SER-1','SER-4','SER-5','SER-7'],
 'Tyramine':['TYRA-2','TYRA-3','SER-2','LGC-55','LGC-53'],'Octopamine':['OCTR-1','SER-3','SER-6'],
 'PDF-1':['PDFR-1'],'PDF-2':['PDFR-1'],'NLP-12':['CKR-2'],'NTC-1':['NTR-1','NTR-2'],
 'FLP-18':['NPR-1','NPR-4','NPR-5'],'FLP-21':['NPR-1'],'FLP-1':['NPR-4','NPR-11'],'NLP-1':['NPR-11']}
W=np.zeros((M,M))
for L,Rs in pair.items():
    if L not in li: continue
    src=np.where(Lvec[:,li[L]]>0)[0]
    for R in Rs:
        if R in ri:
            tgt=np.where(Rvec[:,ri[R]]>0)[0]
            for a in src:
                for b in tgt:
                    if a!=b: W[a,b]+=1
wedge=np.array([W[I[k],J[k]] for k in range(N)]); wedge_rev=np.array([W[J[k],I[k]] for k in range(N)])
EXP=np.column_stack([wedge,wedge_rev,(wedge+wedge_rev)])
nf={n:i%5 for i,n in enumerate(sorted(set([neurons[i] for i in I]+[neurons[j] for j in J])))}
def nho(Xf):
    pred=np.full(N,np.nan)
    for f in range(5):
        te=[k for k in range(N) if nf[neurons[I[k]]]==f and nf[neurons[J[k]]]==f];tr=[k for k in range(N) if nf[neurons[I[k]]]!=f and nf[neurons[J[k]]]!=f]
        pred[np.array(te)]=GradientBoostingClassifier(random_state=0).fit(Xf[tr],yrep[tr]).predict_proba(Xf[te])[:,1]
    m=~np.isnan(pred); return roc_auc_score(yrep[m],pred[m]),pred
a0,p0=nho(BM); hard=np.abs(r1)>0.3; FP=(r1<-0.3) # co-fasciculate-but-dont-wire (surprising non-connections)
P(f"base+katz+mirror AUC={a0:.3f} | explicit wireless edges: {(wedge>0).sum()} directed pairs have a curated ligand-receptor link")
for lab,F in [("(1) assumption-free ligand/receptor vectors",WV),("(2) explicit wireless-network edges",EXP)]:
    a1,p1=nho(np.hstack([BM,F])); imp=(np.abs(yrep[hard]-p1[hard])<np.abs(yrep[hard]-p0[hard])-1e-6).mean()
    P(f"  +{lab:44s} AUC={a1:.3f} (Δ{a1-a0:+.3f}) | hard-968 improved {imp*100:.0f}%")
# KEY: do the 390 touch-but-dont-wire pairs talk WIRELESSLY more than baseline?
allw=(wedge>0)|(wedge_rev>0)
P(f"\n[co-fasciculate-but-DONT-synapse pairs, n={FP.sum()}] wireless-linked rate = {allw[FP].mean():.3f} vs all-contacting {allw.mean():.3f} vs wired-core {allw[yrep==1].mean():.3f}")
P(f"  (hypothesis: if they signal wirelessly INSTEAD of synaptically, FP rate >> baseline)")
