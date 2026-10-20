import sys, csv, numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
DE="../data/celegans/New Notebooks/data_derived"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); z2=np.load(f"{SC}/resid2.npz")
I,J,y7,y8,base,katz,neurons=z['I'],z['J'],z['y7'],z['y8'],z['base'],z['katz'],list(z['neurons'])
N=len(I); yrep=((y7==1)&(y8==1)).astype(int); idx={n:i for i,n in enumerate(neurons)}
BM=np.hstack([base,katz,z2['MF']]); r1=z2['r1']
ez=np.load(f"{DE}/expression_tpm.npz",allow_pickle=True); exprN=[str(x) for x in ez['neurons']]; tpm=ez['tpm']; wbg=[str(x) for x in ez['genes_wbg']]; erow={n:k for k,n in enumerate(exprN)}
name={}
for row in list(csv.reader(open(f"{DE}/expression_genes.csv"))):
    if len(row)>=2: name[row[0]]=row[1]
gnames=[name.get(g,g) for g in wbg]
panels={
 'neuropeptides(NLP/FLP/INS)':('flp-','nlp-','ins-','pdf-','ntc-','snet-','trh-','nlp'),
 'neuropep-receptors':('npr-','frpr-','dmsr-','seb-','gnrr-','tkr-','ckr-','egl-6','trhr-','aex-2','pdfr-'),
 'monoamine(synth+recep)':('cat-','tph-','bas-1','tbh-','tdc-','mod-','dop-','ser-','tyra-','octr-','lgc-55','ptr-','cat-1','cat-2','cat-4')}
def cols_of(pref): return [c for c,gn in enumerate(gnames) if any(gn.startswith(p) for p in pref)]
SENS={'ADF','ADL','AFD','ASE','ASG','ASH','ASI','ASJ','ASK','AWA','AWB','AWC','BAG','URX','IL1','IL2','OLL','OLQ','CEP','ADE','URY','URB','AVM','ALM','PLM','PVD','FLP'}
def isS(n): c=(n[:-1] if n[-1] in 'LR' else n).rstrip('DV'); return any(c.startswith(s) for s in SENS)
senspair=np.array([isS(neurons[I[k]]) or isS(neurons[J[k]]) for k in range(N)])
nf={n:i%5 for i,n in enumerate(sorted(set([neurons[i] for i in I]+[neurons[j] for j in J])))}
def nho(Xf):
    pred=np.full(N,np.nan)
    for f in range(5):
        te=[k for k in range(N) if nf[neurons[I[k]]]==f and nf[neurons[J[k]]]==f];tr=[k for k in range(N) if nf[neurons[I[k]]]!=f and nf[neurons[J[k]]]!=f]
        pred[np.array(te)]=GradientBoostingClassifier(random_state=0).fit(Xf[tr],yrep[tr]).predict_proba(Xf[te])[:,1]
    m=~np.isnan(pred); return roc_auc_score(yrep[m],pred[m]),pred
def feats(cols):
    sub=np.log1p(tpm[:,cols]); sub=np.nan_to_num((sub-sub.mean(0))/(sub.std(0)+1e-9))
    E={n:sub[erow[n]] for n in neurons if n in erow}; z0=np.zeros(len(cols))
    Ei=np.array([E.get(neurons[i],z0) for i in I]); Ej=np.array([E.get(neurons[j],z0) for j in J])
    cos=np.array([Ei[k]@Ej[k]/(np.linalg.norm(Ei[k])*np.linalg.norm(Ej[k])+1e-9) for k in range(N)])
    k=min(8,len(cols)); Pi=PCA(k,random_state=0).fit(np.vstack([Ei,Ej]))
    return np.column_stack([cos,Pi.transform(Ei),Pi.transform(Ej)])
a0,p0=nho(BM); hard=np.abs(r1)>0.3; shard=hard&senspair
P(f"base+katz+mirror AUC={a0:.3f} | hard-968={hard.sum()} sensory-hard={shard.sum()}")
allcols=[]
for lab,pref in panels.items():
    cols=cols_of(pref); allcols+=cols
    ex=[gnames[c] for c in cols[:6]]
    F=feats(cols); a1,p1=nho(np.hstack([BM,F]))
    imp=(np.abs(yrep[hard]-p1[hard])<np.abs(yrep[hard]-p0[hard])-1e-6).mean()
    simp=(np.abs(yrep[shard]-p1[shard])<np.abs(yrep[shard]-p0[shard])-1e-6).mean()
    P(f"  +{lab:26s} ({len(cols):3d} genes, e.g. {ex}) AUC={a1:.3f} (Δ{a1-a0:+.3f}) | hard {imp*100:.0f}% sens-hard {simp*100:.0f}%")
allcols=sorted(set(allcols)); F=feats(allcols); a1,p1=nho(np.hstack([BM,F]))
imp=(np.abs(yrep[hard]-p1[hard])<np.abs(yrep[hard]-p0[hard])-1e-6).mean()
P(f"  +ALL neuromodulator ({len(allcols)} genes) AUC={a1:.3f} (Δ{a1-a0:+.3f}) | hard {imp*100:.0f}%")
