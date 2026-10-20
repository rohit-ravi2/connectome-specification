import sys, csv, numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
DE="../data/celegans/New Notebooks/data_derived"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); z2=np.load(f"{SC}/resid2.npz")
I,J,y7,y8,base,katz,neurons=z['I'],z['J'],z['y7'],z['y8'],z['base'],z['katz'],list(z['neurons'])
N=len(I); yrep=((y7==1)&(y8==1)).astype(int); idx={n:i for i,n in enumerate(neurons)}
BM=np.hstack([base,katz,z2['MF']]); r1=z2['r1']
# genes + names
ez=np.load(f"{DE}/expression_tpm.npz",allow_pickle=True); exprN=[str(x) for x in ez['neurons']]; tpm=ez['tpm']; wbg=[str(x) for x in ez['genes_wbg']]; erow={n:k for k,n in enumerate(exprN)}
name={}
for row in list(csv.reader(open(f"{DE}/expression_genes.csv"))):
    if len(row)>=2: name[row[0]]=row[1]
gnames=[name.get(g,g) for g in wbg]
panels={
 'chemoreceptor':('srh-','str-','sri-','srd-','srj-','sre-','srsx','srb','sra','srw-','srz-','srx-','srt-','srg-','sru-','srv-','dmsr','srn-','srr-'),
 'sensory-signaling':('gcy-','odr-','osm-','tax-','che-','ocr-','trp-','egl-4','daf-11','grk-','arr-1','ncs-'),
 'adhesion-guidance':('rig-','zig-','casy','nlg-','nrx-','sax-','unc-5','unc-6','unc-40','madd-4','plx-','ptp-3','oig-','lron','igcm','cdh-','fmi-1','cam-1','ina-1','sdn-1','lad-2','wrk-1','dma-1')}
def panel_cols(pref): return [c for c,gn in enumerate(gnames) if any(gn.startswith(p) for p in pref)]
SENS={'ADF','ADL','AFD','ASE','ASG','ASH','ASI','ASJ','ASK','AWA','AWB','AWC','BAG','URX','IL1','IL2','OLL','OLQ','CEP','ADE','URY','URB','AVM','ALM','PLM','PVD','FLP'}
def isS(n): c=(n[:-1] if n[-1] in 'LR' else n).rstrip('DV'); return any(c.startswith(s) for s in SENS)
senspair=np.array([isS(neurons[I[k]]) or isS(neurons[J[k]]) for k in range(N)])
P(f"sensory-involving pairs: {senspair.sum()}/{N}")
# neuron-holdout restricted to sensory pairs
nf={n:i%5 for i,n in enumerate(sorted(set([neurons[i] for i in I]+[neurons[j] for j in J])))}
def nho(Xf,sub):
    pred=np.full(N,np.nan)
    for f in range(5):
        te=[k for k in range(N) if sub[k] and nf[neurons[I[k]]]==f and nf[neurons[J[k]]]==f];tr=[k for k in range(N) if nf[neurons[I[k]]]!=f and nf[neurons[J[k]]]!=f]
        if len(te)<10: continue
        pred[np.array(te)]=GradientBoostingClassifier(random_state=0).fit(Xf[tr],yrep[tr]).predict_proba(Xf[te])[:,1]
    m=(~np.isnan(pred))&sub; return roc_auc_score(yrep[m],pred[m]),pred,m
def panel_feats(cols):
    have=[n for n in neurons if n in erow]; E={}
    sub=np.log1p(tpm[:,cols]); sub=np.nan_to_num((sub-sub.mean(0))/(sub.std(0)+1e-9))
    for n in have: E[n]=sub[erow[n]]
    dim=len(cols); z0=np.zeros(dim)
    Ei=np.array([E.get(neurons[i],z0) for i in I]); Ej=np.array([E.get(neurons[j],z0) for j in J])
    cos=np.array([Ei[k]@Ej[k]/(np.linalg.norm(Ei[k])*np.linalg.norm(Ej[k])+1e-9) for k in range(N)])
    from sklearn.decomposition import PCA
    k=min(8,dim); Pi=PCA(k,random_state=0).fit(np.vstack([Ei,Ej]))
    return np.column_stack([cos,Pi.transform(Ei),Pi.transform(Ej)])
a0,p0,m0=nho(BM,senspair); P(f"[sensory pairs] base+katz+mirror AUC={a0:.3f}")
hard=(np.abs(r1)>0.3)&senspair
for lab,pref in panels.items():
    cols=panel_cols(pref)
    if not cols: P(f"  {lab}: 0 genes found"); continue
    F=panel_feats(cols); a1,p1,m1=nho(np.hstack([BM,F]),senspair)
    imp=(np.abs(yrep[hard]-p1[hard])<np.abs(yrep[hard]-p0[hard])-1e-6).mean()
    P(f"  +{lab:18s} ({len(cols):3d} genes) AUC={a1:.3f} (Δ{a1-a0:+.3f}) | sensory-hard improved {imp*100:.0f}%")
