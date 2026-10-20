import sys, csv, re, json, numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
DE="../data/celegans/New Notebooks/data_derived"
D="../data/celegans/data/connectome"
import openpyxl
wb=openpyxl.load_workbook(f"{D}/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx",read_only=True,data_only=True)
data=list(wb['adult nerve ring neighbors'].iter_rows(values_only=True)); neurons=[h for h in data[0][1:] if isinstance(h,str)]; idx={n:i for i,n in enumerate(neurons)}; M=len(neurons)
A=np.zeros((M,M))
for row in data[1:]:
    if row[0] in idx:
        for j,v in enumerate(row[1:]):
            if v not in (None,''):
                try: A[idx[row[0]],j]=float(v)
                except: pass
A=np.maximum(A,A.T)
# centroids from skeletons (adult ds7,8)
cent={}
for dsn in (7,8):
    sk=json.load(open(f"../seed/data/external/mechanistic_frontier/witvliet2021_developmental/data/skeletons/Dataset{dsn}_skeletons.json"))
    for n,v in sk.items():
        if n in idx and v.get('coords'): cent.setdefault(n,[]).append(np.array(list(v['coords'].values()),float).mean(0))
POS=np.full((M,3),np.nan)
for n,v in cent.items(): POS[idx[n]]=np.mean(v,0)
POS=np.nan_to_num(POS,nan=np.nanmean(POS))
# ADHESION/FASCICULATION TOOLKIT genes (specific, not generic)
ez=np.load(f"{DE}/expression_tpm.npz",allow_pickle=True); exprN=[str(x) for x in ez['neurons']]; tpm=ez['tpm']; wbg=[str(x) for x in ez['genes_wbg']]; erow={n:k for k,n in enumerate(exprN)}
name={}
for row in csv.reader(open(f"{DE}/expression_genes.csv")):
    if len(row)>=2: name[row[0]]=row[1]
gn=[name.get(g,g) for g in wbg]
TK=('sax-7','fmi-1','rig-','hmr-1','cdh-','casy-1','nlg-1','nrx-1','sax-3','lad-2','wrk-1','zig-','oig-','igcm-','dma-1','sdn-1','ptp-3','unc-40','unc-5','unc-6','slt-1','plx-','mab-20','smp-','vab-1','efn-','cam-1','ina-1','lron-','madd-4','him-4','ncam-')
cols=[c for c,g in enumerate(gn) if any(g.startswith(t) for t in TK)]
P(f"adhesion/fasciculation TOOLKIT genes found: {len(cols)} e.g. {sorted(set(gn[c] for c in cols))[:14]}")
Xe=np.log1p(tpm[:,cols]); Xe=np.nan_to_num((Xe-Xe.mean(0))/(Xe.std(0)+1e-9))
E={n:Xe[erow[n]] for n in neurons if n in erow}; z0=np.zeros(len(cols))
Etk=np.array([E.get(n,z0) for n in neurons])
Ptk=PCA(10,random_state=0).fit_transform(Etk)  # toolkit expression PCs per neuron
# ============ TEST A: does TOOLKIT predict CONTACT beyond 3D position? (professor's core claim) ============
ii,jj=np.triu_indices(M,1); dist=np.linalg.norm(POS[ii]-POS[jj],axis=1); contact=(A[ii,jj]>0).astype(int)
POSF=np.column_stack([dist,POS[ii],POS[jj]])
TKF=np.column_stack([Ptk[ii],Ptk[jj],np.sum(Etk[ii]*Etk[jj],1)/(np.linalg.norm(Etk[ii],axis=1)*np.linalg.norm(Etk[jj],axis=1)+1e-9)])  # src+tgt toolkit + homophilic cos
cv=StratifiedKFold(5,shuffle=True,random_state=0)
def auc(X,y): return roc_auc_score(y,cross_val_predict(GradientBoostingClassifier(random_state=0),X,y,cv=cv,method='predict_proba')[:,1])
aP=auc(POSF,contact); aPT=auc(np.hstack([POSF,TKF]),contact)
P(f"\n[TEST A: predict CONTACT among all {len(contact)} neuron pairs, rate {contact.mean():.2f}]")
P(f"  3D-position AUC={aP:.3f} -> +adhesion-toolkit AUC={aPT:.3f} (Δ{aPT-aP:+.3f})")
rng=np.random.default_rng(0); nl=[auc(np.hstack([POSF,TKF[rng.permutation(len(ii))]]),contact)-aP for _ in range(4)]
P(f"  [perm toolkit] real Δ={aPT-aP:+.3f} vs shuffled {np.mean(nl):+.3f}±{np.std(nl):.3f} -> {'toolkit predicts contact beyond geometry' if (aPT-aP)>np.mean(nl)+2*np.std(nl) else 'toolkit adds ~nothing over geometry (contact = geometry-dominated)'}")
# ============ TEST B: does HETEROPHILIC toolkit interaction predict the SYNAPTIC RESIDUAL? ============
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); m0=np.load(f"{SC}/mode0.npz")
I,J,base=z['I'],z['J'],z['base']; N=len(I); FP=m0['FP']; FN=m0['FN']; agree=m0['agree']; area=base[:,0]
GL1=np.zeros((M,M))
with open(f"{D}/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']=='chemical':
            d=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
            if d in (1,2,3,4) and r['pre'] in idx and r['post'] in idx: GL1[idx[r['pre']],idx[r['post']]]=1
Cff=GL1@GL1
PROC=np.column_stack([area,np.log1p(Cff[I,J]),GL1[I,J]])
# heterophilic: source toolkit PCs x target toolkit PCs (let GB learn A_i x B_j complementary pairing)
HET=np.column_stack([Ptk[I][:,:6],Ptk[J][:,:6],(Etk[I]*Etk[J]).sum(1)/(np.linalg.norm(Etk[I],axis=1)*np.linalg.norm(Etk[J],axis=1)+1e-9)])
nf={n:i%5 for i,n in enumerate(sorted(set([neurons[I[k]] for k in range(N)]+[neurons[J[k]] for k in range(N)])))}
def nho(X,y,mask):
    kk=np.where(mask)[0]; pred=np.full(len(kk),np.nan)
    for f in range(5):
        te=[t for t,k in enumerate(kk) if nf[neurons[I[k]]]==f and nf[neurons[J[k]]]==f];tr=[t for t,k in enumerate(kk) if nf[neurons[I[k]]]!=f and nf[neurons[J[k]]]!=f]
        if len(te)<8 or len(set(y[kk][tr]))<2: continue
        pred[np.array(te)]=GradientBoostingClassifier(random_state=0).fit(X[kk][tr],y[kk][tr]).predict_proba(X[kk][te])[:,1]
    m=~np.isnan(pred); return roc_auc_score(y[kk][m],pred[m])
P(f"\n[TEST B: heterophilic toolkit interaction -> synaptic residual, agree pairs]")
for y,l in [(FN.astype(int),'FN'),(FP.astype(int),'FP')]:
    a0=nho(PROC,y,agree); a1=nho(np.hstack([PROC,HET]),y,agree); P(f"  {l}: process AUC={a0:.3f} -> +toolkit-heterophilic AUC={a1:.3f} (Δ{a1-a0:+.3f})")
P("\n(CAVEAT per professor: adult class-level expression = wrong time (adhesion acts at outgrowth) + wrong resolution; a null here is RESOLUTION-BOUNDED, not 'no code')")
