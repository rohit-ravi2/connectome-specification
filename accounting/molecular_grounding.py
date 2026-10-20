import sys, csv, numpy as np
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
DE="../data/celegans/New Notebooks/data_derived"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); z2=np.load(f"{SC}/resid2.npz")
I,J,neurons=z['I'],z['J'],list(z['neurons']); r1=z2['r1']; M=len(neurons)
# per-neuron signed residual bias (out/in/all)
resall=np.array([r1[(I==i)|(J==i)].mean() for i in range(M)])
resout=np.array([r1[I==i].mean() if (I==i).any() else 0 for i in range(M)])
resin =np.array([r1[J==i].mean() if (J==i).any() else 0 for i in range(M)])
# expression
ez=np.load(f"{DE}/expression_tpm.npz",allow_pickle=True); exprN=list(ez['neurons']); tpm=ez['tpm']
erow={n:k for k,n in enumerate(exprN)}
mp={}
for row in list(csv.reader(open(f"{DE}/expression_neuron_mapping.csv")))[1:]:
    mp[row[0]]=row[1]
# map nerve-ring neuron -> class -> expr row; aggregate residual bias to CLASS level
cls={}
for i,n in enumerate(neurons):
    c=mp.get(n); 
    if c and c in erow: cls.setdefault(c,[]).append(i)
classes=sorted(cls); Xe=np.log1p(np.array([tpm[erow[c]] for c in classes]))
def classbias(res): return np.array([np.mean([res[i] for i in cls[c]]) for c in classes])
P(f"classes with expression: {len(classes)} | genes {tpm.shape[1]}")
# variance filter genes, PCA, ridge with class-holdout CV
v=Xe.var(0); Xe=Xe[:,np.argsort(-v)[:3000]]; Xe=(Xe-Xe.mean(0))/(Xe.std(0)+1e-9)
Z=PCA(n_components=30,random_state=0).fit_transform(Xe)
rng=np.random.default_rng(0); kf=KFold(5,shuffle=True,random_state=0)
def cvr2(y):
    pred=np.zeros(len(y))
    for tr,te in kf.split(Z):
        m=Ridge(alpha=10.0).fit(Z[tr],y[tr]); pred[te]=m.predict(Z[te])
    ss=1-((y-pred)**2).sum()/((y-y.mean())**2).sum(); return ss,np.corrcoef(y,pred)[0,1]
for lab,res in [("all-edge bias",classbias(resall)),("OUT (source) bias",classbias(resout)),("IN (target) bias",classbias(resin))]:
    r2,cc=cvr2(res)
    # shuffle null
    nl=[cvr2(res[rng.permutation(len(res))])[0] for _ in range(20)]; nl=np.array(nl)
    P(f"  expr->{lab:18s} CV-R2={r2:+.3f} corr={cc:+.3f} | shuffle-R2 {nl.mean():+.3f}±{nl.std():.3f}  z={(r2-nl.mean())/(nl.std()+1e-9):+.1f}")
