import sys, numpy as np
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
DE="../data/celegans/New Notebooks/data_derived"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); z2=np.load(f"{SC}/resid2.npz")
I,J,neurons=z['I'],z['J'],list(z['neurons']); r1=z2['r1']; M=len(neurons)
ez=np.load(f"{DE}/expression_tpm.npz",allow_pickle=True); exprN=[str(x) for x in ez['neurons']]; tpm=ez['tpm']
erow={n:k for k,n in enumerate(exprN)}
keep=[i for i in range(M) if neurons[i] in erow]
resall=np.array([r1[(I==i)|(J==i)].mean() for i in keep])
resout=np.array([r1[I==i].mean() if (I==i).any() else 0 for i in keep])
resin =np.array([r1[J==i].mean() if (J==i).any() else 0 for i in keep])
Xe=np.log1p(np.array([tpm[erow[neurons[i]]] for i in keep]))
base=[ (neurons[i][:-1] if neurons[i][-1] in 'LR' else neurons[i]) for i in keep]
ub=sorted(set(base)); bfold={b:k%5 for k,b in enumerate(ub)}; folds=np.array([bfold[b] for b in base])
P(f"neurons with expression: {len(keep)} | classes {len(ub)}")
v=Xe.var(0); Xe=Xe[:,np.argsort(-v)[:3000]]; Xe=np.nan_to_num((Xe-Xe.mean(0))/(Xe.std(0)+1e-9))
rng=np.random.default_rng(0)
def cvr2(y,ncomp=20):
    pred=np.zeros(len(y))
    for f in range(5):
        tr=folds!=f; te=folds==f
        Zt=PCA(n_components=ncomp,random_state=0).fit(Xe[tr])
        m=Ridge(alpha=10.0).fit(Zt.transform(Xe[tr]),y[tr]); pred[te]=m.predict(Zt.transform(Xe[te]))
    ss=1-((y-pred)**2).sum()/((y-y.mean())**2).sum(); return ss,np.corrcoef(y,pred)[0,1]
for lab,res in [("all-edge bias",resall),("OUT(source) bias",resout),("IN(target) bias",resin)]:
    r2,cc=cvr2(res); nl=np.array([cvr2(res[rng.permutation(len(res))])[0] for _ in range(15)])
    P(f"  expr->{lab:16s} class-holdout CV-R2={r2:+.3f} corr={cc:+.3f} | shuffle {nl.mean():+.3f}±{nl.std():.3f} z={(r2-nl.mean())/(nl.std()+1e-9):+.1f}")
# also: does expression SIMILARITY predict pair residual? (complementary/homophilic molecular code on the residual)
Ei={neurons[i]:Xe[k] for k,i in enumerate(keep)}
pairs=[k for k in range(len(I)) if neurons[I[k]] in Ei and neurons[J[k]] in Ei]
sim=np.array([Ei[neurons[I[k]]]@Ei[neurons[J[k]]]/(np.linalg.norm(Ei[neurons[I[k]]])*np.linalg.norm(Ei[neurons[J[k]]])+1e-9) for k in pairs])
P(f"  corr(pair residual r1, expression cosine-similarity) = {np.corrcoef(r1[pairs],sim)[0,1]:+.3f}  (n={len(pairs)})")
