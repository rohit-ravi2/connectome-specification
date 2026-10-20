#!/usr/bin/env python3
"""How much of the per-edge RESIDUAL (beyond features) is real signal vs 2-animal estimation noise?
Never asked. Use the 4 L1 replicates: split 2+2, estimate residual from each half independently, correlate.
Split-half corr>0 => the beyond-features per-edge propensity is REAL & identifiable (prior white/null tests
used the noisy 2-adult residual => possibly underpowered; re-run on clean version). Corr~0 => residual is
aggregate-only, not per-edge identifiable (a real result: redirect to aggregate structure)."""
import csv, re, os, math
import numpy as np, openpyxl
from collections import defaultdict
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import pearsonr, spearmanr

C="../data/celegans/data"
OUT=os.path.expanduser("../seed/data/processed/relational_propensity_grammar")
EDGES=f"{C}/connectome/witvliet2020/witvliet_combined_edges.csv"
CONTACT=f"{C}/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx"
NPZ="../data/tier5/connectome.npz"
ae=defaultdict(set)
for r in csv.DictReader(open(EDGES)):
    if r['type']=='electrical': continue
    ae[int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))].add((r['pre'],r['post']))
L1=[1,2,3,4]
wb=openpyxl.load_workbook(CONTACT,read_only=True,data_only=True); ws=wb['adult nerve ring neighbors']
rows=list(ws.iter_rows(values_only=True)); alln=[h for h in rows[0][1:]]; ix={n:i for i,n in enumerate(alln) if n}
M=len(alln); A=np.zeros((M,M))
for row in rows[1:]:
    if row[0] in ix:
        for j,v in enumerate(row[1:]):
            if v not in (None,''):
                try: A[ix[row[0]],j]=float(v)
                except: pass
A=np.maximum(A,A.T); Bnb=(A>0).astype(float); deg=A.sum(1)
d=np.load(NPZ,allow_pickle=True)
def ntn(x):
    for k,v in [("Acetyl","ACh"),("Glu","Glu"),("GABA","GABA")]:
        if k in x: return v
    return "mono" if any(m in x for m in["Dopa","Sero","Octo"]) else "Unk"
NT={a:ntn(str(b)) for a,b in zip([str(x) for x in d["names"]],[str(x) for x in d["nt_primary"]])}
pairs=[(alln[i],alln[j]) for i in range(M) for j in range(M) if i!=j and alln[i] and alln[j] and A[i,j]>0]
pres=np.array([[1 if (a,b) in ae[an] else 0 for an in L1] for a,b in pairs],float)  # n x 4
theta=pres.mean(1)
ntk=sorted(set((NT.get(a,'U'),NT.get(b,'U')) for a,b in pairs)); ki={k:i for i,k in enumerate(ntk)}
def feats(a,b):
    i,j=ix[a],ix[b]; cos=float(A[i]@A[j]/(np.linalg.norm(A[i])*np.linalg.norm(A[j])+1e-9)); oh=[0]*len(ntk); oh[ki[(NT.get(a,'U'),NT.get(b,'U'))]]=1
    return [math.log1p(A[i,j]),cos,math.log1p(deg[i]),math.log1p(deg[j]),Bnb[i].sum(),Bnb[j].sum()]+oh
X=np.array([feats(a,b) for a,b in pairs])
# feature propensity p via neuron-disjoint CV (target = theta)
neu=sorted(set(a for a,b in pairs)|set(b for a,b in pairs))
rng=np.random.default_rng(0); fold=np.array_split(rng.permutation(neu),5); nf={n:i for i,ff in enumerate(fold) for n in ff}
pr=[(nf[a],nf[b]) for a,b in pairs]; p=np.full(len(pairs),np.nan)
for k in range(5):
    te=np.array([fa==k and fb==k for fa,fb in pr]); tr=np.array([fa!=k and fb!=k for fa,fb in pr])
    if te.sum()<10: continue
    mdl=make_pipeline(StandardScaler(),GradientBoostingRegressor(random_state=0)); mdl.fit(X[tr],theta[tr]); p[te]=mdl.predict(X[te])
ok=~np.isnan(p); P=p
print(f"contact-positive pairs: {len(pairs)} | covered by neuron-disjoint CV: {ok.sum()}")
print(f"feature model predicts theta_L1: R (pearson) = {pearsonr(theta[ok],P[ok])[0]:.3f}")

# per-animal residual r_i = present_i - p ; split 2+2, correlate residual across halves
resid=pres - P[:,None]   # n x 4
from itertools import combinations
splits=[((0,1),(2,3)),((0,2),(1,3)),((0,3),(1,2))]
print("\n=== SPLIT-HALF RELIABILITY of the beyond-features per-edge RESIDUAL ===")
corrs=[]; theta_corrs=[]
for (ha,hb) in splits:
    rA=resid[:,list(ha)].mean(1); rB=resid[:,list(hb)].mean(1)
    tA=pres[:,list(ha)].mean(1); tB=pres[:,list(hb)].mean(1)
    c=pearsonr(rA[ok],rB[ok])[0]; ct=pearsonr(tA[ok],tB[ok])[0]; corrs.append(c); theta_corrs.append(ct)
    print(f"  split {ha}|{hb}: residual corr={c:+.3f}  (raw theta corr={ct:+.3f})")
r_hh=np.mean(corrs)  # split-half (2v2) reliability
# Spearman-Brown: full 4-animal reliability from 2v2 split-half
rel_full = (2*r_hh)/(1+r_hh) if r_hh>-1 else np.nan
print(f"\n  mean 2v2 split-half residual corr = {r_hh:+.3f}")
print(f"  Spearman-Brown 4-animal residual reliability = {rel_full:+.3f}  (fraction of residual variance that is REAL per-edge signal)")
# permutation null: shuffle rB edges
nullc=[]
for s in range(200):
    pmt=np.random.default_rng(s).permutation(np.where(ok)[0])
    rA=resid[:,[0,1]].mean(1)[ok]; rB=resid[:,[2,3]].mean(1)[ok][np.argsort(np.random.default_rng(s).random(ok.sum()))]
    nullc.append(pearsonr(rA,rB)[0])
nullc=np.array(nullc); z=(corrs[0]-nullc.mean())/nullc.std()
print(f"  permutation null (shuffled edges): corr {nullc.mean():+.3f}±{nullc.std():.3f}  -> real z={z:.1f}")

# WHERE is the residual reliable? stratify split-half by contact quartile
print("\n=== is the residual reliable in a SUBSET? (split-half residual corr by contact quartile) ===")
ca=np.array([A[ix[a],ix[b]] for a,b in pairs]); q=np.quantile(ca[ok],[0.25,0.5,0.75]); qz=np.digitize(ca,q)
for qi in range(4):
    s=ok&(qz==qi)
    if s.sum()>40:
        rA=resid[:,[0,1]].mean(1)[s]; rB=resid[:,[2,3]].mean(1)[s]
        print(f"  contact Q{qi+1} (n={s.sum()}): residual split-half corr = {pearsonr(rA,rB)[0]:+.3f}")
with open(f"{OUT}/split_half_residual.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["metric","value"])
    w.writerow(["residual_2v2_splithalf_corr",round(r_hh,4)]); w.writerow(["residual_4animal_reliability_SB",round(float(rel_full),4)])
    w.writerow(["perm_null_z",round(float(z),2)]); w.writerow(["theta_splithalf_corr",round(float(np.mean(theta_corrs)),4)])
print("\nsplit-half residual reliability done ->",OUT)
