#!/usr/bin/env python3
"""STAGE-0 HARDENING CONTROLS (hard gate for generative parts). Two controls:
(1) L1-REPLICATE baseline: dispersion ratio + present-count distribution across the 4 true L1 replicates
    (richer than 2 adults). Tests under- vs over-dispersion with real same-stage replicates + 0.5 ceiling.
(2) OBSERVATION-WINDOW control: edges appearing early are observed at more timepoints. Does formation
    stage (labeled from datasets 1-6) predict W7/W8 disagreement BEYOND confirmation count (# of datasets
    1-6 the edge appears in)? If the timing effect vanishes controlling for observation count => artifact.
DECISION: crossover (two-regime) vs gradient (one-field). If crossover fails => target=GRADIENT, two-object FORBIDDEN."""
import csv, re, os, math
import numpy as np, openpyxl
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

C="../data/celegans/data"
OUT=os.path.expanduser("../seed/data/processed/relational_propensity_grammar")
EDGES=f"{C}/connectome/witvliet2020/witvliet_combined_edges.csv"
CONTACT=f"{C}/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx"
NPZ="../data/tier5/connectome.npz"
ae=defaultdict(set)
for r in csv.DictReader(open(EDGES)):
    if r['type']=='electrical': continue
    ae[int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))].add((r['pre'],r['post']))
L1a=[1,2,3,4]; W7,W8=ae[7],ae[8]
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
    return "mono" if any(m in x for m in["Dopa","Sero","Octo","Tyra"]) else "Unk"
NT={a:ntn(str(b)) for a,b in zip([str(x) for x in d["names"]],[str(x) for x in d["nt_primary"]])}
pairs=[(alln[i],alln[j]) for i in range(M) for j in range(M) if i!=j and alln[i] and alln[j] and A[i,j]>0]
ntk=sorted(set((NT.get(a,'Unk'),NT.get(b,'Unk')) for a,b in pairs)); ki={k:i for i,k in enumerate(ntk)}
def feats(a,b):
    i,j=ix[a],ix[b]; cos=float(A[i]@A[j]/(np.linalg.norm(A[i])*np.linalg.norm(A[j])+1e-9)); oh=[0]*len(ntk); oh[ki[(NT.get(a,'Unk'),NT.get(b,'Unk'))]]=1
    return [math.log1p(A[i,j]),cos,math.log1p(deg[i]),math.log1p(deg[j])]+oh
X=np.array([feats(a,b) for a,b in pairs])
neu=sorted(set(a for a,b in pairs)|set(b for a,b in pairs))
rng=np.random.default_rng(0); f=np.array_split(rng.permutation(neu),5); nf={n:i for i,ff in enumerate(f) for n in ff}
def ndcv(y):
    y=np.array(y); p=np.full(len(y),np.nan)
    for k in range(5):
        te=np.array([nf[a]==k and nf[b]==k for a,b in pairs]); tr=np.array([nf[a]!=k and nf[b]!=k for a,b in pairs])
        if te.sum()<10 or y[tr].sum()<5: continue
        clf=make_pipeline(StandardScaler(with_mean=False),GradientBoostingClassifier(random_state=0)); clf.fit(X[tr],y[tr]); p[te]=clf.predict_proba(X[te])[:,1]
    return p

# ===== CONTROL 1: L1-replicate baseline (4 true same-stage replicates) =====
pres=np.array([[1 if (a,b) in ae[an] else 0 for an in L1a] for a,b in pairs])  # n x 4
k=pres.sum(1)
# per-pair propensity at L1 (avg of per-animal CV predictions)
pL1=np.nanmean(np.vstack([ndcv([1 if (a,b) in ae[an] else 0 for a,b in pairs]) for an in L1a]),axis=0); m=~np.isnan(pL1)
# dispersion: mean pairwise disagreement over 6 animal-pairs vs 2p(1-p)
pair_dis=k*(4-k)/6.0   # fraction of the 6 pairs that disagree
obs=pair_dis[m].sum(); pred=(2*pL1[m]*(1-pL1[m])).sum()
print("=== CONTROL 1: L1-REPLICATE baseline (4 true same-stage animals) ===")
print(f"  L1 dispersion ratio (4-replicate) = {obs/pred:.3f}  (<1 under-dispersed; reproduces adult ~0.54?)")
print(f"  mean pairwise disagreement among 4 L1 animals = {pair_dis[m].mean():.3f}  (ceiling 0.5 under independence)")
kd=[int((k[m]==kk).sum()) for kk in range(5)]
print(f"  present-count distribution k=0..4: {kd}  (bimodal k0/k4=canalized; excess k=2=over-dispersed regime)")
# over-dispersion test: is variance of k GREATER than Binomial(4,theta) expectation?
theta=k/4.0; binom_var=(4*theta*(1-theta)).mean(); obs_var=((k-4*theta)**2).mean()  # ~0 by construction; use cross-pair
print(f"  0.5-ceiling breach among high-contact L1 pairs? mean-disagree(top20% contact)={pair_dis[m&(np.array([A[ix[a],ix[b]] for a,b in pairs])>=np.quantile([A[ix[a],ix[b]] for a,b in pairs],0.8))].mean():.3f}")

# ===== CONTROL 2: observation-window (does formation stage beat confirmation count?) =====
L1set=ae[1]|ae[2]|ae[3]|ae[4]; L2set=ae[5]; L3set=ae[6]
def fstage(a,b):
    if (a,b) in L1set: return 0
    if (a,b) in L2set: return 1
    if (a,b) in L3set: return 2
    return 3   # not seen in 1-6
fs=np.array([fstage(a,b) for a,b in pairs])
conf=np.array([sum(1 for S in [ae[1],ae[2],ae[3],ae[4],ae[5],ae[6]] if (a,b) in S) for a,b in pairs])  # # of datasets 1-6
y7=np.array([1 if (a,b) in W7 else 0 for a,b in pairs]); y8=np.array([1 if (a,b) in W8 else 0 for a,b in pairs])
disagree=np.abs(y7-y8)
seen=fs<3  # appeared in datasets 1-6 (has a real formation stage)
print("\n=== CONTROL 2: OBSERVATION-WINDOW (formation stage vs confirmation count) ===")
print(f"  corr(formation_stage, confirmation_count) = {np.corrcoef(fs[seen],conf[seen])[0,1]:.3f}  (collinear)")
# does formation stage predict W7/W8 disagreement BEYOND confirmation count?
Xc=StandardScaler().fit_transform(np.column_stack([fs[seen],conf[seen]]))
lr=LogisticRegression(max_iter=2000).fit(Xc,disagree[seen])
print(f"  logistic disagree ~ [formation_stage, confirmation_count]: beta_stage={lr.coef_[0][0]:+.3f}  beta_conf={lr.coef_[0][1]:+.3f}")
print(f"  (frame real iff beta_stage stays POSITIVE [later-formed=>more disagreement] after controlling confirmation)")
# stratified: within each confirmation-count bin, disagreement by formation stage
print("  disagreement by formation stage, WITHIN confirmation-count strata (observation-window-matched):")
for cc in sorted(set(conf[seen])):
    for st in [0,1,2]:
        s=seen&(conf==cc)&(fs==st)
        if s.sum()>=15: print(f"    conf={cc} stage={st}: n={s.sum():4d} disagree={disagree[s].mean():.3f}")

print("\n=== GATE DECISION ===")
print("  (interpret: over-dispersion crossover survives ONLY if some clean stratum breaches 0.5 ceiling with 4-replicate baseline;")
print("   timing gradient survives ONLY if formation stage predicts disagreement beyond confirmation count)")
with open(f"{OUT}/gate_controls.csv","w",newline="") as fo:
    w=csv.writer(fo); w.writerow(["control","value"])
    w.writerow(["L1_4replicate_dispersion",round(obs/pred,4)]); w.writerow(["L1_mean_pairwise_disagree",round(pair_dis[m].mean(),4)])
    w.writerow(["beta_formation_stage_ctrl_conf",round(float(lr.coef_[0][0]),4)]); w.writerow(["beta_confirmation_count",round(float(lr.coef_[0][1]),4)])
print("gate controls done ->",OUT)
