#!/usr/bin/env python3
"""STAGE 0 GATE (developmental-generative redirect): does the UNDER-DISPERSION have developmental-timing
structure? Prediction of the shared-clocked-trajectory frame: edges that FORM EARLIER are more
under-dispersed (dispersion ratio further below 1) than late-forming edges. Flat ratio => under-dispersion
is NOT timing-structured => temporal frame takes a hit.

Guards: (1) dispersion ratio normalizes out 'early edges are higher-propensity' via CV-fit p;
(2) matched-propensity test (calibration-robust): at matched p, do early edges agree more?
(3) eligibility control: restrict to pairs with BOTH neurons present at L1 (edge COULD have formed early)."""
import csv, re, os, math
import numpy as np, openpyxl
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

C="../data/celegans/data"
OUT=os.path.expanduser("../seed/data/processed/relational_propensity_grammar")
EDGES=f"{C}/connectome/witvliet2020/witvliet_combined_edges.csv"
CONTACT=f"{C}/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx"
NPZ="../data/tier5/connectome.npz"

# ---- per-animal directed chemical edges; stage groups L1={1,2,3,4} L2={5} L3={6} adult={7,8} ----
animal_edges=defaultdict(set)
for r in csv.DictReader(open(EDGES)):
    if r['type']=='electrical': continue
    a=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1)); animal_edges[a].add((r['pre'],r['post']))
L1set=animal_edges[1]|animal_edges[2]|animal_edges[3]|animal_edges[4]
L2set=animal_edges[5]; L3set=animal_edges[6]; W7,W8=animal_edges[7],animal_edges[8]
# neurons present at L1 (appear in any L1 edge) -> eligibility
L1neurons=set(x for e in L1set for x in e)
# ---- adult contact area ----
wb=openpyxl.load_workbook(CONTACT,read_only=True,data_only=True); ws=wb['adult nerve ring neighbors']
rows=list(ws.iter_rows(values_only=True)); alln=[h for h in rows[0][1:]]; ix={n:i for i,n in enumerate(alln) if n}
M=len(alln); A=np.zeros((M,M))
for row in rows[1:]:
    if row[0] in ix:
        for j,v in enumerate(row[1:]):
            if v not in (None,''):
                try:A[ix[row[0]],j]=float(v)
                except:pass
A=np.maximum(A,A.T); Bnb=(A>0).astype(float); deg=A.sum(1); degn=Bnb.sum(1)
d=np.load(NPZ,allow_pickle=True)
def ntn(x):
    for k,v in [("Acetyl","ACh"),("Glu","Glu"),("GABA","GABA")]:
        if k in x: return v
    return "mono" if any(m in x for m in["Dopa","Sero","Octo","Tyra"]) else "Unk"
NT={a:ntn(str(b)) for a,b in zip([str(x) for x in d["names"]],[str(x) for x in d["nt_primary"]])}
pairs=[(alln[i],alln[j]) for i in range(M) for j in range(M) if i!=j and alln[i] and alln[j] and A[i,j]>0]
y7=np.array([1 if (a,b) in W7 else 0 for a,b in pairs]); y8=np.array([1 if (a,b) in W8 else 0 for a,b in pairs])

# ---- first-appearance stage of each edge ----
def first_stage(a,b):
    if (a,b) in L1set: return 'L1'
    if (a,b) in L2set: return 'L2'
    if (a,b) in L3set: return 'L3'
    if (a,b) in W7 or (a,b) in W8: return 'adult'
    return 'never'
fstage=np.array([first_stage(a,b) for a,b in pairs])
elig=np.array([1 if (a in L1neurons and b in L1neurons) else 0 for a,b in pairs])  # both present at L1

# ---- CV-fit propensity p (contact+NT+topology, neuron-disjoint) ----
ntk=sorted(set((NT.get(a,'Unk'),NT.get(b,'Unk')) for a,b in pairs)); ki={k:i for i,k in enumerate(ntk)}
def feats(a,b):
    i,j=ix[a],ix[b]; cos=float(A[i]@A[j]/(np.linalg.norm(A[i])*np.linalg.norm(A[j])+1e-9))
    cn=int(((Bnb[i]>0)&(Bnb[j]>0)).sum()); oh=[0]*len(ntk); oh[ki[(NT.get(a,'Unk'),NT.get(b,'Unk'))]]=1
    return [math.log1p(A[i,j]),cos,cn,math.log1p(deg[i]),math.log1p(deg[j]),degn[i],degn[j]]+oh
X=np.array([feats(a,b) for a,b in pairs])
neu=sorted(set(a for a,b in pairs)|set(b for a,b in pairs))
rng=np.random.default_rng(0); f=np.array_split(rng.permutation(neu),5); nf={n:i for i,ff in enumerate(f) for n in ff}
def ndcv(y):
    p=np.full(len(y),np.nan)
    for k in range(5):
        te=np.array([nf[a]==k and nf[b]==k for a,b in pairs]); tr=np.array([nf[a]!=k and nf[b]!=k for a,b in pairs])
        if te.sum()<10 or y[tr].sum()<5: continue
        clf=make_pipeline(StandardScaler(with_mean=False),GradientBoostingClassifier(random_state=0)); clf.fit(X[tr],y[tr]); p[te]=clf.predict_proba(X[te])[:,1]
    return p
p=np.nanmean(np.vstack([ndcv(y7),ndcv(y8)]),axis=0); m=~np.isnan(p)

def disp_ratio(mask):
    mm=mask&m
    if mm.sum()<15: return None,int(mm.sum()),None
    obs=np.abs(y7[mm]-y8[mm]).sum(); pred=(2*p[mm]*(1-p[mm])).sum()
    return obs/pred if pred>0 else None,int(mm.sum()),obs/mm.sum()
def boot_ratio(mask,nb=400):
    idx=np.where(mask&m)[0]
    if len(idx)<15: return (np.nan,np.nan)
    rs=[]
    for s in range(nb):
        bi=np.random.default_rng(s).choice(idx,len(idx),replace=True)
        pred=(2*p[bi]*(1-p[bi])).sum(); rs.append(np.abs(y7[bi]-y8[bi]).sum()/pred if pred>0 else np.nan)
    return (np.nanpercentile(rs,5),np.nanpercentile(rs,95))

print("=== STAGE 0: dispersion ratio by EDGE FORMATION TIME (prediction: ratio rises L1->adult) ===")
print(f"{'first-appearance':16s} {'n':>5s} {'edge-rate':>9s} {'disp_ratio':>10s}  {'90% CI':>16s}")
for st in ['L1','L2','L3','adult']:
    r,n,er=disp_ratio(fstage==st); ci=boot_ratio(fstage==st)
    if r is not None: print(f"{st:16s} {n:5d} {er:9.3f} {r:10.3f}  [{ci[0]:.3f},{ci[1]:.3f}]")
    else: print(f"{st:16s} {n:5d}  (too few)")
r_all,n_all,_=disp_ratio(np.ones(len(pairs),bool)); print(f"{'ALL':16s} {n_all:5d} {'':9s} {r_all:10.3f}")

# ---- eligibility-controlled (both neurons present at L1) ----
print("\n=== eligibility-controlled (both neurons present at L1; edge COULD form early) ===")
for st in ['L1','L2','L3','adult']:
    r,n,er=disp_ratio((fstage==st)&(elig==1)); ci=boot_ratio((fstage==st)&(elig==1))
    if r is not None: print(f"{st:16s} {n:5d} {er:9.3f} {r:10.3f}  [{ci[0]:.3f},{ci[1]:.3f}]")
    else: print(f"{st:16s} {n:5d}  (too few)")

# ---- calibration-robust: MATCHED-PROPENSITY test. Among intermediate-p pairs (room to vary), do
# early-formed edges disagree LESS than late-formed, at matched p? ----
print("\n=== matched-propensity (p in [0.25,0.75]): agreement by formation time (calibration-robust) ===")
band=(p>=0.25)&(p<=0.75)&m
for st in ['L1','L2','L3','adult']:
    s=band&(fstage==st)
    if s.sum()>=15:
        dis=np.abs(y7[s]-y8[s]).mean(); print(f"  {st:8s} n={s.sum():4d}  mean_p={p[s].mean():.3f}  disagreement={dis:.3f}")
    else: print(f"  {st:8s} n={s.sum():4d} (too few)")
print("  (frame predicts: at matched p, EARLY-formed edges disagree LESS than late-formed)")

# spearman trend: stage-order vs per-stratum ratio
from scipy.stats import spearmanr
order={'L1':0,'L2':1,'L3':2,'adult':3}
xs=[];ys=[]
for st in ['L1','L2','L3','adult']:
    r,n,_=disp_ratio(fstage==st)
    if r is not None: xs.append(order[st]); ys.append(r)
if len(xs)>=3:
    rho,pv=spearmanr(xs,ys); print(f"\n  monotonic trend (stage-order vs ratio): Spearman rho={rho:.2f} (positive=frame supported), stratum ratios={['%.2f'%v for v in ys]}")
with open(f"{OUT}/stage0_dispersion_by_formation_time.csv","w",newline="") as fo:
    w=csv.writer(fo); w.writerow(["first_stage","n","edge_rate","disp_ratio","ci_lo","ci_hi","elig_ratio"])
    for st in ['L1','L2','L3','adult']:
        r,n,er=disp_ratio(fstage==st); ci=boot_ratio(fstage==st); re_,_,_=disp_ratio((fstage==st)&(elig==1))
        w.writerow([st,n,round(er,4) if er else '',round(r,4) if r else '',round(ci[0],4),round(ci[1],4),round(re_,4) if re_ else ''])
print("\nStage 0 done ->",OUT)
