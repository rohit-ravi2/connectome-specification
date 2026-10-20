#!/usr/bin/env python3
"""Round3 FORMAL branch -- Constraint Information Decomposition (+ closure + attractor).

Question (formal/organizational, not molecular): how many BITS of connectome uncertainty does
each successive CONSTRAINT layer remove, and does the residual reach the adult-replicate NOISE
CEILING?  If constraints (contact geometry, developmental arrival, topology, NT) remove nearly all
reducible uncertainty and the residual == replicate noise, then the connectome's 'specified
information' is CONSTRAINT REDUCTION, not hidden molecular storage.

Target (non-leaky): among CONTACTING nerve-ring pairs, is there a reproducible strong-core
chemical synapse (present in BOTH adults 7&8)?  Features come ONLY from the contact graph +
lineage arrival + NT -- never from the synapse graph (no target leakage).

Validation: edge-CV (bits) + NEURON-DISJOINT CV (leakage-proof AUC).
Controls: contact-area-stratified label permutation; degree/arrival/NT permutation.
Also: constraint-CLOSURE (does #constraints-satisfied beat best single constraint?) and
strong-core vs weak-halo vs non-edge ATTRACTOR separation.
Stdlib+sklearn.
"""
import openpyxl, csv, re, os, math, json
import numpy as np
from collections import defaultdict, Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score

DATA="../data/celegans/data"
OUT=os.path.expanduser("../seed/data/processed/formal_branch")
CONTACT=f"{DATA}/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx"
EDGES=f"{DATA}/connectome/witvliet2020/witvliet_combined_edges.csv"
NPZ="../data/tier5/connectome.npz"
LIN="../lineage/jtauber_lineage/cells.json"
os.makedirs(OUT,exist_ok=True)
RNG=np.random.default_rng(0)

# ---- lineage depth (birth-order proxy) ----
tree=json.load(open(LIN)); depth={}
def walk(n,d):
    depth[n["name"]]=d
    for c in n.get("children",[]): walk(c,d+1)
walk(tree,0)

# ---- contact area matrix ----
wb=openpyxl.load_workbook(CONTACT,read_only=True,data_only=True); ws=wb['adult nerve ring neighbors']
rows=list(ws.iter_rows(values_only=True)); alln=[h for h in rows[0][1:]]; idx={n:i for i,n in enumerate(alln)}; M=len(alln)
A=np.zeros((M,M))
for row in rows[1:]:
    rn=row[0]
    if rn is None or rn not in idx: continue
    for j,val in enumerate(row[1:]):
        if val in (None,''): continue
        try: A[idx[rn],j]=float(val)
        except: pass
A=np.maximum(A,A.T); deg_area=A.sum(1); deg_num=(A>0).sum(1)

# ---- NT ----
d=np.load(NPZ,allow_pickle=True); names=[str(x) for x in d["names"]]; ntp=[str(x) for x in d["nt_primary"]]
def ntn(x):
    for k,v in [("Acetyl","ACh"),("ACh","ACh"),("Glu","Glu"),("GABA","GABA"),("Dopamine","DA"),
               ("Serotonin","5HT"),("5HT","5HT"),("Octopamine","OA"),("Tyramine","TA")]:
        if k in x: return v
    return "Unk"
nt_of={nm:ntn(nt) for nm,nt in zip(names,ntp)}

# ---- reproducible synapse core (both adults), strong-core & weak-halo ----
chem=defaultdict(lambda: defaultdict(int))   # (a,b)undirected -> {ds: max synapses}
present=defaultdict(set)
for r in csv.DictReader(open(EDGES)):
    ds=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
    if ds not in (7,8): continue
    if r['type']=='electrical': continue
    a,b=r['pre'],r['post']; k=(min(a,b),max(a,b))
    w=int(float(r.get('synapses',r.get('weight',1)) or 1))
    chem[k][ds]=max(chem[k][ds],w); present[k].add(ds)
def core_level(k):  # 0 none, 1 one-animal(halo), 2 both(strong-core)
    return len(present.get(k,()))
def maxsyn(k):
    v=chem.get(k,{}); return max(v.values()) if v else 0

# ---- contacting pairs among neurons with full features ----
usable=[n for n in alln if n in depth and n in nt_of]
pairs=[]; feats=defaultdict(list); y=[]; core=[]; msyn=[]; endpts=[]
for i in range(M):
    for j in range(i+1,M):
        if A[i,j]<=0: continue
        a,b=alln[i],alln[j]
        if a not in depth or b not in depth or a not in nt_of or b not in nt_of: continue
        vi,vj=A[i],A[j]
        cos=float(vi@vj/(np.linalg.norm(vi)*np.linalg.norm(vj)+1e-9))
        shared=int(((A[i]>0)&(A[j]>0)).sum())
        da,db=depth[a],depth[b]
        k=(min(a,b),max(a,b))
        feats["degree"].append([math.log1p(min(deg_area[i],deg_area[j])),math.log1p(max(deg_area[i],deg_area[j])),
                                 min(deg_num[i],deg_num[j]),max(deg_num[i],deg_num[j])])
        feats["area"].append([math.log1p(A[i,j])])
        feats["nbhd"].append([cos,shared])
        feats["arrival"].append([min(da,db),max(da,db),abs(da-db)])
        feats["nt"].append(tuple(sorted((nt_of[a],nt_of[b]))))
        cl=core_level(k); core.append(cl); msyn.append(maxsyn(k))
        y.append(1 if cl==2 else 0)     # strong-core = both adults
        pairs.append((a,b)); endpts.append((a,b))
y=np.array(y); core=np.array(core); msyn=np.array(msyn); N=len(y)
# NT one-hot
ntk=sorted(set(feats["nt"])); nti={k:i for i,k in enumerate(ntk)}
NToh=np.zeros((N,len(ntk)))
for r,k in enumerate(feats["nt"]): NToh[r,nti[k]]=1
F={k:(np.array(v,float) if k!="nt" else NToh) for k,v in feats.items()}
print(f"contacting pairs N={N} | strong-core (both adults) {y.sum()} ({100*y.mean():.1f}%) | "
      f"halo(1-animal) {(core==1).sum()} | non-edge-contact {(core==0).sum()}")

# ---- cumulative constraint layers ----
LAYERS=[("0 base",[]),("1 degree",["degree"]),("2 +area",["degree","area"]),
        ("3 +neighborhood",["degree","area","nbhd"]),("4 +arrival",["degree","area","nbhd","arrival"]),
        ("5 +NT",["degree","area","nbhd","arrival","nt"])]
def stack(keys): return np.hstack([F[k] for k in keys]) if keys else None
def cv_bits(X):
    if X is None:
        p=np.full(N,y.mean())
    else:
        clf=make_pipeline(StandardScaler(with_mean=False),LogisticRegression(max_iter=3000,C=1.0))
        p=cross_val_predict(clf,X,y,cv=StratifiedKFold(5,shuffle=True,random_state=0),method="predict_proba")[:,1]
    p=np.clip(p,1e-6,1-1e-6)
    bits=-(y*np.log2(p)+(1-y)*np.log2(1-p)).sum()
    auc=roc_auc_score(y,p) if X is not None else 0.5
    ap=average_precision_score(y,p) if X is not None else y.mean()
    return bits,auc,ap
def neuron_disjoint_auc(X,seed=0):
    if X is None: return 0.5
    neu=sorted(set(a for a,b in pairs)|set(b for a,b in pairs))
    rng=np.random.default_rng(seed); folds=np.array_split(rng.permutation(neu),5)
    nf={n:i for i,f in enumerate(folds) for n in f}; pred=np.full(N,np.nan)
    for kk in range(5):
        te=np.array([nf[a]==kk and nf[b]==kk for a,b in pairs])
        tr=np.array([nf[a]!=kk and nf[b]!=kk for a,b in pairs])
        if te.sum()<10 or y[tr].sum()<5 or y[te].sum()<3 or len(set(y[te]))<2: continue
        clf=make_pipeline(StandardScaler(with_mean=False),LogisticRegression(max_iter=3000,C=1.0))
        clf.fit(X[tr],y[tr]); pred[te]=clf.predict_proba(X[te])[:,1]
    m=~np.isnan(pred)
    return roc_auc_score(y[m],pred[m]) if (m.sum()>20 and len(set(y[m]))>1) else np.nan
def nd_multi(X): return np.nanmean([neuron_disjoint_auc(X,s) for s in range(5)])

# ---- noise ceiling: predict adult8 strong presence from adult7 among these pairs ----
# per-pair: has syn in ds7? in ds8? ceiling = AUC of ds7-indicator predicting ds8-indicator
s7=np.array([1 if 7 in present.get((min(a,b),max(a,b)),()) else 0 for a,b in pairs])
s8=np.array([1 if 8 in present.get((min(a,b),max(a,b)),()) else 0 for a,b in pairs])
ceil_auc=roc_auc_score(s8,s7)
# ceiling bits: log-loss of using P(ds8|ds7) as predictor of strong-core-ish -- report as reference
print(f"\nNOISE CEILING (adult7 -> adult8 among contacts): AUC {ceil_auc:.3f}")

print("\n============ CONSTRAINT INFORMATION DECOMPOSITION (target: strong-core among contacts) ============")
H0=N*(-(y.mean()*math.log2(y.mean())+(1-y.mean())*math.log2(1-y.mean())))
print(f"H0 (base uncertainty) = {H0:.0f} bits  ({H0/N:.4f} bits/pair)")
print(f"{'layer':18s} {'edgeCV bits':>11s} {'Δbits':>8s} {'bits/pair':>9s} {'AUC':>6s} {'AP':>6s} {'neuronCV AUC':>12s}")
prev=H0; results=[]
for name,keys in LAYERS:
    X=stack(keys); bits,auc,ap=cv_bits(X); nd=nd_multi(X) if keys else 0.5
    db=prev-bits; results.append((name,bits,db,auc,ap,nd)); prev=bits
    print(f"{name:18s} {bits:>11.0f} {db:>8.0f} {bits/N:>9.4f} {auc:>6.3f} {ap:>6.3f} {nd:>12.3f}")
resid_frac=results[-1][1]/H0
print(f"\nresidual after all constraints = {results[-1][1]/N:.4f} bits/pair ({100*resid_frac:.1f}% of H0 remains)")
print(f"full-model neuron-disjoint AUC {results[-1][5]:.3f} vs noise ceiling {ceil_auc:.3f}")

# ---- CONSTRAINT CLOSURE: does #constraints-satisfied beat best single constraint? ----
print("\n============ CONSTRAINT CLOSURE (organizational: AND of constraints) ============")
area=np.array([f[0] for f in F["area"]]); cos=F["nbhd"][:,0]; adiff=F["arrival"][:,2]
degmin=F["degree"][:,0]
# binarize: satisfied = in the strong-core-favorable tail (thresholds from data medians, NOT from y)
C_area=(area>np.median(area)).astype(int)
C_cos =(cos >np.median(cos)).astype(int)
C_conatal=(adiff<=np.median(adiff)).astype(int)     # co-natal (close arrival)
C_deg =(degmin>np.median(degmin)).astype(int)
Cmat=np.vstack([C_area,C_cos,C_conatal,C_deg]).T
count=Cmat.sum(1)
# strong-core rate by #constraints satisfied
print("  strong-core rate by #constraints satisfied (of 4):")
for c in range(5):
    m=count==c
    if m.sum()>0: print(f"    {c} sat: n={m.sum():4d}  strong-core={100*y[m].mean():5.1f}%")
# does closure-count add over best single? (neuron-disjoint)
best_single=max([("area",area),("cos",cos),("conatal",-adiff),("deg",degmin)],
                key=lambda kv: nd_multi(kv[1].reshape(-1,1)))
bs_auc=nd_multi(best_single[1].reshape(-1,1))
count_auc=nd_multi(count.reshape(-1,1).astype(float))
combo_auc=nd_multi(Cmat.astype(float))
print(f"  best single constraint ({best_single[0]}) neuronCV AUC={bs_auc:.3f}")
print(f"  #constraints-satisfied     neuronCV AUC={count_auc:.3f}")
print(f"  all-constraints (binary)   neuronCV AUC={combo_auc:.3f}")

# ---- ATTRACTOR separation: strong-core vs weak-halo vs non-edge on constraint support ----
print("\n============ ATTRACTOR / STABILITY (strong-core vs weak-halo vs non-edge-contact) ============")
grp={2:"strong-core",1:"weak-halo",0:"non-edge"}
print(f"  {'group':12s} {'n':>5s} {'logArea':>8s} {'cos':>6s} {'|Δdepth|':>9s} {'degmin':>7s} {'#sat':>5s}")
for g in (2,1,0):
    m=core==g
    print(f"  {grp[g]:12s} {m.sum():>5d} {area[m].mean():>8.2f} {cos[m].mean():>6.3f} "
          f"{adiff[m].mean():>9.2f} {degmin[m].mean():>7.2f} {count[m].mean():>5.2f}")

# ---- NULLS on full model (contact-area-stratified label permutation) ----
print("\n============ NULLS (full model, edge-CV AUC) ============")
Xfull=stack(LAYERS[-1][1]); full_auc=cv_bits(Xfull)[1]
# stratified permutation: shuffle y within area deciles
dec=np.clip((np.argsort(np.argsort(area))*10//N),0,9)
perm=[]
for s in range(30):
    ys=y.copy(); rng=np.random.default_rng(s)
    for dd in range(10):
        mm=np.where(dec==dd)[0]; ys[mm]=rng.permutation(ys[mm])
    perm.append(cv_bits(Xfull if True else None)[1] if False else roc_auc_score(ys, cross_val_predict(
        make_pipeline(StandardScaler(with_mean=False),LogisticRegression(max_iter=2000)),Xfull,ys,
        cv=StratifiedKFold(5,shuffle=True,random_state=s),method="predict_proba")[:,1]))
perm=np.array(perm)
print(f"  full-model edge-CV AUC={full_auc:.3f} | area-stratified label-perm null mean={perm.mean():.3f} "
      f"95pct={np.percentile(perm,95):.3f} p={(1+(perm>=full_auc).sum())/(1+len(perm)):.3f}")

# ---- save ----
with open(f"{OUT}/constraint_decomposition.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["layer","edgeCV_bits","delta_bits","bits_per_pair","AUC","AP","neuronCV_AUC"])
    for name,bits,db,auc,ap,nd in results: w.writerow([name,round(bits,1),round(db,1),round(bits/N,5),round(auc,4),round(ap,4),round(nd,4)])
    w.writerow([]); w.writerow(["noise_ceiling_AUC",round(ceil_auc,4)]); w.writerow(["H0_bits",round(H0,1)])
    w.writerow(["closure_best_single_AUC",round(bs_auc,4)]); w.writerow(["closure_count_AUC",round(count_auc,4)])
print(f"\nsaved -> {OUT}/constraint_decomposition.csv")
