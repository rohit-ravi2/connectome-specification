#!/usr/bin/env python3
"""Reconcile the falsification: equivalence PROMOTES globally but the prior same-NT residual SUPPRESSES.
Test whether shared-partner equivalence flips sign inside same-NT high-contact subset. Plus a lightweight
generative check: does contact+relational theta reproduce the same-NT residual & reciprocity better than
contact-only? Determines: clean grammar vs context-dependent re-description."""
import csv, re, os, math
import numpy as np, openpyxl
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

C="../data/celegans/data"
OUT=os.path.expanduser("../seed/data/processed/relational_propensity_grammar")
EDGES=f"{C}/connectome/witvliet2020/witvliet_combined_edges.csv"
CONTACT=f"{C}/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx"
NPZ="../data/tier5/connectome.npz"
animal_edges=defaultdict(set)
for r in csv.DictReader(open(EDGES)):
    if r['type']=='electrical': continue
    a=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1)); animal_edges[a].add((r['pre'],r['post']))
W7,W8=animal_edges[7],animal_edges[8]
wb=openpyxl.load_workbook(CONTACT,read_only=True,data_only=True); ws=wb['adult nerve ring neighbors']
rows=list(ws.iter_rows(values_only=True)); alln=[h for h in rows[0][1:]]; ix={n:i for i,n in enumerate(alln) if n}
M=len(alln); A=np.zeros((M,M))
for row in rows[1:]:
    rn=row[0]
    if rn in ix:
        for j,v in enumerate(row[1:]):
            if v not in (None,''):
                try:A[ix[rn],j]=float(v)
                except:pass
A=np.maximum(A,A.T); Bnb=(A>0).astype(float)
d=np.load(NPZ,allow_pickle=True)
def ntn(x):
    for k,v in [("Acetyl","ACh"),("Glu","Glu"),("GABA","GABA"),("Dopamine","mono"),("Serotonin","mono"),("Octopamine","mono"),("Tyramine","mono")]:
        if k in x: return v
    return "Unk"
NT={a:ntn(str(b)) for a,b in zip([str(x) for x in d["names"]],[str(x) for x in d["nt_primary"]])}
pairs=[(alln[i],alln[j]) for i in range(M) for j in range(M) if i!=j and alln[i] and alln[j] and A[i,j]>0]
y8=np.array([1 if (a,b) in W8 else 0 for a,b in pairs])
same_nt=np.array([1 if (NT.get(a,'U')==NT.get(b,'x') and NT.get(a,'U')!='Unk') else 0 for a,b in pairs])
carea=np.array([A[ix[a],ix[b]] for a,b in pairs])
hc=carea>=np.quantile(carea,0.80)  # high-contact top20
def relfeats(ctx):
    outn=defaultdict(set); inn=defaultdict(set)
    for u,v in ctx: outn[u].add(v); inn[v].add(u)
    return np.array([[len(inn[a]&inn[b]),len(outn[a]&outn[b]),len(outn[a]&inn[b])] for a,b in pairs],float)
R=relfeats(W7)  # context

# ---- equivalence direction GLOBAL vs within same-NT high-contact ----
def logit_dir(mask,label):
    if mask.sum()<40 or y8[mask].sum()<8: print(f"  {label}: n={mask.sum()} too few/degenerate"); return None
    Xm=np.column_stack([np.log1p(carea[mask]),R[mask,0],R[mask,1],R[mask,2]])
    Xs=StandardScaler().fit_transform(Xm)
    lr=LogisticRegression(max_iter=2000).fit(Xs,y8[mask]); c=lr.coef_[0]
    print(f"  {label:34s} n={mask.sum():4d} edge%={100*y8[mask].mean():4.1f}  "
          f"shared_in={c[1]:+.2f} shared_out={c[2]:+.2f} feedfwd={c[3]:+.2f}")
    return c
print("=== equivalence(shared-partner) direction: GLOBAL vs same-NT-high-contact subset ===")
allm=np.ones(len(pairs),bool)
c_all=logit_dir(allm,"ALL contact-positive")
c_hc=logit_dir(hc,"high-contact (top20)")
c_snt=logit_dir(same_nt.astype(bool),"same-NT")
c_snthc=logit_dir((same_nt&hc).astype(bool),"same-NT AND high-contact")
c_dnthc=logit_dir(((1-same_nt)&hc).astype(bool),"diff-NT AND high-contact")

# ---- generative: does contact+relational reproduce the same-NT residual (suppression) that contact-only misses? ----
# fit p on W7-derived features (leakage-safe: predict W8), then compare same-NT vs diff-NT edge-rate at matched contact
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
neu=sorted(set(a for a,b in pairs)|set(b for a,b in pairs))
rng=np.random.default_rng(0); f=np.array_split(rng.permutation(neu),5); nf={n:i for i,ff in enumerate(f) for n in ff}
def ndcv(Xf):
    p=np.full(len(y8),np.nan)
    for k in range(5):
        te=np.array([nf[a]==k and nf[b]==k for a,b in pairs]); tr=np.array([nf[a]!=k and nf[b]!=k for a,b in pairs])
        if te.sum()<10 or y8[tr].sum()<5: continue
        clf=make_pipeline(StandardScaler(with_mean=False),GradientBoostingClassifier(random_state=0)); clf.fit(Xf[tr],y8[tr]); p[te]=clf.predict_proba(Xf[te])[:,1]
    return p
Xc=np.column_stack([np.log1p(carea),Bnb.sum(1)[[ix[a] for a,b in pairs]],Bnb.sum(1)[[ix[b] for a,b in pairs]]])
p_contact=ndcv(Xc); p_rel=ndcv(np.hstack([Xc,R]))
m=(~np.isnan(p_contact))&(~np.isnan(p_rel))&hc
print("\n=== generative: predicted edge-prob in HIGH-CONTACT same-NT vs diff-NT (real residual = same-NT LOWER) ===")
for lab,md in [("REAL",y8),("contact-only model",p_contact),("contact+relational",p_rel)]:
    s=same_nt.astype(bool)&m; dd=(1-same_nt).astype(bool)&m
    print(f"  {lab:20s} same-NT={md[s].mean():.3f}  diff-NT={md[dd].mean():.3f}  gap(diff-same)={md[dd].mean()-md[s].mean():+.3f}")
print("  (real residual: diff-NT synapses MORE than same-NT at high contact. Does relational reproduce the gap contact-only misses?)")
with open(f"{OUT}/reconcile_results.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["subset","shared_in","shared_out","feedfwd"])
    for lab,c in [("global",c_all),("high_contact",c_hc),("same_nt",c_snt),("same_nt_high_contact",c_snthc),("diff_nt_high_contact",c_dnthc)]:
        if c is not None: w.writerow([lab,round(c[1],3),round(c[2],3),round(c[3],3)])
print("\nreconcile done ->",OUT)
