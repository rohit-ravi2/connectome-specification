#!/usr/bin/env python3
"""Part A + decisive relational test. VERIFY the premise before building grammar:
(1) reproduce UNDER-DISPERSION (obs cross-adult disagreement / predicted 2p(1-p) < 1),
(2) test whether RELATIONAL context moves the dispersion ratio toward 1 under LEAKAGE-SAFE + DEGREE-
    PRESERVING controls. An earlier run gave higher-order Delta=+0.000; a large positive was expected.
"""
import csv, re, os, math
import numpy as np, openpyxl
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score

C="../data/celegans/data"
OUT=os.path.expanduser("../seed/data/processed/relational_propensity_grammar")
EDGES=f"{C}/connectome/witvliet2020/witvliet_combined_edges.csv"
CONTACT=f"{C}/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx"
NPZ="../data/tier5/connectome.npz"
os.makedirs(OUT,exist_ok=True)

# ---- directed chemical edges per animal ----
animal_edges=defaultdict(set)   # animal -> set (pre,post)
for r in csv.DictReader(open(EDGES)):
    if r['type']=='electrical': continue
    a=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
    animal_edges[a].add((r['pre'],r['post']))
W7,W8=animal_edges[7],animal_edges[8]; L1=[animal_edges[i] for i in [1,2,3,4]]
# ---- contact area (adult) ----
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
A=np.maximum(A,A.T); Bnb=(A>0).astype(float); deg=A.sum(1); degn=Bnb.sum(1)
# ---- NT ----
d=np.load(NPZ,allow_pickle=True); nm=[str(x) for x in d["names"]]; ntp=[str(x) for x in d["nt_primary"]]
def ntn(x):
    for k,v in [("Acetyl","ACh"),("Glu","Glu"),("GABA","GABA"),("Dopamine","mono"),("Serotonin","mono"),("Octopamine","mono"),("Tyramine","mono")]:
        if k in x: return v
    return "Unk"
NT={a:ntn(b) for a,b in zip(nm,ntp)}
neurons=[n for n in alln if n]
# ---- directed contact-positive pairs (opportunity gate) ----
pairs=[]
for i in range(M):
    for j in range(M):
        if i==j: continue
        a,b=alln[i],alln[j]
        if not a or not b or A[i,j]<=0: continue
        pairs.append((a,b))
print(f"directed contact-positive pairs: {len(pairs)}")
# ---- pairwise features (contact + NT + degree + topology) ----
ntk=sorted(set((NT.get(a,'Unk'),NT.get(b,'Unk')) for a,b in pairs)); ki={k:i for i,k in enumerate(ntk)}
def feats(a,b):
    i,j=ix[a],ix[b]
    cos=float(A[i]@A[j]/(np.linalg.norm(A[i])*np.linalg.norm(A[j])+1e-9))
    cn=int(((Bnb[i]>0)&(Bnb[j]>0)).sum())
    oh=[0]*len(ntk); oh[ki[(NT.get(a,'Unk'),NT.get(b,'Unk'))]]=1
    return [math.log1p(A[i,j]),cos,cn,math.log1p(deg[i]),math.log1p(deg[j]),degn[i],degn[j]]+oh
X=np.array([feats(a,b) for a,b in pairs])
# target = edge in adult (union W7,W8) for propensity fit; and per-animal for dispersion
y_any=np.array([1 if ((a,b) in W7 or (a,b) in W8) else 0 for a,b in pairs])
y7=np.array([1 if (a,b) in W7 else 0 for a,b in pairs]); y8=np.array([1 if (a,b) in W8 else 0 for a,b in pairs])
print(f"edge rate (any adult): {y_any.mean():.3f} | W7 {y7.mean():.3f} | W8 {y8.mean():.3f}")

# ---- neuron-disjoint CV propensity p ----
def ndcv_proba(X,y,seed=0):
    neu=sorted(set(a for a,b in pairs)|set(b for a,b in pairs))
    rng=np.random.default_rng(seed); f=np.array_split(rng.permutation(neu),5); nf={n:i for i,ff in enumerate(f) for n in ff}
    p=np.full(len(y),np.nan)
    for k in range(5):
        te=np.array([nf[a]==k and nf[b]==k for a,b in pairs]); tr=np.array([nf[a]!=k and nf[b]!=k for a,b in pairs])
        if te.sum()<10 or y[tr].sum()<5: continue
        clf=make_pipeline(StandardScaler(with_mean=False),GradientBoostingClassifier(random_state=seed)); clf.fit(X[tr],y[tr]); p[te]=clf.predict_proba(X[te])[:,1]
    return p
# fit propensity on per-animal presence (predict each adult's edge), average -> p_hat
p7=ndcv_proba(X,y7); p8=ndcv_proba(X,y8)
p=np.nanmean(np.vstack([p7,p8]),axis=0)
m=~np.isnan(p)
# ---- DISPERSION RATIO ----
def dispersion(p,ya,yb,mask):
    obs=np.abs(ya[mask]-yb[mask]).sum()           # observed W7/W8 disagreement
    pred=(2*p[mask]*(1-p[mask])).sum()            # Bernoulli predicted disagreement
    return obs/pred, obs, pred
ratio,obs,pred=dispersion(p,y7,y8,m)
auc=roc_auc_score(y_any[m],p[m])
print(f"\n=== DISPERSION (contact+NT+topology model) ===")
print(f"  propensity AUC (any-adult, neuron-disjoint) = {auc:.3f}")
print(f"  observed W7/W8 disagreement = {obs:.0f} | predicted 2p(1-p) = {pred:.0f}")
print(f"  DISPERSION RATIO = {ratio:.3f}   (<1 = under-dispersed = edges sharper than model; memory says ~0.48)")

# ---- theta table + save ----
theta_L1=np.array([sum(1 for s in L1 if (a,b) in s)/4 for a,b in pairs])
theta_adult=np.array([( (1 if (a,b) in W7 else 0)+(1 if (a,b) in W8 else 0) )/2 for a,b in pairs])
with open(f"{OUT}/latent_theta_table.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["pre","post","contact_area","theta_L1","theta_adult","p_model","w7","w8"])
    for k,(a,b) in enumerate(pairs):
        w.writerow([a,b,round(A[ix[a],ix[b]],1),round(theta_L1[k],3),round(theta_adult[k],3),round(p[k],4) if not np.isnan(p[k]) else '',y7[k],y8[k]])
print(f"\ntheta table -> {OUT}/latent_theta_table.csv")

# ============ DECISIVE: does RELATIONAL context move the ratio, LEAKAGE-SAFE + DEGREE-NULL? ============
# Relational context from an INDEPENDENT graph (W7) to predict W8 propensity, target-edge held out.
# Feature: shared inputs/outputs + feedforward support computed from W7 graph.
def build_relfeats(context_edges, pairs):
    outn=defaultdict(set); inn=defaultdict(set)
    for (u,v) in context_edges: outn[u].add(v); inn[v].add(u)
    F=[]
    for a,b in pairs:
        sh_in=len(inn[a]&inn[b]); sh_out=len(outn[a]&outn[b])   # equivalence (shared in/out)
        ff=len(outn[a]&inn[b])                                   # a->k->b feedforward support (complementarity)
        fb=len(outn[b]&inn[a])
        F.append([sh_in,sh_out,ff,fb])
    return np.array(F,float)
# predict W8 using contact+NT+topology + relational-from-W7 (W7 is independent of W8 label; no W8 leakage)
Xrel_w7=build_relfeats(W7 - {(a,b) for a,b in pairs}, pairs)  # W7 as context (target graph W8)
def ndcv_p_for(Xf,ytarget,seed=0):
    return ndcv_proba(Xf,ytarget,seed)
p8_base=ndcv_proba(X,y8);
p8_rel=ndcv_proba(np.hstack([X,Xrel_w7]),y8)
# dispersion using symmetric: predict both animals' propensity with cross-animal context
p7_relctx=ndcv_proba(np.hstack([X,build_relfeats(W8,pairs)]),y7)  # W8 context -> predict W7
p8_relctx=ndcv_proba(np.hstack([X,build_relfeats(W7,pairs)]),y8)  # W7 context -> predict W8
p_rel=np.nanmean(np.vstack([p7_relctx,p8_relctx]),axis=0); mr=~np.isnan(p_rel)
ratio_rel,_,_=dispersion(p_rel,y7,y8,mr)
auc_rel=roc_auc_score(y_any[mr],p_rel[mr])
print(f"\n=== RELATIONAL (cross-animal context, leakage-safe) ===")
print(f"  AUC base={auc:.3f} -> +relational={auc_rel:.3f}  (dAUC={auc_rel-auc:+.3f})")
print(f"  DISPERSION RATIO base={ratio:.3f} -> +relational={ratio_rel:.3f}  (moved toward 1 by {ratio_rel-ratio:+.3f})")
# DEGREE-PRESERVING NULL: relational feats from a degree-preserving randomization of W7/W8
def degree_preserving(edges, nswap=20000, seed=0):
    E=list(edges); rng=np.random.default_rng(seed); E=[list(e) for e in E]
    for _ in range(nswap):
        i,j=rng.integers(0,len(E),2)
        if i==j: continue
        a,b=E[i]; c,dd=E[j]
        if len({a,b,c,dd})<4: continue
        if (a,dd) not in edges and (c,b) not in edges:
            E[i]=[a,dd]; E[j]=[c,b]
    return set((e[0],e[1]) for e in E)
null_ratios=[]
for s in range(6):
    w7n=degree_preserving(W7,seed=s); w8n=degree_preserving(W8,seed=100+s)
    p7n=ndcv_proba(np.hstack([X,build_relfeats(w8n,pairs)]),y7,s); p8n=ndcv_proba(np.hstack([X,build_relfeats(w7n,pairs)]),y8,s)
    pn=np.nanmean(np.vstack([p7n,p8n]),axis=0); mn=~np.isnan(pn)
    null_ratios.append(dispersion(pn,y7,y8,mn)[0])
print(f"  DEGREE-PRESERVING NULL relational ratio: mean={np.mean(null_ratios):.3f} (real relational={ratio_rel:.3f})")
print(f"  => relational is REAL (not degree) iff real ratio moves toward 1 MORE than degree-null")
with open(f"{OUT}/dispersion_baselines.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["model","AUC","dispersion_ratio"])
    w.writerow(["contact+NT+topology",round(auc,4),round(ratio,4)])
    w.writerow(["+relational(cross-animal)",round(auc_rel,4),round(ratio_rel,4)])
    w.writerow(["degree-preserving-null(relational)","",round(float(np.mean(null_ratios)),4)])
print("\nPart A + decisive test done ->",OUT)
