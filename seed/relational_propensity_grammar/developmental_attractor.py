"""Phase 6: readout vs specification. Does relational structure from L1 (early) predict ADULT edges &
which L1 contacts persist to adult? If L1-relational predicts adult beyond contact, the field is present
early (attractor/specification-like); if only adult-context works, it's a downstream readout."""
import csv,re,os,numpy as np,openpyxl
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
C="../data/celegans/data"
ae=defaultdict(set)
for r in csv.DictReader(open(f"{C}/connectome/witvliet2020/witvliet_combined_edges.csv")):
    if r['type']=='electrical':continue
    ae[int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))].add((r['pre'],r['post']))
L1=ae[1]|ae[2]|ae[3]|ae[4]; W7,W8=ae[7],ae[8]  # L1 aggregate context
wb=openpyxl.load_workbook(f"{C}/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx",read_only=True,data_only=True);ws=wb['adult nerve ring neighbors']
rows=list(ws.iter_rows(values_only=True));alln=[h for h in rows[0][1:]];ix={n:i for i,n in enumerate(alln) if n};Mn=len(alln);A=np.zeros((Mn,Mn))
for row in rows[1:]:
    if row[0] in ix:
        for j,v in enumerate(row[1:]):
            if v not in (None,''):
                try:A[ix[row[0]],j]=float(v)
                except:pass
A=np.maximum(A,A.T);Bnb=(A>0).astype(float)
pairs=[(alln[i],alln[j]) for i in range(Mn) for j in range(Mn) if i!=j and alln[i] and alln[j] and A[i,j]>0]
y8=np.array([1 if (a,b) in W8 else 0 for a,b in pairs])
inL1=np.array([1 if (a,b) in L1 else 0 for a,b in pairs])
def relf(ctx):
    o=defaultdict(set);i2=defaultdict(set)
    for u,v in ctx:o[u].add(v);i2[v].add(u)
    return np.array([[len(i2[a]&i2[b]),len(o[a]&o[b]),len(o[a]&i2[b])] for a,b in pairs],float)
neu=sorted(set(a for a,b in pairs)|set(b for a,b in pairs))
rng=np.random.default_rng(0);f=np.array_split(rng.permutation(neu),5);nf={n:i for i,ff in enumerate(f) for n in ff}
Xc=np.column_stack([np.log1p([A[ix[a],ix[b]] for a,b in pairs]),Bnb.sum(1)[[ix[a] for a,b in pairs]],Bnb.sum(1)[[ix[b] for a,b in pairs]]])
def ndcv(Xf,y):
    p=np.full(len(y),np.nan)
    for k in range(5):
        te=np.array([nf[a]==k and nf[b]==k for a,b in pairs]);tr=np.array([nf[a]!=k and nf[b]!=k for a,b in pairs])
        if te.sum()<10 or y[tr].sum()<5:continue
        clf=make_pipeline(StandardScaler(with_mean=False),GradientBoostingClassifier(random_state=0));clf.fit(Xf[tr],y[tr]);p[te]=clf.predict_proba(Xf[te])[:,1]
    return p
# predict ADULT W8 from: contact-only vs +L1-context-relational vs +adult(W7)-context-relational
p_c=ndcv(Xc,y8); p_l1=ndcv(np.hstack([Xc,relf(L1)]),y8); p_ad=ndcv(np.hstack([Xc,relf(W7)]),y8)
def au(p):m=~np.isnan(p);return roc_auc_score(y8[m],p[m])
print("=== predict ADULT W8 edge ===")
print(f"  contact-only            AUC={au(p_c):.3f}")
print(f"  +L1-context relational  AUC={au(p_l1):.3f}  (dvs contact={au(p_l1)-au(p_c):+.3f})  <- EARLY structure")
print(f"  +adult(W7) relational   AUC={au(p_ad):.3f}  (dvs contact={au(p_ad)-au(p_c):+.3f})")
# does L1-relational predict which L1 contacts PERSIST/appear in adult (among L1 edges -> stay in adult)?
m=inL1.astype(bool)&(~np.isnan(p_l1))
persist=np.array([1 if (a,b) in W8 else 0 for a,b in pairs])[m]
print(f"\n=== among L1 edges (n={m.sum()}), does L1-relational predict adult persistence? ===")
print(f"  L1-relational AUC for L1->adult persistence = {roc_auc_score(persist,p_l1[m]):.3f}  (base rate {persist.mean():.2f})")
open(f"{os.path.expanduser('../seed/data/processed/relational_propensity_grammar')}/developmental_attractor.csv","w").write(
    f"contact_only_AUC,{au(p_c):.4f}\nL1_context_AUC,{au(p_l1):.4f}\nadult_context_AUC,{au(p_ad):.4f}\nL1_persist_AUC,{roc_auc_score(persist,p_l1[m]):.4f}\n")
