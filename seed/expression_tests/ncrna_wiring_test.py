#!/usr/bin/env python3
"""Round3 -- test the NON-CODING RNA layer (challenged: never tested; scRNA-seq can't see it).
Acquired CeNGEN bulk ncRNA table (Zenodo 6426323): 41 neuron types x lincRNA(181)+ncRNA(7375),
rRNA-depleted random-primed (captures non-polyA ncRNA the 3' scRNA-seq misses). NOTE: mature miRNA
still ABSENT (field gap) -- this tests the lincRNA/ncRNA layer.

Two questions, honest validation:
  SANITY : does ncRNA expression carry neuron IDENTITY (NT / modality)?  -> confirms it conveys info
  WIRING : does ncRNA ADD to contact TOPOLOGY for reproducible strong-core synapse, beyond
           protein-coding, under NEURON-DISJOINT CV, vs a random-matched gene null?
"""
import openpyxl, csv, re, os, math, json
import numpy as np
from collections import defaultdict, Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

DATA="../data/celegans/data"
OUT=os.path.expanduser("../seed/data/processed/formal_branch")
NCRNA=f"{DATA}/expression/cengen/bulk_ncrna_zenodo/Long_form_ncRNA_table_040822.tsv"
PC_BULK=f"{DATA}/expression/cengen/bulk_barrett2022/Average_integrated_TMM_counts_lengthNormalized_111521.tsv"
CONTACT=f"{DATA}/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx"
EDGES=f"{DATA}/connectome/witvliet2020/witvliet_combined_edges.csv"
NPZ="../data/tier5/connectome.npz"
PROF=os.path.expanduser("../data/celegans/New Notebooks/data_derived/nb28_neuron_profile.csv")
RNG=np.random.default_rng(0)

# ---- ncRNA matrix: celltype x gene (lincRNA+ncRNA), GeTMM log1p ----
cell_gene=defaultdict(dict)
for i,r in enumerate(csv.reader(open(NCRNA),delimiter='\t')):
    if i==0: continue
    r=[x.strip('"') for x in r]
    if r[3] in ("ncRNA_gene","lincRNA_gene"):
        try: cell_gene[r[4]][r[1]]=float(r[5])
        except: pass
cells=sorted(cell_gene); genes=sorted({g for c in cell_gene for g in cell_gene[c]})
gi={g:i for i,g in enumerate(genes)}
X=np.zeros((len(cells),len(genes)))
for ci,c in enumerate(cells):
    for g,v in cell_gene[c].items(): X[ci,gi[g]]=v
X=np.log1p(X)
# keep expressed+variable ncRNA genes
keep=(X>0).mean(0)>=0.15
Xk=X[:,keep]
Xk=Xk[:,Xk.std(0)>0]
print(f"ncRNA: {len(cells)} neuron types x {Xk.shape[1]} expressed/variable ncRNA genes (of {len(genes)})")
cidx={c:i for i,c in enumerate(cells)}

# ---- protein-coding bulk for same neuron types (comparator) ----
pc_rows=[]; pc_cols=None
with open(PC_BULK) as f:
    pc_cols=[h.strip().strip('"') for h in f.readline().split()]   # whitespace+quote delimited
    for line in f:
        p=line.split(); pc_rows.append((p[0].strip('"'), [float(x) for x in p[1:]]))
def repl_class(x):  # ADLr171 -> ADL
    return re.match(r'^([A-Z0-9]+?)r\d+$', x).group(1) if re.match(r'^([A-Z0-9]+?)r\d+$', x) else x
col_class=[repl_class(c) for c in pc_cols]
PCgenes=[g for g,_ in pc_rows]; PCmat=np.array([v for _,v in pc_rows]).T  # replicate x gene
# average replicates per class
pc_by_class={}
for cl in set(col_class):
    idx=[i for i,c in enumerate(col_class) if c==cl]
    pc_by_class[cl]=PCmat[idx].mean(0)
pc_classes=[c for c in cells if c in pc_by_class]
PCX=np.log1p(np.array([pc_by_class[c] for c in cells if c in pc_by_class]))
print(f"protein-coding bulk available for {len(pc_classes)}/{len(cells)} ncRNA neuron types")

# ---- labels: NT + modality ----
d=np.load(NPZ,allow_pickle=True); names=[str(x) for x in d["names"]]; ntp=[str(x) for x in d["nt_primary"]]
def ntn(x):
    for k,v in [("Acetyl","ACh"),("Glu","Glu"),("GABA","GABA"),("Dopamine","mono"),("Serotonin","mono"),("Octopamine","mono"),("Tyramine","mono")]:
        if k in x: return v
    return None
prof={r["neuron"]:r for r in csv.DictReader(open(PROF))}
MOTOR={'VA','VB','VD','DA','DB','DD','AS','VC','RMD','RME','RIM','RMF','RMH','RIV','SMD','SMB','URA'}
def canon(nm,tgt):
    n=nm
    while True:
        if n in tgt: return n
        if len(n)>2 and n[-1] in "LRDV": n=n[:-1]
        elif re.search(r'\d$',n): n=re.sub(r'\d+$','',n)
        else: return None
def modality(nm):
    r=prof.get(nm)
    if r:
        if r.get("is_sensory")in("1","1.0","True"): return "sensory"
        if r.get("is_command")in("1","1.0","True"): return "command"
    base=nm[:-1] if nm[-1:] in "LR" else nm
    return "motor" if (base in MOTOR or nm in MOTOR) else "inter"
cellset=set(cells)
ntv=defaultdict(list); modv=defaultdict(list)
for nm,nt in zip(names,ntp):
    c=canon(nm,cellset); v=ntn(nt)
    if c and v: ntv[c].append(v)
for nm in names:
    c=canon(nm,cellset)
    if c: modv[c].append(modality(nm))
cls_nt={c:Counter(v).most_common(1)[0][0] for c,v in ntv.items()}
cls_mod={c:Counter(v).most_common(1)[0][0] for c,v in modv.items()}

def cv_multi(Xf, labels_by_cell, feat_cells):
    idx=[i for i,c in enumerate(feat_cells) if c in labels_by_cell]
    Xu=Xf[idx]; y=np.array([labels_by_cell[feat_cells[i]] for i in idx])
    cnt=Counter(y); keep=np.array([cnt[v]>=2 for v in y]); Xu=Xu[keep]; y=y[keep]
    k=min(5,min(Counter(y).values()))
    if k<2 or len(set(y))<2: return None
    clf=make_pipeline(StandardScaler(),PCA(n_components=min(15,Xu.shape[0]-1,Xu.shape[1])),LogisticRegression(max_iter=2000))
    yp=cross_val_predict(clf,Xu,y,cv=StratifiedKFold(k,shuffle=True,random_state=0))
    return dict(n=len(y),K=len(set(y)),f1=f1_score(y,yp,average='macro'),ba=balanced_accuracy_score(y,yp))

print("\n=========== SANITY: does ncRNA carry neuron identity? ===========")
for tgt,lab in [("NT",cls_nt),("modality",cls_mod)]:
    r=cv_multi(Xk,lab,cells)
    rp=cv_multi(PCX,lab,pc_classes)
    # random null: shuffle labels
    nulls=[]
    for s in range(80):
        idx=[i for i,c in enumerate(cells) if c in lab]
        y=np.array([lab[cells[i]] for i in idx]); ys=RNG.permutation(y)
        d2={cells[idx[j]]:ys[j] for j in range(len(idx))}
        rr=cv_multi(Xk,d2,cells)
        if rr: nulls.append(rr['f1'])
    nulls=np.array(nulls); p=(1+(nulls>=r['f1']).sum())/(1+len(nulls))
    print(f"  {tgt:9s}: ncRNA macroF1={r['f1']:.3f} (n={r['n']},K={r['K']}) | protein-coding={rp['f1']:.3f} | "
          f"shuffle-null {nulls.mean():.3f} p={p:.3f}")

# =========== WIRING: does ncRNA add to contact topology? ===========
# contact matrix
wb=openpyxl.load_workbook(CONTACT,read_only=True,data_only=True); ws=wb['adult nerve ring neighbors']
rows=list(ws.iter_rows(values_only=True)); alln=[h for h in rows[0][1:]]; idx={n:i for i,n in enumerate(alln)}; M=len(alln)
A=np.zeros((M,M))
for row in rows[1:]:
    rn=row[0]
    if rn in idx:
        for j,val in enumerate(row[1:]):
            if val not in (None,''):
                try:A[idx[rn],j]=float(val)
                except:pass
A=np.maximum(A,A.T);B=(A>0).astype(float);deg=A.sum(1)
pres=defaultdict(set)
for r in csv.DictReader(open(EDGES)):
    ds=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
    if ds in(7,8) and r['type']!='electrical':
        pres[(min(r['pre'],r['post']),max(r['pre'],r['post']))].add(ds)
# map ncRNA neuron type -> contact neurons (collapse L/R): class c contacts via its member neurons
def members(c):
    return [n for n in alln if n and canon(n,{c})==c]
# ncRNA PCA embedding per class
emb=make_pipeline(StandardScaler(),PCA(n_components=15)).fit_transform(Xk)
embd={cells[i]:emb[i] for i in range(len(cells))}
# build contacting class pairs (both classes have ncRNA + >=1 contact member)
classes=[c for c in cells if members(c)]
def cls_contact(ca,cb):
    ma,mb=members(ca),members(cb); best=0
    for a in ma:
        for b in mb:
            best=max(best,A[idx[a],idx[b]])
    return best
def cls_strong(ca,cb):
    ma,mb=members(ca),members(cb)
    for a in ma:
        for b in mb:
            k=(min(a,b),max(a,b))
            if len(pres.get(k,()))==2: return 1
    return 0
pairs=[];y=[];Xtopo=[];Xnc=[]
for i in range(len(classes)):
    for j in range(i+1,len(classes)):
        ca,cb=classes[i],classes[j]; area=cls_contact(ca,cb)
        if area<=0: continue
        # topo features at class level
        ma,mb=members(ca)[0],members(cb)[0]; ia,ib=idx[ma],idx[mb]
        cos=float(A[ia]@A[ib]/(np.linalg.norm(A[ia])*np.linalg.norm(A[ib])+1e-9))
        cn=int(((B[ia]>0)&(B[ib]>0)).sum())
        Xtopo.append([math.log1p(area),cos,cn,math.log1p(deg[ia]),math.log1p(deg[ib])])
        va,vb=embd[ca],embd[cb]; Xnc.append(np.concatenate([va+vb,np.abs(va-vb)]))
        pairs.append((ca,cb)); y.append(cls_strong(ca,cb))
y=np.array(y);Xtopo=np.array(Xtopo);Xnc=np.array(Xnc)
print(f"\n=========== WIRING: contacting class-pairs N={len(y)} strong-core {y.sum()} ({100*y.mean():.0f}%), {len(classes)} classes ===========")
def cdcv(X,seed):
    rng=np.random.default_rng(seed);f=np.array_split(rng.permutation(classes),5);cf={c:i for i,ff in enumerate(f) for c in ff}
    pred=np.full(len(y),np.nan)
    for k in range(5):
        te=np.array([cf[a]==k and cf[b]==k for a,b in pairs]);tr=np.array([cf[a]!=k and cf[b]!=k for a,b in pairs])
        if te.sum()<8 or y[tr].sum()<4 or y[te].sum()<2 or len(set(y[te]))<2: continue
        c=make_pipeline(StandardScaler(with_mean=False),LogisticRegression(max_iter=2000));c.fit(X[tr],y[tr]);pred[te]=c.predict_proba(X[te])[:,1]
    m=~np.isnan(pred);return roc_auc_score(y[m],pred[m]) if (m.sum()>15 and len(set(y[m]))>1) else np.nan
def cdm(X): return np.nanmean([cdcv(X,s) for s in range(6)])
topo=cdm(Xtopo); tnc=cdm(np.hstack([Xtopo,Xnc]))
print(f"  class-disjoint AUC: topology={topo:.3f} | topology+ncRNA={tnc:.3f} | Δ={tnc-topo:+.3f}")
# random null: does a random ncRNA subset add as much? (shuffle ncRNA class labels)
nulls=[]
for s in range(20):
    rng=np.random.default_rng(500+s); perm=rng.permutation(len(classes)); shuf={classes[i]:embd[classes[perm[i]]] for i in range(len(classes))}
    Xr=np.array([np.concatenate([shuf[a]+shuf[b],np.abs(shuf[a]-shuf[b])]) for a,b in pairs])
    nulls.append(cdm(np.hstack([Xtopo,Xr])))
nulls=np.array(nulls)
print(f"  shuffled-ncRNA null: topology+shuf mean={nulls.mean():.3f} 95pct={np.percentile(nulls,95):.3f} | P(null>=real)={(1+(nulls>=tnc).sum())/(1+len(nulls)):.3f}")
print("\nVERDICT: ncRNA carries identity iff SANITY beats shuffle; ncRNA adds to wiring iff Δ>0 AND real>shuffled-ncRNA null.")
