#!/usr/bin/env python3
"""Round3 -- INDIRECT miRNA-activity footprint test (mature miRNA is unmeasurable at neuron
resolution -> infer ACTIVITY from target repression in mRNA we DO have).

For each conserved miRNA family (TargetScanWorm 6.2), define target genes. In each neuron type,
activity = coordinated REPRESSION of targets vs background:
    activity[fam,neuron] = mean_z(non-targets) - mean_z(targets)   (z across neuron types per gene)
High = targets collectively low in that neuron = miRNA active. Gives a (neuron x family) inferred
activity matrix -- a proxy for the unmeasured mature-miRNA layer.

Then: SANITY (does inferred activity carry identity: NT/modality?) and WIRING (does it add to
contact topology for reproducible strong-core, neuron-disjoint CV, vs shuffled-activity null?).
Honest caveat: indirect + underpowered; targets are predictions.
"""
import openpyxl, csv, re, os, math
import numpy as np
from collections import defaultdict, Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

DATA="../data/celegans/data"
OUT=os.path.expanduser("../seed/data/processed/formal_branch")
TS=f"{DATA}/mirna_targets/targetscan_worm62/Conserved_Family_Info.txt"
FAMINFO=f"{DATA}/mirna_targets/targetscan_worm62/miR_Family_Info.txt"
GTF=f"{DATA}/expression/scrna_seq_other/GSE136049/GSE136049_c_elegans.PRJNA13758.WS273.canonical_geneset.extend3UTR_optimized.gtf"
PC_BULK=f"{DATA}/expression/cengen/bulk_barrett2022/Average_integrated_TMM_counts_lengthNormalized_111521.tsv"
CONTACT=f"{DATA}/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx"
EDGES=f"{DATA}/connectome/witvliet2020/witvliet_combined_edges.csv"
NPZ="../data/tier5/connectome.npz"
PROF=os.path.expanduser("../data/celegans/New Notebooks/data_derived/nb28_neuron_profile.csv")
RNG=np.random.default_rng(0)

# ---- seqname/transcript -> WBGene from GTF ----
seq2wb={}
for l in open(GTF):
    if "\ttranscript\t" not in l and "\tgene\t" not in l: continue
    g=re.search(r'gene_id "([^"]+)"',l); t=re.search(r'transcript_id "([^"]+)"',l)
    if g:
        wb=g.group(1)
        if t:
            tid=t.group(1); seq2wb[tid]=wb
            seq2wb[re.sub(r'[a-z]$','',tid)]=wb          # strip isoform letter
            seq2wb[re.sub(r'\.\d+$','',tid)]=wb          # strip .N
print(f"seqname->WBGene keys: {len(seq2wb)}")

# ---- conserved miRNA family -> target WBGenes (species 6239 = C. elegans) ----
fam_targets=defaultdict(set)
with open(TS) as f:
    hdr=f.readline().rstrip('\n').split('\t')
    ci={h:i for i,h in enumerate(hdr)}
    for line in f:
        p=line.rstrip('\n').split('\t')
        if p[ci['Species ID']]!='6239': continue
        sym=p[ci['Gene Symbol']]
        wb=seq2wb.get(sym) or seq2wb.get(re.sub(r'\.\d+$','',sym)) or seq2wb.get(re.sub(r'[a-z]$','',sym))
        if wb: fam_targets[p[ci['miR Family']]].add(wb)
print(f"miRNA families with mapped targets: {len(fam_targets)} | median targets/family: "
      f"{int(np.median([len(v) for v in fam_targets.values()]))}")

# ---- CeNGEN bulk mRNA (41 neuron types) ----
with open(PC_BULK) as f:
    cols=[h.strip().strip('"') for h in f.readline().split()]
    genes=[]; mat=[]
    for line in f:
        pp=line.split(); genes.append(pp[0].strip('"')); mat.append([float(x) for x in pp[1:]])
mat=np.array(mat)  # gene x replicate
def repl_class(x):
    m=re.match(r'^([A-Z0-9]+?)r\d+$',x); return m.group(1) if m else x
col_class=[repl_class(c) for c in cols]
classes=sorted(set(col_class))
# average replicates per class -> class x gene (log)
byc=np.array([np.log1p(mat[:,[i for i,c in enumerate(col_class) if c==cl]].mean(1)) for cl in classes])  # class x gene
gi={g:i for i,g in enumerate(genes)}
# z-score each gene across classes
Z=(byc-byc.mean(0))/(byc.std(0)+1e-9)   # class x gene
print(f"CeNGEN bulk: {len(classes)} neuron types x {len(genes)} genes")

# ---- activity matrix: class x family ----
fams=[fam for fam,t in fam_targets.items() if len([g for g in t if g in gi])>=15]
print(f"families with >=15 targets present in bulk: {len(fams)}")
ACT=np.zeros((len(classes),len(fams)))
for fj,fam in enumerate(fams):
    tgt=[gi[g] for g in fam_targets[fam] if g in gi]
    tmask=np.zeros(len(genes),bool); tmask[tgt]=True
    zt=Z[:,tmask].mean(1); zn=Z[:,~tmask].mean(1)
    ACT[:,fj]=zn-zt        # targets repressed => positive activity
cls_i={c:i for i,c in enumerate(classes)}

# ---- labels ----
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
        if n and len(n)>2 and n[-1] in "LRDV": n=n[:-1]
        elif n and re.search(r'\d$',n): n=re.sub(r'\d+$','',n)
        else: return None
def modality(nm):
    r=prof.get(nm)
    if r:
        if r.get("is_sensory")in("1","1.0","True"): return "sensory"
        if r.get("is_command")in("1","1.0","True"): return "command"
    base=nm[:-1] if nm[-1:] in "LR" else nm
    return "motor" if (base in MOTOR or nm in MOTOR) else "inter"
cset=set(classes); ntv=defaultdict(list); modv=defaultdict(list)
for nm,nt in zip(names,ntp):
    c=canon(nm,cset); v=ntn(nt)
    if c and v: ntv[c].append(v)
for nm in names:
    c=canon(nm,cset)
    if c: modv[c].append(modality(nm))
cls_nt={c:Counter(v).most_common(1)[0][0] for c,v in ntv.items()}
cls_mod={c:Counter(v).most_common(1)[0][0] for c,v in modv.items()}

def cv_multi(Xf,lab,feat_cells):
    idx=[i for i,c in enumerate(feat_cells) if c in lab]
    Xu=Xf[idx]; y=np.array([lab[feat_cells[i]] for i in idx])
    cnt=Counter(y); k=np.array([cnt[v]>=2 for v in y]); Xu=Xu[k]; y=y[k]
    kk=min(5,min(Counter(y).values()))
    if kk<2 or len(set(y))<2: return None
    clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,C=0.5))
    yp=cross_val_predict(clf,Xu,y,cv=StratifiedKFold(kk,shuffle=True,random_state=0))
    return dict(n=len(y),K=len(set(y)),f1=f1_score(y,yp,average='macro'))
print("\n=========== SANITY: does inferred miRNA ACTIVITY carry identity? ===========")
for tgt,lab in [("NT",cls_nt),("modality",cls_mod)]:
    r=cv_multi(ACT,lab,classes)
    nulls=[]
    for s in range(100):
        idx=[i for i,c in enumerate(classes) if c in lab]; y=np.array([lab[classes[i]] for i in idx])
        ys=RNG.permutation(y); d2={classes[idx[j]]:ys[j] for j in range(len(idx))}
        rr=cv_multi(ACT,d2,classes)
        if rr: nulls.append(rr['f1'])
    nulls=np.array(nulls); p=(1+(nulls>=r['f1']).sum())/(1+len(nulls))
    print(f"  {tgt:9s}: miRNA-activity macroF1={r['f1']:.3f} (n={r['n']},K={r['K']}) | shuffle-null {nulls.mean():.3f} p={p:.3f}")

# =========== WIRING ===========
wb=openpyxl.load_workbook(CONTACT,read_only=True,data_only=True); ws=wb['adult nerve ring neighbors']
rows=list(ws.iter_rows(values_only=True)); alln=[h for h in rows[0][1:]]; idx2={n:i for i,n in enumerate(alln)}; M=len(alln)
A=np.zeros((M,M))
for row in rows[1:]:
    rn=row[0]
    if rn in idx2:
        for j,val in enumerate(row[1:]):
            if val not in (None,''):
                try:A[idx2[rn],j]=float(val)
                except:pass
A=np.maximum(A,A.T);B=(A>0).astype(float);deg=A.sum(1)
pres=defaultdict(set)
for r in csv.DictReader(open(EDGES)):
    ds=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
    if ds in(7,8) and r['type']!='electrical':
        pres[(min(r['pre'],r['post']),max(r['pre'],r['post']))].add(ds)
def members(c): return [n for n in alln if n and canon(n,{c})==c]
wclasses=[c for c in classes if members(c)]
# activity embedding per class (already ACT rows)
actd={classes[i]:ACT[i] for i in range(len(classes))}
def strong(ca,cb):
    for a in members(ca):
        for b in members(cb):
            if len(pres.get((min(a,b),max(a,b)),()))==2: return 1
    return 0
def contact(ca,cb):
    best=0
    for a in members(ca):
        for b in members(cb): best=max(best,A[idx2[a],idx2[b]])
    return best
pairs=[];y=[];Xtopo=[];Xact=[]
for i in range(len(wclasses)):
    for j in range(i+1,len(wclasses)):
        ca,cb=wclasses[i],wclasses[j]; ar=contact(ca,cb)
        if ar<=0: continue
        ia,ib=idx2[members(ca)[0]],idx2[members(cb)[0]]
        cos=float(A[ia]@A[ib]/(np.linalg.norm(A[ia])*np.linalg.norm(A[ib])+1e-9))
        cn=int(((B[ia]>0)&(B[ib]>0)).sum())
        Xtopo.append([math.log1p(ar),cos,cn,math.log1p(deg[ia]),math.log1p(deg[ib])])
        va,vb=actd[ca],actd[cb]; Xact.append(np.concatenate([va+vb,np.abs(va-vb)]))
        pairs.append((ca,cb)); y.append(strong(ca,cb))
y=np.array(y);Xtopo=np.array(Xtopo);Xact=np.array(Xact)
print(f"\n=========== WIRING: {len(y)} contacting class-pairs, strong-core {y.sum()} ({100*y.mean():.0f}%), {len(wclasses)} classes ===========")
def cdcv(X,seed):
    rng=np.random.default_rng(seed);f=np.array_split(rng.permutation(wclasses),5);cf={c:i for i,ff in enumerate(f) for c in ff}
    pred=np.full(len(y),np.nan)
    for k in range(5):
        te=np.array([cf[a]==k and cf[b]==k for a,b in pairs]);tr=np.array([cf[a]!=k and cf[b]!=k for a,b in pairs])
        if te.sum()<8 or y[tr].sum()<4 or y[te].sum()<2 or len(set(y[te]))<2: continue
        c=make_pipeline(StandardScaler(with_mean=False),LogisticRegression(max_iter=2000,C=0.5));c.fit(X[tr],y[tr]);pred[te]=c.predict_proba(X[te])[:,1]
    m=~np.isnan(pred);return roc_auc_score(y[m],pred[m]) if (m.sum()>15 and len(set(y[m]))>1) else np.nan
def cdm(X): return np.nanmean([cdcv(X,s) for s in range(8)])
topo=cdm(Xtopo); tact=cdm(np.hstack([Xtopo,Xact]))
print(f"  class-disjoint AUC: topology={topo:.3f} | topology+miRNA-activity={tact:.3f} | Δ={tact-topo:+.3f}")
nulls=[]
for s in range(20):
    rng=np.random.default_rng(700+s); perm=rng.permutation(len(wclasses)); sh={wclasses[i]:actd[wclasses[perm[i]]] for i in range(len(wclasses))}
    Xr=np.array([np.concatenate([sh[a]+sh[b],np.abs(sh[a]-sh[b])]) for a,b in pairs]); nulls.append(cdm(np.hstack([Xtopo,Xr])))
nulls=np.array(nulls)
print(f"  shuffled-activity null: mean={nulls.mean():.3f} 95pct={np.percentile(nulls,95):.3f} P(null>=real)={(1+(nulls>=tact).sum())/(1+len(nulls)):.3f}")

# save
with open(f"{OUT}/mirna_footprint_results.csv","w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["test","value","note"])
    w.writerow(["n_families",len(fams),">=15 targets in bulk"])
    w.writerow(["wiring_topology_AUC",round(topo,4),"class-disjoint"])
    w.writerow(["wiring_topology+miRNAactivity_AUC",round(tact,4),f"delta={tact-topo:+.4f}"])
    w.writerow(["wiring_shuffled_null_mean",round(nulls.mean(),4),f"P(null>=real)={(1+(nulls>=tact).sum())/(1+len(nulls)):.3f}"])
print(f"\nsaved -> {OUT}/mirna_footprint_results.csv")
print("VERDICT: inferred miRNA activity is a proxy (indirect). Adds to wiring iff Δ>0 AND real>shuffled null.")
