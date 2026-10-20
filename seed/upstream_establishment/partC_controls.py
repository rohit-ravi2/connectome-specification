#!/usr/bin/env python3
"""Part C controls / nulls for the regulatory add-value (GBM, where the signal lives).
Nulls: (1) neuron-label shuffle (break identity<->wiring), (2) random matched-gene sets
(is it THESE curated genes?), (3) lineage-shuffle, (4) contact-stratified add.
Reference claim = REG_all (GRN TF-act + terminal-selector + module scores) delta-AUC over
contact+birth+lineage+NT baseline, neuron-disjoint CV.
"""
import json, gzip, re, os, warnings; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT=os.path.expanduser("../seed")
OUT=f"{ROOT}/data/processed/upstream_early_establishment"
CENGEN="../data/celegans/data/expression/cengen/thresholded/021821_liberal_threshold1.csv"
GRN=os.path.expanduser("../grn/CelEsT-app/www/CelEsT_GRN.txt")
TFLIST=os.path.expanduser("../grn/CelEsT-app/www/fullset_TFs_BM_with_MOR_for_TS2.txt")
GENEIDS="../data/celegans/data/raw/wormbase_full_release/Downloads/wormbase_full_release/c_elegans/PRJNA13758/annotation/c_elegans.PRJNA13758.WS297.geneIDs.txt.gz"
GENESETS=f"{ROOT}/data/processed/regulatory_branch/gene_sets_effector.json"
NDRAW=20

df=pd.read_csv(f"{OUT}/partC_pairwise_features.csv")
neurons=sorted(set(df.pre)|set(df.post))
nmap=json.load(open(f"{OUT}/partC_neuron_class_map.json"))
cg=pd.read_csv(CENGEN); cg_classes=list(cg.columns[3:])
wbid=cg["Wormbase_ID"].astype(str).values
X=cg[cg_classes].apply(pd.to_numeric,errors="coerce").fillna(0.0).values; Xln=np.log1p(X)
wb2row={w:i for i,w in enumerate(wbid)}
neuron_cols={n:[cg_classes.index(c) for c in nmap[n]] for n in neurons}
Z=(Xln-Xln.mean(1,keepdims=True))/(Xln.std(1,keepdims=True)+1e-9)

# rebuild GRN tf_act (class x TF)
seq2wb={}
with gzip.open(GENEIDS,'rt') as f:
    for line in f:
        p=line.rstrip().split(",")
        if len(p)>=4:
            if p[3]: seq2wb[p[3]]=p[1]
            if p[2]: seq2wb.setdefault(p[2],p[1])
grn=pd.read_csv(GRN,sep="\t"); grn.columns=[c.strip('"') for c in grn.columns]
for c in ['source','target']: grn[c]=grn[c].astype(str).str.strip('"')
grn['weight']=pd.to_numeric(grn['weight'],errors='coerce').fillna(0.0)
tfl=pd.read_csv(TFLIST,sep="\t")
g2wb=dict(zip(tfl['wormbase_gseq'].astype(str),tfl['wormbase_gene'].astype(str)))
p2wb=dict(zip(tfl['wormbase_locus'].astype(str),tfl['wormbase_gene'].astype(str)))
grn['tgt_row']=grn['target'].map(lambda s: wb2row.get(seq2wb.get(s,"")))
grn['src_wb']=grn['source'].map(lambda s:g2wb.get(s) or p2wb.get(s) or seq2wb.get(s))
grn=grn.dropna(subset=['tgt_row']); grn['tgt_row']=grn['tgt_row'].astype(int)
tfs=sorted(grn['src_wb'].dropna().unique())
tf_act=np.zeros((len(tfs),len(cg_classes)))
for ti,tf in enumerate(tfs):
    sub=grn[grn['src_wb']==tf]; r=sub['tgt_row'].values; w=sub['weight'].values
    if len(r): tf_act[ti]=(w[:,None]*Z[r]).sum(0)/(np.abs(w).sum()+1e-9)

gs=json.load(open(GENESETS))
tsel_rows=[wb2row[w] for w in gs['categories']['terminal_selector_identity']['genes'] if w in wb2row]
cats=sorted(gs['categories']); cat_rows={c:[wb2row[w] for w in gs['categories'][c]['genes'] if w in wb2row] for c in cats}
allrows=list(range(len(wbid)))

def profiles(assign, tsel_r, cat_r, tf_mat):
    """assign: neuron->column-list. returns dict of per-neuron feature-profile parts."""
    P={}
    for n in neurons:
        idx=assign[n]
        P[n]=(tf_mat[:,idx].mean(1), Xln[np.ix_(tsel_r,idx)].mean(1),
              np.array([Xln[np.ix_(cat_r[c],idx)].mean() for c in cats]))
    return P
def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
def pear(a,b): a=a-a.mean(); b=b-b.mean(); return cos(a,b)
def build_reg(P):
    rows=[]
    for pre,post in zip(df.pre,df.post):
        tfp,tsp,mp=P[pre]; tfq,tsq,mq=P[post]
        r=[cos(tfp,tfq),pear(tfp,tfq),float(np.linalg.norm(tfp-tfq)),
           float(min(np.linalg.norm(tfp),np.linalg.norm(tfq))),float(max(np.linalg.norm(tfp),np.linalg.norm(tfq))),
           cos(tsp,tsq),pear(tsp,tsq),float(((tsp>0.5)&(tsq>0.5)).sum()),
           float(min((tsp>0.5).sum(),(tsq>0.5).sum())),float(max((tsp>0.5).sum(),(tsq>0.5).sum()))]
        r+=[float(mp[i]*mq[i]) for i in range(len(cats))]+[float(abs(mp[i]-mq[i])) for i in range(len(cats))]
        rows.append(r)
    return np.array(rows)

# baseline
def block_birth():
    pb=df.pre_birth.values.astype(float); qb=df.post_birth.values.astype(float)
    mn=np.nanmin(np.c_[pb,qb],1); mx=np.nanmax(np.c_[pb,qb],1)
    ad=np.where(np.isnan(pb)|np.isnan(qb),0,np.abs(pb-qb))
    ld=np.nan_to_num(df.lin_dist.values.astype(float),nan=20.); sf=np.nan_to_num(df.same_founder.values.astype(float))
    return np.nan_to_num(np.c_[mn,mx,ad,ld,sf])
Xc=df[["contact_area","contact_cos","contact_cn"]].values.astype(float)
Xn=pd.get_dummies(df.pre_nt.astype(str)+"|"+df.post_nt.astype(str)).values.astype(float)
BASE=np.hstack([Xc,block_birth(),Xn])

def folds(seed):
    rng=np.random.default_rng(seed); f=np.array_split(rng.permutation(neurons),5)
    return {n:i for i,ff in enumerate(f) for n in ff}
Ppre=df.pre.values; Qpost=df.post.values
def auc_gbm(Xf,y,seeds):
    Xf=np.nan_to_num(Xf); a=[]
    for s in seeds:
        nf=folds(s); P=np.array([nf[n] for n in Ppre]); Q=np.array([nf[n] for n in Qpost]); pr=np.full(len(y),np.nan)
        for k in range(5):
            te=(P==k)&(Q==k); tr=(P!=k)&(Q!=k)
            if te.sum()<8 or y[tr].sum()<5 or len(set(y[te]))<2: continue
            clf=GradientBoostingClassifier(random_state=s,n_estimators=100,max_depth=3).fit(Xf[tr],y[tr])
            pr[te]=clf.predict_proba(Xf[te])[:,1]
        m=~np.isnan(pr)
        if m.sum()>15 and len(set(y[m]))>1: a.append(roc_auc_score(y[m],pr[m]))
    return np.mean(a) if a else np.nan
def delta(REG,y,seeds): return auc_gbm(np.hstack([BASE,REG]),y,seeds)-auc_gbm(BASE,y,seeds)

TARGETS=[("L1_reproducible","T_L1_repro"),("preadult_persistent","T_preadult_persist"),("first_synapse_L1","T_first_L1")]
real_assign={n:neuron_cols[n] for n in neurons}
P_real=profiles(real_assign,tsel_rows,cat_rows,tf_act)
REG_real=build_reg(P_real)
rng=np.random.default_rng(1)
res=[]
for tname,tk in TARGETS:
    y=df[tk].values.astype(int)
    d_real=delta(REG_real,y,range(3))
    # (1) neuron-label shuffle
    ns=[]
    for _ in range(NDRAW):
        perm=rng.permutation(neurons); amap={neurons[i]:neuron_cols[perm[i]] for i in range(len(neurons))}
        Pn=profiles(amap,tsel_rows,cat_rows,tf_act); ns.append(delta(build_reg(Pn),y,range(2)))
    ns=np.array(ns)
    # (2) random matched-gene sets (tsel + module) + shuffled tf_act columns
    rg=[]
    for _ in range(NDRAW):
        rt=list(rng.choice(allrows,size=len(tsel_rows),replace=False))
        rc={c:list(rng.choice(allrows,size=max(1,len(cat_rows[c])),replace=False)) for c in cats}
        tfa=tf_act[:,rng.permutation(len(cg_classes))]
        Pr=profiles(real_assign,rt,rc,tfa); rg.append(delta(build_reg(Pr),y,range(2)))
    rg=np.array(rg)
    z_ns=(d_real-np.nanmean(ns))/(np.nanstd(ns)+1e-9); p_ns=(np.sum(ns>=d_real)+1)/(len(ns)+1)
    z_rg=(d_real-np.nanmean(rg))/(np.nanstd(rg)+1e-9); p_rg=(np.sum(rg>=d_real)+1)/(len(rg)+1)
    print(f"[{tname}] d_real={d_real:+.4f} | neuron-shuffle {np.nanmean(ns):+.4f}±{np.nanstd(ns):.4f} z={z_ns:+.2f} p={p_ns:.3f} | random-gene {np.nanmean(rg):+.4f}±{np.nanstd(rg):.4f} z={z_rg:+.2f} p={p_rg:.3f}")
    res.append(dict(target=tname,d_real=d_real,neuron_shuffle_mean=float(np.nanmean(ns)),neuron_shuffle_sd=float(np.nanstd(ns)),z_neuron_shuffle=float(z_ns),p_neuron_shuffle=float(p_ns),
                    random_gene_mean=float(np.nanmean(rg)),random_gene_sd=float(np.nanstd(rg)),z_random_gene=float(z_rg),p_random_gene=float(p_rg)))

# (4) contact-stratified: does REG add within contact-strength quartiles?
print("\ncontact-stratified add (GBM, REG_all over baseline within contact_area quartiles):")
qa=pd.qcut(df.contact_area,4,labels=False,duplicates='drop').values
for tname,tk in TARGETS:
    y=df[tk].values.astype(int); ds=[]
    for q in sorted(set(qa)):
        sel=qa==q
        if y[sel].sum()<15: ds.append(np.nan); continue
        # within-stratum: reuse global folds but restrict rows (approx): fit on stratum only
        sub=np.where(sel)[0]
        # quick: AUC delta on stratum using seed range(2)
        def aq(Xf):
            Xf=np.nan_to_num(Xf); a=[]
            for s in range(2):
                nf=folds(s); P=np.array([nf[n] for n in Ppre]); Q=np.array([nf[n] for n in Qpost]); pr=np.full(len(y),np.nan)
                for k in range(5):
                    te=(P==k)&(Q==k)&sel; tr=((P!=k)&(Q!=k));
                    if te.sum()<6 or y[tr].sum()<5 or len(set(y[te]))<2: continue
                    clf=GradientBoostingClassifier(random_state=s,n_estimators=80,max_depth=3).fit(Xf[tr],y[tr]); pr[te]=clf.predict_proba(Xf[te])[:,1]
                m=~np.isnan(pr)
                if m.sum()>10 and len(set(y[m]))>1: a.append(roc_auc_score(y[m],pr[m]))
            return np.mean(a) if a else np.nan
        ds.append(aq(np.hstack([BASE,REG_real]))-aq(BASE))
    print(f"  [{tname}] per-quartile REG add: "+" ".join(f"{d:+.3f}" if d==d else "na" for d in ds))
    res.append(dict(target=tname+"_contactstrat",d_real=float(np.nanmean(ds))))
pd.DataFrame(res).to_csv(f"{OUT}/partC_controls.csv",index=False)
print("\nsaved partC_controls.csv")
