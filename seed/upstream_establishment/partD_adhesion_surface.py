#!/usr/bin/env python3
"""Part D: does LOCAL cell-surface / synaptogenic-machinery expression predict
PRE-ADULT connectome establishment beyond contact/birth/lineage/NT?

- Master table targets are PRE-ADULT ONLY. adult_strong_core* NEVER used as predictor.
- Baseline (Part A analogue) = contact(area,cos,cn)+birth(min,max,|diff|)+lineage(dist,founder)+NT-pair one-hot,
  rebuilt from master columns, on identical neuron-disjoint folds.
- Modules are PRE-REGISTERED (see MODULES below); no cherry-picking.
- CONTROLS: each module vs >=20 expression-matched, size-matched random gene modules;
  neuron-disjoint CV; empirical null p; BH multiple-testing correction over all module x target tests.

CeNGEN source: full canonical scRNA neuron-class TPM (021821_liberal_threshold1.csv, 128 classes,
symbols + Wormbase IDs). NOTE: the originally planned derived/expression_neuron_mean.csv is a REDUCED
91-class subset (covers only 2373->fewer pairs); we use the full release for coverage/power and
document the choice. log2(1+TPM) transform. Class-collapse maps L/R/D/V neuron names to CeNGEN classes.
"""
import os, re, json, math, time, csv
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

RNG_MASTER = 20260701
ROOT = os.path.expanduser("../seed")
OUT  = f"{ROOT}/data/processed/upstream_early_establishment"
DOC  = f"{ROOT}/docs/upstream_early_establishment"
os.makedirs(DOC, exist_ok=True)
CENGEN = "../data/celegans/data/expression/cengen/thresholded/021821_liberal_threshold1.csv"
GAF = "../data/celegans/data/wormbase_release_WS297/associations/c_elegans.PRJNA13758.WS297.gene_association.wb"
MASTER = f"{OUT}/upstream_master_table.csv"

N_SEEDS = 3          # neuron-disjoint CV seeds (paired base vs add)
N_NULL  = 20         # matched-random draws per module (>= 20 required)
MAXIT   = 100        # HistGB iterations

# ---------------------------------------------------------------------------
# D1  PRE-REGISTERED curated gene modules (symbols). Source = WormBase WS297 GAF
#     symbol resolution + CeNGEN scRNA. Curated from canonical C. elegans
#     cell-surface / synaptogenesis literature (as enumerated in the task spec).
# ---------------------------------------------------------------------------
MODULES = {
 'IgSF':        ['rig-1','rig-3','rig-4','rig-5','rig-6','zig-1','zig-2','zig-3','zig-4','zig-5','zig-8','oig-1','oig-4','wrk-1','syg-1','syg-2','lad-2','sax-7','dig-1'],
 'cadherin':    ['cdh-1','cdh-3','cdh-4','cdh-6','cdh-7','cdh-8','cdh-11','hmr-1','fmi-1','casy-1'],
 'nrx_nlg':     ['nrx-1','nlg-1'],
 'netrin':      ['unc-6','unc-40','unc-5'],
 'slit_robo':   ['slt-1','sax-3','eva-1'],
 'sema_plexin': ['smp-1','smp-2','plx-1','plx-2','mab-20'],
 'wnt':         ['lin-44','egl-20','cwn-1','cwn-2','mom-2','lin-17','mig-1','cam-1'],
 'active_zone': ['syd-1','syd-2','unc-10','unc-13','cla-1','elks-1','rimb-1'],
 'vesicle':     ['rab-3','snb-1','unc-18','unc-64','unc-104','snt-1'],
 'camkii_prune':['unc-43','pkc-2'],
 'glial':       ['ced-1'],
}
# D3 complementarity: directed source-ligand x target-receptor pairs (pre-registered)
COMPLEMENTARITY = [   # (source_gene, target_gene, label)
 ('unc-6','unc-40','netrin_attr'),
 ('unc-6','unc-5','netrin_rep'),
 ('slt-1','sax-3','slit_robo'),
 ('slt-1','eva-1','slit_eva'),
 ('syg-2','syg-1','syg'),
 ('nrx-1','nlg-1','nrx_nlg'),
 ('smp-1','plx-1','sema1'),
 ('smp-2','plx-2','sema2'),
 ('efn-4','vab-1','ephrin'),   # extra canonical L-R pair
]
# presynaptic-assembly modules (source-side biology) for D4 directionality
PRESYN_MODULES = ['active_zone','vesicle']

# ---------------------------------------------------------------------------
def load_symbol_map():
    sym2wb={}
    with open(GAF) as f:
        for line in f:
            if line.startswith('!'): continue
            p=line.rstrip('\n').split('\t')
            if len(p)<3: continue
            wb,sym=p[1],p[2]
            if wb.startswith('WBGene') and sym: sym2wb.setdefault(sym.lower(),wb)
            if len(p)>10 and p[10]:
                for s in p[10].split('|'):
                    s=s.strip()
                    if s: sym2wb.setdefault(s.lower(),wb)
    return sym2wb

def load_expression():
    df=pd.read_csv(CENGEN)
    meta=['','gene_name','Wormbase_ID','Unnamed: 0']
    classes=[c for c in df.columns if c not in meta and not c.startswith('Unnamed')]
    # symbol -> row (log2 TPM+1) ; also WBGene index
    df['gene_name']=df['gene_name'].astype(str)
    E = np.log2(1.0+df[classes].values.astype(float))     # genes x classes
    sym_row = {s.lower():i for i,s in enumerate(df['gene_name'])}
    wb_row  = {w:i for i,w in enumerate(df['Wormbase_ID'].astype(str))}
    classidx={c:j for j,c in enumerate(classes)}
    gene_mean = E.mean(1)                                   # for expression matching
    return E, classes, classidx, sym_row, wb_row, gene_mean

def collapse_map(neurons, cset):
    def collapse(n):
        if n in ('ASEL','ASER','AWC_ON','AWC_OFF'): return n
        if n in ('AWCL','AWCR'): return ('AWC_ON','AWC_OFF')      # avg of both
        if n.startswith('IL2'):
            if n in ('IL2L','IL2R'): return 'IL2_LR'
            return 'IL2_DV'
        if n.startswith('RMD'):
            if n in ('RMDL','RMDR'): return 'RMD_LR'
            return 'RMD_DV'
        if n.startswith('RME'):
            if n in ('RMEL','RMER'): return 'RME_LR'
            return 'RME_DV'
        cands=[n, re.sub(r'(L|R)$','',n), re.sub(r'(DL|DR|VL|VR)$','',n),
               re.sub(r'(DL|DR|VL|VR|D|V|L|R)$','',n)]
        for c in cands:
            if c in cset: return c
        m=re.match(r'^(DA|VA|DB|VB|DD|VD|AS|VC|SMD|SMB|SAA|SIA|SIB|URA|URY|OLQ|SAB|IL1)',n)
        if m and m.group(1) in cset: return m.group(1)
        return None
    return {n:collapse(n) for n in neurons}

def class_vec(E, classidx, cmap_entry):
    """expression column(s) for a mapped class (avg if tuple)."""
    if cmap_entry is None: return None
    if isinstance(cmap_entry, tuple):
        cols=[classidx[c] for c in cmap_entry if c in classidx]
        if not cols: return None
        return E[:,cols].mean(1)
    if cmap_entry not in classidx: return None
    return E[:,classidx[cmap_entry]]

# ---------------------------------------------------------------------------
# neuron-disjoint CV (mirrors Part A ndcv), model = HistGradientBoosting
def ndcv(X,y,pairs,neurons,seed):
    rng=np.random.default_rng(seed)
    folds=np.array_split(rng.permutation(neurons),5)
    nf={n:i for i,ff in enumerate(folds) for n in ff}
    pred=np.full(len(y),np.nan)
    for k in range(5):
        te=np.array([nf[a]==k and nf[b]==k for a,b in pairs])
        trm=np.array([nf[a]!=k and nf[b]!=k for a,b in pairs])
        if te.sum()<10 or y[trm].sum()<5 or y[te].sum()<3 or len(set(y[te]))<2: continue
        clf=HistGradientBoostingClassifier(max_iter=MAXIT,random_state=seed)
        clf.fit(X[trm],y[trm]); pred[te]=clf.predict_proba(X[te])[:,1]
    m=~np.isnan(pred)
    return roc_auc_score(y[m],pred[m]) if (m.sum()>15 and len(set(y[m]))>1) else np.nan

def base_auc_by_seed(Xbase,y,pairs,neurons,seeds):
    return {s:ndcv(Xbase,y,pairs,neurons,s) for s in seeds}

def paired_delta(Xbase,Xadd,y,pairs,neurons,seeds,base_seed):
    """add-model only; delta vs precomputed baseline per seed (identical folds)."""
    as_,dl=[],[]
    for s in seeds:
        a=ndcv(np.hstack([Xbase,Xadd]),y,pairs,neurons,s)
        b=base_seed[s]; as_.append(a)
        if not(np.isnan(a) or np.isnan(b)): dl.append(a-b)
    return None,np.nanmean(as_),(np.mean(dl) if dl else np.nan)

# ---------------------------------------------------------------------------
def build_module_features(gene_syms, E, classidx, sym_row, wb_row, sym2wb, cvecs_pre, cvecs_tgt, active):
    """5 pairwise features per module: src_score,tgt_score,|diff|,product,cosine over module genes."""
    rows=[]
    for g in gene_syms:
        i=sym_row.get(g.lower())
        if i is None:
            wb=sym2wb.get(g.lower())
            if wb is not None: i=wb_row.get(wb)
        if i is not None: rows.append(i)
    rows=np.array(sorted(set(rows)))
    if len(rows)==0: return None,0
    feats=np.zeros((len(active),5))
    for idx,(vp,vt) in enumerate(zip(cvecs_pre,cvecs_tgt)):
        sp=vp[rows]; st=vt[rows]
        src=sp.mean(); tgt=st.mean()
        cos=float(sp@st/(np.linalg.norm(sp)*np.linalg.norm(st)+1e-9))
        feats[idx]=[src,tgt,abs(src-tgt),src*tgt,cos]
    return feats,len(rows)

def build_comp_features(pairs_lr, E, classidx, sym_row, wb_row, sym2wb, cvecs_pre, cvecs_tgt, active):
    """directed source-ligand x target-receptor products."""
    cols=[]; used=[]
    def rowof(g):
        i=sym_row.get(g.lower())
        if i is None:
            wb=sym2wb.get(g.lower());  i=wb_row.get(wb) if wb else None
        return i
    ri=[(rowof(s),rowof(t),lab) for s,t,lab in pairs_lr]
    ri=[(a,b,lab) for a,b,lab in ri if a is not None and b is not None]
    F=np.zeros((len(active),len(ri)))
    for idx,(vp,vt) in enumerate(zip(cvecs_pre,cvecs_tgt)):
        for j,(a,b,lab) in enumerate(ri):
            F[idx,j]=vp[a]*vt[b]
    return F,[lab for _,_,lab in ri]

# ---------------------------------------------------------------------------
def main():
    t0=time.time()
    print("loading expression + symbol map ...")
    sym2wb=load_symbol_map()
    E,classes,classidx,sym_row,wb_row,gene_mean=load_expression()
    cset=set(classes)

    m=pd.read_csv(MASTER)
    neurons_all=sorted(set(m.pre)|set(m.post))
    cmap=collapse_map(neurons_all,cset)

    # restrict to pairs where BOTH neurons map to a CeNGEN class
    def mapped(n): return cmap.get(n) is not None
    keep=m.apply(lambda r: mapped(r['pre']) and mapped(r['post']), axis=1).values
    m=m[keep].reset_index(drop=True)
    print(f"pairs with both neurons mapped: {len(m)} / {len(keep)}")

    pairs=list(zip(m.pre,m.post))
    neurons=sorted(set(m.pre)|set(m.post))

    # precompute per-pair source/target expression column vectors
    cvecs_pre=[class_vec(E,classidx,cmap[p]) for p,_ in pairs]
    cvecs_tgt=[class_vec(E,classidx,cmap[q]) for _,q in pairs]
    active=np.arange(len(pairs))

    # ---- baseline blocks from master columns ----
    Xc=m[['contact_area','contact_cos','contact_cn']].values.astype(float)
    pb=m['pre_birth'].values; qb=m['post_birth'].values
    bmin=np.nanmin(np.vstack([pb,qb]),0); bmax=np.nanmax(np.vstack([pb,qb]),0)
    bdiff=np.where(np.isnan(pb)|np.isnan(qb),0,np.abs(pb-qb))
    ld=np.where(np.isnan(m['lin_dist'].values),20,m['lin_dist'].values)
    Xb=np.nan_to_num(np.vstack([bmin,bmax,bdiff,ld,m['same_founder'].values]).T)
    nt_keys=sorted(set(zip(m.pre_nt,m.post_nt))); nki={k:i for i,k in enumerate(nt_keys)}
    Xn=np.zeros((len(m),len(nt_keys)))
    for i,(a,b) in enumerate(zip(m.pre_nt,m.post_nt)): Xn[i,nki[(a,b)]]=1
    Xbase=np.hstack([Xc,Xb,Xn])
    print(f"baseline dims: contact{Xc.shape[1]} birth/lin{Xb.shape[1]} NT{Xn.shape[1]} -> {Xbase.shape[1]}")

    targets=[('T_L1_repro','L1_reproducible'),('T_preadult_persist','preadult_persistent'),('T_first_L1','first_synapse_L1')]
    seeds=list(range(N_SEEDS))

    # expression-matching: decile bins by gene_mean
    order=np.argsort(gene_mean); bins=np.array_split(order,10)
    gene_bin=np.zeros(len(gene_mean),dtype=int)
    for bi,b in enumerate(bins): gene_bin[b]=bi
    all_rows=np.arange(len(gene_mean))
    rng_null=np.random.default_rng(RNG_MASTER)

    # gene rows per module (for size + bin profile of matched null)
    def module_rows(gene_syms):
        rr=[]
        for g in gene_syms:
            i=sym_row.get(g.lower())
            if i is None:
                wb=sym2wb.get(g.lower()); i=wb_row.get(wb) if wb else None
            if i is not None: rr.append(i)
        return np.array(sorted(set(rr)))

    def matched_random_rows(ref_rows):
        # draw one gene per ref gene from the same expression decile
        picks=[]
        for r in ref_rows:
            b=gene_bin[r]; pool=bins[b]
            picks.append(int(rng_null.choice(pool)))
        return np.array(picks)

    def feats_from_rows(rows):
        F=np.zeros((len(pairs),5))
        for idx,(vp,vt) in enumerate(zip(cvecs_pre,cvecs_tgt)):
            sp=vp[rows]; st=vt[rows]
            src=sp.mean(); tgt=st.mean()
            cos=float(sp@st/(np.linalg.norm(sp)*np.linalg.norm(st)+1e-9))
            F[idx]=[src,tgt,abs(src-tgt),src*tgt,cos]
        return F

    # ---- gene-module inventory (D1 output) ----
    inv=[]
    for mod,genes in MODULES.items():
        rr=module_rows(genes)
        found=[]; miss=[]
        for g in genes:
            if module_rows([g]).size>0: found.append(g)
            else: miss.append(g)
        inv.append(dict(module=mod,n_genes_listed=len(genes),n_genes_found=len(rr),
                        genes_found=';'.join(found),genes_missing=';'.join(miss),
                        source='WormBase WS297 GAF symbol + CeNGEN scRNA (021821 liberal TPM)'))
    pd.DataFrame(inv).to_csv(f"{OUT}/partD_gene_modules.csv",index=False)
    print("wrote partD_gene_modules.csv")

    # =====================================================================
    results=[]
    print("\n== module ADD over baseline (neuron-disjoint, paired) + matched-random null ==")
    for tkey,tname in targets:
        y=m[tkey].values.astype(int)
        base_seed=base_auc_by_seed(Xbase,y,pairs,neurons,seeds)
        base_only=np.nanmean(list(base_seed.values()))
        print(f"\n--- target {tname}  pos={y.sum()}/{len(y)}  baseline AUC={base_only:.4f} ---",flush=True)
        for mod,genes in list(MODULES.items()):
            rr=module_rows(genes)
            if rr.size==0:
                print(f"  {mod:12s} NO GENES"); continue
            Xmod=feats_from_rows(rr)
            b,a,delta=paired_delta(Xbase,Xmod,y,pairs,neurons,seeds,base_seed)
            # matched-random null
            nulls=[]
            for _ in range(N_NULL):
                Xr=feats_from_rows(matched_random_rows(rr))
                _,_,dn=paired_delta(Xbase,Xr,y,pairs,neurons,seeds,base_seed)
                if not np.isnan(dn): nulls.append(dn)
            nulls=np.array(nulls)
            p_emp=(np.sum(nulls>=delta)+1)/(len(nulls)+1) if len(nulls) else np.nan
            null_mean=np.mean(nulls) if len(nulls) else np.nan
            null_sd=np.std(nulls) if len(nulls) else np.nan
            z=(delta-null_mean)/(null_sd+1e-9) if len(nulls) else np.nan
            results.append(dict(target=tname,module=mod,n_genes=int(rr.size),
                base_auc=round(base_only,4),add_auc=round(a,4),delta_auc=round(delta,4),
                null_mean_delta=round(null_mean,4),null_sd=round(null_sd,4),
                z_vs_matched=round(z,3),p_matched_emp=round(p_emp,4),n_null=len(nulls)))
            print(f"  {mod:12s} n={rr.size:2d}  delta={delta:+.4f}  null={null_mean:+.4f}±{null_sd:.4f}  z={z:+.2f}  p={p_emp:.3f}")
        # complementarity module
        Fc,labs=build_comp_features(COMPLEMENTARITY,E,classidx,sym_row,wb_row,sym2wb,cvecs_pre,cvecs_tgt,active)
        b,a,delta=paired_delta(Xbase,Fc,y,pairs,neurons,seeds,base_seed)
        # null: random directed gene-pair products, matched expression, same count
        comp_ref=[]
        for s,t,_ in COMPLEMENTARITY:
            si=sym_row.get(s.lower()) or wb_row.get(sym2wb.get(s.lower(),''))
            ti=sym_row.get(t.lower()) or wb_row.get(sym2wb.get(t.lower(),''))
            if si is not None and ti is not None: comp_ref.append((si,ti))
        nulls=[]
        for _ in range(N_NULL):
            Fr=np.zeros((len(pairs),len(comp_ref)))
            picks=[(int(rng_null.choice(bins[gene_bin[si]])),int(rng_null.choice(bins[gene_bin[ti]]))) for si,ti in comp_ref]
            for idx,(vp,vt) in enumerate(zip(cvecs_pre,cvecs_tgt)):
                for j,(si,ti) in enumerate(picks): Fr[idx,j]=vp[si]*vt[ti]
            _,_,dn=paired_delta(Xbase,Fr,y,pairs,neurons,seeds,base_seed)
            if not np.isnan(dn): nulls.append(dn)
        nulls=np.array(nulls); p_emp=(np.sum(nulls>=delta)+1)/(len(nulls)+1)
        z=(delta-np.mean(nulls))/(np.std(nulls)+1e-9)
        results.append(dict(target=tname,module='complementarity_LR',n_genes=len(comp_ref),
            base_auc=round(base_only,4),add_auc=round(a,4),delta_auc=round(delta,4),
            null_mean_delta=round(float(np.mean(nulls)),4),null_sd=round(float(np.std(nulls)),4),
            z_vs_matched=round(float(z),3),p_matched_emp=round(float(p_emp),4),n_null=len(nulls)))
        print(f"  {'compl_LR':12s} n={len(comp_ref):2d}  delta={delta:+.4f}  null={np.mean(nulls):+.4f}±{np.std(nulls):.4f}  z={z:+.2f}  p={p_emp:.3f}")

        # D4 directionality: presyn-assembly source-side-only vs pairwise
        for mod in PRESYN_MODULES:
            rr=module_rows(MODULES[mod])
            # source-side score only (1 feature: presynaptic assembly is a source property)
            Xsrc=np.array([[vp[rr].mean()] for vp in cvecs_pre])
            _,a_src,d_src=paired_delta(Xbase,Xsrc,y,pairs,neurons,seeds,base_seed)
            results.append(dict(target=tname,module=f'{mod}_SOURCEONLY',n_genes=int(rr.size),
                base_auc=round(base_only,4),add_auc=round(a_src,4),delta_auc=round(d_src,4),
                null_mean_delta=np.nan,null_sd=np.nan,z_vs_matched=np.nan,p_matched_emp=np.nan,n_null=0))
            print(f"  {mod+'_src':12s} delta_src={d_src:+.4f}")

    res=pd.DataFrame(results)
    # BH correction over tests that have an empirical p (curated + complementarity)
    tested=res[res['p_matched_emp'].notna()].copy()
    pv=tested['p_matched_emp'].values
    ordv=np.argsort(pv); n=len(pv); bh=np.empty(n)
    prev=1.0
    for rank in range(n-1,-1,-1):
        i=ordv[rank]; val=pv[i]*n/(rank+1); prev=min(prev,val); bh[i]=prev
    tested['p_BH']=np.round(bh,4)
    res=res.merge(tested[['target','module','p_BH']],on=['target','module'],how='left')
    res.to_csv(f"{OUT}/partD_module_results.csv",index=False)
    print(f"\nwrote partD_module_results.csv  ({len(tested)} null-tested module x target cells, BH corrected)")

    # ---- D5 known cases ----
    known=[]
    kc_pairs={'ASE_L_R':('ASEL','ASER'),'HSN_present':('HSN',None)}
    for lab,(a,b) in kc_pairs.items():
        pres=(a in cset) or (a in [cmap[n] for n in neurons if not isinstance(cmap[n],tuple)])
        known.append(dict(case=lab,cengen_class_present=(a in cset),note='class present in CeNGEN' if a in cset else 'absent'))
    # module expression profile for known identity neurons if present
    for kn in ['ASEL','ASER','HSN','AWA','AFD']:
        if kn in cset:
            j=classidx[kn]
            row={'case':f'{kn}_module_logTPM'}
            for mod,genes in MODULES.items():
                rr=module_rows(genes)
                row[mod]=round(float(E[rr,j].mean()),3) if rr.size else np.nan
            known.append(row)
    pd.DataFrame(known).to_csv(f"{OUT}/partD_known_case_results.csv",index=False)
    print("wrote partD_known_case_results.csv")

    # ---- verdict ----
    best=res[res['p_matched_emp'].notna()].sort_values('delta_auc',ascending=False).head(1)
    print("\n==== SUMMARY ====")
    print(res[['target','module','delta_auc','z_vs_matched','p_matched_emp','p_BH']].to_string(index=False))
    print(f"\nelapsed {time.time()-t0:.0f}s")
    return res

if __name__=='__main__':
    main()
