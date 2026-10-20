#!/usr/bin/env python3
"""Part C — build REGULATORY / lineage-program features for pre-adult establishment.

Regulatory features (inferred programs, NOT raw expression snapshots):
  - GRN TF-activity per neuron class (CelEsT GRN x CeNGEN expression, VIPER-style)
  - Terminal-selector identity code (curated Hobert selectors, per-class expression)
  - Regulatory module scores (5 effector categories) per class

Expression-snapshot control: full-transcriptome cosine similarity (adult CeNGEN).
All per-class profiles collapsed to PAIRWISE features (similarity + magnitudes).
"""
import json, gzip, re, os
import numpy as np, pandas as pd

ROOT=os.path.expanduser("../seed")
OUT=f"{ROOT}/data/processed/upstream_early_establishment"
CENGEN=f"../data/celegans/data/expression/cengen/thresholded/021821_liberal_threshold1.csv"
GRN=os.path.expanduser("../grn/CelEsT-app/www/CelEsT_GRN.txt")
TFLIST=os.path.expanduser("../grn/CelEsT-app/www/fullset_TFs_BM_with_MOR_for_TS2.txt")
GENEIDS="../data/celegans/data/raw/wormbase_full_release/Downloads/wormbase_full_release/c_elegans/PRJNA13758/annotation/c_elegans.PRJNA13758.WS297.geneIDs.txt.gz"
GENESETS=f"{ROOT}/data/processed/regulatory_branch/gene_sets_effector.json"

# ---------- master ----------
m=pd.read_csv(f"{OUT}/upstream_master_table.csv")
neurons=sorted(set(m.pre)|set(m.post))

# ---------- adult CeNGEN expression (128 classes, WBGene, thresholded TPM) ----------
cg=pd.read_csv(CENGEN)
cg_classes=list(cg.columns[3:])
wbid=cg["Wormbase_ID"].astype(str).values
gname=cg["gene_name"].astype(str).values
X=cg[cg_classes].apply(pd.to_numeric,errors="coerce").fillna(0.0).values  # genes x classes
Xln=np.log1p(X)
wb2row={w:i for i,w in enumerate(wbid)}

# ---------- neuron -> CeNGEN class mapper ----------
CS=set(cg_classes)
def mapcls(n):
    if n in CS: return [n]
    # explicit special cases -> averaged columns
    if n.startswith("AWC"): return ["AWC_ON","AWC_OFF"]
    if n.startswith("IL2"):
        return ["IL2_DV"] if re.search(r'[DV]',n[3:]) else ["IL2_LR"]
    if n.startswith("RMD"):
        return ["RMD_DV"] if re.search(r'[DV]',n[3:]) else ["RMD_LR"]
    if n.startswith("RME"):
        return ["RME_DV"] if n[3:] in ("D","V") else ["RME_LR"]
    for pref in ["CEP","OLQ","SAA","SIA","SIB","SMB","SMD","URY","URA","SAB"]:
        if n.startswith(pref) and pref in CS: return [pref]
    # ventral-cord motor neurons VAn DAn DBn VBn VDn DDn ASn VCn
    mmo=re.match(r'^(VA|DA|DB|VB|VD|DD|AS|VC)\d+$',n)
    if mmo:
        base=mmo.group(1)
        if n in CS: return [n]
        if base in CS: return [base]
        if base=="VD" or base=="DD": return ["VD_DD"] if "VD_DD" in CS else ([base] if base in CS else [])
    # strip trailing L/R then D/V
    for cand in [re.sub(r'[LR]$','',n), re.sub(r'[DV]$','',n),
                 re.sub(r'[DV]$','',re.sub(r'[LR]$','',n)),
                 re.sub(r'[LR]$','',re.sub(r'[DV]$','',n))]:
        if cand in CS: return [cand]
    return []

nmap={n:mapcls(n) for n in neurons}
missing=[n for n,c in nmap.items() if not c]
print(f"neurons {len(neurons)} | unmapped {len(missing)}: {missing}")

def class_profile(cols_idx_fn, cls_list):
    """average the mapped class columns"""
    idxs=[cg_classes.index(c) for c in cls_list if c in cg_classes]
    return idxs

# per-neuron adult expression column-index list
neuron_cols={n:[cg_classes.index(c) for c in nmap[n]] for n in neurons if nmap[n]}
valid_neurons=set(neuron_cols)

def prof(n, mat):
    return mat[:, neuron_cols[n]].mean(axis=1)  # genes-vector

# ---------- geneIDs: seqname -> WBGene ----------
seq2wb={}
with gzip.open(GENEIDS,'rt') as f:
    for line in f:
        p=line.rstrip("\n").split(",")
        if len(p)>=4:
            wb=p[1]; pub=p[2]; seq=p[3]
            if seq: seq2wb[seq]=wb
            if pub: seq2wb[pub]=wb  # allow public-name fallback

# ---------- CelEsT GRN -> TF-activity ----------
grn=pd.read_csv(GRN, sep="\t")
grn.columns=[c.strip('"') for c in grn.columns]
for c in ['source','target']:
    grn[c]=grn[c].astype(str).str.strip('"')
grn['weight']=pd.to_numeric(grn['weight'],errors='coerce').fillna(0.0)
# TF MOR (mode of regulation): from TF list (mostly unsigned/0 -> treat as activation +1)
tfl=pd.read_csv(TFLIST,sep="\t")
tf_gseq2wb=dict(zip(tfl['wormbase_gseq'].astype(str), tfl['wormbase_gene'].astype(str)))
tf_pub2wb=dict(zip(tfl['wormbase_locus'].astype(str), tfl['wormbase_gene'].astype(str)))
def tf_to_wb(s):
    return tf_gseq2wb.get(s) or tf_pub2wb.get(s) or seq2wb.get(s)
# map target seqnames -> expression rows
grn['tgt_row']=grn['target'].map(lambda s: wb2row.get(seq2wb.get(s,"")) )
grn['src_wb']=grn['source'].map(tf_to_wb)
grn2=grn.dropna(subset=['tgt_row']).copy()
grn2['tgt_row']=grn2['tgt_row'].astype(int)
tfs=sorted(grn2['src_wb'].dropna().unique())
print(f"GRN edges {len(grn)} | mappable-target edges {len(grn2)} | TFs {len(tfs)}")

# z-score target expression across classes (regulatory activity relative to mean)
Z=(Xln - Xln.mean(axis=1,keepdims=True))/(Xln.std(axis=1,keepdims=True)+1e-9)  # genes x classes
tf_act=np.zeros((len(tfs), len(cg_classes)))
for ti,tf in enumerate(tfs):
    sub=grn2[grn2['src_wb']==tf]
    rows=sub['tgt_row'].values; w=sub['weight'].values
    if len(rows)==0: continue
    tf_act[ti]=(w[:,None]*Z[rows]).sum(0)/ (np.abs(w).sum()+1e-9)  # weighted mean target z per class
# tf_act: nTF x classes  -> per neuron: average mapped columns
def tf_act_prof(n):
    idx=neuron_cols[n]; return tf_act[:,idx].mean(1)

# ---------- terminal-selector identity code ----------
gs=json.load(open(GENESETS))
tsel=gs['categories']['terminal_selector_identity']['genes']
tsel_wb=[w for w in tsel if w in wb2row]
tsel_rows=[wb2row[w] for w in tsel_wb]
print(f"terminal selectors: {len(tsel)} curated, {len(tsel_wb)} in CeNGEN")
def tsel_prof(n):
    idx=neuron_cols[n]; return Xln[np.ix_(tsel_rows,idx)].mean(1)

# ---------- module scores (5 effector categories) from adult CeNGEN ----------
cat_rows={}
for cat,cd in gs['categories'].items():
    rws=[wb2row[w] for w in cd['genes'] if w in wb2row]
    cat_rows[cat]=rws
cats=sorted(cat_rows)
def module_prof(n):
    idx=neuron_cols[n]
    return np.array([Xln[np.ix_(cat_rows[c],idx)].mean() for c in cats])

# ---------- precompute per-neuron profiles ----------
P_tf={n:tf_act_prof(n) for n in valid_neurons}
P_tsel={n:tsel_prof(n) for n in valid_neurons}
P_mod={n:module_prof(n) for n in valid_neurons}
P_full={n:prof(n,Xln) for n in valid_neurons}  # full transcriptome (expression snapshot)

def cos(a,b):
    na=np.linalg.norm(a); nb=np.linalg.norm(b)
    return float(a@b/(na*nb+1e-9))
def pear(a,b):
    a=a-a.mean(); b=b-b.mean(); return cos(a,b)

# ---------- pairwise feature table ----------
rows=[]
for _,r in m.iterrows():
    p,q=r['pre'],r['post']
    if p not in valid_neurons or q not in valid_neurons:
        rows.append(None); continue
    tfp,tfq=P_tf[p],P_tf[q]; tsp,tsq=P_tsel[p],P_tsel[q]
    mp,mq=P_mod[p],P_mod[q]; fp,fq=P_full[p],P_full[q]
    d=dict(pre=p,post=q,
        # GRN TF-activity pairwise
        tf_cos=cos(tfp,tfq), tf_pear=pear(tfp,tfq),
        tf_l2=float(np.linalg.norm(tfp-tfq)),
        tf_mag_min=float(min(np.linalg.norm(tfp),np.linalg.norm(tfq))),
        tf_mag_max=float(max(np.linalg.norm(tfp),np.linalg.norm(tfq))),
        # terminal-selector code pairwise
        tsel_cos=cos(tsp,tsq), tsel_pear=pear(tsp,tsq),
        tsel_shared=float(((tsp>0.5)&(tsq>0.5)).sum()),
        tsel_rich_min=float(min((tsp>0.5).sum(),(tsq>0.5).sum())),
        tsel_rich_max=float(max((tsp>0.5).sum(),(tsq>0.5).sum())),
        # module scores pairwise (per category product + absdiff)
        **{f"mod_{c}_prod":float(mp[i]*mq[i]) for i,c in enumerate(cats)},
        **{f"mod_{c}_absd":float(abs(mp[i]-mq[i])) for i,c in enumerate(cats)},
        # expression snapshot control
        full_cos=cos(fp,fq), full_pear=pear(fp,fq),
    )
    rows.append(d)
feat=pd.DataFrame([x for x in rows if x is not None])
# join targets + baseline cols from master (align on index of valid rows)
mask=[x is not None for x in rows]
mm=m[mask].reset_index(drop=True)
assert len(mm)==len(feat)
out=pd.concat([mm.reset_index(drop=True), feat.drop(columns=['pre','post']).reset_index(drop=True)],axis=1)
out.to_csv(f"{OUT}/partC_pairwise_features.csv", index=False)
print(f"pairwise feature rows: {len(out)} / {len(m)} (dropped {len(m)-len(out)} for missing expression)")
print("saved partC_pairwise_features.csv")
# save class map + tf list for provenance
json.dump({n:nmap[n] for n in neurons}, open(f"{OUT}/partC_neuron_class_map.json","w"), indent=0)
print("cats:",cats)
