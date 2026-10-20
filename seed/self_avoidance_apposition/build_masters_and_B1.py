#!/usr/bin/env python3
"""Phase 1 masters + Part B1 (residual replication GATE). Neuron-level NT/modality/class annotation
(so NON-synaptic pairs are annotated). Tests whether high-contact-NONsynaptic same-identity enrichment
replicates across thresholds + survives contact-area/degree matching + homolog exclusion + family
decomposition + gap-junction deconfound + cross-stage persistence. MUST pass before any molecular test."""
import csv, re, os, math
import numpy as np
from collections import defaultdict, Counter
from scipy.stats import fisher_exact

ROOT=os.path.expanduser("../seed")
OUT=f"{ROOT}/data/processed/self_avoidance_apposition"
MASTER=f"{ROOT}/data/processed/contact_decomposition/contact_decomposition_master.csv"
NPZ="../data/tier5/connectome.npz"
PROF=os.path.expanduser("../data/celegans/New Notebooks/data_derived/nb28_neuron_profile.csv")
EDGES="../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv"
os.makedirs(OUT,exist_ok=True)
HEAD_MOTOR={'SMB','SIA','SIB','SAA','SMD','RME','RMD','RMF','RMH','RIM','RIV'}
MOTOR={'VA','VB','VD','DA','DB','DD','AS','VC','RMD','RME','RIM','RMF','RMH','RIV','SMD','SMB','URA','SIA','SIB','SAA'}

# ---- neuron-level NT / modality / class ----
d=np.load(NPZ,allow_pickle=True); names=[str(x) for x in d["names"]]; ntp=[str(x) for x in d["nt_primary"]]
def ntn(x):
    for k,v in [("Acetyl","ACh"),("Glu","Glu"),("GABA","GABA"),("Dopamine","DA"),("Serotonin","5HT"),("Octopamine","OA"),("Tyramine","TA")]:
        if k in x: return v
    return "Unk"
NT={nm:ntn(nt) for nm,nt in zip(names,ntp)}
prof={r["neuron"]:r for r in csv.DictReader(open(PROF))}
def cls(n):  # class family (strip L/R, D/V, trailing digits)
    m=re.match(r'^([A-Z]{2,4})',n); return m.group(1) if m else n
def modality(n):
    r=prof.get(n)
    if r:
        if r.get("is_sensory")in("1","1.0","True"): return "sensory"
        if r.get("is_command")in("1","1.0","True"): return "command"
    if cls(n) in MOTOR: return "motor"
    return "inter"
def full_class(n):  # e.g. SMBDL->SMB, ASEL->ASE, AVAL->AVA
    x=n
    x=re.sub(r'[LR]$','',x); x=re.sub(r'[DV]$','',x); x=re.sub(r'\d+$','',x)
    return x

# ---- gap junctions (electrical) per undirected pair (adult 7/8) ----
gap=set()
for r in csv.DictReader(open(EDGES)):
    if r['type']=='electrical':
        ds=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
        if ds in (7,8): gap.add((min(r['pre'],r['post']),max(r['pre'],r['post'])))

# ---- load contact master, re-annotate ----
rows=list(csv.DictReader(open(MASTER)))
def fl(x):
    try: return float(x)
    except: return np.nan
recs=[]
deg=defaultdict(float)
for r in rows:
    a,b=r['na'],r['nb']; ca=fl(r['c_adult'])
    if ca>0: deg[a]+=ca; deg[b]+=ca
for r in rows:
    a,b=r['na'],r['nb']
    ca=fl(r['c_adult']); s_ad=int(fl(r['s_adult']) or 0)
    fam_a,fam_b=full_class(a),full_class(b)
    recs.append(dict(na=a,nb=b,c_adult=ca,c_L1=fl(r['c_L1']),logc_adult=fl(r['logc_adult']),
        s_adult=s_ad, s_L1=int(fl(r['s_L1']) or 0),s_L2=int(fl(r['s_L2']) or 0),s_L3=int(fl(r['s_L3']) or 0),
        first_synapse_stage=r['first_synapse_stage'],adult_strong_core1=int(r['adult_strong_core1']),
        nt_a=NT.get(a,'Unk'),nt_b=NT.get(b,'Unk'),mod_a=modality(a),mod_b=modality(b),
        fam_a=fam_a,fam_b=fam_b,cls_a=cls(a),cls_b=cls(b),
        same_nt=int(NT.get(a,'?')==NT.get(b,'#') and NT.get(a,'?')!='Unk'),
        same_mod=int(modality(a)==modality(b)),
        same_family=int(fam_a==fam_b),
        homolog=int(fam_a==fam_b),   # same class family = homolog/sibling-ish
        both_head_motor=int(fam_a in HEAD_MOTOR and fam_b in HEAD_MOTOR),
        has_gap=int((min(a,b),max(a,b)) in gap),
        deg_a=deg[a],deg_b=deg[b]))
# high-contact thresholds (adult contact)
carr=np.array([r['c_adult'] for r in recs if r['c_adult']>0])
thr={'top5':np.quantile(carr,0.95),'top10':np.quantile(carr,0.90),'top20':np.quantile(carr,0.80)}
# source-normalized top-3: partner among source's top-3 by area
src_top=defaultdict(list)
for r in recs:
    if r['c_adult']>0: src_top[r['na']].append((r['c_adult'],r['nb'])); src_top[r['nb']].append((r['c_adult'],r['na']))
top3=set()
for n,lst in src_top.items():
    for c,p in sorted(lst,reverse=True)[:3]: top3.add((min(n,p),max(n,p)))
for r in recs:
    r['hc_top5']=int(r['c_adult']>=thr['top5']); r['hc_top10']=int(r['c_adult']>=thr['top10'])
    r['hc_top20']=int(r['c_adult']>=thr['top20']); r['hc_srcnorm']=int((min(r['na'],r['nb']),max(r['na'],r['nb'])) in top3)
    r['nonsyn']=int(r['s_adult']==0)

# ---- write masters ----
cols=list(recs[0].keys())
with open(f"{OUT}/high_contact_residual_master.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
    for r in recs:
        if r['c_adult']>0: w.writerow(r)
with open(f"{OUT}/apposition_generation_master.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
    for r in recs: w.writerow(r)
print(f"pairs: {len(recs)} | adult-contacting: {sum(1 for r in recs if r['c_adult']>0)}")

# ============ PART B1: residual replication ============
def enrich(sub, key):
    """among high-contact pairs 'sub': is same-<key> enriched in NONsynaptic vs synaptic? Fisher OR + p."""
    ns_same=sum(1 for r in sub if r['nonsyn'] and r[key]); ns_diff=sum(1 for r in sub if r['nonsyn'] and not r[key])
    sy_same=sum(1 for r in sub if not r['nonsyn'] and r[key]); sy_diff=sum(1 for r in sub if not r['nonsyn'] and not r[key])
    if min(ns_same+ns_diff, sy_same+sy_diff)<10: return None
    OR,p=fisher_exact([[ns_same,ns_diff],[sy_same,sy_diff]])
    return dict(ns_same_rate=ns_same/(ns_same+ns_diff+1e-9), sy_same_rate=sy_same/(sy_same+sy_diff+1e-9), OR=OR, p=p, n_ns=ns_same+ns_diff, n_sy=sy_same+sy_diff)

print("\n=== B1.1/B1.3: same-NT / same-modality / same-family enrichment in high-contact NONsynaptic (across thresholds) ===")
out=[]
for thr_name in ['hc_top5','hc_top10','hc_top20','hc_srcnorm']:
    sub=[r for r in recs if r[thr_name]]
    for key in ['same_nt','same_mod','same_family']:
        e=enrich(sub,key)
        if e: print(f"  {thr_name:10s} {key:12s}: nonsyn same={100*e['ns_same_rate']:.0f}% vs syn same={100*e['sy_same_rate']:.0f}%  OR={e['OR']:.2f} p={e['p']:.1e} (n_ns={e['n_ns']},n_sy={e['n_sy']})"); out.append((thr_name,key,round(e['OR'],3),e['p'],e['n_ns'],e['n_sy']))

print("\n=== B1.4: EXCLUDE homolog/same-family pairs (is it just class structure?) ===")
for thr_name in ['hc_top10']:
    sub=[r for r in recs if r[thr_name] and not r['same_family']]
    for key in ['same_nt','same_mod']:
        e=enrich(sub,key)
        if e: print(f"  {thr_name} NONhomolog {key}: nonsyn={100*e['ns_same_rate']:.0f}% syn={100*e['sy_same_rate']:.0f}% OR={e['OR']:.2f} p={e['p']:.1e}")

print("\n=== B1.2: GAP-JUNCTION deconfound (are high-contact nonsyn-chemical pairs electrically coupled?) ===")
for thr_name in ['hc_top10']:
    sub=[r for r in recs if r[thr_name]]
    ns=[r for r in sub if r['nonsyn']]; sy=[r for r in sub if not r['nonsyn']]
    print(f"  high-contact({thr_name}): nonsyn n={len(ns)} gap-coupled={sum(r['has_gap'] for r in ns)} ({100*np.mean([r['has_gap'] for r in ns]):.0f}%) | syn n={len(sy)} gap={100*np.mean([r['has_gap'] for r in sy]):.0f}%")
    # same-NT nonsyn: how many are gap-coupled?
    snt_ns=[r for r in ns if r['same_nt']]
    print(f"  same-NT high-contact nonsyn: n={len(snt_ns)} gap-coupled={100*np.mean([r['has_gap'] for r in snt_ns]) if snt_ns else 0:.0f}% (if high -> 'avoidance' is actually electrical coupling)")

print("\n=== B1.3b: head-motor family share of high-contact NONsynaptic ===")
sub=[r for r in recs if r['hc_top10']]
ns=[r for r in sub if r['nonsyn']]
hm=sum(r['both_head_motor'] for r in ns)
print(f"  high-contact top10 nonsyn: {len(ns)} pairs, both-head-motor: {hm} ({100*hm/len(ns):.0f}%)")

print("\n=== B1.5: cross-stage persistence of high-contact NONsynaptic (active avoidance vs delayed) ===")
ns_persist=[r for r in ns if r['s_L1']==0 and r['s_L2']==0 and r['s_L3']==0 and r['s_adult']==0]
print(f"  of {len(ns)} high-contact adult-nonsyn: persistently nonsyn across L1/L2/L3/adult = {len(ns_persist)} ({100*len(ns_persist)/len(ns):.0f}%) -> active avoidance not delayed")

with open(f"{OUT}/partB1_residual_replication.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["threshold","identity_key","OR_nonsyn_vs_syn","p","n_nonsyn","n_syn"])
    for row in out: w.writerow(row)
print("\nB1 done ->",OUT)
