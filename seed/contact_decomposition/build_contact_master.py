#!/usr/bin/env python3
"""Phase 1: contact-decomposition master with PER-STAGE membrane contact (Witvliet L1..adult).
Rows = undirected neuron pairs contacting at ANY stage. Per-stage contact area + per-stage synapse +
timing/persistence + eligibility + baseline metadata. Enables the eligibility-restricted contact-timing
test (attacks the birth-time definitional confound)."""
import csv, re, os, math
import numpy as np
from collections import defaultdict

ROOT=os.path.expanduser("../seed")
PC=f"{ROOT}/data/external/mechanistic_frontier/witvliet2021_developmental/physical_contact"
EDGES="../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv"
TRAJ=f"{ROOT}/data/processed/mechanistic_perturbation/dev_edge_trajectory_table.csv"
OUT=f"{ROOT}/data/processed/contact_decomposition"
os.makedirs(OUT,exist_ok=True)
STAGE_DS={'L1':[1,2,3,4],'L2':[5],'L3':[6],'adult':[8]}   # ds7 has no contact matrix
STAGES=['L1','L2','L3','adult']

def is_neuron(n):
    return bool(n) and not n.startswith('BWM') and 'BWM' not in n and not n.startswith('hyp') \
        and 'exc' not in n.lower() and not n.startswith('MC') and re.match(r'^[A-Z]',n)

# ---- per-stage undirected contact area ----
def load_adj(ds):
    f=f"{PC}/Dataset{ds}_adjacency.csv"
    if not os.path.exists(f): return None,None
    rows=list(csv.reader(open(f))); hdr=rows[0][1:]
    C={}
    for r in rows[1:]:
        a=r[0]
        for j,v in enumerate(r[1:]):
            if v and v!='0':
                b=hdr[j]
                if a!=b:
                    k=(min(a,b),max(a,b)); C[k]=max(C.get(k,0.0),float(v))
    return set(hdr),C
stage_contact={}; stage_nodes={}
for st,dss in STAGE_DS.items():
    merged=defaultdict(list); nodes=set()
    for ds in dss:
        nd,C=load_adj(ds)
        if C is None: continue
        nodes|=nd
        for k,v in C.items(): merged[k].append(v)
    # L1: mean over animals present + count; others: the single value
    stage_contact[st]={k:(np.mean(v)) for k,v in merged.items()}
    if st=='L1':
        stage_contact['L1_count']={k:len(v) for k,v in merged.items()}
    stage_nodes[st]=nodes
print("per-stage contacts:",{st:len(stage_contact[st]) for st in STAGES})

# ---- per-stage directed chemical synapse counts ----
stagemap={1:'L1',2:'L1',3:'L1',4:'L1',5:'L2',6:'L3',7:'adult',8:'adult'}
syn_stage=defaultdict(lambda: defaultdict(int))   # (pre,post) -> stage -> max synapses
adult78=defaultdict(dict)
for r in csv.DictReader(open(EDGES)):
    if r['type']=='electrical': continue
    ds=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
    k=(r['pre'],r['post']); s=int(r['synapses'])
    syn_stage[k][stagemap[ds]]=max(syn_stage[k][stagemap[ds]],s)
    if ds in (7,8): adult78[k][ds]=s
def undirected_syn(a,b,st):
    return max(syn_stage.get((a,b),{}).get(st,0), syn_stage.get((b,a),{}).get(st,0))

# ---- metadata from trajectory table ----
meta={}
for r in csv.DictReader(open(TRAJ)):
    meta[(r['pre'],r['post'])]=r
def get_meta(a,b,field,default=''):
    r=meta.get((a,b)) or meta.get((b,a))
    return r.get(field,default) if r else default

# ---- neuron set + all contacting pairs across stages ----
neurons=set(n for st in STAGES for n in stage_nodes[st] if is_neuron(n))
allpairs=set()
for st in STAGES:
    for (a,b) in stage_contact[st]:
        if a in neurons and b in neurons: allpairs.add((a,b))
print(f"neurons {len(neurons)} | undirected contacting pairs (any stage) {len(allpairs)}")

# ---- contact degree/budget per neuron per stage (for rank/normalization) ----
deg_area=defaultdict(lambda: defaultdict(float))
for st in STAGES:
    for (a,b),v in stage_contact[st].items():
        if a in neurons and b in neurons: deg_area[st][a]+=v; deg_area[st][b]+=v

# ---- build rows ----
def birthf(x):
    try: return float(x)
    except: return np.nan
rows=[]
for (a,b) in sorted(allpairs):
    ca={st:stage_contact[st].get((a,b),0.0) for st in STAGES}
    l1cnt=stage_contact['L1_count'].get((a,b),0)
    first_contact=next((st for st in STAGES if ca[st]>0),'none')
    contact_persist=sum(1 for st in STAGES if ca[st]>0)
    # synapse per stage (undirected)
    sy={st:undirected_syn(a,b,st) for st in STAGES}
    first_syn=next((st for st in STAGES if sy[st]>0),'none')
    # adult strong-core
    a7=max(adult78.get((a,b),{}).get(7,0),adult78.get((b,a),{}).get(7,0))
    a8=max(adult78.get((a,b),{}).get(8,0),adult78.get((b,a),{}).get(8,0))
    sc1=int(a7>0 and a8>0); sc2=int(a7>=2 and a8>=2)
    # timing
    order={'L1':0,'L2':1,'L3':2,'adult':3,'none':9}
    delay=(order[first_syn]-order[first_contact]) if (first_syn!='none' and first_contact!='none') else np.nan
    # metadata
    pb=birthf(get_meta(a,b,'pre_birth')); qb=birthf(get_meta(a,b,'post_birth'))
    rows.append(dict(na=a,nb=b,
        c_L1=ca['L1'],c_L2=ca['L2'],c_L3=ca['L3'],c_adult=ca['adult'],c_L1_count=l1cnt,
        logc_L1=math.log1p(ca['L1']),logc_adult=math.log1p(ca['adult']),
        first_contact_stage=first_contact,contact_persist=contact_persist,
        frac_src=ca['adult']/(deg_area['adult'][a]+1e-9),frac_tgt=ca['adult']/(deg_area['adult'][b]+1e-9),
        frac_src_L1=ca['L1']/(deg_area['L1'][a]+1e-9),
        s_L1=sy['L1'],s_L2=sy['L2'],s_L3=sy['L3'],s_adult=sy['adult'],
        first_synapse_stage=first_syn,contact_syn_delay=delay,
        contact_no_syn=int(ca['adult']>0 and sy['adult']==0),
        # TARGETS
        T_first_L1=int(sy['L1']>0), T_L1_repro=int(sum(1 for st in ['L1'] for _ in [0] if sy['L1']>0)>0),
        T_preadult_persist=int(sum(1 for st in ['L1','L2','L3'] if sy[st]>0)>=2),
        T_preadult_strength=math.log1p(max(sy['L1'],sy['L2'],sy['L3'])),
        adult_strong_core1=sc1,adult_strong_core2=sc2,
        pre_birth=pb,post_birth=qb,
        both_born_preL1=int((not np.isnan(pb)) and (not np.isnan(qb)) and pb<560 and qb<560),
        pre_nt=get_meta(a,b,'pre_nt'),post_nt=get_meta(a,b,'post_nt'),
        pre_mod=get_meta(a,b,'pre_mod'),post_mod=get_meta(a,b,'post_mod'),
        lin_dist=get_meta(a,b,'lin_dist')))
cols=list(rows[0].keys())
with open(f"{OUT}/contact_decomposition_master.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
    for r in rows: w.writerow(r)
n=len(rows)
print(f"\nmaster rows (undirected contacting pairs): {n}")
print(f"  contact at L1: {sum(1 for r in rows if r['c_L1']>0)} | any synapse: {sum(1 for r in rows if r['first_synapse_stage']!='none')}")
print(f"  high-contact NON-synaptic (adult contact>0, no adult syn): {sum(r['contact_no_syn'] for r in rows)}")
print(f"  both born pre-L1: {sum(r['both_born_preL1'] for r in rows)}")
print("  target prevalences:", {t:sum(r[t] for r in rows) for t in ['T_first_L1','T_preadult_persist','adult_strong_core1']})
print("  first_contact vs first_synapse stage cross-tab (contact -> synapse):")
from collections import Counter
ct=Counter((r['first_contact_stage'],r['first_synapse_stage']) for r in rows)
for st in STAGES:
    print(f"    contact first={st}: "+", ".join(f"syn@{s}={ct[(st,s)]}" for s in STAGES+['none']))
print("saved ->",f"{OUT}/contact_decomposition_master.csv")
