"""
Robustness of the ablation verdict across strong-core thresholds for the BASE graph.
Rebuild the network at strong_core1 and strong_core2, and for a representative set of
classes test whether they beat a contact-matched null on the two fast core metrics
(sensory->motor reachability, global efficiency). 20 nulls/class.
"""
import numpy as np, pandas as pd, networkx as nx
from numpy.random import default_rng
RNG = default_rng(7); OUT="../seed/data/processed/mechanistic_perturbation"
d = pd.read_csv(f"{OUT}/dev_edge_trajectory_table.csv")
mr = pd.concat([d[['pre','pre_mod']].rename(columns={'pre':'n','pre_mod':'m'}),
                d[['post','post_mod']].rename(columns={'post':'n','post_mod':'m'})])
MOD = mr.groupby('n')['m'].agg(lambda s: s.value_counts().index[0]).to_dict()
N_NULL=20

def build(colmask):
    core=d[colmask].reset_index(drop=True)
    G=nx.DiGraph()
    for n in set(core.pre)|set(core.post): G.add_node(n)
    for r in core.itertuples(): G.add_edge(r.pre,r.post,area=r.contact_area)
    return G,core

def geff(G):
    nodes=list(G.nodes()); n=len(nodes); s=0.0
    for u,dd in nx.all_pairs_shortest_path_length(G):
        for v,dist in dd.items():
            if v!=u: s+=1.0/dist
    return s/(n*(n-1))

def smreach(G,SENS,MOTOR):
    tot=rc=0
    for sc in SENS:
        if sc not in G: continue
        des=nx.descendants(G,sc)
        for mo in MOTOR:
            if mo==sc: continue
            tot+=1; rc+= (mo in des)
    return rc/tot if tot else np.nan

def run(colmask,label,rows):
    G0,core=build(colmask)
    SENS=[n for n in G0 if MOD.get(n)=='sensory']; MOTOR=[n for n in G0 if MOD.get(n)=='motor']
    EDGES=list(G0.edges()); EIDX={e:i for i,e in enumerate(EDGES)}
    AREA=np.array([G0[u][v]['area'] for u,v in EDGES])
    qs=np.quantile(AREA,np.linspace(0,1,9)); qs[0]-=1e-9; qs[-1]+=1e-9
    ABIN=np.clip(np.digitize(AREA,qs)-1,0,7); ALL=np.arange(len(EDGES))
    base_g=geff(G0); base_s=smreach(G0,SENS,MOTOR)
    def abl(idx):
        G=G0.copy(); G.remove_edges_from([EDGES[i] for i in idx]); return geff(G),smreach(G,SENS,MOTOR)
    def matched(cei):
        k=len(cei); pool=np.setdiff1d(ALL,cei)
        if len(pool)<k: pool=ALL
        tc=pd.Series(ABIN[cei]).value_counts().to_dict(); picks=[]
        for b,c in tc.items():
            cand=pool[ABIN[pool]==b]
            if len(cand)>=c: picks+=RNG.choice(cand,c,replace=False).tolist()
            else:
                picks+=cand.tolist(); rest=np.setdiff1d(pool,np.array(picks,int))
                picks+=RNG.choice(rest,c-len(cand),replace=False).tolist()
        return np.array(picks[:k],int)
    classes={
      'strong_core2(subset)': core.strong_core2==1 if 'strong_core2' in core else None,
      'strong_core4(subset)': core.strong_core4==1,
      'preadult_persistent': core.n_preadult_groups>=2,
      'sensory->inter': (core.pre_mod=='sensory')&(core.post_mod=='inter'),
      'post_embryonic_integration': core.classification=='post-embryonic brain integration',
      'NT_ACh': core.pre_nt=='ACh',
    }
    for cn,mask in classes.items():
        if mask is None or mask.sum()<3: continue
        cei=np.array([EIDX[(r.pre,r.post)] for r in core[mask.values].itertuples() if (r.pre,r.post) in EIDX])
        if len(cei)<3: continue
        cg,cs=abl(cei)
        ng=[]; ns=[]
        for _ in range(N_NULL):
            g,s=abl(matched(cei)); ng.append(g); ns.append(s)
        ng=np.array(ng); ns=np.array(ns)
        rows.append(dict(base=label, edge_class=cn, n_edges=int(len(cei)),
            geff_class_pct_drop=round((base_g-cg)/base_g*100,2),
            geff_null_pct_drop=round((base_g-ng.mean())/base_g*100,2),
            geff_p=round((np.sum(ng<=cg)+1)/(N_NULL+1),3),
            geff_more_disruptive=bool(cg<ng.mean()),
            smreach_class_lost=round(base_s-cs,4),
            smreach_null_lost=round(base_s-ns.mean(),4),
            smreach_p=round((np.sum(ns<=cs)+1)/(N_NULL+1),3),
            smreach_more_disruptive=bool(cs<ns.mean())))

rows=[]
run(d.strong_core1==1,'strong_core1(1469e)',rows)
run(d.strong_core2==1,'strong_core2(903e)',rows)
df=pd.DataFrame(rows)
df.to_csv(f"{OUT}/B2_robustness_thresholds.csv",index=False)
pd.set_option('display.width',220,'display.max_columns',30)
print(df.to_string(index=False))
