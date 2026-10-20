#!/usr/bin/env python3
"""B1 Test 3: do STRUCTURAL communities align with FUNCTIONAL communities?

Structural graph = adult chemical synapse (strong_core1) + contact area.
Functional graph = |Randi funconn| (WT). Community detection on each; compare
partitions via Adjusted Rand Index / NMI on the shared neuron set, vs a
label-permutation null. Functional coupling is response coupling, not anatomy.
"""
import os, sys, math
import numpy as np, pandas as pd, networkx as nx
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
sys.path.insert(0, os.path.dirname(__file__))
from B1_common import load_funconn, funconn_matrix, load_contact, load_edge_table, ROOT
OUT = f"{ROOT}/data/processed/mechanistic_perturbation"

def communities(G):
    part = nx.community.greedy_modularity_communities(G, weight='weight')
    lab = {}
    for k,c in enumerate(part):
        for n in c: lab[n]=k
    return lab

def run(fc_kind='wt'):
    fc=load_funconn(fc_kind); fn,fix,M=funconn_matrix(fc)
    cneur,cix,A=load_contact()
    edge=load_edge_table()
    # shared neurons: in funconn (recorded or stimulated) and contact
    use=[n for n in fn if n in cix]
    # functional graph (undirected, |val| symmetrized)
    Fg=nx.Graph()
    Fg.add_nodes_from(use)
    for i in use:
        for j in use:
            if i<j:
                w=abs(M[fix[i],fix[j]])+abs(M[fix[j],fix[i]])
                if w>0: Fg.add_edge(i,j,weight=w)
    # structural: contact graph
    Cg=nx.Graph(); Cg.add_nodes_from(use)
    for i in use:
        for j in use:
            if i<j and A[cix[i],cix[j]]>0:
                Cg.add_edge(i,j,weight=A[cix[i],cix[j]])
    # structural: synaptic graph (strong_core1, undirected)
    Sg=nx.Graph(); Sg.add_nodes_from(use); uset=set(use)
    for _,r in edge.iterrows():
        if r['strong_core1']==1 and r['pre'] in uset and r['post'] in uset:
            p,q=r['pre'],r['post']
            if Sg.has_edge(p,q): Sg[p][q]['weight']+=1
            else: Sg.add_edge(p,q,weight=1)
    # restrict to neurons with functional edges (largest functional component context)
    funced=[n for n in use if Fg.degree(n)>0]
    res=[]
    for sname,Gs in [('contact',Cg),('synapse_strongcore',Sg)]:
        # common neurons with edges in both structural and functional
        common=[n for n in funced if Gs.degree(n)>0]
        if len(common)<20:
            res.append(dict(fc=fc_kind,structure=sname,status='too_few',n=len(common))); continue
        fl=communities(Fg.subgraph(common)); sl=communities(Gs.subgraph(common))
        labs_f=[fl[n] for n in common]; labs_s=[sl[n] for n in common]
        ari=adjusted_rand_score(labs_s,labs_f); nmi=normalized_mutual_info_score(labs_s,labs_f)
        # permutation null
        rng=np.random.default_rng(0); nari=[]; nnmi=[]
        for _ in range(500):
            perm=rng.permutation(labs_f)
            nari.append(adjusted_rand_score(labs_s,perm)); nnmi.append(normalized_mutual_info_score(labs_s,perm))
        nari=np.array(nari); nnmi=np.array(nnmi)
        p_ari=(1+np.sum(nari>=ari))/(1+len(nari)); p_nmi=(1+np.sum(nnmi>=nmi))/(1+len(nnmi))
        print(f"[{fc_kind}] {sname:18s} n={len(common)} "
              f"n_struct_comm={len(set(labs_s))} n_func_comm={len(set(labs_f))} "
              f"ARI={ari:.3f}(null{nari.mean():.3f},p={p_ari:.3f}) "
              f"NMI={nmi:.3f}(null{nnmi.mean():.3f},p={p_nmi:.3f})")
        res.append(dict(fc=fc_kind,structure=sname,status='ok',n=int(len(common)),
                        n_struct_comm=len(set(labs_s)),n_func_comm=len(set(labs_f)),
                        ARI=round(float(ari),4),ARI_null=round(float(nari.mean()),4),p_ARI=round(float(p_ari),4),
                        NMI=round(float(nmi),4),NMI_null=round(float(nnmi.mean()),4),p_NMI=round(float(p_nmi),4)))
    return res

if __name__=="__main__":
    allres=[]
    for k in ['wt','leifer']:
        allres+=run(k)
    df=pd.DataFrame(allres)
    df.to_csv(f"{OUT}/B1_activity_structure_results.csv",index=False)
    print(f"\nwrote {OUT}/B1_activity_structure_results.csv ({len(df)} rows)")
