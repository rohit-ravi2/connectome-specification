"""Variants of Generator A staying strictly hierarchy+exponent only (no contact-area values)."""
import sys; sys.path.insert(0,'../accounting')
import numpy as np, networkx as nx
from networkx.algorithms.community import louvain_communities
from bulk_benchmark import *

G = nx.Graph(); G.add_nodes_from(range(M))
for k in np.where(CONTACT>0)[0]: G.add_edge(int(PI[k]),int(PJ[k]))

def build(resolution, min_split=4, maxdepth=20):
    modules={}; mod_size={}; node_path={n:[] for n in range(M)}; nx_=[0]
    def rec(nodes,depth):
        mid=nx_[0]; nx_[0]+=1; modules[mid]=set(nodes); mod_size[mid]=len(nodes)
        for n in nodes: node_path[n].append(mid)
        if len(nodes)<min_split or depth>=maxdepth: return
        comms=[c for c in louvain_communities(G.subgraph(nodes),seed=1,resolution=resolution) if len(c)>0]
        if len(comms)<=1: return
        for c in comms: rec(sorted(c),depth+1)
    rec(list(range(M)),0)
    nlev=max(len(p) for p in node_path.values())
    return modules,mod_size,node_path,nlev

def rent(modules,mod_size):
    Ns,Ts=[],[]
    for mid,s in modules.items():
        N=len(s)
        if N<2 or N==M: continue
        T=sum(1 for u in s for v in G.neighbors(u) if v not in s)
        if T>0: Ns.append(N); Ts.append(T)
    Ns,Ts=np.array(Ns,float),np.array(Ts,float)
    p,logk=np.polyfit(np.log(Ns),np.log(Ts),1)
    yh=p*np.log(Ns)+logk; R2=1-((np.log(Ts)-yh)**2).sum()/((np.log(Ts)-np.log(Ts).mean())**2).sum()
    return p,R2,len(Ns)

def prop_lca(node_path,mod_size,p):
    def lca(i,j):
        a,b=node_path[i],node_path[j]; c=M
        for l in range(min(len(a),len(b))):
            if a[l]==b[l]: c=mod_size[a[l]]
            else: break
        return c
    return np.array([lca(int(PI[k]),int(PJ[k]))**(p-2.0) for k in range(NP)],float)

def prop_pathsum(node_path,mod_size,p):
    # sum over ALL shared levels of N_l^(p-2): rewards staying together deep, continuous-valued
    def ps(i,j):
        a,b=node_path[i],node_path[j]; s=0.0
        for l in range(min(len(a),len(b))):
            if a[l]==b[l]: s+=mod_size[a[l]]**(p-2.0)
            else: break
        return s
    return np.array([ps(int(PI[k]),int(PJ[k])) for k in range(NP)],float)

for resn in [1.0,1.5,2.0,3.0,4.0]:
    mod,msz,npth,nlev=build(resn)
    p,R2,nm=rent(mod,msz)
    g_lca=neuron_disjoint_auc(prop_lca(npth,msz,p))
    g_ps =neuron_disjoint_auc(prop_pathsum(npth,msz,p))
    print(f"res={resn}: levels={nlev} modules={len(mod)} | Rent p={p:.3f} R2={R2:.3f} | "
          f"LCA gen_auc={g_lca:.3f} (margin {g_lca-0.700:+.3f}) | pathsum gen_auc={g_ps:.3f} (margin {g_ps-0.700:+.3f})")
