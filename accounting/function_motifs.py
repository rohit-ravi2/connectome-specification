"""NEW AXIS — WHAT DOES THE CONNECTOME COMPUTE? First dive: signed (E/I) circuit-motif census. The arc asked
what SPECIFIES the wiring for 20 rounds; never once what the wiring is FOR. Sign each edge by presynaptic NT
(GABA=inhibitory, ACh/Glu=excitatory), enumerate the computational primitives — feedforward inhibition,
disinhibition, mutual inhibition, feedback inhibition, coherent/incoherent feedforward loops — and ask which are
over-represented vs a degree+sign-preserving null. What primitives is C. elegans built from?
"""
import warnings,sys,openpyxl; warnings.simplefilter("ignore")
import numpy as np, networkx as nx
from collections import defaultdict, Counter
def P(*a): print(*a); sys.stdout.flush()
RNG=np.random.default_rng(0); D="../data/celegans/data"
# NT signs
t5=np.load("../data/tier5/connectome.npz",allow_pickle=True)
nt={str(n):str(v) for n,v in zip(t5["names"],t5["nt_primary"])}
def bc(n):
    for s in "LRDV":
        if n.endswith(s): return n[:-1]
    return n
def sign(n):
    v=nt.get(n,nt.get(bc(n),"?"))
    if v.startswith("GABA"): return -1
    if v.startswith(("Acetyl","Glu","ACh")): return +1
    return 0   # monoamine/unknown = modulatory, excluded from E/I motif signs
# Cook herm directed chemical connectome
wb=openpyxl.load_workbook(f"{D}/connectome/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
A=list(wb['hermaphrodite chemical'].iter_rows(values_only=True))
# find header row/col
def looks(n): return isinstance(n,str) and len(n)>=2 and n[0].isalpha()
hr=max(range(len(A)),key=lambda r:sum(looks(x) for x in A[r]))
cols=[(str(x).strip() if x is not None else None) for x in A[hr]]
hc=max(range(min(6,len(A[0]))),key=lambda c:sum(looks(A[r][c]) for r in range(len(A))))
edges=[]  # (pre,post)
neurons=set()
for r in range(hr+1,len(A)):
    pre=A[r][hc]; pre=pre.strip() if isinstance(pre,str) else None
    if not pre: continue
    for c in range(hc+1,len(A[r])):
        post=cols[c]; v=A[r][c]
        if post and v not in (None,''):
            try:
                if float(v)>0: edges.append((pre,post)); neurons.add(pre); neurons.add(post)
            except: pass
neurons=sorted(neurons); idx={n:i for i,n in enumerate(neurons)}; M=len(neurons)
G=nx.DiGraph(); G.add_nodes_from(range(M))
for a,b in edges:
    if a in idx and b in idx and a!=b: G.add_edge(idx[a],idx[b])
sgn=np.array([sign(n) for n in neurons])
P(f"Cook directed connectome: {M} neurons, {G.number_of_edges()} edges | E {int((sgn>0).sum())} I {int((sgn<0).sum())} mod/unk {int((sgn==0).sum())}")

def census(Gx,sg):
    A=nx.to_numpy_array(Gx,dtype=int)   # A[i,j]=1 if i->j
    recip_I=recip_E=recip_mix=0; ffl_co=ffl_inco=ffi=disinh=0
    # reciprocal dyads
    for i in range(M):
        for j in range(i+1,M):
            if A[i,j] and A[j,i]:
                s=(sg[i],sg[j])
                if s==(-1,-1): recip_I+=1
                elif s==(1,1): recip_E+=1
                elif -1 in s and 1 in s: recip_mix+=1
    # feedforward loops A->B, A->C, B->C (B!=C, no C->B needed)
    for a in range(M):
        outs=np.where(A[a]>0)[0]
        for bi in range(len(outs)):
            for ci in range(len(outs)):
                b,c=outs[bi],outs[ci]
                if b!=c and A[b,c] and not A[c,b]:
                    # coherent iff sign(direct a->c via sg[a]) == sign(path a->b->c via sg[a]*sg[b])
                    direct=sg[a]; path=sg[a]*sg[b]
                    if direct*path>0: ffl_co+=1
                    else: ffl_inco+=1
                    if sg[b]<0: ffi+=1                 # B inhibitory -> feedforward inhibition
    # disinhibition A-|B-|C (A inhibits B, B inhibits C)
    for a in range(M):
        for b in np.where(A[a]>0)[0]:
            if sg[a]<0:
                for c in np.where(A[b]>0)[0]:
                    if sg[b]<0 and c!=a: disinh+=1
    return dict(recip_I=recip_I,recip_E=recip_E,recip_mix=recip_mix,ffl_co=ffl_co,ffl_inco=ffl_inco,ffi=ffi,disinh=disinh)
obs=census(G,sgn)
# degree+sign-preserving null: directed edge swap (preserves in/out degree); signs stay on presynaptic node -> preserved
def null_census(K=40):
    accs=defaultdict(list)
    for _ in range(K):
        H=G.copy()
        try: nx.directed_edge_swap(H,nswap=3*G.number_of_edges(),max_tries=30*G.number_of_edges())
        except: pass
        c=census(H,sgn)
        for k,v in c.items(): accs[k].append(v)
    return accs
nul=null_census()
P("\n"+"="*82); P("SIGNED CIRCUIT-MOTIF CENSUS — computational primitives vs degree+sign-preserving null"); P("="*82)
labels={'recip_I':'mutual inhibition (winner-take-all / bistable)','recip_E':'mutual excitation (amplifier)',
        'recip_mix':'reciprocal E-I (feedback gain-control)','ffl_co':'coherent feedforward loop (sign-consistent, filtering)',
        'ffl_inco':'incoherent feedforward loop (pulse/adaptation)','ffi':'feedforward inhibition (gain control/timing)',
        'disinh':'disinhibition (A-|B-|C, gating)'}
for k in ['recip_I','recip_E','recip_mix','ffi','disinh','ffl_co','ffl_inco']:
    m=np.mean(nul[k]); s=np.std(nul[k])+1e-9; z=(obs[k]-m)/s
    tag="OVER-represented" if z>2 else ("under" if z<-2 else "~null")
    P(f"  {labels[k]:52s} obs={obs[k]:5d} null={m:7.1f} z={z:+6.1f}  {tag}")
P("\n[read] over-represented signed motifs = the circuit primitives C. elegans wiring is enriched for = a first")
P("  glimpse of what it COMPUTES, beyond what its degree/sign distribution trivially gives. (opens: which neurons/")
P("  circuits carry each primitive, sensory->motor flow, dynamical modes — many follow-ons.)")
P("="*82)
