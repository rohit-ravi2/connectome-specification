import numpy as np
from scipy.sparse.csgraph import shortest_path, connected_components
from scipy.sparse import csr_matrix

def build_A(vec, I, J, M):
    A=np.zeros((M,M),dtype=float)
    on=np.where(vec>0)[0]
    A[I[on],J[on]]=1.0
    np.fill_diagonal(A,0)
    return A

def entropy(deg):
    d=deg[deg>0]
    if len(d)==0: return 0.0
    p=d/d.sum()
    return float(-(p*np.log(p)).sum())

def invariants(A):
    M=A.shape[0]
    Au=((A+A.T)>0).astype(float)
    do=A.sum(1); di=A.sum(0); dt=Au.sum(1)
    ne=A.sum(); mutual=(A*A.T).sum()/2.0
    inv={}
    inv['n_edges']=ne
    inv['reciprocity']=mutual*2.0/max(ne,1)
    inv['deg_out_var']=do.var(); inv['deg_in_var']=di.var(); inv['deg_tot_var']=dt.var()
    inv['deg_out_max']=do.max(); inv['deg_in_max']=di.max(); inv['deg_tot_max']=dt.max()
    inv['deg_out_ent']=entropy(do); inv['deg_in_ent']=entropy(di); inv['deg_tot_ent']=entropy(dt)
    # triangles / clustering on undirected
    Au3=np.linalg.matrix_power(Au,3)
    tri=np.trace(Au3)/6.0
    twop=(Au@Au).sum()-np.trace(Au@Au)  # ordered 2-paths
    inv['triangles']=tri
    inv['transitivity']=(3*2*tri)/max(twop,1)  # 6*tri/ordered2paths = global clustering
    # directed motifs (algebraic)
    A2=A@A
    inv['dir_3cycles']=np.trace(A@A@A)/3.0
    inv['dir_transtri']=(A2*A).sum()      # transitive triads i->k paths with i->k edge
    inv['dir_2paths']=A2.sum()-np.trace(A2)
    # adjacency spectrum (complex -> magnitudes)
    ev=np.abs(np.linalg.eigvals(A)); ev.sort(); ev=ev[::-1]
    inv['spec_radius']=ev[0]
    inv['spec_gap']=ev[0]-ev[1]
    for k in range(1,6): inv[f'adj_ev{k}']=ev[k-1]
    inv['adj_ev_sum5']=ev[:5].sum()
    # symmetric adjacency spectrum
    evs=np.linalg.eigvalsh(Au); evs=evs[::-1]
    for k in range(1,4): inv[f'symadj_ev{k}']=evs[k-1]
    inv['symadj_evmin']=evs[-1]
    # Laplacian spectrum (undirected)
    L=np.diag(dt)-Au
    evl=np.linalg.eigvalsh(L)
    inv['lap_lambda2']=evl[1]          # algebraic connectivity
    inv['lap_max']=evl[-1]
    inv['lap_ev_sum5']=evl[-5:].sum()
    inv['lap_energy']=np.abs(evl).sum()
    # rich club at degree thresholds
    for kk in (5,10,20):
        sel=dt>kk
        if sel.sum()>1:
            sub=Au[np.ix_(sel,sel)]; poss=sel.sum()*(sel.sum()-1)
            inv[f'richclub{kk}']=sub.sum()/max(poss,1)
        else:
            inv[f'richclub{kk}']=0.0
    # connectivity / path length (undirected)
    ncomp,_=connected_components(csr_matrix(Au),directed=False)
    inv['n_components']=ncomp
    Dsp=shortest_path(csr_matrix(Au),directed=False,unweighted=True)
    finite=np.isfinite(Dsp)&(Dsp>0)
    inv['mean_path']=Dsp[finite].mean() if finite.any() else 0.0
    inv['global_eff']=(1.0/Dsp[finite]).mean() if finite.any() else 0.0
    inv['diameter']=Dsp[finite].max() if finite.any() else 0.0
    # directed strong components
    nsc,_=connected_components(csr_matrix(A),directed=True,connection='strong')
    inv['n_strongcomp']=nsc
    return inv

INV_KEYS=None
def vecify(A):
    global INV_KEYS
    d=invariants(A)
    if INV_KEYS is None: INV_KEYS=sorted(d.keys())
    return np.array([d[k] for k in INV_KEYS]), INV_KEYS
