import sys,numpy as np,networkx as nx
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True)
I,J=z['I'].astype(int),z['J'].astype(int); y7,y8=z['y7'],z['y8']
neurons=list(z['neurons']); M=len(neurons); N=len(I)
base=z['base']
yrep=((y7==1)&(y8==1)).astype(int)
# reconstruct contact area matrix A (symmetric) and binary Ab
A=np.zeros((M,M));
for k in range(N): A[I[k],J[k]]=np.expm1(base[k,0])
A=np.maximum(A,A.T); Ab=(A>0).astype(int)
# directed synapse matrix Y (yrep) among contacting pairs
Y=np.zeros((M,M),int)
for k in range(N): Y[I[k],J[k]]=yrep[k]
# pair index lookup
pidx={(int(I[k]),int(J[k])):k for k in range(N)}

# ---- BUNDLE DEFINITIONS ----
G=nx.Graph()
G.add_nodes_from(range(M))
for k in range(N):
    if I[k]<J[k]: G.add_edge(int(I[k]),int(J[k]),weight=float(A[I[k],J[k]]))
Gw=G  # weighted

# (a) greedy modularity communities (partition) -- weighted
comm_a=list(nx.community.greedy_modularity_communities(G,weight='weight'))
part_a=np.full(M,-1);
for c,s in enumerate(comm_a):
    for n in s: part_a[n]=c
P(f"[def a] greedy-modularity communities: {len(comm_a)} sizes={sorted([len(s) for s in comm_a],reverse=True)}")

# (b) spectral clustering on Ab (binary contact)
from sklearn.cluster import SpectralClustering
K=len(comm_a)  # match count
sc=SpectralClustering(n_clusters=K,affinity='precomputed',assign_labels='discretize',random_state=0)
part_b=sc.fit_predict(A)  # weighted affinity
sz_b=np.bincount(part_b)
P(f"[def b] spectral (K={K}) sizes={sorted(sz_b.tolist(),reverse=True)}")

# (c) k-cores (overlapping-ish; assign each node its max core number, and use k-core subgraphs as bundles)
core=nx.core_number(G)
core_arr=np.array([core[n] for n in range(M)])
P(f"[def c] k-core numbers: max={core_arr.max()} distn={np.bincount(core_arr).tolist()}")

# (d) maximal cliques (overlapping hyperedges) on binary contact graph
cliques=[c for c in nx.find_cliques(G) if len(c)>=4]
P(f"[def d] maximal cliques (>=4): {len(cliques)} sizes={sorted([len(c) for c in cliques],reverse=True)[:15]}...")

# ---- FEATURE BUILDER for a PARTITION (each node one bundle) ----
def partition_feats(part):
    F=np.zeros((N,8))
    # precompute per-bundle within-contact directed pair list & synapse counts
    for k in range(N):
        i,j=int(I[k]),int(J[k])
        bi,bj=part[i],part[j]
        same=int(bi==bj and bi!=-1)
        F[k,0]=same
        if not same:
            continue
        members=[n for n in range(M) if part[n]==bi]
        msize=len(members)
        F[k,1]=msize
        # within-bundle contact degree of i and j (binary)
        F[k,2]=sum(Ab[i,n] for n in members if n!=i)
        F[k,3]=sum(Ab[j,n] for n in members if n!=j)
        # within-bundle directed contacting pairs (a,b) both in bundle, Ab>0
        # internal synapse density leave-this-pair-out
        num=0; den=0
        for a in members:
            for b in members:
                if a!=b and Ab[a,b]>0:
                    if a==i and b==j: continue
                    den+=1; num+=Y[a,b]
        F[k,4]=num/den if den>0 else 0.0
        # i's within-bundle out-synapse rate (leave pair out)
        oi_num=sum(Y[i,b] for b in members if b!=i and b!=j and Ab[i,b]>0)
        oi_den=sum(1 for b in members if b!=i and b!=j and Ab[i,b]>0)
        F[k,5]=oi_num/oi_den if oi_den>0 else 0.0
        # j's within-bundle in-synapse rate (leave pair out)
        ij_num=sum(Y[a,j] for a in members if a!=j and a!=i and Ab[a,j]>0)
        ij_den=sum(1 for a in members if a!=j and a!=i and Ab[a,j]>0)
        F[k,6]=ij_num/ij_den if ij_den>0 else 0.0
        # competition/saturation: i's within-bundle out-synapse COUNT (leave pair out)
        F[k,7]=oi_num
    return F

Fa=partition_feats(part_a)
Fb=partition_feats(part_b)

# ---- k-core feature: core number of i, j, min core, and within-shared-kcore synapse density ----
# treat each k-core (k = min(core_i,core_j)) subgraph as bundle
def kcore_feats():
    F=np.zeros((N,5))
    for k in range(N):
        i,j=int(I[k]),int(J[k])
        ci,cj=core_arr[i],core_arr[j]
        kk=min(ci,cj)
        F[k,0]=ci; F[k,1]=cj; F[k,2]=kk
        members=[n for n in range(M) if core_arr[n]>=kk]
        num=0;den=0
        for a in members:
            for b in members:
                if a!=b and Ab[a,b]>0 and core_arr[a]>=kk and core_arr[b]>=kk:
                    if a==i and b==j: continue
                    den+=1;num+=Y[a,b]
        F[k,3]=num/den if den>0 else 0.0
        F[k,4]=len(members)
    return F
Fc=kcore_feats()

# ---- clique (overlapping) features: number of shared cliques, max shared-clique syn density (LPO) ----
node_cliques=[[] for _ in range(M)]
for ci,c in enumerate(cliques):
    for n in c: node_cliques[n].append(ci)
clique_syn={}  # per clique: (num,den) of within-clique directed synapse
for ci,c in enumerate(cliques):
    cs=set(c); num=0;den=0
    for a in c:
        for b in c:
            if a!=b and Ab[a,b]>0:
                den+=1;num+=Y[a,b]
    clique_syn[ci]=(num,den)
def clique_feats():
    F=np.zeros((N,3))
    for k in range(N):
        i,j=int(I[k]),int(J[k])
        shared=set(node_cliques[i])&set(node_cliques[j])
        F[k,0]=len(shared)
        if shared:
            # max clique size shared, and LPO synapse density over union of shared cliques
            F[k,1]=max(len(cliques[ci]) for ci in shared)
            num=0;den=0
            seen=set()
            for ci in shared:
                for a in cliques[ci]:
                    for b in cliques[ci]:
                        if a!=b and Ab[a,b]>0 and (a,b) not in seen:
                            seen.add((a,b))
                            if a==i and b==j: continue
                            den+=1;num+=Y[a,b]
            F[k,2]=num/den if den>0 else 0.0
    return F
Fd=clique_feats()

np.savez(f"{SC}/bundle_feats.npz",Fa=Fa,Fb=Fb,Fc=Fc,Fd=Fd,part_a=part_a,part_b=part_b,core_arr=core_arr)
P("saved bundle_feats.npz shapes:",Fa.shape,Fb.shape,Fc.shape,Fd.shape)
P("Fa same-bundle frac:",Fa[:,0].mean().round(3),"Fb same-bundle frac:",Fb[:,0].mean().round(3))
