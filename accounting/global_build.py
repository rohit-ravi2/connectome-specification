import sys, csv, re, numpy as np
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); m0=np.load(f"{SC}/mode0.npz")
I,J,y7,y8,base,neurons=z['I'],z['J'],z['y7'],z['y8'],z['base'],list(z['neurons'])
idx={n:i for i,n in enumerate(neurons)}; M=len(neurons); N=len(I)
coinflip=m0['coinflip']; agree=m0['agree']
# ---- L1 backbone graph (Witvliet ds1-4) ----
GL1=np.zeros((M,M))
with open("../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']=='chemical':
            d=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
            if d in (1,2,3,4) and r['pre'] in idx and r['post'] in idx: GL1[idx[r['pre']],idx[r['post']]]=1
# ---- White N2U 3rd adult ----
GN=np.zeros((M,M)); covered=set()
for line in open("../data/celegans/data/connectome/white_MoW/aconnectome_white_1986_A.csv"):
    p=line.strip().split('\t')
    if len(p)<4 or p[0]=='pre': continue
    pre,post,typ=p[0],p[1],p[2]
    if pre in idx: covered.add(idx[pre])
    if post in idx: covered.add(idx[post])
    if typ=='chemical' and pre in idx and post in idx: GN[idx[pre],idx[post]]=1
yn=GN[I,J].astype(int)
cov=np.array([ (I[k] in covered) and (J[k] in covered) for k in range(N)])
P(f"White covered {len(covered)}/{M} neurons, {int(cov.sum())}/{N} pairs both-covered")
P(f"edge rates on covered: y7={y7[cov].mean():.3f} y8={y8[cov].mean():.3f} yn={yn[cov].mean():.3f}")
np.savez(f"{SC}/global_base.npz",I=I,J=J,y7=y7,y8=y8,yn=yn,cov=cov,coinflip=coinflip,agree=agree,
         GL1=GL1,area=base[:,0],neurons=np.array(neurons,dtype=object))
P("saved global_base.npz")
