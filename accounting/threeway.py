import sys, csv, re, numpy as np
from collections import defaultdict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); m0=np.load(f"{SC}/mode0.npz")
I,J,y7,y8,base,neurons=z['I'],z['J'],z['y7'],z['y8'],z['base'],list(z['neurons']); idx={n:i for i,n in enumerate(neurons)}
M=len(neurons); N=len(I); area=base[:,0]; L1c=m0['L1c']
# ---- load White 1986 N2U adult (3rd independent adult) ----
GN=np.zeros((M,M)); covered=set()
for line in open("../data/celegans/data/connectome/white_MoW/aconnectome_white_1986_A.csv"):
    p=line.strip().split('\t')
    if len(p)<4 or p[0]=='pre': continue
    pre,post,typ=p[0],p[1],p[2]
    if pre in idx: covered.add(idx[pre])
    if post in idx: covered.add(idx[post])
    if typ=='chemical' and pre in idx and post in idx: GN[idx[pre],idx[post]]=1
yn=GN[I,J]
cov=np.array([ (I[k] in covered) and (J[k] in covered) for k in range(N)])  # pairs both endpoints reconstructed in N2U
P(f"White N2U aligned: {len(covered)}/{M} nerve-ring neurons covered | {int(cov.sum())}/{N} pairs both-covered")
P(f"[BATCH CHECK] edge rate (both-covered pairs): Witvliet7={y7[cov].mean():.3f} Witvliet8={y8[cov].mean():.3f} N2U={yn[cov].mean():.3f}")
c=cov  # restrict all to covered
# ---- 3-way agreement structure ----
k3=(y7+y8+yn)[c]
P(f"[3-WAY STRUCTURE] present-in-k-of-3: k0={(k3==0).mean():.3f} k1={(k3==1).mean():.3f} k2={(k3==2).mean():.3f} k3={(k3==3).mean():.3f} (bimodal 0/3 = determinism)")
# ---- does 2-adult determinism hold under the 3rd (independent) adult? ----
ap=(y7==1)&(y8==1)&c; aa=(y7==0)&(y8==0)&c; cf=(y7!=y8)&c
P(f"[DETERMINISM CONFIRMATION by N2U]")
P(f"  2-adult AGREE-PRESENT (n={ap.sum()}): N2U also present = {yn[ap].mean():.3f}  (high => present-determinism real; N2U-present is batch-INFLATED so this is upper-ish)")
P(f"  2-adult AGREE-ABSENT  (n={aa.sum()}): N2U also absent  = {(1-yn[aa]).mean():.3f}  (high => absent-determinism real; N2U spurious-adds so this is a LOWER bound)")
P(f"  2-adult COIN-FLIP     (n={cf.sum()}): N2U present = {yn[cf].mean():.3f}  (~0.5 => genuinely variable)")
# ---- dispersion with 3 animals (does under-dispersion survive?) ----
GL1=np.zeros((M,M))
with open("../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']=='chemical':
            d=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
            if d in (1,2,3,4) and r['pre'] in idx and r['post'] in idx: GL1[idx[r['pre']],idx[r['post']]]=1
Cff=GL1@GL1; L1syn=GL1[I,J]; feats=np.column_stack([area,np.log1p(Cff[I,J]),L1syn])
cv=StratifiedKFold(5,shuffle=True,random_state=0)
def disp3(animals):
    A=np.array(animals)  # (n_animals, Npairs) on covered
    yr=(A.sum(0)>=A.shape[0]).astype(int)  # present in ALL
    dis=np.mean([ (A[a]!=A[b]).mean() for a in range(A.shape[0]) for b in range(a+1,A.shape[0])])
    Y=np.concatenate([A[a] for a in range(A.shape[0])]); X=np.vstack([feats[c]]*A.shape[0])
    p=np.clip(cross_val_predict(GradientBoostingClassifier(random_state=0),X,Y,cv=cv,method='predict_proba')[:,1][:c.sum()],1e-4,1-1e-4)
    return dis/(2*p*(1-p)).mean()
d2=disp3([y7[c],y8[c]]); d3=disp3([y7[c],y8[c],yn[c]])
P(f"[DISPERSION under process-model features contact+closure+persistence, covered pairs]")
P(f"  2 adults (Witvliet 7,8): {d2:.2f}")
P(f"  3 adults (+White N2U):   {d3:.2f}   (batch noise INFLATES disagreement => this is CONSERVATIVE; <1 = determinism survives 3rd animal)")
P(f"  => per-edge determinism {'SURVIVES the independent 3rd adult (real, not 2-adult artifact)' if d3<0.9 else 'weakens with 3rd adult (was partly 2-adult noise)'}")
