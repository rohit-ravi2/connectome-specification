import sys, csv, re, numpy as np
from collections import defaultdict
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); z2=np.load(f"{SC}/resid2.npz")
I,J,y7,y8,neurons=z['I'],z['J'],z['y7'],z['y8'],list(z['neurons']); idx={n:i for i,n in enumerate(neurons)}
N=len(I); r1=z2['r1']; yrep=((y7==1)&(y8==1)).astype(int)
dp=defaultdict(set)
with open("../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']!='chemical': continue
        d=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
        if r['pre'] in idx and r['post'] in idx: dp[(idx[r['pre']],idx[r['post']])].add(d)
L1c=np.array([len(dp[(I[k],J[k])]&{1,2,3,4}) for k in range(N)])
agree=(y7==y8)  # both adults agree (deterministic); else coin-flip
coinflip=(y7!=y8)
P(f"ALL contacting pairs: {N} | coin-flip (present in exactly 1 adult): {coinflip.sum()} ({coinflip.mean()*100:.1f}%) | agree: {agree.sum()}")
hard=np.abs(r1)>0.3; FN=(r1>0.3); FP=(r1<-0.3)
P(f"\nHARD CORE {hard.sum()}: FN(r1>.3)={FN.sum()} FP(r1<-.3)={FP.sum()}")
# FN: yrep=1 => both present => agree-present (deterministic by construction)
P(f"  FN: agree-present(y7=y8=1)={((FN)&(y7==1)&(y8==1)).sum()} | coin-flip(among FN)={((FN)&coinflip).sum()}")
# FP: yrep=0 => split agree-absent vs coin-flip
fp_absent=(FP)&(y7==0)&(y8==0); fp_coin=(FP)&coinflip
P(f"  FP: agree-ABSENT(y7=y8=0, deterministic pruned/repelled)={fp_absent.sum()} | coin-flip(present 1 adult)={fp_coin.sum()} ({fp_coin.sum()/FP.sum()*100:.0f}% of FP)")
det_core=hard&agree; P(f"\n=> DETERMINISTIC hard core (adults agree) = {det_core.sum()} ({det_core.sum()/hard.sum()*100:.0f}% of 968); coin-flip dilution = {(hard&coinflip).sum()}")
P(f"   deterministic breakdown: FN(added,reproducible)={((det_core)&FN).sum()} | FP-absent(pruned/repelled,reproducible)={((det_core)&FP).sum()}")
# save clean masks + L1 count
np.savez(f"{SC}/mode0.npz",det_core=det_core,coinflip=coinflip,agree=agree,L1c=L1c,FN=FN,FP=FP,fp_absent=fp_absent)
P("saved mode0.npz")
