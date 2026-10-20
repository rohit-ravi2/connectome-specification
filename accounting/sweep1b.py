import csv, numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
R=list(csv.DictReader(open("data/processed/dev_stabilization_program/master_directed_chemical_pairs.csv")))
def c(n,f=float): return np.array([f(r[n]) for r in R])
a7=c('a7',int); a8=c('a8',int); disagree=(a7!=a8).astype(int)
cv=StratifiedKFold(5,shuffle=True,random_state=0)
def disp(keys,label):
    Xb=np.column_stack([c(k) for k in keys]); y2=np.concatenate([a7,a8]); X2=np.vstack([Xb,Xb])
    p=np.clip(cross_val_predict(GradientBoostingClassifier(random_state=0),X2,y2,cv=cv,method='predict_proba')[:,1][:len(R)],1e-4,1-1e-4)
    obs=disagree.mean(); pred=(2*p*(1-p)).mean()
    print(f"  {label:34s} obs_disagree={obs:.3f} pred(2p(1-p))={pred:.3f} ratio={obs/pred:.2f}")
    return obs/pred
base=['log_area','shared','cos','cdeg_s','cdeg_t','ntsame','src_share','tgt_share','src_rank','tgt_rank']
print("DISPERSION follow-up: does developmental persistence close the under-dispersion?")
r1=disp(base,"contact/topology/NT/competition")
r2=disp(base+['devpre'],"+ developmental persistence")
print(f"\n  ratio {r1:.2f} -> {r2:.2f} with persistence")
print(f"  -> {'persistence CLOSES under-dispersion: the missing propensity = developmental history' if r2>0.80 else 'persistence does NOT fully close it: reproducible structure remains beyond contact AND persistence (ratio still '+f'{r2:.2f})'}")
