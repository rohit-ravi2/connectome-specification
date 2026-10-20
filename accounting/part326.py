import csv, math, numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
R=list(csv.DictReader(open("data/processed/dev_stabilization_program/master_directed_chemical_pairs.csv")))
def c(n,f=float): return np.array([f(r[n]) for r in R])
def X(keys): return np.column_stack([c(k) for k in keys])
cv=StratifiedKFold(5,shuffle=True,random_state=0)
def h(p): return 0.0 if p<=0 or p>=1 else -p*math.log2(p)-(1-p)*math.log2(1-p)
nf={n:i%5 for i,n in enumerate(sorted(set([r['src'] for r in R]+[r['tgt'] for r in R])))}
def nho(keys,y):
    XX=X(keys); pred=np.full(len(R),np.nan)
    for f in range(5):
        te=[k for k,r in enumerate(R) if nf[r['src']]==f and nf[r['tgt']]==f]; tr=[k for k,r in enumerate(R) if nf[r['src']]!=f and nf[r['tgt']]!=f]
        if len(te)<20 or len(set(y[tr]))<2: continue
        pred[np.array(te)]=GradientBoostingClassifier(random_state=0).fit(XX[tr],y[tr]).predict_proba(XX[te])[:,1]
    m=~np.isnan(pred); return roc_auc_score(y[m],pred[m])
up=['log_area','src_share','tgt_share','src_rank','tgt_rank','shared','cos','cdeg_s','cdeg_t','ntsame']

print("="*72); print("PART 3 — is developmental PERSISTENCE upstream-determined or its own variable?"); print("="*72)
persist=(c('devpre')>=3).astype(int)
print(f"  persistent edges (devpre>=3): {persist.sum()}/{len(R)} ({persist.mean():.2f})")
a_area=roc_auc_score(persist,cross_val_predict(GradientBoostingClassifier(random_state=0),X(['log_area']),persist,cv=cv,method='predict_proba')[:,1])
a_up=roc_auc_score(persist,cross_val_predict(GradientBoostingClassifier(random_state=0),X(up),persist,cv=cv,method='predict_proba')[:,1])
a_up_nho=nho(up,persist)
print(f"  predict persistence | contact-area only:  AUC={a_area:.3f}  (circularity check: is persistence just big contacts?)")
print(f"  predict persistence | ALL upstream:        AUC={a_up:.3f} (edge-CV) / {a_up_nho:.3f} (neuron-holdout)")
print(f"  -> persistence is {'MOSTLY contact-area (circular w/ geometry)' if a_area>a_up-0.03 else 'MORE than contact-area (genuine dev variable, +'+f'{a_up-a_area:.3f})'}")

print("\n"+"="*72); print("PART 2 — rate-distortion: minimal bits, WITHOUT vs WITH persistence (predict adult reproducible)"); print("="*72)
y=((c('a7',int)==1)&(c('a8',int)==1)).astype(int)  # reproducible chemical directed
def rd(keys,label):
    print(f"  {label}")
    for leaves in [2,4,8,16]:
        p=cross_val_predict(DecisionTreeClassifier(max_leaf_nodes=leaves,random_state=0),X(keys),y,cv=cv,method='predict_proba')[:,1]
        pp=np.clip(p,1e-6,1-1e-6); d=-(y*np.log2(pp)+(1-y)*np.log2(1-pp)).mean()
        print(f"     leaves={leaves:2d} (~{int(math.log2(leaves))}b)  distortion={d:.4f}  AUC={roc_auc_score(y,p):.3f}")
rd(['log_area','shared','cos','cdeg_s','cdeg_t','ntsame'],"WITHOUT persistence:")
rd(['log_area','shared','cos','cdeg_s','cdeg_t','ntsame','devpre'],"WITH persistence:")

print("\n"+"="*72); print("PART 6 — threshold generalization: does persistence contribution hold at >=1/>=2/>=4?"); print("="*72)
# need synapse counts -> not in this table; use presence-based proxy across thresholds via devpre bins as robustness
base=['log_area','shared','cos','cdeg_s','cdeg_t','ntsame']
for sub,mask in [("all contacting",np.ones(len(R),bool)),
                 ("high-contact (area>median)",c('log_area')>np.median(c('log_area'))),
                 ("low-contact (area<=median)",c('log_area')<=np.median(c('log_area')))]:
    idx=np.where(mask)[0]
    if len(idx)<200: continue
    yy=y[idx]
    if len(set(yy))<2: continue
    Xb=X(base)[idx]; Xp=X(base+['devpre'])[idx]
    ab=roc_auc_score(yy,cross_val_predict(GradientBoostingClassifier(random_state=0),Xb,yy,cv=cv,method='predict_proba')[:,1])
    ap=roc_auc_score(yy,cross_val_predict(GradientBoostingClassifier(random_state=0),Xp,yy,cv=cv,method='predict_proba')[:,1])
    print(f"  {sub:28s} n={len(idx):5d}  structure AUC={ab:.3f} -> +persistence {ap:.3f}  (Δ={ap-ab:+.3f})")
