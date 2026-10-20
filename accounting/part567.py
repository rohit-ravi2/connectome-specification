import csv, math, numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy import sparse
R=[r for r in csv.DictReader(open("data/processed/specification_accounting/master_edge_table.csv")) if r['contact']=='1']
def col(n,f=float): return np.array([f(r[n]) for r in R])
Xs=np.column_stack([col('log_area'),col('shared_nb'),col('cos_sim'),col('cdeg_min'),col('cdeg_max'),
                    np.array([1.0 if r['src_nt']==r['tgt_nt'] else 0.0 for r in R])])
yrep=col('reproducible',int); cv=StratifiedKFold(5,shuffle=True,random_state=0)

print("="*70); print("PART 7 re-check — is the residual really structured? (GB baseline, not LR)"); print("="*70)
cats=sorted(set([r['src_class'] for r in R]+[r['tgt_class'] for r in R])); cix={c:i for i,c in enumerate(cats)}; K=len(cats)
CP=np.zeros((len(R),K*K))
for k,r in enumerate(R):
    a,b=cix[r['src_class']],cix[r['tgt_class']]; lo,hi=min(a,b),max(a,b); CP[k,lo*K+hi]=1
ag=roc_auc_score(yrep,cross_val_predict(GradientBoostingClassifier(random_state=0),Xs,yrep,cv=cv,method='predict_proba')[:,1])
agc=roc_auc_score(yrep,cross_val_predict(GradientBoostingClassifier(random_state=0),np.hstack([Xs,CP]),yrep,cv=cv,method='predict_proba')[:,1])
print(f"  GB structure AUC={ag:.3f} -> +class-pair AUC={agc:.3f}  (Δ={agc-ag:+.3f})")
print(f"  verdict: {'residual STRUCTURED (class adds beyond GB-structure)' if agc-ag>0.02 else 'residual ~NOISE (class Δ collapses under strong baseline = LR artifact)'}")

print("\n"+"="*70); print("PART 5 — macro (contact-area) vs micro (class/NT) determinism for the edge"); print("="*70)
def det_deg(binfeat):
    # determinism = 1 - H(edge|bin)/1 ; degeneracy proxy = H(bin|edge=1)/H(bin)
    vals=np.array(binfeat); Hcond=0.0
    for v in np.unique(vals):
        m=vals==v; p=yrep[m].mean()
        Hcond+=m.mean()*(0 if p<=0 or p>=1 else -p*math.log2(p)-(1-p)*math.log2(1-p))
    return 1-Hcond  # higher = more deterministic
area_bin=np.digitize(col('log_area'),np.quantile(col('log_area'),[.2,.4,.6,.8]))
clspair=np.array([cix[r['src_class']]*K+cix[r['tgt_class']] for r in R])
ntpair=np.array([hash((r['src_nt'],r['tgt_nt']))%997 for r in R])
for nm,b in [("contact-area bins (MACRO)",area_bin),("class-pair (MICRO identity)",clspair),("NT-pair",ntpair)]:
    print(f"  {nm:28s} determinism(1-H(edge|feat))={det_deg(b):.3f}  #states={len(np.unique(b))}")
print("  (macro should be high-determinism with FEW states; micro high-determinism only via MANY states=degeneracy)")

print("\n"+"="*70); print("PART 6 — functional: do strong-core edges carry more circuit load than halo?"); print("="*70)
# build undirected adult graph (union), global efficiency; matched knockout
neur=sorted(set([r['src'] for r in R]+[r['tgt'] for r in R])); nix={n:i for i,n in enumerate(neur)}; n=len(neur)
def build(exclude=set()):
    adj=[[] for _ in range(n)]
    for r in R:
        if (r['ad7']=='1' or r['ad8']=='1') and (r['src'],r['tgt']) not in exclude:
            adj[nix[r['src']]].append(nix[r['tgt']]); adj[nix[r['tgt']]].append(nix[r['src']])
    return adj
def gleff(adj):
    from collections import deque
    tot=0.0
    for s in range(n):
        d=[-1]*n; d[s]=0; q=deque([s])
        while q:
            u=q.popleft()
            for v in adj[u]:
                if d[v]<0: d[v]=d[u]+1; q.append(v)
        tot+=sum(1/d[t] for t in range(n) if d[t]>0)
    return tot/(n*(n-1))
base=gleff(build())
rng=np.random.default_rng(0)
core=[(r['src'],r['tgt']) for r in R if r['reproducible']=='1']
halo=[(r['src'],r['tgt']) for r in R if r['halo']=='1']
k=min(len(core),len(halo),300)
def drop(edges): return base-gleff(build(set(rng.choice(len(edges),k,replace=False).tolist() and [edges[i] for i in rng.choice(len(edges),k,replace=False)])))
dc=base-gleff(build(set(core[i] for i in rng.choice(len(core),k,replace=False))))
dh=base-gleff(build(set(halo[i] for i in rng.choice(len(halo),k,replace=False))))
print(f"  global efficiency baseline={base:.4f}")
print(f"  remove {k} strong-core edges: efficiency drop={dc:.5f}")
print(f"  remove {k} weak-halo edges:   efficiency drop={dh:.5f}")
print(f"  -> strong-core {'MORE' if dc>dh else 'NOT more'} load-bearing (ratio {dc/max(dh,1e-9):.2f}x)")
with open("data/processed/specification_accounting/final_probes.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["probe","value"])
    w.writerow(["residual_class_delta_GB",round(agc-ag,4)]); w.writerow(["gleff_base",round(base,5)])
    w.writerow(["core_knockout_drop",round(dc,6)]); w.writerow(["halo_knockout_drop",round(dh,6)])
print("\nartifact: final_probes.csv")
