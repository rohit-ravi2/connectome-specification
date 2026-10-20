"""TEST 2 — NT-selector TF -> NT-synthesis battery (the failed sign-layer done at the layer the mechanism
actually builds). Sign (Layer C) was emergent -> null. NT identity is a directly-built expression structure.
Per-rule: does a specific NT-selector TF predict the NT-synthesis battery, beyond (a) generic identity =
random same-size gene set (the GRN-killer control) and (b) a permutation/degree null?
Positive control: unc-3 -> cholinergic MN battery (must land hard, like mec-3->mec-4).
"""
import json,warnings,sys; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
def P(*a): print(*a); sys.stdout.flush()
RNG=np.random.default_rng(0)
CE="../data/celegans/data/expression/cengen/thresholded/021821_liberal_threshold1.csv"
cg=pd.read_csv(CE); classes=list(cg.columns[3:]); X=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values
gname=cg["gene_name"].astype(str).str.lower().values; nC=len(classes); row={g:i for i,g in enumerate(gname)}

BATTERY={"ACh":["unc-17","cha-1","cho-1"],"Glu":["eat-4"],"GABA":["unc-25","unc-47","unc-46"],
         "monoamine":["tph-1","cat-1","cat-2","bas-1","tdc-1","tbh-1"]}
batt=[g for fam in BATTERY.values() for g in fam if g in row]
SELECTORS=["unc-3","unc-30","cfi-1","unc-86","ceh-14","ttx-3","che-1","ceh-10","unc-42","lin-11",
           "lim-6","mec-3","ast-1","pag-3","ceh-43","unc-4","ceh-2","nhr-67","lin-39","bnc-1"]
SELECTORS=[g for g in SELECTORS if g in row]
Yb=(X[[row[g] for g in batt]]>0).astype(float)  # battery x class
breadth=Yb.sum(0)
residB={}
for g in batt:
    y=(X[row[g]]>0).astype(float)
    if 3<=y.sum()<=nC-3:
        p=LogisticRegression(max_iter=500).fit(breadth.reshape(-1,1),y).predict_proba(breadth.reshape(-1,1))[:,1]
        residB[g]=y-p
def rule_z(sel,tgt,reps=2000):
    if sel not in row or tgt not in residB: return None
    ix=(X[row[sel]]>0)
    if ix.sum()<4 or (~ix).sum()<4: return None
    r=residB[tgt]; obs=r[ix].mean()-r[~ix].mean()
    nul=np.array([(lambda p:r[p].mean()-r[~p].mean())(RNG.permutation(ix)) for _ in range(reps)])
    return (obs-nul.mean())/(nul.std()+1e-9),obs,int(ix.sum())
P(f"NT-selectors in CeNGEN: {len(SELECTORS)} | battery genes: {len(batt)}")
P("\n"+"="*74); P("POSITIVE CONTROLS (known NT-selector -> its battery)"); P("="*74)
for sel,fam in [("unc-3","ACh"),("unc-30","GABA"),("cfi-1","ACh"),("che-1","Glu")]:
    for t in BATTERY[fam]:
        r=rule_z(sel,t)
        if r: P(f"  {sel:6s} -> {t:7s} ({fam}): z={r[0]:+.1f} enrich={r[1]:+.3f} (n={r[2]})"+("  HIT" if r[0]>3 else ""))
P("\n"+"="*74); P("FULL PER-RULE SCAN (selector x battery), permutation null"); P("="*74)
res=[]
for s in SELECTORS:
    for t in batt:
        r=rule_z(s,t)
        if r: res.append((s,t,r[0],r[1]))
res.sort(key=lambda x:-x[2]); nsig=sum(1 for x in res if x[2]>3)
P(f"rules {len(res)} | z>3: {nsig} (chance ~{len(res)*0.00135:.2f})")
fam_of={g:f for f,gs in BATTERY.items() for g in gs}
for s,t,z,o in res[:12]: P(f"   {s:6s} -> {t:7s} z={z:+.1f} enrich={o:+.3f} [{fam_of.get(t)}]"+("  HIT" if z>3 else ""))
# ---- GRN-killer control: NT-selectors vs RANDOM same-size gene set ----
P("\n"+"="*74); P("RANDOM-GENE CONTROL (GRN-killer): do NT-selectors beat random same-size genes?"); P("="*74)
pool=[i for i in range(len(gname)) if 10<=(X[i]>0).sum()<=nC-10 and gname[i] not in set(SELECTORS)|set(batt)]
def best_z_per_target(sel_rows):
    # for each battery gene, best |enrichment-z| among the given selector rows
    tot=[]
    for t in batt:
        if t not in residB: continue
        r=residB[t]; best=0
        for si in sel_rows:
            ix=(X[si]>0)
            if ix.sum()<4 or (~ix).sum()<4: continue
            obs=r[ix].mean()-r[~ix].mean()
            nul=np.array([(lambda p:r[p].mean()-r[~p].mean())(np.random.default_rng(k).permutation(ix)) for k in range(200)])
            best=max(best,(obs-nul.mean())/(nul.std()+1e-9))
        tot.append(best)
    return np.mean(tot)
sel_rows=[row[g] for g in SELECTORS]
real=best_z_per_target(sel_rows)
rand=[best_z_per_target(list(np.random.default_rng(s).choice(pool,len(SELECTORS),replace=False))) for s in range(5)]
rand=np.array(rand)
P(f"  mean best-per-battery-gene z: NT-selectors={real:.2f}  vs random {len(SELECTORS)}-gene sets={rand.mean():.2f}±{rand.std():.2f}")
P(f"  Δ = {real-rand.mean():+.2f}  -> "+("NT-SELECTOR-SPECIFIC (beats random = real regulatory code)" if real>rand.mean()+2*rand.std() else "NOT selector-specific (generic identity, same as GRN null)"))
P("\n"+"="*74)
P("VERDICT:", "NT-selector code REAL at NT-identity layer" if (nsig>3*max(1,len(res)*0.00135) and real>rand.mean()+2*rand.std())
  else "per-rule recovers known selectors but check random-gene control")
P("="*74)
