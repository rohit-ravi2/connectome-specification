"""PER-RULE terminal-selector homeobox -> effector, Barabási-style (aggregate was marginal +0.014;
does the per-rule pass recover the KNOWN terminal-selector rules the way adhesion recovered netrin/DMA-1?).
rule(X homeobox, Y effector): is Y enriched in classes expressing X, beyond class-breadth, vs a
permutation null? Positive controls: unc-30->GABA battery, mec-3->touch DEG/ENaC, ttx-3->AIY cholinergic.
"""
import json, warnings; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
RNG=np.random.default_rng(0)
CE="../data/celegans/data/expression/cengen/thresholded/021821_liberal_threshold1.csv"
PANEL="../data/celegans/data/processed/grn_branch/panel_biophysical_targets.json"
cg=pd.read_csv(CE); classes=list(cg.columns[3:]); X=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values
gname=cg["gene_name"].astype(str).str.lower().values; nC=len(classes)
row={g:i for i,g in enumerate(gname)}

NONCEH=["unc-4","unc-30","unc-39","unc-42","unc-62","unc-86","lin-11","lin-39","lim-4","lim-6","lim-7",
        "mec-3","mab-5","egl-5","php-3","nob-1","vab-7","vab-15","ttx-1","ttx-3","cog-1","alr-1","tab-1",
        "dsc-1","pha-2","dve-1","pros-1","zfh-2","zag-1","hmbx-1","psa-3","eyg-1"]
hb=[g for g in gname if g.startswith("ceh-") or g in NONCEH]
hb=[g for g in dict.fromkeys(hb)]  # unique, keep order
panel=json.loads(Path(PANEL).read_text()); eff=[g.lower() for fam in panel.values() for g in fam["genes"]]
eff=[g for g in dict.fromkeys(eff) if g in row]
Ye=(X[[row[g] for g in eff]]>0).astype(float)  # nEff x nC
breadth=Ye.sum(0)  # effectors per class
# residualize each effector on breadth (logistic)
residY=np.zeros_like(Ye)
for k in range(len(eff)):
    y=Ye[k]
    if 3<=y.sum()<=nC-3:
        p=LogisticRegression(max_iter=500).fit(breadth.reshape(-1,1),y).predict_proba(breadth.reshape(-1,1))[:,1]
        residY[k]=y-p
    else: residY[k]=np.nan
HBind={g:(X[row[g]]>0) for g in hb if g in row}  # class expresses homeobox g

def rule_z(xg, yg, reps=200):
    if xg not in HBind or yg not in row: return None
    ix=HBind[xg]
    if ix.sum()<4 or (~ix).sum()<4: return None
    yi=eff.index(yg) if yg in eff else None
    r=residY[yi] if yi is not None else (X[row[yg]]>0).astype(float)-0  # if effector not in panel, raw
    if np.isnan(r).any(): return None
    obs=r[ix].mean()-r[~ix].mean()
    nul=np.array([ (lambda p: r[p].mean()-r[~p].mean())(RNG.permutation(ix)) for _ in range(reps)])
    return (obs-nul.mean())/(nul.std()+1e-9), obs, int(ix.sum())

print("="*78); print("POSITIVE CONTROLS (known terminal-selector -> effector battery)"); print("="*78)
PC=[("unc-30","GABA",["unc-25","unc-47","unc-46"]),
    ("mec-3","touch",["mec-4","mec-10"]),
    ("ttx-3","AIY-ACh",["cho-1","unc-17"])]
for sel,fate,tgts in PC:
    for t in tgts:
        res=rule_z(sel,t)
        if res: z,obs,n=res; print(f"  {sel:7s} -> {t:8s} ({fate}): z={z:+.1f} enrich={obs:+.3f} (X in {n} classes)"+("  <-- HIT" if z>3 else ""))
        else: print(f"  {sel:7s} -> {t:8s}: N/A (gene missing/degenerate)")

print("\n"+"="*78); print("FULL SCAN — all homeobox x effector rules"); print("="*78)
allz=[]
for xg in hb:
    if xg not in HBind: continue
    ix=HBind[xg]
    if ix.sum()<4 or (~ix).sum()<4: continue
    for k,yg in enumerate(eff):
        if np.isnan(residY[k]).any(): continue
        r=residY[k]; obs=r[ix].mean()-r[~ix].mean()
        # fast z: analytic mean-diff permutation approx via variance of resid
        n1=ix.sum(); n0=nC-n1; v=r.var()
        se=np.sqrt(v*(1/n1+1/n0)); z=obs/(se+1e-9)
        allz.append((xg,yg,obs,z))
allz.sort(key=lambda t:-t[3])
nsig4=sum(1 for t in allz if t[3]>4); nsig3=sum(1 for t in allz if t[3]>3)
exp4=len(allz)*(1-0.99997)  # ~z>4 one-sided
print(f"  total rules {len(allz)} | z>4: {nsig4} (chance ~{exp4:.0f}) | z>3: {nsig3} (chance ~{len(allz)*0.00135:.0f})")
print("  top rules:")
for xg,yg,obs,z in allz[:15]:
    fam=[f for f,d in panel.items() if yg in [g.lower() for g in d["genes"]]]
    print(f"    {xg:7s} -> {yg:9s} z={z:+.1f} enrich={obs:+.3f}  [{fam[0] if fam else '?'}]")
print("\n"+"="*78)
pc_ok=sum(1 for sel,_,tg in PC for t in tg if (lambda r: r and r[0]>3)(rule_z(sel,t)))
print(f"VERDICT: positive-control HITS (z>3): {pc_ok} | scan z>4 enrichment: {nsig4} vs ~{exp4:.0f} chance")
print("  ->", "PER-RULE RECOVERS TERMINAL-SELECTOR RULES (regulatory layer real at discrete level)" if pc_ok>=2 or nsig4>3*max(1,exp4)
      else "per-rule does NOT clearly recover known rules — regulatory code below class resolution (same as GRN)")
print("="*78)
