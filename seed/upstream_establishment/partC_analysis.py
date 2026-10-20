#!/usr/bin/env python3
"""Part C analysis: do regulatory programs add over contact+birth+lineage+NT for
pre-adult establishment?  Neuron-disjoint CV.  Tests C2-C5 + controls.
Primary model = L2-logistic (fast, stable, clean added-value test); GBM confirm on headline.
"""
import json, os, warnings; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, r2_score

ROOT=os.path.expanduser("../seed")
OUT=f"{ROOT}/data/processed/upstream_early_establishment"
CENGEN="../data/celegans/data/expression/cengen/thresholded/021821_liberal_threshold1.csv"
RNG=np.random.default_rng(0)

df=pd.read_csv(f"{OUT}/partC_pairwise_features.csv")
pairs=list(zip(df.pre,df.post))
neurons=sorted(set(df.pre)|set(df.post))
TARGETS=[("L1_reproducible","T_L1_repro"),("preadult_persistent","T_preadult_persist"),("first_synapse_L1","T_first_L1")]

# ---------- baseline feature blocks (Part A) ----------
def block_contact():
    return df[["contact_area","contact_cos","contact_cn"]].values.astype(float)
def block_birth():
    pb=df.pre_birth.values.astype(float); qb=df.post_birth.values.astype(float)
    mn=np.nanmin(np.c_[pb,qb],1); mx=np.nanmax(np.c_[pb,qb],1)
    ad=np.where(np.isnan(pb)|np.isnan(qb),0,np.abs(pb-qb))
    ld=np.nan_to_num(df.lin_dist.values.astype(float),nan=20.0)
    sf=df.same_founder.values.astype(float)
    return np.nan_to_num(np.c_[mn,mx,ad,ld,sf])
def block_nt():
    key=(df.pre_nt.astype(str)+"|"+df.post_nt.astype(str))
    return pd.get_dummies(key).values.astype(float)
def block_lineage_only():
    ld=np.nan_to_num(df.lin_dist.values.astype(float),nan=20.0); sf=df.same_founder.values.astype(float)
    return np.c_[ld,sf]

# regulatory blocks
REG_GRN=["tf_cos","tf_pear","tf_l2","tf_mag_min","tf_mag_max"]
REG_TSEL=["tsel_cos","tsel_pear","tsel_shared","tsel_rich_min","tsel_rich_max"]
REG_MOD=[c for c in df.columns if c.startswith("mod_")]
EXPR=["full_cos","full_pear"]
def cols(names): return df[names].values.astype(float)

Xc=block_contact(); Xb=block_birth(); Xn=block_nt()
BASE=np.hstack([Xc,Xb,Xn])                    # Part A full baseline
BASE_NOcontact=np.hstack([Xb,Xn])

# ---------- neuron-disjoint CV ----------
def folds(seed):
    rng=np.random.default_rng(seed); f=np.array_split(rng.permutation(neurons),5)
    return {n:i for i,ff in enumerate(f) for n in ff}
def oof(X,y,seed,model="lr"):
    X=np.nan_to_num(X.astype(float))
    nf=folds(seed); pred=np.full(len(y),np.nan)
    P=np.array([nf[a] for a in df.pre]); Q=np.array([nf[b] for b in df.post])
    for k in range(5):
        te=(P==k)&(Q==k); tr=(P!=k)&(Q!=k)
        if te.sum()<8 or y[tr].sum()<5 or len(set(y[te]))<2: continue
        if model=="lr":
            clf=LogisticRegression(max_iter=2000,C=1.0)
            Xs=StandardScaler().fit(X[tr]); clf.fit(Xs.transform(X[tr]),y[tr])
            pred[te]=clf.predict_proba(Xs.transform(X[te]))[:,1]
        else:
            clf=GradientBoostingClassifier(random_state=seed,n_estimators=100,max_depth=3)
            clf.fit(X[tr],y[tr]); pred[te]=clf.predict_proba(X[te])[:,1]
    return pred
def auc(X,y,seeds=range(5),model="lr"):
    a=[]
    for s in seeds:
        p=oof(X,y,s,model); m=~np.isnan(p)
        if m.sum()>15 and len(set(y[m]))>1: a.append(roc_auc_score(y[m],p[m]))
    return (np.mean(a),np.std(a)) if a else (np.nan,np.nan)

# ================= C2: does regulatory add over baseline? =================
print("="*70,"\nC2  regulatory add over contact+birth+lineage+NT (neuron-disjoint AUC)\n"+"="*70)
c2rows=[]
for tname,tk in ([] if os.environ.get("SKIP_C2") else TARGETS):
    y=df[tk].values.astype(int)
    if y.sum()<15: continue
    b_lr=auc(BASE,y); b_gb=auc(BASE,y,range(3),"gbm")
    print(f"\n[{tname}] pos={y.sum()}/{len(y)}  BASE AUC lr={b_lr[0]:.3f}±{b_lr[1]:.3f}  gbm={b_gb[0]:.3f}")
    for bn,bc in [("+GRN_TFact",REG_GRN),("+TerminalSelector",REG_TSEL),("+ModuleScores",REG_MOD),
                  ("+REG_all",REG_GRN+REG_TSEL+REG_MOD),("+EXPR_snapshot",EXPR),
                  ("+REG_all+EXPR",REG_GRN+REG_TSEL+REG_MOD+EXPR)]:
        X=np.hstack([BASE,cols(bc)]); a_lr=auc(X,y); a_gb=auc(X,y,range(3),"gbm")
        c2rows.append(dict(target=tname,block=bn,base_lr=b_lr[0],auc_lr=a_lr[0],d_lr=a_lr[0]-b_lr[0],
                           base_gbm=b_gb[0],auc_gbm=a_gb[0],d_gbm=a_gb[0]-b_gb[0]))
        print(f"   {bn:20s} lr {a_lr[0]:.3f} (d={a_lr[0]-b_lr[0]:+.3f})   gbm {a_gb[0]:.3f} (d={a_gb[0]-b_gb[0]:+.3f})")
if c2rows: pd.DataFrame(c2rows).to_csv(f"{OUT}/partC_c2_addvalue.csv",index=False)

# ================= C3: lineage x molecular interaction =================
print("\n"+"="*70,"\nC3  lineage/birth x molecular interaction\n"+"="*70)
ld=np.nan_to_num(df.lin_dist.values.astype(float),nan=20.0); sf=df.same_founder.values.astype(float)
bd=np.nan_to_num(np.abs(df.pre_birth.values.astype(float)-df.post_birth.values.astype(float)),nan=0.0)
INT=np.nan_to_num(np.c_[ld*df.tsel_cos, sf*df.tsel_cos, bd*df.tsel_cos, ld*df.tf_cos, sf*df.tf_cos])
c3rows=[]
for tname,tk in TARGETS:
    y=df[tk].values.astype(int)
    if y.sum()<15: continue
    reg=cols(REG_TSEL+REG_GRN)
    a_add=auc(np.hstack([BASE,reg]),y)[0]
    a_int=auc(np.hstack([BASE,reg,INT]),y)[0]
    c3rows.append(dict(target=tname,base_plus_reg=a_add,plus_interaction=a_int,d_interaction=a_int-a_add))
    print(f"[{tname}] base+reg={a_add:.3f}  +interaction={a_int:.3f}  d={a_int-a_add:+.3f}")
pd.DataFrame(c3rows).to_csv(f"{OUT}/partC_c3_interaction.csv",index=False)

# ================= C4: MEDIATION (direct vs contact-mediated) =================
print("\n"+"="*70,"\nC4  mediation: is molecular effect direct or contact-mediated?\n"+"="*70)
# (a) predict contact geometry FROM regulatory features (neuron-disjoint R2)
REGX=cols(REG_GRN+REG_TSEL+REG_MOD)
med_rows=[]
for cvar in ["contact_cos","contact_area","contact_cn"]:
    yv=df[cvar].values.astype(float); r2s=[]
    for s in range(5):
        nf=folds(s); P=np.array([nf[a] for a in df.pre]); Q=np.array([nf[b] for b in df.post])
        pr=np.full(len(yv),np.nan)
        for k in range(5):
            te=(P==k)&(Q==k); tr=(P!=k)&(Q!=k)
            if te.sum()<8 or tr.sum()<30: continue
            sc=StandardScaler().fit(REGX[tr]); rg=Ridge(alpha=1.0).fit(sc.transform(REGX[tr]),yv[tr])
            pr[te]=rg.predict(sc.transform(REGX[te]))
        m=~np.isnan(pr)
        if m.sum()>30: r2s.append(r2_score(yv[m],pr[m]))
    print(f"  regulatory -> {cvar}: neuron-disjoint R2 = {np.mean(r2s):+.3f}")
    med_rows.append(dict(kind="reg_predicts_contact",var=cvar,r2=float(np.mean(r2s))))
# (b) reg lift WITHOUT contact vs WITH contact  -> mediated fraction
print("\n  reg lift with/without contact in the model:")
for tname,tk in TARGETS:
    y=df[tk].values.astype(int)
    if y.sum()<15: continue
    reg=cols(REG_TSEL+REG_GRN+REG_MOD)
    base_nc=auc(BASE_NOcontact,y)[0]; reg_nc=auc(np.hstack([BASE_NOcontact,reg]),y)[0]
    base_wc=auc(BASE,y)[0];           reg_wc=auc(np.hstack([BASE,reg]),y)[0]
    d_nc=reg_nc-base_nc; d_wc=reg_wc-base_wc
    med_frac=1-(d_wc/d_nc) if abs(d_nc)>1e-6 else np.nan
    # also: contact's own lift, and how much survives adding reg
    print(f"  [{tname}] reg lift no-contact={d_nc:+.3f} | after-contact={d_wc:+.3f} | mediated_frac={med_frac:+.2f}")
    med_rows.append(dict(kind="mediation",var=tname,d_reg_nocontact=d_nc,d_reg_withcontact=d_wc,mediated_fraction=float(med_frac)))
pd.DataFrame(med_rows).to_csv(f"{OUT}/partC_mediation_results.csv",index=False)

# ================= C5: regulatory explains baseline residual? =================
print("\n"+"="*70,"\nC5  does regulatory activity explain Part-A residuals?\n"+"="*70)
c5rows=[]
for tname,tk in TARGETS:
    y=df[tk].values.astype(int)
    if y.sum()<15: continue
    # OOF baseline prob (avg over seeds), residual = y - p
    ps=[]
    for s in range(5):
        p=oof(BASE,y,s); ps.append(p)
    P=np.nanmean(np.vstack(ps),0); m=~np.isnan(P); resid=np.where(m,y-P,0.0)
    # correlation of key reg similarities with residual
    for f in ["tsel_cos","tf_cos","full_cos"]:
        c=np.corrcoef(df[f].values[m],resid[m])[0,1]
        c5rows.append(dict(target=tname,feature=f,resid_corr=float(c)))
    # can reg predict residual sign beyond chance? ridge on residual, neuron-disjoint R2
    r2s=[]
    for s in range(5):
        nf=folds(s); Pn=np.array([nf[a] for a in df.pre]); Qn=np.array([nf[b] for b in df.post])
        pr=np.full(len(y),np.nan)
        for k in range(5):
            te=(Pn==k)&(Qn==k)&m; tr=(Pn!=k)&(Qn!=k)&m
            if te.sum()<8 or tr.sum()<30: continue
            sc=StandardScaler().fit(REGX[tr]); rg=Ridge(alpha=1.0).fit(sc.transform(REGX[tr]),resid[tr])
            pr[te]=rg.predict(sc.transform(REGX[te]))
        mm=(~np.isnan(pr))&m
        if mm.sum()>30: r2s.append(r2_score(resid[mm],pr[mm]))
    print(f"  [{tname}] resid corr tsel_cos={c5rows[-3]['resid_corr']:+.3f} tf_cos={c5rows[-2]['resid_corr']:+.3f} full_cos={c5rows[-1]['resid_corr']:+.3f} | reg->resid R2={np.mean(r2s):+.3f}")
    c5rows.append(dict(target=tname,feature="reg_resid_R2",resid_corr=float(np.mean(r2s))))
pd.DataFrame(c5rows).to_csv(f"{OUT}/partC_c5_residual.csv",index=False)
print("\nCORE tests done. Run partC_controls.py for nulls.")
