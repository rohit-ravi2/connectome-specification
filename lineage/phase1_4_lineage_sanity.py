"""Developmental-rule branch — Phase 1 (neuron master table) + Phase 4 sanity gate.

Phase-0 data reality: full Sulston tree NOT on disk; what IS = 279-neuron pairwise lineage-
RELATEDNESS matrix (WormAtlas, 1-25, higher=closer; L/R homologs ~17, sisters ~25). Packer
2019 birth-time/dev-expression covers only ~36-55 named neurons (embryo dataset) -> too sparse
for neuron-level backbone; used later on covered subset. Lineage feature = relatedness (all 279).

PHASE 4 SANITY GATE (must pass or feature-mapping is wrong): does LINEAGE predict NT identity
and modality? Test = lineage-kNN (leave-one-neuron-out): predict a neuron's NT from its lineage-
nearest neighbors' NT. vs prevalence baseline + NT-label permutation. Also modality + region.
"""
import warnings; warnings.simplefilter("ignore"); from pathlib import Path
import numpy as np, pandas as pd
RNG=np.random.default_rng(0)
D="../data/celegans/data"
OUT=Path(D+"/processed/developmental_rules"); OUT.mkdir(parents=True,exist_ok=True)

# ---- lineage relatedness matrix ----
rel=pd.read_csv(D+"/lineage/merged_lineage_table.csv")
neurons=sorted(set(rel.NeuronA)|set(rel.NeuronB)); idx={n:i for i,n in enumerate(neurons)}; N=len(neurons)
R=np.zeros((N,N))
for a,b,r in zip(rel.NeuronA,rel.NeuronB,rel.Relatedness):
    R[idx[a],idx[b]]=r; R[idx[b],idx[a]]=r
np.fill_diagonal(R,25.0)  # self = max

# ---- NT (Tier5), modality (nb28), region group ----
t5=np.load("../data/tier5/connectome.npz",allow_pickle=True)
t5nt={str(n):str(v) for n,v in zip(t5["names"],t5["nt_primary"])}
def bclass(n):
    b=n
    for s in "LRDV":
        if b.endswith(s): return b[:-1]
    return b
def normnt(x):
    x=str(x)
    if x.startswith("ACh") or x.startswith("Acetyl"): return "ACh"
    if x.startswith("Glu"): return "Glu"
    for k,l in [("GABA","GABA"),("Dopamine","DA"),("Serot","5HT"),("Octopamine","OA"),("Tyramine","TA")]:
        if k in x: return l
    return "Unknown"
prof=pd.read_csv(D_pr:="../data/celegans/New Notebooks/data_derived/nb28_neuron_profile.csv").set_index("neuron")
MOTOR={'VA','VB','VD','DA','DB','DD','AS','VC','RMD','RME','RIM','SMD','SMB','URA','RMF','RMH','RIV'}
reg=pd.read_csv(D+"/lineage/lineage_region_groups.csv"); reg["neuron_id"]=reg["neuron_id"].astype(str).str.strip()
reg_of=dict(zip(reg.neuron_id,reg.lineage_group))
def modality(n):
    if n in prof.index:
        if int(prof.loc[n,"is_sensory"])==1: return "sensory"
        if int(prof.loc[n,"is_command"])==1: return "command"
    if bclass(n) in MOTOR or n in MOTOR: return "motor"
    return "inter"

nt=np.array([normnt(t5nt.get(n,t5nt.get(bclass(n),"Unknown"))) for n in neurons],dtype=object)
mod=np.array([modality(n) for n in neurons],dtype=object)
regg=np.array([reg_of.get(n,-1) for n in neurons])

# Packer birth-time (covered subset only)
pk=pd.read_csv(D+"/expression/scrna_seq_other/GSE126954/GSE126954_cell_annotation.csv",
               usecols=["cell.subtype","embryo.time","lineage","cell.type"])
pk=pk[pk["cell.subtype"].notna()]
bt={}; fnd={}
for sub,g in pk.groupby("cell.subtype"):
    bt[str(sub)]=g["embryo.time"].median()
    lp=g["lineage"].dropna().astype(str)
    if len(lp): fnd[str(sub)]=lp.mode().iloc[0][0]  # founder = first char of modal lineage path
def lookup(n,dct):
    return dct.get(n, dct.get(bclass(n), np.nan))
birthtime=np.array([lookup(n,bt) for n in neurons],dtype=float)
founder=np.array([lookup(n,fnd) for n in neurons],dtype=object)

# ---- save master table ----
master=pd.DataFrame({"neuron":neurons,"nt":nt,"modality":mod,"region_group":regg,
                     "birth_time_packer":birthtime,"founder_packer":founder,
                     "has_NT":[t5nt.get(n,t5nt.get(bclass(n)))is not None for n in neurons]})
master.to_csv(OUT/"neuron_development_master.csv",index=False)
print(f"PHASE 1 — master table: {N} neurons | NT-covered {int((nt!='Unknown').sum())} | "
      f"modality {dict(pd.Series(mod).value_counts())} | region-covered {int((regg>=0).sum())} | "
      f"birthtime-covered {int(np.isfinite(birthtime).sum())} | founder-covered {int((founder!=None).sum() - int(pd.isna(pd.Series(founder)).sum()))}")
print("  NT dist:",dict(pd.Series(nt).value_counts()))
print("  saved",OUT/"neuron_development_master.csv")

# ================= PHASE 4 SANITY GATE: lineage-kNN predicts NT / modality / region =================
def knn_loo(labels, K=7):
    """leave-one-out: predict each neuron's label = weighted vote of K lineage-nearest neighbors."""
    lab=np.array(labels); pred=np.empty(N,dtype=object)
    for i in range(N):
        order=np.argsort(-R[i]); order=order[order!=i][:K]     # K most-related (excl self)
        w=R[i,order]                                            # relatedness as weight
        votes={}
        for j,wj in zip(order,w): votes[lab[j]]=votes.get(lab[j],0)+wj
        pred[i]=max(votes,key=votes.get)
    return pred
def scores(true,pred,mask):
    from sklearn.metrics import f1_score,balanced_accuracy_score,accuracy_score
    t,p=np.array(true)[mask],np.array(pred)[mask]
    return accuracy_score(t,p),f1_score(t,p,average="macro"),balanced_accuracy_score(t,p)

print("\n"+"="*84); print("PHASE 4 SANITY — lineage-kNN (leave-one-out) predicts NT / modality / region"); print("="*84)
for name,lab,mask in [("NT identity",nt,nt!="Unknown"),("modality",mod,np.ones(N,bool)),
                       ("region_group",regg.astype(object),regg>=0)]:
    pred=knn_loo(lab)
    acc,mf1,bacc=scores(lab,pred,mask)
    # prevalence baseline
    from collections import Counter
    maj=Counter(np.array(lab)[mask]).most_common(1)[0][0]
    base_acc=np.mean(np.array(lab)[mask]==maj)
    # permutation null (shuffle labels, redo kNN)
    perm=[]
    for s in range(20):
        pl=RNG.permutation(np.array(lab))
        pp=knn_loo(pl); perm.append(scores(pl,pp,mask)[1])
    perm=np.array(perm)
    z=(mf1-perm.mean())/(perm.std()+1e-9)
    print(f"  {name:14s} n={int(mask.sum()):3d} | acc={acc:.3f} (base {base_acc:.3f}) | macroF1={mf1:.3f} "
          f"| perm-F1 {perm.mean():.3f}±{perm.std():.3f} z={z:+.1f}")

print("\n"+"="*84)
# gate: lineage must predict NT above permutation clearly
pred_nt=knn_loo(nt); _,mf1_nt,_=scores(nt,pred_nt,nt!="Unknown")
perm_nt=np.array([scores(pl:=RNG.permutation(nt),knn_loo(pl),nt!="Unknown")[1] for _ in range(20)])
ok = mf1_nt > perm_nt.mean()+3*perm_nt.std()
print("PHASE-4 SANITY VERDICT:", "PASS — lineage predicts NT/identity; feature mapping valid -> proceed to Phase 5/6"
      if ok else "FAIL — lineage does NOT predict NT; feature mapping wrong -> fix before continuing")
print("="*84)
