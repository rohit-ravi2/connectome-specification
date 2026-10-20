"""STAGE 1 — IS THE CONTACT GEOMETRY MOLECULAR? The decisive measurement.
Target = contact PRESENCE/STRENGTH itself (NOT residual, NOT synapse-beyond-contact). Predictor = molecular
families in native config (matching-complementary over Nawrocka LR pairs + barcode-similarity over IgSF/cadherin/
zig). NO geometric baseline subtracted. Capacity null = random genes (matched count), which controls overfitting
NOT the mediator. Report molecular AUC/R2 ALONGSIDE the pure-geometry (soma-distance) prior, never subtracted.
DECIDES: molecules explain contact >> random -> geometry is molecular, the mediated path we deleted is real.
"""
import warnings,sys,json; warnings.simplefilter("ignore")
import numpy as np, pandas as pd, openpyxl
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, r2_score
def P(*a): print(*a); sys.stdout.flush()
RNG=np.random.default_rng(0); D="../data/celegans/data"; SUP="../data/adhesion/nawrocka_supp"
pj=json.load(open("../data/network_positions.json"))
pos={c["cell"]:np.array([c["x"],c["y"],c["z"]],float) for c in pj["cells"] if None not in (c.get("x"),c.get("y"),c.get("z"))}
cg=pd.read_csv(D+"/expression/cengen/thresholded/021821_liberal_threshold1.csv")
classes=list(cg.columns[3:]); Xt=cg[classes].apply(pd.to_numeric,errors="coerce").fillna(0).values
gname=cg["gene_name"].astype(str).str.lower().values; wbid=cg["Wormbase_ID"].astype(str).values
cls_idx={c:k for k,c in enumerate(classes)}; nCls=len(classes); Xb=(Xt>0).astype(np.float32)
def n2c(n):
    if n in cls_idx: return cls_idx[n]
    b=n
    for s in "LR":
        if b.endswith(s): b=b[:-1]; break
    if b in cls_idx: return cls_idx[b]
    for s in "DV":
        if b.endswith(s) and b[:-1] in cls_idx: return cls_idx[b[:-1]]
    return cls_idx.get(b,None)
# contact
wb0=openpyxl.load_workbook(f"{D}/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx",read_only=True,data_only=True)
ws=wb0['adult nerve ring neighbors']; dat=list(ws.iter_rows(values_only=True)); hdr=[h for h in dat[0][1:]]
carea=defaultdict(float)
for r in dat[1:]:
    rn=r[0]
    if rn is None: continue
    for jj,val in enumerate(r[1:]):
        cn=hdr[jj]
        if cn and val not in (None,'') and rn!=cn:
            try: carea[(min(rn,cn),max(rn,cn))]=max(carea[(min(rn,cn),max(rn,cn))],float(val))
            except: pass
ring=[n for n in hdr if n in pos and n2c(n) is not None]
pairs=[(ring[i],ring[j]) for i in range(len(ring)) for j in range(i+1,len(ring))]; Np=len(pairs)
Ci=np.array([n2c(a) for a,b in pairs]); Cj=np.array([n2c(b) for a,b in pairs])
yC=np.array([1 if carea[(min(a,b),max(a,b))]>0 else 0 for a,b in pairs])
area=np.array([carea[(min(a,b),max(a,b))] for a,b in pairs])
somad=np.array([np.linalg.norm(pos[a]-pos[b]) for a,b in pairs])
P(f"ring neurons {len(ring)} | pairs {Np} | contact-rate {yC.mean():.3f}")

# Nawrocka LR pairs
s1=pd.read_excel(f"{SUP}/media-1.xlsx",sheet_name="Table S1",header=1); prot2wb={};seq2wb={}
for _,r in s1.iterrows():
    w=str(r.get("WormBase Gene ID","")).strip()
    if w.startswith("WBGene"):
        pn=str(r.get("Protein name","")).strip().upper(); sq=str(r.get("Sequence ID","")).strip()
        if pn and pn!="NAN": prot2wb[pn]=w
        if sq and sq!="nan": seq2wb[sq]=w
def wbrow(w):
    i=np.where(wbid==w)[0]; return int(i[0]) if len(i) else None
s3=pd.read_excel(f"{SUP}/media-3.xlsx",sheet_name="Table S3",header=1); LR=[]
for _,r in s3.iterrows():
    wa=prot2wb.get(str(r["Protein Name"]).strip().upper(),seq2wb.get(str(r["Sequence Name"]).strip()))
    wbb=prot2wb.get(str(r["Protein Name.1"]).strip().upper(),seq2wb.get(str(r["Sequence Name.1"]).strip()))
    if wa and wbb and wa!=wbb:
        ra,rb=wbrow(wa),wbrow(wbb)
        if ra is not None and rb is not None: LR.append((ra,rb))
LR=list(set(LR)); P(f"Nawrocka LR pairs mapped: {len(LR)}")
def matching_feats(lrpairs):
    F=np.zeros((Np,len(lrpairs)),np.float32)
    for c,(a,b) in enumerate(lrpairs):
        ea,eb=Xb[a],Xb[b]; F[:,c]=(ea[Ci]*eb[Cj]+eb[Ci]*ea[Cj])
    return F
def barcode_feats(prefs_list):
    cols=[]
    for prefs in prefs_list:
        g=[i for i in range(len(gname)) if any(gname[i].startswith(p) for p in prefs)]
        if len(g)<4: continue
        B=Xb[g]; cols.append((B[:,Ci]*B[:,Cj]).sum(0))
    return np.array(cols).T if cols else np.zeros((Np,0))
Fmatch=matching_feats(LR)
Fbar=barcode_feats([['zig-'],['cdh-'],['oig-','igcm-','igdb-','lron-'],['casy-'],['rig-']])
Fmol=np.hstack([Fmatch,Fbar]); P(f"molecular features: matching {Fmatch.shape[1]} + barcode {Fbar.shape[1]} = {Fmol.shape[1]}")

skf=StratifiedKFold(5,shuffle=True,random_state=0); kf=KFold(5,shuffle=True,random_state=0)
def auc_of(F):
    p=cross_val_predict(HistGradientBoostingClassifier(max_iter=200,random_state=0),F,yC,cv=skf,method="predict_proba")[:,1]
    return roc_auc_score(yC,p)
def r2_of(F):
    p=cross_val_predict(HistGradientBoostingRegressor(max_iter=200,random_state=0),F,np.log1p(area),cv=kf)
    return r2_score(np.log1p(area),p)
# random-gene capacity null: random gene pairs (matched count) + random barcode modules
pool=[i for i in range(len(gname)) if 0.05<=Xb[i].mean()<=0.60]
def random_mol():
    a=RNG.choice(pool,len(LR)); b=RNG.choice(pool,len(LR)); Fm=matching_feats(list(zip(a,b)))
    mods=[RNG.choice(pool,15) for _ in range(Fbar.shape[1] or 5)]
    Fb=np.array([(Xb[m][:,Ci]*Xb[m][:,Cj]).sum(0) for m in mods]).T
    return np.hstack([Fm,Fb])

P("\n"+"="*80); P("STAGE 1 — CONTACT PRESENCE (AUC): molecular vs random-capacity vs geometry prior"); P("="*80)
a_mol=auc_of(Fmol); a_geo=auc_of(somad.reshape(-1,1))
a_rand=[auc_of(random_mol()) for _ in range(3)]
P(f"  MOLECULAR (matching+barcode, native config)  AUC = {a_mol:.3f}")
P(f"  random-gene capacity null                    AUC = {np.mean(a_rand):.3f} ± {np.std(a_rand):.3f}")
P(f"  pure-geometry prior (soma distance)          AUC = {a_geo:.3f}   [reported alongside, NOT subtracted]")
P(f"  molecular + geometry together                AUC = {auc_of(np.hstack([Fmol,somad.reshape(-1,1)])):.3f}")
P("\n"+"="*80); P("STAGE 1 — CONTACT STRENGTH (R2, log area)"); P("="*80)
r_mol=r2_of(Fmol); r_geo=r2_of(somad.reshape(-1,1)); r_rand=[r2_of(random_mol()) for _ in range(3)]
P(f"  MOLECULAR R2 = {r_mol:.3f} | random-capacity R2 = {np.mean(r_rand):.3f}±{np.std(r_rand):.3f} | geometry R2 = {r_geo:.3f}")
P("\n"+"="*80)
molspec = a_mol > np.mean(a_rand)+3*np.std(a_rand)+0.01
P(f"VERDICT: contact geometry is molecularly predictable beyond capacity? {'YES' if molspec else 'NO'}")
P(f"  molecular AUC {a_mol:.3f} vs random {np.mean(a_rand):.3f} (Δ={a_mol-np.mean(a_rand):+.3f}); geometry-prior {a_geo:.3f}")
if molspec:
    P("  => the GEOMETRY IS MOLECULAR. The mediated path (molecule->contact->synapse) we deleted is REAL.")
    P("     'molecularly singular' was a method artifact; proceed to Stage 2 mediation decomposition.")
else:
    P("  => geometry is NOT molecularly specific (molecular ~= random capacity). Residual-hunting was valid;")
    P("     contact geometry has a non-molecular origin (mechanical/lineage/timing). Molecular story genuinely thin.")
P("="*80)
