"""A5 HARDENING — decouple synapse STRENGTH from the animals that define reproducibility k.
Confound: k = fraction of animals with the edge; 'mean strength' is averaged over THOSE SAME animals ->
mechanically linked, so R2(k~strength)=0.545 may be arithmetic. Null: LEAVE-ONE-ANIMAL-OUT. For each held-out
animal a, predict the edge's PRESENCE in a from (i) mean strength in the OTHER animals (disjoint from a) vs
(ii) contact area. If out-of-sample strength still beats area, weight is a REAL independent reproducibility axis;
if it collapses toward area, it was a shared-synapse-count artifact. Accept either outcome.
"""
import warnings,sys,csv; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
def P(*a): print(*a); sys.stdout.flush()
D="../data/celegans/data"; AG="../accounting/principles"
# contacting-pair universe + area (reuse A5 output)
pairs=pd.read_csv(f"{AG}/A5_pairs_all.csv")   # pre,post,area,k,bin
area={(r.pre,r.post):r.area for r in pairs.itertuples()}
# per-animal synapse counts
syn=defaultdict(dict)
with open(f"{D}/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']!='chemical': continue
        ds=int(r['source_file'].split('_')[2].split('.')[0])
        syn[(r['pre'],r['post'])][ds]=int(r['synapses'])
def loo(dslist,tag):
    P("\n"+"="*66); P(f"LEAVE-ONE-ANIMAL-OUT over datasets {dslist}  [{tag}]")
    aucs_str=[]; aucs_area=[]; aucs_kstr=[]
    for a in dslist:
        others=[d for d in dslist if d!=a]
        y=[]; xs=[]; xa=[]
        for (pre,post),ar in area.items():
            sc=syn.get((pre,post),{})
            y.append(1 if sc.get(a,0)>0 else 0)
            xs.append(np.mean([sc.get(d,0) for d in others]))   # OUT-OF-SAMPLE strength (disjoint from a)
            xa.append(ar)
        y=np.array(y)
        if y.sum()<5 or y.sum()==len(y): continue
        aucs_str.append(roc_auc_score(y,xs)); aucs_area.append(roc_auc_score(y,xa))
    P(f"  predict HELD-OUT animal's edge PRESENCE (mean over {len(aucs_str)} held-out animals):")
    P(f"    out-of-sample STRENGTH (other animals)  AUC = {np.mean(aucs_str):.3f} ± {np.std(aucs_str):.3f}")
    P(f"    contact AREA                            AUC = {np.mean(aucs_area):.3f} ± {np.std(aucs_area):.3f}")
    P(f"    -> strength {'STILL beats' if np.mean(aucs_str)>np.mean(aucs_area)+0.02 else ('~= ' if abs(np.mean(aucs_str)-np.mean(aucs_area))<=0.02 else 'now BELOW')} area out-of-sample")
    return np.mean(aucs_str),np.mean(aucs_area)
# also reproduce the IN-SAMPLE (definitional) claim for contrast: k vs mean-strength-over-present + area
def insample(dslist,tag):
    rows=[]
    for (pre,post),ar in area.items():
        sc=syn.get((pre,post),{}); present=[sc[d] for d in dslist if sc.get(d,0)>0]
        k=len(present); ms=np.mean(present) if present else 0.0
        rows.append((k/len(dslist),ms,ar))
    df=pd.DataFrame(rows,columns=['kfrac','meanstr_present','area'])
    r_str=spearmanr(df.kfrac,df.meanstr_present)[0]; r_area=spearmanr(df.kfrac,df.area)[0]
    P(f"\n  IN-SAMPLE (definitional, for contrast) [{tag}]: Spearman k-vs-meanStrength(present)={r_str:.3f}  k-vs-area={r_area:.3f}")
P("A5 HARDENING — is synapse strength a REAL reproducibility axis or a shared-count artifact?")
insample([1,2,3,4,5,6,7,8],"ALL 1-8")
loo([1,2,3,4,5,6,7,8],"ALL 8 (mixed developmental stages)")
loo([1,2,3,4],"L1 replicates 1-4 (same-stage, cleanest)")
loo([5,6,7,8],"LATE 5-8")
P("\nVERDICT: read the out-of-sample strength vs area AUCs above. Out-of-sample strength predicting a held-out")
P("animal's presence is DISJOINT from that animal's own count, so if it beats area, weight is a genuine second")
P("reproducibility dimension; if it falls to ~area, the R2=0.545 was definitional coupling.")
