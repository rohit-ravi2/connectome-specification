import sys,numpy as np,pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import shap
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/feats.npz",allow_pickle=True)
I,J=z['I'],z['J']; yrep=z['yrep']; N=len(I); neurons=list(z['neurons'])
CONTACT,HO=z['CONTACT'],z['HO']
CN=list(z['CONTACT_NAMES']); HN=list(z['HO_NAMES'])
# best RELATIONAL model = contact + Cook-HO + RDPG-16 + Katz  (== CONTACT + HO here)
X=np.hstack([CONTACT,HO]); names=[f"C:{n}" for n in CN]+[f"HO:{n}" for n in HN]
interp={
 'C:c_L1':'contact area L1 (anatomy)','C:c_L2':'contact area L2','C:c_L3':'contact area L3',
 'C:c_adult':'adult contact area (Witvliet)','C:log_c_adult':'log adult contact area',
 'C:wcdeg_i':'pre contact degree','C:wcdeg_j':'post contact degree',
 'C:shared_contact_nbr':'#shared contact neighbors (contact transitivity)',
 'C:log_brittin_area':'Brittin adult contact area',
 'HO:log_Cin(sharedIN)':'READOUT/transitivity: shared presynaptic inputs (assortativity)',
 'HO:log_Cout(sharedOUT)':'READOUT/transitivity: shared postsynaptic targets',
 'HO:log_Cff(feedfwd)':'READOUT: 2-hop feedforward paths (transitivity)',
 'HO:log_comm(commonNbr)':'READOUT: common connectome neighbors',
 'HO:RDPG16_ij':'READOUT-DIRECT: rank-16 reconstruction of Cook edge i->j (contains direct edge)',
 'HO:RDPG16_ji':'READOUT-DIRECT: rank-16 reconstruction of Cook edge j->i',
 'HO:Katz_ij':'READOUT-DIRECT: Katz i->j (1st term = direct Cook edge)',
 'HO:Katz_ji':'READOUT-DIRECT: Katz j->i',
 'HO:log_C3_ij':'READOUT: 3-hop paths i->j (transitivity)','HO:log_C3_ji':'READOUT: 3-hop paths j->i'}
m=GradientBoostingClassifier(random_state=0).fit(X,yrep)
expl=shap.TreeExplainer(m); sv=expl.shap_values(X)
msh=np.abs(sv).mean(0)
order=np.argsort(-msh)
rows=[]
P("=== TASK 1: SHAP |mean| ranking (best relational model = contact+HO+RDPG+Katz) ===")
for r in order:
    tag='CONTACT' if names[r].startswith('C:') else 'HIGHER-ORDER'
    rows.append(dict(feature=names[r],block=tag,mean_abs_shap=float(msh[r]),interpretation=interp.get(names[r],'')))
    P(f"  {msh[r]:.4f}  {names[r]:26s} [{tag}] {interp.get(names[r],'')}")
pd.DataFrame(rows).to_csv(f"{SC}/part1_shap.csv",index=False)
# group SHAP by category: direct-edge readout vs transitivity/assortativity vs contact-anatomy
direct=['HO:RDPG16_ij','HO:RDPG16_ji','HO:Katz_ij','HO:Katz_ji']
transit=['HO:log_Cin(sharedIN)','HO:log_Cout(sharedOUT)','HO:log_Cff(feedfwd)','HO:log_comm(commonNbr)','HO:log_C3_ij','HO:log_C3_ji']
def gsum(gr): return sum(msh[names.index(g)] for g in gr)
tot=msh.sum()
P("\n=== SHAP mass by interpretive group (fraction of total |SHAP|) ===")
P(f"  HO direct-edge-readout (Katz/RDPG contain C_ij) : {gsum(direct)/tot:.3f}")
P(f"  HO transitivity/assortativity (sharedIN/OUT/ff/C3): {gsum(transit)/tot:.3f}")
P(f"  CONTACT anatomy                                   : {sum(msh[k] for k in range(len(names)) if names[k].startswith('C:'))/tot:.3f}")
