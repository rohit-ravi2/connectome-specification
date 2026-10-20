"""SHARED BULK-GENERATOR BENCHMARK (importable). All four generators (A Rentian, B Criticality, C Constraint,
D Minimal-embedding) are scored identically here so the head-to-head is honest. TARGET = the real CONTACT
structure (the bulk). A generator provides a per-pair contact PROPENSITY; score() returns:
  gen_auc      = neuron-disjoint held-out AUC(propensity vs real contact)   [generation]
  degnull_auc  = AUC of the Chung-Lu degree-preserving null (deg_i*deg_j)   [the null to beat]
  margin       = gen_auc - degnull_auc  [organizing structure BEYOND degree — the win condition]
  cook_auc     = AUC(propensity vs COOK synapses among eligible)  [held-out: connectome reads out from contact]
  params       = compression (part reports; needing ~#edges params = no compression)
NO connectome-from-connectome leak: build propensity from organizing principles / geometry, NOT the contact matrix
values themselves (Generator D low-rank-embeds contact and is the explicit benchmark that IS allowed to see contact).
"""
import numpy as np, openpyxl, json, csv, re, pandas as pd
from collections import defaultdict
from sklearn.metrics import roc_auc_score
_D="../data/celegans/data"
_SKD="../seed/data/external/mechanistic_frontier/witvliet2021_developmental/data/skeletons"
_cg=pd.read_csv(_D+"/expression/cengen/thresholded/021821_liberal_threshold1.csv"); _classes=list(_cg.columns[3:]); _ci={c:k for k,c in enumerate(_classes)}
def _n2c(n):
    if n in _ci: return _ci[n]
    b=n
    for s in "LR":
        if b.endswith(s): b=b[:-1]; break
    if b in _ci: return _ci[b]
    for s in "DV":
        if b.endswith(s) and b[:-1] in _ci: return _ci[b[:-1]]
    return _ci.get(b,None)
_wb=openpyxl.load_workbook(f"{_D}/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx",read_only=True,data_only=True)
_ws=_wb['adult nerve ring neighbors']; _dd=list(_ws.iter_rows(values_only=True)); _hdr=[h for h in _dd[0][1:]]
_ca=defaultdict(float)
for _r in _dd[1:]:
    _rn=_r[0]
    if _rn is None: continue
    for _jj,_v in enumerate(_r[1:]):
        _cn=_hdr[_jj]
        if _cn and _v not in (None,'') and _rn!=_cn:
            try: _ca[(min(_rn,_cn),max(_rn,_cn))]=max(_ca[(min(_rn,_cn),max(_rn,_cn))],float(_v))
            except: pass
SK=json.load(open(f"{_SKD}/Dataset7_skeletons.json"))
RING=[n for n in _hdr if n in SK and SK[n].get('coords') and _n2c(n) is not None]; M=len(RING); IDX={n:i for i,n in enumerate(RING)}
PAIRS=[(RING[i],RING[j]) for i in range(M) for j in range(i+1,M)]; NP=len(PAIRS)
PI=np.array([IDX[a] for a,b in PAIRS]); PJ=np.array([IDX[b] for a,b in PAIRS])
AREA=np.array([_ca.get((min(a,b),max(a,b)),0.0) for a,b in PAIRS])
CONTACT=(AREA>0).astype(int)                     # TARGET = the bulk contact structure
# contact-degree, positions, skeleton point clouds available to generators
_CADJ=np.zeros((M,M))
for _k in np.where(CONTACT>0)[0]: _CADJ[PI[_k],PJ[_k]]=1; _CADJ[PJ[_k],PI[_k]]=1
CADJ=_CADJ; CDEG=CADJ.sum(1)
_pj=json.load(open("../data/network_positions.json"))
_pos={c["cell"]:np.array([c["x"],c["y"],c["z"]],float) for c in _pj["cells"] if None not in (c.get("x"),c.get("y"),c.get("z"))}
POS=np.array([_pos.get(n,[np.nan]*3) for n in RING])
# Cook synapses (downstream held-out target: connectome reads out from contact)
_w2=openpyxl.load_workbook(f"{_D}/connectome/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
_A=list(_w2['hermaphrodite chemical'].iter_rows(values_only=True)); _nset=set(RING)
def _nm(seq): return sum(1 for x in seq if isinstance(x,str) and x.strip() in _nset)
_hr=int(np.argmax([_nm(r) for r in _A])); _cols=[(str(x).strip() if x is not None else None) for x in _A[_hr]]
_hc=int(np.argmax([_nm([_A[r][c] for r in range(len(_A))]) for c in range(min(6,len(_A[0])))]))
_GC=np.zeros((M,M))
for _r in range(_hr+1,len(_A)):
    _rn=_A[_r][_hc]; _rn=_rn.strip() if isinstance(_rn,str) else None
    if _rn not in IDX: continue
    for _c in range(_hc+1,len(_A[_r])):
        _cn=_cols[_c]; _v=_A[_r][_c]
        if _cn in IDX and _v not in (None,''):
            try:
                if float(_v)>0: _GC[IDX[_rn],IDX[_cn]]=1
            except: pass
YC=np.array([1 if (_GC[PI[k],PJ[k]] or _GC[PJ[k],PI[k]]) else 0 for k in range(NP)])   # Cook synapse per pair
# neuron-disjoint folds
_rng=np.random.default_rng(0); FOLD=np.array([_rng.integers(5) for _ in RING])
PFOLD=np.array([max(FOLD[PI[k]],FOLD[PJ[k]]) if FOLD[PI[k]]==FOLD[PJ[k]] else -1 for k in range(NP)])
def neuron_disjoint_auc(prop):
    aus=[]
    for f in range(5):
        te=(PFOLD==f)
        if CONTACT[te].sum()>3 and (CONTACT[te]==0).sum()>3: aus.append(roc_auc_score(CONTACT[te],prop[te]))
    return float(np.mean(aus))
_CL=CDEG[PI]*CDEG[PJ]; DEGNULL_AUC=roc_auc_score(CONTACT,_CL); DEGNULL_ND=neuron_disjoint_auc(_CL)
def score(propensity, n_params, name):
    propensity=np.asarray(propensity,float)
    gen=neuron_disjoint_auc(propensity); full=roc_auc_score(CONTACT,propensity)
    cook=roc_auc_score(YC,propensity)
    d={'name':name,'params':n_params,'gen_auc_neurondisjoint':round(gen,3),'gen_auc_full':round(full,3),
       'degnull_auc_neurondisjoint':round(DEGNULL_ND,3),'margin_over_degree':round(gen-DEGNULL_ND,3),
       'cook_synapse_auc':round(cook,3)}
    print("BENCHMARK", d)
    return d
if __name__=="__main__":
    print(f"ring {M} pairs {NP} contact-rate {CONTACT.mean():.3f} | Cook-syn {YC.sum()} | degree-null contact AUC (neuron-disjoint) {DEGNULL_ND:.3f}")
