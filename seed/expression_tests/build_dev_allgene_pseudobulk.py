#!/usr/bin/env python3
"""Phase 3b -- ALL-GENE dev pseudobulk background per window (E2,E3,E4) for random-matched
gene-set nulls that mirror the 15-dim (E2/E3/E4 x 5 category) effector features.
Streams Packer once. Output per window: dev_allgene_pb_<win>.csv (class x 20222 logCPM)
plus dev_allgene_meanexpr.csv (per-gene mean logCPM over E2-E4, detection rate).
"""
import gzip, csv, os, re
import numpy as np
from collections import defaultdict

DATA="../data/celegans/data"
OUT=os.path.expanduser("../seed/data/processed/regulatory_branch")
MTX=f"{DATA}/expression/scrna_seq_other/GSE126954/GSE126954_gene_by_cell_count_matrix.txt.gz"
GENE=f"{DATA}/expression/scrna_seq_other/GSE126954/GSE126954_gene_annotation.csv.gz"
CELL=f"{DATA}/expression/scrna_seq_other/GSE126954/GSE126954_cell_annotation.csv"
WINS=["E2_270_390","E3_390_510","E4_ge510"]

NONNEUR=("parent","grandparent","Neuroblast","lineage","blast","possibly","_and_","Parent","great",
 "hyp","Hypodermis","Seam","Intestin","muscle","Muscle","BWM","Germline","Coelomocyte","Glia","sheath",
 "socket","Arcade","arcade","gland","marginal","Pharyngeal_m","Rectal","Excretory","valve","XXX","Z1",
 "g1","g2","pm","mc","mu_","P_cell","M_cell","_cell","Tail","ant_arc","Anterior","Posterior","Body_wall",
 "GLR","hmc","Somatic","precursor","bin_")
def clean(s):
    if not s or s=="NA" or any(t in s for t in NONNEUR): return None
    return s if re.match(r'^[A-Z]{2,4}\d?$', s) else None
def win_of(t):
    try: t=float(t)
    except: return None
    if 270<=t<390: return "E2_270_390"
    if 390<=t<510: return "E3_390_510"
    if t>=510:     return "E4_ge510"
    return None

row2wb={}
with gzip.open(GENE,"rt") as f:
    for i,r in enumerate(csv.reader(f)):
        if i>0: row2wb[i]=r[1].strip().strip('"')
NG=len(row2wb); wb_order=[row2wb[i] for i in range(1,NG+1)]
col_cls={}; col_umi={}; col_win={}; ncell=0
with open(CELL) as f:
    for j,r in enumerate(csv.DictReader(f),1):
        ncell+=1
        col_umi[j]=float(r["n.umi"]) if r["n.umi"] not in("","NA") else 0.0
        col_cls[j]=clean(r["plot.cell.type"]) or clean(r["cell.subtype"])
        col_win[j]=win_of(r["embryo.time"])
classes=sorted(set(v for v in col_cls.values() if v)); cidx={c:k for k,c in enumerate(classes)}; C=len(classes)
print(f"cells {ncell}, classes {C}, genes {NG}")
# accumulators per window
acc={w:np.zeros((C,NG)) for w in WINS}
umi={w:np.zeros(C) for w in WINS}; ncnt={w:np.zeros(C,int) for w in WINS}
for j in range(1,ncell+1):
    c=col_cls[j]; w=col_win[j]
    if c and w: k=cidx[c]; umi[w][k]+=col_umi[j]; ncnt[w][k]+=1
nl=0
with gzip.open(MTX,"rt") as f:
    hdr=False
    for line in f:
        if line.startswith("%"): continue
        if not hdr: hdr=True; continue
        nl+=1; sp=line.split(); g=int(sp[0]); cc=int(sp[1]); v=float(sp[2])
        c=col_cls[cc]; w=col_win[cc]
        if c and w: acc[w][cidx[c],g-1]+=v
        if nl%25_000_000==0: print(f"  {nl/1e6:.0f}M")
print("streamed",nl)
for w in WINS:
    with open(f"{OUT}/dev_allgene_pb_{w}.csv","w",newline="") as fo:
        wr=csv.writer(fo); wr.writerow(["class"]+wb_order)
        for c in classes:
            k=cidx[c]
            if ncnt[w][k]<3 or umi[w][k]<=0: continue
            wr.writerow([c]+[round(float(x),5) for x in np.log1p(acc[w][k]/umi[w][k]*1e4)])
# per-gene mean logCPM over E2-E4 (for expression-level matching)
allrows=[];
for w in WINS:
    for c in classes:
        k=cidx[c]
        if ncnt[w][k]>=3 and umi[w][k]>0: allrows.append(np.log1p(acc[w][k]/umi[w][k]*1e4))
L=np.array(allrows); gmean=L.mean(0); gdet=(L>0).mean(0)
with open(f"{OUT}/dev_allgene_meanexpr.csv","w",newline="") as fo:
    wr=csv.writer(fo); wr.writerow(["WBGene","mean_logcpm","detect_rate"])
    for i,wb in enumerate(wb_order): wr.writerow([wb,round(float(gmean[i]),5),round(float(gdet[i]),4)])
print("DONE per-window all-gene pseudobulk ->",OUT)
