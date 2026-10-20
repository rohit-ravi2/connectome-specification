#!/usr/bin/env python3
"""Phase 3 (Round 3) -- map developmental expression to neuron classes.

PRIMARY dev source: Packer 2019 embryo (GSE126954). Streams the sparse count matrix
(only the 510 frozen effector-gene rows) and pseudobulks to (neuron_class x embryo_time_window)
with CPM normalization using per-cell n.umi. ADULT baseline: CeNGEN expression_neuron_mean.csv.

Outputs (data/processed/regulatory_branch/):
  developmental_effector_features_by_class.csv   (class x effector-gene logCPM, wiring window)
  developmental_effector_expression_long.csv     (class,window,gene,... all windows)
  adult_effector_features_by_class.csv           (class x effector-gene log-expr, CeNGEN)
  developmental_module_scores.csv / adult_module_scores.csv
  mapping_quality_report.txt

Alignment assumptions (VERIFIED in report):
  - matrix row i (1-based) <-> gene_annotation data row i (per-dataset order; Packer!=Cao)
  - matrix col j (1-based) <-> cell_annotation data row j
  A pan-neuronal marker sanity check (neuron vs non-neuron) is emitted to confirm.
"""
import gzip, csv, json, os, re, math
import numpy as np
from collections import defaultdict

DATA = "../data/celegans/data"
OUT  = os.path.expanduser("../seed/data/processed/regulatory_branch")
PACK_MTX  = f"{DATA}/expression/scrna_seq_other/GSE126954/GSE126954_gene_by_cell_count_matrix.txt.gz"
PACK_GENE = f"{DATA}/expression/scrna_seq_other/GSE126954/GSE126954_gene_annotation.csv.gz"
PACK_CELL = f"{DATA}/expression/scrna_seq_other/GSE126954/GSE126954_cell_annotation.csv"
CENGEN    = f"{DATA}/expression/cengen/derived/expression_neuron_mean.csv"
GENESETS  = f"{OUT}/gene_sets_effector.json"

# pan-neuronal / marker WBGenes for alignment sanity (rab-3, unc-119, ric-19, snb-1) + a muscle marker (myo-3)
MARKERS = {"WBGene00004267":"rab-3(neuronal)", "WBGene00006843":"unc-119(neuronal)",
           "WBGene00003514":"myo-3(muscle)"}

# ---------------- effector gene sets ----------------
gs = json.load(open(GENESETS))
eff_wb = {}          # WBGene -> list of categories
for cat, c in gs["categories"].items():
    for wb in c["genes"]:
        eff_wb.setdefault(wb, []).append(cat)
CATS = list(gs["categories"].keys())
print(f"effector genes: {len(eff_wb)} across {len(CATS)} categories")

# ---------------- neuron-class canonicalization ----------------
NONNEUR_TOK = ("parent","grandparent","Neuroblast","lineage","blast","possibly","_and_",
               "Parent","great","hyp","Hypodermis","Seam","Intestin","muscle","Muscle","BWM",
               "Germline","Coelomocyte","Glia","sheath","socket","Arcade","arcade","gland",
               "marginal","Pharyngeal_m","Rectal","Excretory","valve","XXX","Z1","g1","g2",
               "pm","mc","mu_","P_cell","M_cell","_cell","Tail","ant_arc","Anterior","Posterior",
               "Body_wall","sh","so","GLR","hmc","Somatic","precursor","to.filter","bin_")
def is_progenitor_or_nonneuron(s):
    return any(t in s for t in NONNEUR_TOK)

def clean_identity(s):
    """Return a clean class-like neuron identity token or None."""
    if not s or s == "NA": return None
    if is_progenitor_or_nonneuron(s): return None
    # accept simple identity tokens: 2-4 letters + optional single digit (ASE, AVA, IL1, RMD, DA)
    if re.match(r'^[A-Z]{2,4}\d?$', s): return s
    return None

def canon(name):
    """collapse a neuron NAME (ASEL, RMDDL, VA12...) toward a class label by stripping
    trailing L/R then D/V then digits, returning progressively shorter candidates."""
    cands = [name]
    n = name
    # strip trailing L/R
    if len(n) > 2 and n[-1] in "LR": n = n[:-1]; cands.append(n)
    # strip trailing D/V
    if len(n) > 2 and n[-1] in "DV": n = n[:-1]; cands.append(n)
    # strip trailing digits (motor VA1.. -> VA) but keep IL1/IL2/RME etc as-is candidates
    m = re.match(r'^([A-Z]{2,4})(\d+)$', name)
    if m: cands.append(m.group(1))
    return cands

# ---------------- Packer gene rows ----------------
pack_row2wb = {}
with gzip.open(PACK_GENE, "rt") as f:
    for i, row in enumerate(csv.reader(f)):
        if i == 0: continue
        pack_row2wb[i] = row[1].strip().strip('"')   # data row i (1-based) -> WBGene
eff_row = {r: wb for r, wb in pack_row2wb.items() if wb in eff_wb}
marker_row = {r: wb for r, wb in pack_row2wb.items() if wb in MARKERS}
print(f"Packer effector rows: {len(eff_row)} | marker rows: {len(marker_row)}")

# ---------------- Packer cells -> (class, window, umi) ----------------
def window_of(t):
    if t == "" or t is None: return None
    try: t = float(t)
    except ValueError: return None
    if t < 270:  return "E1_lt270"
    if t < 390:  return "E2_270_390"
    if t < 510:  return "E3_390_510"
    return "E4_ge510"
WIRING = "E3_390_510"   # nerve-ring assembly window (late embryo) -- primary

col_class = {}; col_win = {}; col_umi = {}; col_isneur = {}
n_cell = 0; n_neur = 0
with open(PACK_CELL) as f:
    for j, row in enumerate(csv.DictReader(f), start=1):
        n_cell += 1
        umi = float(row["n.umi"]) if row["n.umi"] not in ("","NA") else 0.0
        col_umi[j] = umi
        # neuron identity from plot.cell.type then cell.subtype
        lab = clean_identity(row["plot.cell.type"]) or clean_identity(row["cell.subtype"])
        col_class[j] = lab
        col_win[j] = window_of(row["embryo.time"])
        # crude neuron flag for marker sanity
        ct = row["cell.type"] or ""
        col_isneur[j] = ("neuron" in ct.lower())
        if lab: n_neur += 1
print(f"Packer cells: {n_cell} | cells with terminal neuron identity: {n_neur}")

# dev class set (Packer terminal identities present) -> target for adult label matching
dev_set = set(v for v in col_class.values() if v)

def match_to_class(name, target):
    """greedily collapse a CeNGEN label (ASEL, RMDDL, DA1..) to a dev class in `target`."""
    n = name
    while True:
        if n in target: return n
        if len(n) > 2 and n[-1] in "LRDV": n = n[:-1]
        elif re.search(r'\d$', n):        n = re.sub(r'\d+$', '', n)
        else: return None

# ---------------- adult CeNGEN per-class (matched to dev labels) ----------------
with open(CENGEN) as f:
    rdr = csv.reader(f); hdr = next(rdr)
    genes = hdr[1:]
    keep_idx = [i for i,g in enumerate(genes) if g in eff_wb]
    adult_genes = [genes[i] for i in keep_idx]
    cengen_by_class = defaultdict(list)
    for r in rdr:
        vals = np.array([float(r[1+i]) for i in keep_idx])
        lab = match_to_class(r[0], dev_set) or clean_identity(r[0]) or r[0]
        cengen_by_class[lab].append(vals)
adult_present = set(adult_genes)
adult_feat = {lab: np.mean(vs, axis=0) for lab, vs in cengen_by_class.items()}
print(f"CeNGEN: {len(cengen_by_class)} class labels, {len(adult_genes)}/{len(eff_wb)} effector genes present")

# ---------------- stream matrix, accumulate ----------------
gene_sum = defaultdict(lambda: np.zeros(len(eff_wb)))   # (class,window) -> sums over ORDERED eff genes
eff_order = sorted(eff_wb)                               # canonical column order for eff genes
eff_pos = {wb:i for i,wb in enumerate(eff_order)}
umi_sum = defaultdict(float); cell_n = defaultdict(int)
# precompute per-group umi + cell count (over neuron cells with a window)
for j in range(1, n_cell+1):
    lab = col_class[j]; w = col_win[j]
    if lab and w:
        umi_sum[(lab,w)] += col_umi[j]; cell_n[(lab,w)] += 1
# marker accumulators
mk_neur = defaultdict(float); mk_nonneur = defaultdict(float)
neur_umi = 0.0; nonneur_umi = 0.0
for j in range(1, n_cell+1):
    (neur_umi if col_isneur[j] else 0)  # noop
for j in range(1, n_cell+1):
    if col_isneur[j]: neur_umi += col_umi[j]
    else: nonneur_umi += col_umi[j]

nlines = 0
with gzip.open(PACK_MTX, "rt") as f:
    header_done = False
    for line in f:
        if line.startswith("%"): continue
        if not header_done:      # dims line
            header_done = True; continue
        nlines += 1
        sp = line.split()
        g = int(sp[0])
        if g in eff_row:
            c = int(sp[1]); v = float(sp[2])
            lab = col_class[c]; w = col_win[c]
            if lab and w:
                gene_sum[(lab,w)][eff_pos[eff_row[g]]] += v
        elif g in marker_row:
            c = int(sp[1]); v = float(sp[2])
            wbm = marker_row[g]
            if col_isneur[c]: mk_neur[wbm] += v
            else: mk_nonneur[wbm] += v
        if nlines % 20_000_000 == 0:
            print(f"  ...streamed {nlines/1e6:.0f}M nnz")
print(f"streamed {nlines} nonzeros")

# ---------------- build feature tables ----------------
# developmental: logCPM per (class,window)
dev_long_rows = []
dev_feat_wiring = {}
for (lab,w), sums in gene_sum.items():
    denom = umi_sum[(lab,w)]
    if denom <= 0 or cell_n[(lab,w)] < 3:   # require >=3 cells
        continue
    cpm = sums / denom * 1e4
    logcpm = np.log1p(cpm)
    if w == WIRING:
        dev_feat_wiring[lab] = logcpm
    for i, wb in enumerate(eff_order):
        if cpm[i] > 0:
            dev_long_rows.append((lab, w, wb, eff_wb[wb][0],
                                  cell_n[(lab,w)], round(float(cpm[i]),4), round(float(logcpm[i]),5)))

# module scores (mean logCPM over category genes) per (class,window)
def module_scores(logcpm_vec):
    out = {}
    for cat in CATS:
        idx = [eff_pos[wb] for wb in gs["categories"][cat]["genes"] if wb in eff_pos]
        out[cat] = float(np.mean(logcpm_vec[idx])) if idx else 0.0
    return out

os.makedirs(OUT, exist_ok=True)
# dev features wide (wiring window)
with open(f"{OUT}/developmental_effector_features_by_class.csv","w",newline="") as f:
    wr = csv.writer(f); wr.writerow(["class"]+eff_order)
    for lab in sorted(dev_feat_wiring):
        wr.writerow([lab]+[round(float(x),5) for x in dev_feat_wiring[lab]])
# dev long (all windows)
with open(f"{OUT}/developmental_effector_expression_long.csv","w",newline="") as f:
    wr = csv.writer(f); wr.writerow(["class","window","WBGene","category","n_cells","cpm","logcpm"])
    wr.writerows(dev_long_rows)
# dev module scores (all windows)
with open(f"{OUT}/developmental_module_scores.csv","w",newline="") as f:
    wr = csv.writer(f); wr.writerow(["class","window","n_cells"]+CATS)
    for (lab,w), sums in sorted(gene_sum.items()):
        if cell_n[(lab,w)]<3 or umi_sum[(lab,w)]<=0: continue
        logcpm = np.log1p(sums/umi_sum[(lab,w)]*1e4)
        ms = module_scores(logcpm)
        wr.writerow([lab,w,cell_n[(lab,w)]]+[round(ms[c],5) for c in CATS])

# adult features wide (effector genes present in CeNGEN) + module scores
adult_eff_order = [wb for wb in eff_order if wb in adult_present]
adpos = {wb:i for i,wb in enumerate(adult_genes)}
with open(f"{OUT}/adult_effector_features_by_class.csv","w",newline="") as f:
    wr = csv.writer(f); wr.writerow(["class"]+adult_eff_order)
    for lab in sorted(adult_feat):
        v = adult_feat[lab]
        wr.writerow([lab]+[round(float(np.log1p(v[adpos[wb]])),5) for wb in adult_eff_order])
with open(f"{OUT}/adult_module_scores.csv","w",newline="") as f:
    wr = csv.writer(f); wr.writerow(["class"]+CATS)
    for lab in sorted(adult_feat):
        vlog = np.log1p(adult_feat[lab])
        row=[lab]
        for cat in CATS:
            idx=[adpos[wb] for wb in gs["categories"][cat]["genes"] if wb in adpos]
            row.append(round(float(np.mean(vlog[idx])) if idx else 0.0,5))
        wr.writerow(row)

# ---------------- mapping quality report ----------------
dev_classes = set(dev_feat_wiring)
dev_classes_all = set(l for (l,w) in gene_sum if cell_n[(l,w)]>=3)
adult_classes = set(adult_feat)
both = sorted(dev_classes_all & adult_classes & set(clean_identity(x) or x for x in adult_classes))
both = sorted([c for c in dev_classes_all if c in adult_feat])
with open(f"{OUT}/mapping_quality_report.txt","w") as f:
    def p(*a): print(*a); f.write(" ".join(str(x) for x in a)+"\n")
    p("=== Phase 3 mapping quality report (Round 3) ===")
    p(f"Packer cells total: {n_cell} | with terminal neuron identity: {n_neur} ({100*n_neur/n_cell:.1f}%)")
    p(f"effector genes: {len(eff_wb)} | in Packer: {len(eff_row)} | in CeNGEN: {len(adult_genes)}")
    p(f"\nALIGNMENT SANITY (neuron vs non-neuron mean CPM of markers):")
    for wb,name in MARKERS.items():
        n_c = mk_neur.get(wb,0)/neur_umi*1e4 if neur_umi else 0
        nn_c = mk_nonneur.get(wb,0)/nonneur_umi*1e4 if nonneur_umi else 0
        p(f"   {name:22s} neuron={n_c:8.2f}  nonneuron={nn_c:8.2f}  ratio={ (n_c/(nn_c+1e-9)):6.2f}")
    p(f"   (neuronal markers should be >>1; muscle marker <<1 -> confirms col alignment)")
    p(f"\nDEV classes with >=3 cells (any window): {len(dev_classes_all)}")
    p(f"DEV classes at WIRING window {WIRING}: {len(dev_classes)}")
    p(f"ADULT (CeNGEN) class labels: {len(adult_classes)}")
    p(f"CLASSES WITH BOTH dev(any-window) + adult: {len(both)}")
    p(f"   both: {both}")
    p(f"\nDEV-only (no adult match): {sorted(dev_classes_all - adult_feat.keys())}")
    p(f"\ncells per (class,window) [wiring-relevant windows]:")
    for (lab,w) in sorted(cell_n, key=lambda k:(k[0],k[1])):
        if cell_n[(lab,w)]>=3:
            p(f"   {lab:6s} {w:12s} n={cell_n[(lab,w)]:4d}")
print("\nDONE. wrote features + report to", OUT)
