#!/usr/bin/env python3
"""Phase 2 (Round 3, developmental-effector branch) -- FREEZE effector gene sets.

Builds pre-registered effector gene sets BEFORE any outcome (wiring/contact) is examined.
Two grounded sources per category:
  (1) CURATED SEEDS  -- canonical C. elegans neurodevelopment genes (symbol lists).
                        Every symbol is validated against the WormBase WS297 GAF;
                        any symbol not found in the GAF is DROPPED and logged
                        (=> no hallucinated / non-existent gene can survive).
  (2) GO EXPANSION   -- genes DIRECTLY annotated to a frozen list of GO IDs per category
                        in the WS297 c_elegans GAF.
                        CAVEAT: no OBO on disk => no DAG propagation; direct annotations only.
                        Curated seeds cover canonical genes GO direct-annotation may miss.

Output: data/processed/regulatory_branch/gene_sets_effector.json  (+ .freeze manifest w/ sha256).
Provenance is recorded PER GENE (curated?, which GO ids, in Packer matrix?, in CeNGEN?).
Stdlib only.
"""
import csv, json, gzip, hashlib, datetime, os
from collections import defaultdict

ROOT = "../data/celegans/data"
GAF  = f"{ROOT}/wormbase_release_WS297/associations/c_elegans.PRJNA13758.WS297.gene_association.wb"
PACKER_GENES = f"{ROOT}/expression/scrna_seq_other/GSE126954/GSE126954_gene_annotation.csv.gz"
CENGEN_GENES = f"{ROOT}/expression/scrna_seq_other/GSE136049/GSE136049_all_cells_gene_annotations.csv"
OUT_DIR = os.path.expanduser("../seed/data/processed/regulatory_branch")
OUT = f"{OUT_DIR}/gene_sets_effector.json"

# ------------------------------------------------------------------ CURATED SEEDS
# Canonical worm genes; validated against GAF at resolve time.
CURATED = {
  "axon_guidance": {
    "netrin":        ["unc-6","unc-5","unc-40","unc-129"],
    "slit_robo":     ["slt-1","sax-3","eva-1"],
    "ephrin_eph":    ["vab-1","efn-1","efn-2","efn-3","efn-4"],
    "wnt_frizzled":  ["egl-20","cwn-1","cwn-2","lin-44","mom-2","lin-17","mig-1","cfz-2","mom-5","cam-1"],
    "sema_plexin":   ["smp-1","smp-2","plx-1","plx-2","mab-20"],
    "cytoskel_guid": ["unc-44","unc-33","unc-51","unc-14","unc-76","ced-10","mig-2","unc-73",
                       "unc-53","sax-1","sax-2","vab-8","unc-71","unc-34"],
  },
  "adhesion_recognition": {
    "cadherin":      ["hmr-1","cdh-3","cdh-4","cdh-1","fmi-1","casy-1"],
    "igsf":          ["sax-7","rig-1","rig-3","rig-4","rig-5","rig-6","wrk-1","oig-1","oig-4",
                       "zig-1","zig-2","zig-3","zig-4","zig-5","zig-8","syg-1","syg-2","lad-2","dig-1"],
    "nrx_nlg":       ["nrx-1","nlg-1"],
    "lar_hspg":      ["ptp-3","clr-1","sdn-1","unc-52","him-4","mig-6"],
  },
  "synaptogenesis": {
    "active_zone":   ["unc-10","syd-1","syd-2","elks-1","cla-1","rimb-1","ida-1"],
    "release_mach":  ["unc-13","unc-18","unc-64","snb-1","snt-1","ric-4","unc-2","unc-31","ric-19"],
    "syn_organizer": ["madd-4","lev-9","lev-10","oig-4","rpm-1","fsn-1","sad-1","nid-1"],
  },
  "nt_fate_receptor": {
    "nt_synthesis":  ["cha-1","unc-17","unc-25","unc-46","unc-47","eat-4","cat-1","cat-2",
                       "tph-1","tbh-1","bas-1","tdc-1"],
    "nachr":         ["unc-38","unc-29","unc-63","lev-1","lev-8","acr-2","acr-3","acr-5","acr-16",
                       "deg-3","des-2"],
    "glucl_iglur":   ["avr-14","avr-15","glc-1","glc-2","glc-3","glc-4","glr-1","glr-2","glr-3",
                       "glr-4","glr-5","glr-6","glr-7","glr-8","nmr-1","nmr-2"],
    "gaba_r":        ["unc-49","gab-1","lgc-35","exp-1"],
    "monoamine_r":   ["mgl-1","mgl-2","mgl-3","dop-1","dop-2","dop-3","dop-4","ser-1","ser-2",
                       "ser-3","ser-4","ser-5","ser-6","ser-7","mod-1","octr-1","tyra-2","tyra-3",
                       "gbb-1","gbb-2"],
    "peptide_r":     ["npr-1","npr-2","npr-3","npr-4","npr-5","npr-6","npr-9","npr-11","frpr-3","egl-6"],
  },
  "terminal_selector_identity": {
    "terminal_sel":  ["unc-3","unc-4","unc-30","unc-42","unc-86","ttx-3","ceh-10","ceh-14","ceh-36",
                       "ceh-37","mec-3","che-1","cog-1","lim-6","lim-4","ast-1","tab-1","nhr-67",
                       "fax-1","vab-3","egl-5","egl-43","mab-5","lin-39","ceh-6","ttx-1","unc-55",
                       "ceh-43","ceh-16","zag-1","die-1","sem-4","pag-3"],
  },
}

# ------------------------------------------------------------------ GO ID SETS (frozen)
GO_TERMS = {
  "axon_guidance": ["GO:0007411","GO:0007409","GO:0048846","GO:0008046","GO:0097485",
                     "GO:0048841","GO:0007413","GO:0038007","GO:0048842","GO:0097486"],
  "adhesion_recognition": ["GO:0007155","GO:0007156","GO:0098742","GO:0050839","GO:0007157",
                            "GO:0016339","GO:0098609","GO:0098742"],
  "synaptogenesis": ["GO:0007416","GO:0050808","GO:0099054","GO:0051963","GO:0048786",
                      "GO:0007269","GO:0016079","GO:0048488","GO:0098793","GO:0007268"],
  "nt_fate_receptor": ["GO:0030594","GO:0005326","GO:0001505","GO:0042165","GO:0005230",
                        "GO:0004890","GO:0015464","GO:0099589","GO:0022824","GO:0004993"],
  "terminal_selector_identity": ["GO:0048665","GO:0045664","GO:0048663","GO:0021782"],
}

# ------------------------------------------------------------------ load GAF
sym2wb = {}; wb2sym = {}; go2wb = defaultdict(set)
with open(GAF) as f:
    for line in f:
        if line.startswith("!"): continue
        c = line.rstrip("\n").split("\t")
        if len(c) < 9: continue
        wb, sym, go = c[1], c[2], c[4]
        if sym: sym2wb.setdefault(sym, wb); wb2sym.setdefault(wb, sym)
        go2wb[go].add(wb)
print(f"GAF: {len(wb2sym)} genes, {len(go2wb)} GO ids")

# fallback symbol map from Packer gene_annotation (covers genes lacking GO annotations)
pk_sym2wb = {}
with gzip.open(PACKER_GENES, "rt") as f:
    for i, row in enumerate(csv.reader(f)):
        if i == 0 or len(row) < 3: continue
        wb, sname = row[1].strip().strip('"'), row[2].strip().strip('"')
        if wb.startswith("WBGene") and sname:
            pk_sym2wb.setdefault(sname, wb); wb2sym.setdefault(wb, sname)
def resolve(sym):
    return sym2wb.get(sym) or pk_sym2wb.get(sym)
print(f"Packer symbol map: {len(pk_sym2wb)} symbols")

# matrix universes
def load_wbset(path, opener=open, col=0, has_header=True):
    s = set()
    with (gzip.open(path,'rt') if opener==gzip.open else open(path)) as f:
        rdr = csv.reader(f)
        for i,row in enumerate(rdr):
            if has_header and i==0: continue
            if not row: continue
            tok = row[col].strip().strip('"')
            if tok.startswith("WBGene"): s.add(tok)
            elif len(row)>1 and row[1].strip().strip('"').startswith("WBGene"): s.add(row[1].strip().strip('"'))
    return s
packer_univ = load_wbset(PACKER_GENES, gzip.open)          # 20222
cengen_univ = set(l.strip() for l in open(CENGEN_GENES) if l.startswith("WBGene"))
print(f"Packer matrix genes: {len(packer_univ)} | CeNGEN genes: {len(cengen_univ)}")

# ------------------------------------------------------------------ build sets
categories = {}
unresolved = {}
for cat, subs in CURATED.items():
    genes = {}   # wb -> record
    unres = []
    # curated
    for subcat, syms in subs.items():
        for sym in syms:
            sym = sym.strip()
            if not sym or sym.endswith("?"):   # guard against stray tokens
                if sym and sym.endswith("?"): unres.append(sym)
                continue
            wb = resolve(sym)
            if wb is None:
                unres.append(sym); continue
            rec = genes.setdefault(wb, {"symbol": wb2sym.get(wb, sym), "curated": True,
                                        "curated_subcats": [], "go_ids": []})
            rec["curated"] = True
            if subcat not in rec["curated_subcats"]: rec["curated_subcats"].append(subcat)
    # GO expansion (direct annotations)
    for go in GO_TERMS.get(cat, []):
        for wb in go2wb.get(go, ()):
            rec = genes.setdefault(wb, {"symbol": wb2sym.get(wb, wb), "curated": False,
                                        "curated_subcats": [], "go_ids": []})
            if go not in rec["go_ids"]: rec["go_ids"].append(go)
    # membership flags
    for wb, rec in genes.items():
        rec["in_packer"] = wb in packer_univ
        rec["in_cengen"] = wb in cengen_univ
    categories[cat] = {
        "go_terms": GO_TERMS.get(cat, []),
        "curated_subcats": list(subs.keys()),
        "n_genes": len(genes),
        "n_curated": sum(1 for r in genes.values() if r["curated"]),
        "n_go_only": sum(1 for r in genes.values() if not r["curated"]),
        "n_in_packer": sum(1 for r in genes.values() if r["in_packer"]),
        "n_in_cengen": sum(1 for r in genes.values() if r["in_cengen"]),
        "genes": genes,
    }
    unresolved[cat] = unres

# union across categories (a gene may be in >1)
all_wb = set()
for c in categories.values(): all_wb |= set(c["genes"])

payload = {
  "meta": {
    "phase": "Phase 2 -- pre-frozen effector gene sets",
    "branch": "developmental-effector-expression gate (Round 3)",
    "frozen_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "sources": {"gaf": GAF, "packer_genes": PACKER_GENES, "cengen_genes": CENGEN_GENES},
    "method": "curated canonical seeds (GAF-validated) UNION GO direct-annotation expansion",
    "caveats": ["no OBO on disk => GO direct annotations only, no DAG propagation",
                "curated symbols not found in GAF are dropped+logged (see unresolved_curated)"],
    "n_categories": len(categories), "n_unique_genes": len(all_wb),
  },
  "categories": categories,
  "unresolved_curated": unresolved,
}

os.makedirs(OUT_DIR, exist_ok=True)
blob = json.dumps(payload, indent=2, sort_keys=True)
with open(OUT, "w") as f: f.write(blob)
# freeze manifest: hash of the gene membership only (stable across reruns)
member = {cat: sorted(categories[cat]["genes"].keys()) for cat in categories}
h = hashlib.sha256(json.dumps(member, sort_keys=True).encode()).hexdigest()
with open(OUT + ".freeze", "w") as f:
    json.dump({"frozen_at": payload["meta"]["frozen_at"], "sha256_membership": h,
               "n_unique_genes": len(all_wb),
               "per_category": {c: categories[c]["n_genes"] for c in categories}}, f, indent=2)

print("\n=== FROZEN EFFECTOR GENE SETS ===")
for cat, c in categories.items():
    print(f"  {cat:28s} n={c['n_genes']:4d}  (curated {c['n_curated']:3d}, GO-only {c['n_go_only']:4d}) "
          f" inPacker {c['n_in_packer']:4d}  inCeNGEN {c['n_in_cengen']:4d}")
    if unresolved[cat]:
        print(f"      dropped (not in GAF): {unresolved[cat]}")
print(f"\n  UNIQUE genes across all categories: {len(all_wb)}")
print(f"  membership sha256: {h[:16]}...")
print(f"  written: {OUT}")
