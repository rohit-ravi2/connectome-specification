# connectome-specification

Where does the information that specifies the *C. elegans* wiring diagram live? The worm genome is too
small to list ~7,000 synapses one by one, yet the nerve ring is largely the same from animal to animal.
This repo is the analysis code behind that question: about 300 scripts that test candidate "hidden wiring
codes" against matched nulls on the Witvliet et al. (2021) developmental connectomes, CeNGEN expression,
the Sulston lineage and the Cook et al. (2019) adult connectome.

The short version of the result: physical contact, developmental birth timing and a small sender-side
transmitter bias carry most of the reproducible strong core. On directed contacting pairs, contact
geometry alone gives AUC 0.8245, adding developmental arrival gives 0.8851, adding the presynaptic
transmitter gives 0.8925, and the full Sulston lineage on top adds only +0.0018
(`lineage/results/phase7.txt`). Every aggregate molecular code I tested dissolved under a proper control.
A few rule-specific codes did survive, including a Slit-Robo (SLT-1 to SAX-3) signal that replicates on a
held-out connectome.

Write-up: https://www.rohitravi.com/projects/specified-information

## Results

All numbers below are copied from files in this repo.

**Contact geometry versus two-adult agreement** (`seed/results/null_controls.txt`, `seed/null_controls.py`).
The reproducible core is edges present in both Witvliet adults (datasets 7 and 8), among 5,030 nerve-ring
contacting pairs.

| min. synapses | core N | adult 7 predicts adult 8 (AUC) | contact geometry (AUC) |
|---|---|---|---|
| 1 | 1,257 | 0.819 | 0.839 |
| 2 | 784 | 0.843 | 0.879 |
| 4 | 458 | 0.880 | 0.905 |

I first read this as geometry reaching the animal-to-animal noise ceiling. That reading was wrong. One adult
predicting another is only a lower bound on the ceiling, and on harder splits contact does much worse: the
contact-area model scores 0.855 in-sample, 0.763 on neuron-disjoint cross-validation and 0.746 when scored
on the Cook adult as a held-out animal (`accounting/results/stage1_noise.txt`).

**The residual is structured, not noise.** Using the four L1 animals, the per-connection residual left after
a contact, degree and transmitter feature model (fit with neuron-disjoint CV) has a four-animal Spearman-Brown reliability of 0.855, split-half correlation 0.747, and
permutation z = 31.4 (`seed/data/processed/relational_propensity_grammar/split_half_residual.csv`).

**Specification ledger** (`lineage/results/phase7.txt`, 9,878 directed contacting pairs, 1,217 reproducible
chemical synapses, 5-fold CV over pairs):

| model | AUC |
|---|---|
| contact geometry | 0.8245 |
| + developmental arrival (L1 to L3 stages) | 0.8851 |
| + presynaptic transmitter | 0.8925 |
| + lineage (tree distance, birth class, founder) | 0.8943 |

**Hypotheses that did not survive** (one line each; the scripts print their own verdicts):

- Low-rank pairwise compatibility code. A learned source-by-target latent code of rank 2 to 16 adds 0 bits
  over geometry plus identity; the latent vectors collapse to mean norm 0.001
  (`seed/results/lowrank_pair_compatibility.txt`).
- Transmitter "grammar". The transmitter class-pair lift over geometry is real but small, ΔAUC +0.0147
  against a permuted-transmitter control of -0.0002 ± 0.0012 (`seed/results/null_controls.txt`); most of the
  pairwise part was birth timing (`seed/nt_pairwise_dev_check.py`).
- A compact program behind the lineage. The real Sulston tree compresses 5.07x by subtree sharing against
  4.14x for random trees with the same leaves, a ratio of only 1.23x (`lineage/results/E0_mdl_compression.txt`).
- Lineage as a wiring blueprint: +0.0018 AUC beyond contact, timing and transmitter (above).
- Expression homophily, gene-regulatory-network repertoire, developmental-window expression and
  structure-to-function were also tested (`seed/upstream_establishment/`, `grn/`, `seed/expression_tests/`,
  `seed/perturbation/`); see the write-up for the details of each control.

**Rule-specific codes that do hold up:**

- Slit-Robo. Among synapses the contact model misses, SLT-1 to SAX-3 pairs are enriched (degree-null
  z = +25.8, expression-permutation z = 3.2, n = 50; `accounting/results/residual_adhesion_v2.txt`) and the
  signal replicates on the Cook connectome (n = 150, expression-permutation z = 2.8, region z = 2.0,
  contact-residual z = 4.5; `accounting/results/held_out_cook.txt`). Two other candidates, Notch-Delta and
  MIG-13 to MIG-6, did not replicate.
- Terminal selectors. Homeobox genes recover known selector-to-effector rules (mec-3 to mec-4 and mec-10)
  and 189 rules at z > 4 against about 1 expected by chance (`grn/results/homeobox_per_rule.txt`). Known
  transmitter selectors beat random same-size gene sets, mean best z 3.09 vs 1.96 ± 0.20
  (`grn/results/nt_selector.txt`).

**Reproducibility baseline** (`seed/results/reproducibility.txt`): 48.6% of distinct adult edges appear in
both adults (chemical 50.3%, electrical 36.9%); 24.8% of L1 edges appear in all four L1 animals.

## What the nulls do and do not rule out

Every molecular test used mature mRNA at the level of neuron classes (CeNGEN). MicroRNAs and other small
RNAs were not measured, left and right cells cannot be told apart, and there is no surface-protein data from
the time the wiring forms. The negative results are bounded by that resolution. The fair summary is that,
at this resolution, the genome looks like it specifies a physical-and-temporal scaffold plus a sparse set of
rule-specific codes, not an edge-by-edge wiring table. Contact dominating the wiring echoes Brittin et al.
(2021); reproducibility echoes Witvliet et al. (2021). What is new here is the timing mediation, the
controlled transmitter bias, and the systematic testing of hidden-code hypotheses against matched nulls.

## Repo layout

```
seed/          core decomposition: reproducibility, contact geometry, nulls, NT gates, low-rank code
  contact_decomposition/  developmental_window/  expression_tests/  perturbation/
  relational_propensity_grammar/  self_avoidance_apposition/  subcellular_tiling/
  upstream_establishment/  simulation/
  data/processed/         small result tables (CSV/JSON) written by the scripts
  results/                saved stdout of the key runs
grn/           gene-regulatory layer: CelEsT regulators, homeobox and NT-selector rules
accounting/    specification accounting: held-out Cook replication, adhesion and Slit-Robo screens,
               generative and dynamical models, and a set of shorter exploratory branches (overnight/)
lineage/       Sulston lineage features, birth timing, the Phase 7 ledger, lineage MDL
```

Script names follow the order the work was done in (`partA_`, `partB_`, ..., `stage1_`, `phase7`), not a
pipeline. The `seed/` scripts and the files in `results/` are the best place to start.

## How to run

Python 3.11+ with numpy, scipy, pandas, scikit-learn, openpyxl and networkx. A few scripts use torch or brian2.

```
pip install numpy scipy pandas scikit-learn openpyxl networkx
cd seed && python null_controls.py           # geometry vs two-adult agreement, NT lift
cd lineage && python phase7.py               # the specification ledger
```

Each script is run from its own directory and reads inputs from `../data/` (see below). Some later scripts
read tables written by earlier ones (`seed/data/processed/`, `../work/`), so run the `build_*` scripts in a
directory before its analysis scripts.

## Data

No third-party data is included. Scripts expect it under `data/` at the repo root; `data/celegans/`
mirrors my local data tree, and the path strings in each script show exactly where a file goes.
Sources:

- `data/celegans/data/connectome/witvliet2020/` and Witvliet et al. 2021 datasets 1 to 8 (synapses, skeletons,
  physical contact): https://github.com/dwitvliet/nature2021 and https://nemanode.org
- `data/celegans/data/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx` and
  `cook2019/SI 3 Synapse lists.xlsx`: supplementary tables of Cook et al. 2019 (Nature 571:63) and
  Brittin et al. 2021 (Nature 591:105), https://wormwiring.org
- `data/celegans/data/expression/cengen/`: CeNGEN thresholded expression (021821 release), https://cengen.org
- Packer et al. 2019 embryonic single-cell expression (GEO GSE126954)
- Randi et al. 2023 functional connectivity atlas: https://github.com/leiferlab/worm-functional-connectivity
- Sulston lineage tree as `lineage/jtauber_lineage/cells.json`: https://gist.github.com/jtauber/ef6ff50e9e14900e00bd
- Sulston neuron birth times (Sulston et al. 1983, Dev Biol 100:64)
- CelEsT gene regulatory network and Shiny app (Perez 2024, Genetics, doi:10.1093/genetics/iyae189),
  placed at `grn/CelEsT-app/`
- WormBase WS297 gene IDs and anatomy associations: https://wormbase.org
- `data/tier5/connectome.npz`: a merged neuron table (names, primary transmitter) built in a separate
  simulator project; any per-neuron transmitter table from Loer and Rand (WormAtlas) can stand in.
- `data/adhesion/nawrocka_supp/`: supplementary tables of the Nawrocka et al. adhesion-molecule study,
  used by the adhesion screens in `accounting/`.
- A few exploratory scripts also read intermediate tables from my other worm projects
  (`data/celegans/New Notebooks/`, `data/celegans/experiments/`, `data/network_positions.json`). Those
  tables (neuron profiles and 2D layout coordinates) are not included.

## License

MIT
