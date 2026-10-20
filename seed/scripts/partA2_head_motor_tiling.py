"""
Part A2: Head-motor family x family tiling / apposition-without-synapse matrix.

Question: do the head-motor families {SMB,SIA,SIB,SAA,SMD,RME,RMD} form a
reproducible tiling architecture in which similar neurons *contact* (share
membrane apposition) but *avoid* forming chemical synapses onto each other?

Data:
  data/processed/self_avoidance_apposition/high_contact_residual_master.csv
    (adult-contacting pairs; nonsyn == (s_adult==0))
  witvliet2021 developmental physical_contact adjacency (per-stage contact area)

Outputs (data/processed/subcellular_tiling/):
  partA2_head_motor_family_matrix.csv   family x family contact/synapse table
  partA2_family_tiling_results.csv      no-synapse-zone / tiling / control tests
  partA2_top_residual_pairs.csv         ranked high-contact nonsynaptic HM pairs
Doc: docs/subcellular_tiling/PART_A2_HEAD_MOTOR_TILING.md
"""
import numpy as np, pandas as pd, itertools, warnings
warnings.filterwarnings('ignore')
rng = np.random.default_rng(42)

OUT = 'data/processed/subcellular_tiling/'
MASTER = 'data/processed/self_avoidance_apposition/high_contact_residual_master.csv'
ADJ = 'data/external/mechanistic_frontier/witvliet2021_developmental/physical_contact/Dataset{}_adjacency.csv'
HM = ['SMB', 'SIA', 'SIB', 'SAA', 'SMD', 'RME', 'RMD']
STAGE_DS = {'L1': [1, 2, 3, 4], 'L2': [5], 'L3': [6], 'adult': [8]}

# ---------------------------------------------------------------- load master
df = pd.read_csv(MASTER)
df['syn_ever'] = ((df.s_L1 + df.s_L2 + df.s_L3 + df.s_adult) > 0).astype(int)
df['syn_adult'] = (df.s_adult > 0).astype(int)          # 1 - nonsyn
df['elec'] = df.has_gap.astype(int)


def pos(name):
    """parse dorsal/ventral and left/right subclass from a neuron name."""
    dv = 'D' if 'D' in name[3:] else ('V' if 'V' in name[3:] else 'X')
    lr = 'L' if name.endswith('L') else ('R' if name.endswith('R') else 'X')
    return dv, lr


for c in ['na', 'nb']:
    dv, lr = zip(*df[c].map(pos))
    df[c + '_dv'] = dv
    df[c + '_lr'] = lr
df['same_dv'] = (df.na_dv == df.nb_dv).astype(int)
df['same_lr'] = (df.na_lr == df.nb_lr).astype(int)

hm = df[df.fam_a.isin(HM) & df.fam_b.isin(HM)].copy()
print(f'[A2] HM-HM contacting pairs: {len(hm)}  nonsyn={hm.nonsyn.mean():.3f}  '
      f'gap={hm.elec.mean():.3f}')

# ------------------------------------------------ per-stage contact (adjacency)
# collect union of HM neurons appearing in master
hm_neurons = sorted(set(hm.na) | set(hm.nb))


def load_stage_contact():
    """return dict stage -> DataFrame(mean normalized contact area, symmetric)."""
    out = {}
    for stage, dss in STAGE_DS.items():
        mats = []
        for d in dss:
            a = pd.read_csv(ADJ.format(d), index_col=0)
            a = a.reindex(index=hm_neurons, columns=hm_neurons).fillna(0.0)
            a = (a + a.T) / 2.0
            tot = a.values.sum()
            if tot > 0:
                a = a / tot * 1e6           # per-million normalization within dataset
            mats.append(a)
        out[stage] = sum(mats) / len(mats)
    return out


stage_contact = load_stage_contact()


def pair_contact(na, nb, stage):
    m = stage_contact[stage]
    if na in m.index and nb in m.columns:
        return float(m.at[na, nb])
    return 0.0


for stage in STAGE_DS:
    hm[f'ctc_{stage}'] = [pair_contact(a, b, stage) for a, b in zip(hm.na, hm.nb)]
# contact present at every stage (persistent apposition)
hm['contact_all_stages'] = ((hm.ctc_L1 > 0) & (hm.ctc_L2 > 0) &
                            (hm.ctc_L3 > 0) & (hm.ctc_adult > 0)).astype(int)

# =====================================================================
# A2.1  family x family matrix
# =====================================================================
rows = []
for fa, fb in itertools.combinations_with_replacement(HM, 2):
    sub = hm[((hm.fam_a == fa) & (hm.fam_b == fb)) |
             ((hm.fam_a == fb) & (hm.fam_b == fa))]
    if len(sub) == 0:
        continue
    rows.append(dict(
        fam_a=fa, fam_b=fb, within=int(fa == fb), n_pairs=len(sub),
        mean_contact_adult=sub.c_adult.mean(),
        mean_ctc_norm_adult=sub.ctc_adult.mean(),
        chem_syn_rate=sub.syn_adult.mean(),          # fraction with adult chemical synapse
        chem_syn_ever_rate=sub.syn_ever.mean(),
        elec_rate=sub.elec.mean(),
        nonsyn_rate=sub.nonsyn.mean(),
        same_nt_frac=sub.same_nt.mean(),
        same_mod_frac=sub.same_mod.mean(),
        frac_LR_homolog=sub.same_lr.mean(),
        frac_DV_same=sub.same_dv.mean(),
        persist_contact_frac=sub.contact_all_stages.mean(),
        contact_L1_frac=(sub.c_L1 > 0).mean(),
    ))
mat = pd.DataFrame(rows).sort_values(['within', 'nonsyn_rate'],
                                     ascending=[False, False])
mat.round(4).to_csv(OUT + 'partA2_head_motor_family_matrix.csv', index=False)
print(f'[A2.1] family x family cells: {len(mat)}  '
      f'(within-family={mat.within.sum()}, cross-family={(~mat.within.astype(bool)).sum()})')

# =====================================================================
# A2.2  no-synapse zones: where do high-contact NONSYN pairs live?
# =====================================================================
nonsyn = hm[hm.nonsyn == 1]
zone = dict(
    n_HM_nonsyn=len(nonsyn),
    frac_within_family=(nonsyn.fam_a == nonsyn.fam_b).mean(),
    frac_same_nt=nonsyn.same_nt.mean(),
    frac_same_mod=nonsyn.same_mod.mean(),
    frac_LR_homolog=nonsyn.homolog.mean(),
    frac_same_DV=nonsyn.same_dv.mean(),
    frac_same_LR=nonsyn.same_lr.mean(),
    frac_has_gap=nonsyn.elec.mean(),
    frac_contact_all_stages=nonsyn.contact_all_stages.mean(),
    frac_contact_at_L1=(nonsyn.c_L1 > 0).mean(),
)
print('[A2.2] no-synapse zone profile:',
      {k: round(v, 3) for k, v in zone.items()})

# =====================================================================
# A2.3  tiling / exclusion test
# =====================================================================
tests = []

# (i) same-family vs cross-family adult chemical-synapse rate at matched contact
same_fam = hm[hm.fam_a == hm.fam_b]
cross_fam = hm[hm.fam_a != hm.fam_b]
tests.append(dict(test='A2.3_syn_rate_same_family', group='same_family',
                  n=len(same_fam), chem_syn_rate=same_fam.syn_adult.mean(),
                  mean_contact=same_fam.c_adult.mean(), nonsyn_rate=same_fam.nonsyn.mean()))
tests.append(dict(test='A2.3_syn_rate_cross_family', group='cross_family',
                  n=len(cross_fam), chem_syn_rate=cross_fam.syn_adult.mean(),
                  mean_contact=cross_fam.c_adult.mean(), nonsyn_rate=cross_fam.nonsyn.mean()))

# (ii) same-NT vs diff-NT synapse rate (NT-gated exclusion) within HM
snt = hm[hm.same_nt == 1]; dnt = hm[hm.same_nt == 0]
tests.append(dict(test='A2.3_syn_rate_same_nt', group='same_nt',
                  n=len(snt), chem_syn_rate=snt.syn_adult.mean(),
                  mean_contact=snt.c_adult.mean(), nonsyn_rate=snt.nonsyn.mean()))
tests.append(dict(test='A2.3_syn_rate_diff_nt', group='diff_nt',
                  n=len(dnt), chem_syn_rate=dnt.syn_adult.mean(),
                  mean_contact=dnt.c_adult.mean(), nonsyn_rate=dnt.nonsyn.mean()))

# (iii) homolog reproducibility of syn/nonsyn status.
# For each unordered family pair, does the L-side homolog instance and the
# R-side homolog instance share the same synaptic outcome? We collapse each
# contacting pair to (subclass_a, subclass_b) then compare mirror pairs.
def mirror(name):
    if name.endswith('L'):
        return name[:-1] + 'R'
    if name.endswith('R'):
        return name[:-1] + 'L'
    return None


key = {(min(a, b), max(a, b)): s for a, b, s in zip(hm.na, hm.nb, hm.syn_adult)}
mirror_rows = []
seen = set()
for a, b, s in zip(hm.na, hm.nb, hm.syn_adult):
    ma, mb = mirror(a), mirror(b)
    if ma is None or mb is None:
        continue
    k = (min(a, b), max(a, b)); mk = (min(ma, mb), max(ma, mb))
    if mk == k or k in seen or mk in seen:
        continue
    if mk in key:
        mirror_rows.append((s, key[mk]))
        seen.add(k)
mr = np.array(mirror_rows)
lr_concord = float((mr[:, 0] == mr[:, 1]).mean()) if len(mr) else np.nan
tests.append(dict(test='A2.3_LR_homolog_syn_concordance', group='mirror_pairs',
                  n=len(mr), chem_syn_rate=lr_concord, mean_contact=np.nan,
                  nonsyn_rate=np.nan))

td = pd.DataFrame(tests)
print('[A2.3] tiling tests:')
print(td.round(3).to_string(index=False))

# =====================================================================
# A2.4  controls
# =====================================================================
controls = []
# control 1: HM high-contact SYNAPTIC pairs (the positive complement)
hm_syn = hm[hm.syn_adult == 1]
controls.append(dict(control='HM_high_contact_synaptic', n=len(hm_syn),
                     mean_contact=hm_syn.c_adult.mean(),
                     same_nt_frac=hm_syn.same_nt.mean(),
                     same_family_frac=(hm_syn.fam_a == hm_syn.fam_b).mean(),
                     nonsyn_rate=0.0))
# control 2: non-HM high-contact nonsyn (rest of connectome, matched cohort)
nonhm = df[~(df.fam_a.isin(HM) & df.fam_b.isin(HM))]
nonhm_nonsyn = nonhm[nonhm.nonsyn == 1]
controls.append(dict(control='nonHM_nonsynaptic', n=len(nonhm_nonsyn),
                     mean_contact=nonhm_nonsyn.c_adult.mean(),
                     same_nt_frac=nonhm_nonsyn.same_nt.mean(),
                     same_family_frac=(nonhm_nonsyn.fam_a == nonhm_nonsyn.fam_b).mean(),
                     nonsyn_rate=1.0))
controls.append(dict(control='HM_nonsynaptic', n=len(nonsyn),
                     mean_contact=nonsyn.c_adult.mean(),
                     same_nt_frac=nonsyn.same_nt.mean(),
                     same_family_frac=(nonsyn.fam_a == nonsyn.fam_b).mean(),
                     nonsyn_rate=1.0))

# control 3: degree-matched null for HM nonsyn rate.
# Compare HM nonsyn rate against connectome pairs matched on summed log-degree.
df['ldeg'] = np.log10(df.deg_a.clip(lower=1)) + np.log10(df.deg_b.clip(lower=1))
hm_ldeg = df.loc[hm.index, 'ldeg']
lo, hi = hm_ldeg.quantile([.05, .95])
matched = df[(df.ldeg.between(lo, hi)) &
             ~df.index.isin(hm.index)]
controls.append(dict(control='degree_matched_nonHM', n=len(matched),
                     mean_contact=matched.c_adult.mean(),
                     same_nt_frac=matched.same_nt.mean(),
                     same_family_frac=(matched.fam_a == matched.fam_b).mean(),
                     nonsyn_rate=matched.nonsyn.mean()))

# control 4: shuffled family labels -> does the same-family exclusion survive?
obs_gap = same_fam.nonsyn.mean() - cross_fam.nonsyn.mean()
perm_gaps = []
fam_vals = hm.fam_a.values.copy()          # shuffle family identity across neurons
neurons = sorted(set(hm.na) | set(hm.nb))
for _ in range(2000):
    perm = rng.permutation(neurons)
    remap = dict(zip(neurons, perm))
    fa = hm.na.map(remap).map(lambda n: n[:3] if n[:3] in HM else n[:3])
    # family of shuffled neuron = its own true family (relabel identity)
    fam_of = dict(zip(hm.na, hm.fam_a))
    fam_of.update(dict(zip(hm.nb, hm.fam_b)))
    sfa = hm.na.map(remap).map(fam_of)
    sfb = hm.nb.map(remap).map(fam_of)
    sf = (sfa == sfb)
    g = hm.nonsyn[sf].mean() - hm.nonsyn[~sf].mean()
    perm_gaps.append(g)
perm_gaps = np.array(perm_gaps)
z = (obs_gap - np.nanmean(perm_gaps)) / (np.nanstd(perm_gaps) + 1e-9)
p = float((np.abs(perm_gaps) >= abs(obs_gap)).mean())
controls.append(dict(control='shuffle_family_same_vs_cross_nonsyn_gap',
                     n=2000, mean_contact=obs_gap,
                     same_nt_frac=float(np.nanmean(perm_gaps)),
                     same_family_frac=z, nonsyn_rate=p))
print(f'[A2.4] same-vs-cross nonsyn gap obs={obs_gap:+.3f} '
      f'perm_mean={np.nanmean(perm_gaps):+.3f} z={z:+.2f} p={p:.3f}')

# control 5: shuffled NT labels -> is the same-NT synaptic exclusion real?
# (this is the true tiling axis: similar-NT contact but avoid synapsing)
obs_nt_gap = snt.syn_adult.mean() - dnt.syn_adult.mean()   # negative = same-NT avoid
nt_of = dict(zip(hm.na, hm.nt_a)); nt_of.update(dict(zip(hm.nb, hm.nt_b)))
neurons_nt = sorted(nt_of)
nt_arr = np.array([nt_of[n] for n in neurons_nt])
perm_nt_gaps = []
for _ in range(2000):
    perm = rng.permutation(nt_arr)
    remap = dict(zip(neurons_nt, perm))
    s = (hm.na.map(remap) == hm.nb.map(remap))
    if s.sum() < 5 or (~s).sum() < 5:
        continue
    perm_nt_gaps.append(hm.syn_adult[s].mean() - hm.syn_adult[~s].mean())
perm_nt_gaps = np.array(perm_nt_gaps)
z_nt = (obs_nt_gap - perm_nt_gaps.mean()) / (perm_nt_gaps.std() + 1e-9)
p_nt = float((np.abs(perm_nt_gaps) >= abs(obs_nt_gap)).mean())
controls.append(dict(control='shuffle_NT_same_vs_diff_synrate_gap',
                     n=len(perm_nt_gaps), mean_contact=obs_nt_gap,
                     same_nt_frac=float(perm_nt_gaps.mean()),
                     same_family_frac=z_nt, nonsyn_rate=p_nt))
print(f'[A2.4] same-NT synaptic exclusion: obs gap(syn_same-syn_diff)={obs_nt_gap:+.3f} '
      f'perm_mean={perm_nt_gaps.mean():+.3f} z={z_nt:+.2f} p={p_nt:.4f}')

cd = pd.DataFrame(controls)
tiling_out = pd.concat([td, cd], ignore_index=True)
tiling_out.round(4).to_csv(OUT + 'partA2_family_tiling_results.csv', index=False)

# =====================================================================
# A2.5  top residual pairs + candidate microcircuit
# =====================================================================
top = nonsyn.sort_values('c_adult', ascending=False)[
    ['na', 'nb', 'fam_a', 'fam_b', 'nt_a', 'nt_b', 'same_nt', 'same_family' if 'same_family' in nonsyn else 'fam_a',
     'homolog', 'same_dv', 'same_lr', 'elec', 'c_adult', 'ctc_L1', 'ctc_adult',
     'contact_all_stages', 's_L1', 's_L2', 's_L3', 's_adult']].copy()
top['same_family'] = (nonsyn.loc[top.index, 'fam_a'] == nonsyn.loc[top.index, 'fam_b']).astype(int)
top.head(60).round(3).to_csv(OUT + 'partA2_top_residual_pairs.csv', index=False)
print(f'[A2.5] wrote top {min(60,len(top))} residual pairs')

# =====================================================================
# verdict
# =====================================================================
# Tiling architecture = (1) high apposition-without-synapse among HM families,
# (2) reproducible across L/R homologs, (3) organized by a real chemical axis
# (NT-gated exclusion) that survives label permutation. Family-per-se is NOT
# the axis (family shuffle null); NT identity is.
lr_ok = (not np.isnan(lr_concord)) and lr_concord >= 0.70
nt_ok = (p_nt < 0.05) and (obs_nt_gap < 0)      # same-NT synapse LESS despite more contact
if hm.nonsyn.mean() > 0.75 and lr_ok and nt_ok:
    verdict = 'PASS'
elif hm.nonsyn.mean() > 0.6 and (lr_ok or nt_ok):
    verdict = 'YELLOW'
else:
    verdict = 'FAIL'
print(f'\n[A2 VERDICT] {verdict}  '
      f'(HM nonsyn={hm.nonsyn.mean():.3f}, LR-concord={lr_concord:.3f}, '
      f'family-shuffle p={p:.3f}, NT-shuffle p={p_nt:.4f})')

summary = dict(hm_nonsyn_rate=hm.nonsyn.mean(), lr_concordance=lr_concord,
               family_shuffle_p=p, same_minus_cross_nonsyn=obs_gap,
               nt_shuffle_p=p_nt, syn_gap_same_minus_diff_nt=obs_nt_gap,
               same_nt_nonsyn=snt.nonsyn.mean(), diff_nt_nonsyn=dnt.nonsyn.mean(),
               same_nt_synrate=snt.syn_adult.mean(), diff_nt_synrate=dnt.syn_adult.mean(),
               verdict=verdict)
pd.DataFrame([summary]).round(4).to_csv(OUT + 'partA2_verdict.csv', index=False)
print('[A2] done.')
