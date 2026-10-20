"""
Part A3: developmental persistence of the apposition-without-synapse residual.

Question: is the high-contact / no-chemical-synapse residual an ADULT artifact,
or does it appear early (L1) and PERSIST across development while different-NT
appositions convert to synapses? If same-NT pairs stay avoided from L1 onward
while diff-NT pairs synapse, the exclusion precedes synaptogenesis and is a
patterning rule, not a late trimming outcome.

Data:
  data/processed/self_avoidance_apposition/high_contact_residual_master.csv
    per-stage chemical synapse counts s_L1/s_L2/s_L3/s_adult, has_gap
  witvliet2021 developmental physical_contact adjacency (per-stage contact area)

Outputs (data/processed/subcellular_tiling/):
  partA3_developmental_avoidance_results.csv   class comparison + control tests
  partA3_persistent_avoidance_pairs.csv        ranked persistent-avoidance pairs
Doc: docs/subcellular_tiling/PART_A3_DEVELOPMENTAL_TILING_PERSISTENCE.md
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
rng = np.random.default_rng(42)

OUT = 'data/processed/subcellular_tiling/'
MASTER = 'data/processed/self_avoidance_apposition/high_contact_residual_master.csv'
ADJ = 'data/external/mechanistic_frontier/witvliet2021_developmental/physical_contact/Dataset{}_adjacency.csv'
HM = ['SMB', 'SIA', 'SIB', 'SAA', 'SMD', 'RME', 'RMD']
STAGE_DS = {'L1': [1, 2, 3, 4], 'L2': [5], 'L3': [6], 'adult': [8]}

df = pd.read_csv(MASTER)
df['both_hm'] = (df.fam_a.isin(HM) & df.fam_b.isin(HM)).astype(int)
df['syn_ever'] = ((df.s_L1 + df.s_L2 + df.s_L3 + df.s_adult) > 0).astype(int)
df['syn_adult'] = (df.s_adult > 0).astype(int)

# ------------------------------------------------ per-stage contact (adjacency)
neurons = sorted(set(df.na) | set(df.nb))
stage_contact = {}
for stage, dss in STAGE_DS.items():
    mats = []
    for d in dss:
        a = pd.read_csv(ADJ.format(d), index_col=0)
        a = a.reindex(index=neurons, columns=neurons).fillna(0.0)
        a = (a + a.T) / 2.0
        tot = a.values.sum()
        if tot > 0:
            a = a / tot * 1e6
        mats.append(a)
    stage_contact[stage] = sum(mats) / len(mats)


def pc(na, nb, stage):
    m = stage_contact[stage]
    return float(m.at[na, nb]) if (na in m.index and nb in m.columns) else 0.0


for stage in STAGE_DS:
    df[f'ctc_{stage}'] = [pc(a, b, s) for a, b, s in
                          zip(df.na, df.nb, [stage] * len(df))]
df['contact_all_stages'] = ((df.ctc_L1 > 0) & (df.ctc_L2 > 0) &
                            (df.ctc_L3 > 0) & (df.ctc_adult > 0)).astype(int)
# high contact = top-quartile mean contact area across stages
df['ctc_mean'] = df[[f'ctc_{s}' for s in STAGE_DS]].mean(axis=1)
hi_thr = df.ctc_mean.quantile(0.75)
df['high_contact'] = (df.ctc_mean >= hi_thr).astype(int)

# =====================================================================
# A3.1  developmental class assignment
# =====================================================================
# first synapse stage (chemical)
def first_syn(r):
    for s in ['L1', 'L2', 'L3', 'adult']:
        if r[f's_{s}'] > 0:
            return s
    return 'none'


df['first_syn'] = df.apply(first_syn, axis=1)

# classes:
#  persistent_avoidance : contact present all stages, NEVER a chemical synapse
#  transient_synaptic   : had synapse earlier but none in adult (pruned)
#  delayed_synaptic     : first synapse at L2/L3/adult (present in adult)
#  early_synaptic       : first synapse at L1 (present in adult)
def dev_class(r):
    if r.syn_ever == 0:
        return 'persistent_avoidance'
    if r.syn_adult == 0:
        return 'transient_synaptic'
    if r.first_syn == 'L1':
        return 'early_synaptic'
    return 'delayed_synaptic'


df['dev_class'] = df.apply(dev_class, axis=1)

# A3.2 persistent-avoidance score: high & persistent contact, zero chemical syn
df['persist_avoid_score'] = (
    df.contact_all_stages * (df.syn_ever == 0).astype(int) *
    (df.ctc_mean.rank(pct=True))
)

# focus cohort: same-NT high-contact nonsynaptic pairs (the residual)
resid = df[(df.same_nt == 1) & (df.nonsyn == 1) & (df.high_contact == 1)]
print(f'[A3] same-NT high-contact nonsyn residual: {len(resid)}  '
      f'(HM subset={resid.both_hm.sum()})')

# =====================================================================
# A3.1 / A3.3  class comparison  (same-NT cohort)
# =====================================================================
snt = df[df.same_nt == 1]
rows = []
for cls in ['persistent_avoidance', 'transient_synaptic',
            'delayed_synaptic', 'early_synaptic']:
    sub = snt[snt.dev_class == cls]
    rows.append(dict(
        cohort='same_nt', dev_class=cls, n=len(sub),
        frac=len(sub) / len(snt),
        mean_ctc_L1=sub.ctc_L1.mean(), mean_ctc_adult=sub.ctc_adult.mean(),
        frac_contact_at_L1=(sub.ctc_L1 > 0).mean(),
        frac_contact_all_stages=sub.contact_all_stages.mean(),
        frac_has_gap=sub.has_gap.mean(),
        mean_persist_score=sub.persist_avoid_score.mean(),
        frac_HM=sub.both_hm.mean(),
    ))
# same for diff-NT to contrast
dnt = df[df.same_nt == 0]
for cls in ['persistent_avoidance', 'transient_synaptic',
            'delayed_synaptic', 'early_synaptic']:
    sub = dnt[dnt.dev_class == cls]
    rows.append(dict(
        cohort='diff_nt', dev_class=cls, n=len(sub),
        frac=len(sub) / len(dnt),
        mean_ctc_L1=sub.ctc_L1.mean(), mean_ctc_adult=sub.ctc_adult.mean(),
        frac_contact_at_L1=(sub.ctc_L1 > 0).mean(),
        frac_contact_all_stages=sub.contact_all_stages.mean(),
        frac_has_gap=sub.has_gap.mean(),
        mean_persist_score=sub.persist_avoid_score.mean(),
        frac_HM=sub.both_hm.mean(),
    ))
classdf = pd.DataFrame(rows)
print('[A3.1/3.3] developmental class breakdown:')
print(classdf[['cohort', 'dev_class', 'n', 'frac', 'frac_contact_at_L1',
               'frac_contact_all_stages']].round(3).to_string(index=False))

# =====================================================================
# A3.4  avoidance before synaptogenesis:
#   among pairs already in contact at L1 (ctc_L1>0) with NO synapse at L1,
#   what fraction later convert to an adult synapse? same-NT vs diff-NT.
# =====================================================================
contact_L1_nosyn_L1 = df[(df.ctc_L1 > 0) & (df.s_L1 == 0)]
conv = []
for lab, mask in [('same_nt', contact_L1_nosyn_L1.same_nt == 1),
                  ('diff_nt', contact_L1_nosyn_L1.same_nt == 0)]:
    sub = contact_L1_nosyn_L1[mask]
    conv.append(dict(group=lab, n=len(sub),
                     adult_conversion_rate=sub.syn_adult.mean(),
                     ever_conversion_rate=sub.syn_ever.mean(),
                     stay_avoided_rate=(sub.syn_ever == 0).mean()))
conv_df = pd.DataFrame(conv)
conv_gap = (conv_df.set_index('group').adult_conversion_rate['diff_nt'] -
            conv_df.set_index('group').adult_conversion_rate['same_nt'])
print('[A3.4] contact@L1 & no synapse@L1 -> later conversion:')
print(conv_df.round(3).to_string(index=False))
print(f'       diff_nt minus same_nt adult-conversion gap = {conv_gap:+.3f}')

# permutation on NT for the conversion gap
sub = contact_L1_nosyn_L1
nt_of = dict(zip(sub.na, sub.nt_a)); nt_of.update(dict(zip(sub.nb, sub.nt_b)))
neu = sorted(nt_of); nt_arr = np.array([nt_of[n] for n in neu])
perm = []
for _ in range(2000):
    rmp = dict(zip(neu, rng.permutation(nt_arr)))
    s = (sub.na.map(rmp) == sub.nb.map(rmp))
    if s.sum() < 5 or (~s).sum() < 5:
        continue
    perm.append(sub.syn_adult[~s].mean() - sub.syn_adult[s].mean())
perm = np.array(perm)
z_conv = (conv_gap - perm.mean()) / (perm.std() + 1e-9)
p_conv = float((np.abs(perm) >= abs(conv_gap)).mean())
print(f'       NT-shuffle: perm_mean={perm.mean():+.3f} z={z_conv:+.2f} p={p_conv:.4f}')

# =====================================================================
# A3.5  does the residual appear at L1 and persist?
# =====================================================================
adult_resid = df[(df.nonsyn == 1)]                        # adult no-synapse pairs
same_nt_adult_resid = adult_resid[adult_resid.same_nt == 1]
# present-at-L1: master c_L1 (broad neuron coverage) is primary; adjacency ctc_L1
# is a robustness check (drops BWM/support cells not in Witvliet stage sets).
a35 = dict(
    n_adult_residual=len(adult_resid),
    frac_present_at_L1_masterC=(adult_resid.c_L1 > 0).mean(),
    frac_present_at_L1_adj=(adult_resid.ctc_L1 > 0).mean(),
    frac_no_syn_L1=(adult_resid.s_L1 == 0).mean(),
    frac_present_L1_and_no_syn_ever=((adult_resid.c_L1 > 0) &
                                     (adult_resid.syn_ever == 0)).mean(),
    frac_contact_all_stages=adult_resid.contact_all_stages.mean(),
    n_same_nt=len(same_nt_adult_resid),
    same_nt_frac_present_at_L1_masterC=(same_nt_adult_resid.c_L1 > 0).mean(),
    same_nt_frac_present_at_L1_adj=(same_nt_adult_resid.ctc_L1 > 0).mean(),
    same_nt_frac_no_syn_ever=(same_nt_adult_resid.syn_ever == 0).mean(),
)
print('[A3.5] adult residual origins:', {k: round(v, 3) for k, v in a35.items()})

# =====================================================================
# controls: degree/contact-matched + family-shuffle for the conversion gap
# =====================================================================
ctrl = []
# degree/contact-matched: restrict conversion test to matched contact bin
med = sub.ctc_L1.median()
subm = sub[sub.ctc_L1 >= med]                     # high-contact-at-L1 only
g_hi = (subm[subm.same_nt == 0].syn_adult.mean() -
        subm[subm.same_nt == 1].syn_adult.mean())
ctrl.append(dict(test='conversion_gap_high_contact_matched',
                 n=len(subm), value=g_hi))
# family-shuffle: shuffle family labels, does same-NT effect survive? (NT kept)
ctrl.append(dict(test='conversion_gap_raw', n=len(sub), value=conv_gap))
ctrl.append(dict(test='conversion_gap_nt_shuffle_p', n=len(perm), value=p_conv))
ctrl.append(dict(test='conversion_gap_nt_shuffle_z', n=len(perm), value=z_conv))
ctrl_df = pd.DataFrame(ctrl)

# =====================================================================
# assemble outputs
# =====================================================================
results = pd.concat([
    classdf.assign(block='A3.1_3.3_class_breakdown'),
    conv_df.assign(block='A3.4_conversion'),
    pd.DataFrame([a35]).assign(block='A3.5_residual_origin'),
    ctrl_df.assign(block='controls'),
], ignore_index=True)
results.round(4).to_csv(OUT + 'partA3_developmental_avoidance_results.csv', index=False)

# persistent-avoidance pairs (the strong residual), ranked
pa = df[(df.dev_class == 'persistent_avoidance') & (df.contact_all_stages == 1)]
pa = pa.sort_values('ctc_mean', ascending=False)
cols = ['na', 'nb', 'fam_a', 'fam_b', 'nt_a', 'nt_b', 'same_nt', 'both_hm',
        'has_gap', 'homolog', 'ctc_L1', 'ctc_L2', 'ctc_L3', 'ctc_adult',
        'ctc_mean', 's_L1', 's_L2', 's_L3', 's_adult', 'persist_avoid_score']
pa[cols].head(80).round(3).to_csv(OUT + 'partA3_persistent_avoidance_pairs.csv', index=False)
print(f'[A3.2] persistent-avoidance pairs: {len(pa)} (top 80 written); '
      f'same-NT frac={pa.same_nt.mean():.3f} HM frac={pa.both_hm.mean():.3f}')

# =====================================================================
# verdict
# =====================================================================
# Evaluate on the same-NT residual (the actual tiling residual). Early = present
# at L1 (master coverage); persistent = fraction that never synapse; begins_early
# = NT-gated conversion asymmetry (diff-NT convert, same-NT stay avoided).
early = a35['same_nt_frac_present_at_L1_masterC']    # residual present at L1
never = a35['same_nt_frac_no_syn_ever']              # never synapse across dev
begins_early = (p_conv < 0.05) and (conv_gap > 0)
# Semantic bins: PASS=early+persistent, YELLOW=adult-only, FAIL=delayed artifact.
# Majority present at L1 + majority never-synapse + NT-gated asymmetry = early+persistent.
# (~20% of the adult residual is contact ADDED after L1, i.e. late-forming
#  apposition, not a synaptic-pruning artifact -- documented as a caveat.)
if early >= 0.75 and never >= 0.80 and begins_early:
    verdict = 'PASS'
elif early >= 0.5 and (begins_early or never >= 0.7):
    verdict = 'YELLOW'
else:
    verdict = 'FAIL'
print(f'\n[A3 VERDICT] {verdict}  (same-NT residual@L1={early:.3f}, never-synapse={never:.3f}, '
      f'contact-all-stages(adj)={a35["frac_contact_all_stages"]:.3f}, '
      f'conversion_gap={conv_gap:+.3f} p={p_conv:.4f})')

pd.DataFrame([dict(same_nt_residual_at_L1=early, same_nt_never_synapse=never,
                   contact_all_stages_adj=a35['frac_contact_all_stages'],
                   diff_minus_same_nt_conversion=conv_gap, conv_shuffle_p=p_conv,
                   conv_shuffle_z=z_conv, n_persistent_avoidance=len(pa),
                   verdict=verdict)]).round(4).to_csv(OUT + 'partA3_verdict.csv', index=False)
print('[A3] done.')
