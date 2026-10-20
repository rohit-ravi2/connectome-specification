"""
Recompute edge-class importance robustly from B2_graph_ablation_results.csv.
z-scores blow up when the matched-null variance ~ 0 (e.g. sm_reachability: almost
no random ablation disconnects any sensory->motor pair). Empirical p-values are the
valid summary, so rank by a signed -log10(p) evidence score and report raw effect sizes.
"""
import numpy as np, pandas as pd
OUT = "../seed/data/processed/mechanistic_perturbation"
res = pd.read_csv(f"{OUT}/B2_graph_ablation_results.csv")

LOWER_DISRUPT = {'global_efficiency','largest_scc','sm_reachability','communicability','n_reachable_pairs'}
KEY = ['sm_reachability','global_efficiency','communicability','largest_scc']
N_NULL = 30
LOGP_MAX = -np.log10(1/(N_NULL+1))   # ~1.505, cap for a single metric

# clip z in the raw results for readability (keep sign)
res['z_vs_null_clipped'] = res['z_vs_null'].clip(-50, 50)
res.to_csv(f"{OUT}/B2_graph_ablation_results.csv", index=False)

def signed_logp(row):
    lp = min(-np.log10(max(row.emp_p_more_disruptive, 1e-9)), LOGP_MAX)
    return lp if row.more_disruptive_than_null else -lp

imp = []
for cls, sub in res.groupby('edge_class'):
    contact = sub[sub.null_scheme == 'contact']
    n_edges = int(sub.n_edges.iloc[0])
    frac = n_edges / 1469
    km = contact[contact.metric.isin(KEY)].copy()
    km['slp'] = km.apply(signed_logp, axis=1)
    evidence = km['slp'].sum()                      # signed evidence, bounded [-6,6]
    n_sig = int(((km.emp_p_more_disruptive <= 0.05) & km.more_disruptive_than_null).sum())
    # raw effect sizes
    def cd(metric):
        r = contact[contact.metric == metric]
        return float(r.class_delta.iloc[0]) if len(r) else np.nan
    def base(metric):
        r = contact[contact.metric == metric]
        return float(r.baseline.iloc[0]) if len(r) else np.nan
    sm_drop = -cd('sm_reachability')                # positive = fraction of s->m pairs lost
    geff_pct = -cd('global_efficiency')/base('global_efficiency')*100
    scc_drop = -cd('largest_scc')
    # robustness: does it also beat the degree-matched null on sm_reach & geff?
    deg = sub[sub.null_scheme == 'degree']
    def robust(metric):
        r = deg[deg.metric == metric]
        return bool(r.more_disruptive_than_null.iloc[0] and r.emp_p_more_disruptive.iloc[0] <= 0.05) if len(r) else False
    imp.append(dict(
        edge_class=cls, n_edges=n_edges, frac_of_core=round(frac,3),
        evidence_signed_logp=round(evidence,3),
        n_key_metrics_sig_disruptive=n_sig,
        sm_reach_frac_lost=round(sm_drop,4),
        global_eff_pct_drop=round(geff_pct,2),
        largest_scc_nodes_lost=round(scc_drop,1),
        sm_reach_p_contact=float(contact[contact.metric=='sm_reachability'].emp_p_more_disruptive.iloc[0]),
        geff_p_contact=float(contact[contact.metric=='global_efficiency'].emp_p_more_disruptive.iloc[0]),
        robust_vs_degree_null=(robust('sm_reachability') or robust('global_efficiency')),
    ))
impdf = pd.DataFrame(imp).sort_values(
    ['evidence_signed_logp','global_eff_pct_drop'], ascending=False)
# per-edge load: evidence normalized by class size (efficiency of disruption)
impdf['evidence_per_100edges'] = (impdf.evidence_signed_logp / impdf.n_edges * 100).round(3)
impdf.to_csv(f"{OUT}/B2_edgeclass_importance.csv", index=False)
pd.set_option('display.width', 200, 'display.max_columns', 30)
print(impdf.to_string(index=False))
