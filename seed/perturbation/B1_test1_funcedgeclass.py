#!/usr/bin/env python3
"""B1 Test 1: does the developmental structural scaffold predict FUNCTIONAL coupling
(Randi funconn) BEYOND degree + contact?

Functional connectivity is measured response coupling, NOT anatomy. y = 1{funconn!=0}.
Structural classes come from the Witvliet developmental synapse trajectory + adult
nerve-ring contact area. Decisive control = degree-matched + contact-matched null.
"""
import os, math, sys
import numpy as np, pandas as pd
from itertools import product
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
import networkx as nx
sys.path.insert(0, os.path.dirname(__file__))
from B1_common import (load_funconn, funconn_matrix, load_contact, load_edge_table, ROOT)

OUT = f"{ROOT}/data/processed/mechanistic_perturbation"
rng_global = np.random.default_rng(0)

def build_pair_table(fc_kind, measured_only=True):
    fc = load_funconn(fc_kind)
    fneur, fidx, M = funconn_matrix(fc)
    cneur_all, cix, A = load_contact()
    cneur = [n for n in cneur_all if n]
    edge = load_edge_table()
    # measurement coverage: only ~half the neurons were stimulated/recorded.
    # Restrict src to stimulated (>=1 nonzero out) and dst to recorded (>=1 nonzero in)
    # so that y=fc_nonzero reflects biology, not whether the neuron was in the experiment.
    out_nz = {n:int((M[fidx[n]]!=0).sum()) for n in fneur}
    in_nz  = {n:int((M[:,fidx[n]]!=0).sum()) for n in fneur}
    stim = set(n for n in fneur if out_nz[n]>0)
    rec  = set(n for n in fneur if in_nz[n]>0)
    # neurons usable = in funconn AND in contact matrix
    use = [n for n in fneur if n in cix]
    uset = set(use)
    # adult synaptic directed graph (strong_core1 edges) for graph distance
    G = nx.DiGraph()
    G.add_nodes_from(use)
    syn = {}  # (pre,post) -> edge row dict
    for _,r in edge.iterrows():
        p,q = r['pre'], r['post']
        syn[(p,q)] = r
        if p in uset and q in uset and r['strong_core1']==1:
            G.add_edge(p,q)
    # distances (cap at 4)
    dist = {}
    for s in use:
        lengths = nx.single_source_shortest_path_length(G, s, cutoff=4)
        dist[s] = lengths
    # degrees
    # synaptic out/in degree (strong_core1)
    out_syn = {n:0 for n in use}; in_syn = {n:0 for n in use}
    for (p,q),r in syn.items():
        if p in uset and q in uset and r['strong_core1']==1:
            out_syn[p]+=1; in_syn[q]+=1
    # contact degree
    Abool = (A>0).astype(int)
    cdeg = {n:int(Abool[cix[n]].sum()) for n in use}
    recs = []
    for i in use:
        if measured_only and i not in stim: continue
        for j in use:
            if i==j: continue
            if measured_only and j not in rec: continue
            v = M[fidx[i], fidx[j]]
            ca = A[cix[i], cix[j]]
            r = syn.get((i,j))
            has_syn = r is not None
            d = dist.get(i,{}).get(j, 99)
            recs.append(dict(
                pre=i, post=j, fc=v, fc_nonzero=int(v!=0), fc_abs=abs(v),
                fc_pos=int(v>0), fc_neg=int(v<0),
                contact=int(ca>0), contact_area=math.log1p(ca),
                gdist=d if d<99 else 5, connected=int(d==1),
                out_syn=out_syn[i], in_syn=in_syn[j],
                cdeg_pre=cdeg[i], cdeg_post=cdeg[j],
                has_syn=int(has_syn),
                strong_core1=int(r['strong_core1']) if has_syn else 0,
                strong_core2=int(r['strong_core2']) if has_syn else 0,
                strong_core4=int(r['strong_core4']) if has_syn else 0,
                variable=int(r['variable']) if has_syn else 0,
                n_preadult=int(r['n_preadult_groups']) if has_syn else 0,
                preadult_persist=int((r['n_preadult_groups']>=2)) if has_syn else 0,
                pre_nt=(r['pre_nt'] if has_syn else 'none'),
                post_nt=(r['post_nt'] if has_syn else 'none'),
                classification=(r['classification'] if has_syn else 'none'),
            ))
    df = pd.DataFrame(recs)
    return df, use

def ndcv_auc(df, feat_cols, target='fc_nonzero', nseeds=5):
    """Neuron-disjoint 5-fold CV AUC, averaged over seeds. Returns (mean, std)."""
    y = df[target].values.astype(int)
    X = df[feat_cols].values.astype(float)
    pairs = list(zip(df.pre, df.post))
    neurons = sorted(set(df.pre)|set(df.post))
    aucs=[]
    for seed in range(nseeds):
        rng = np.random.default_rng(100+seed)
        folds = np.array_split(rng.permutation(neurons), 5)
        nf = {n:k for k,ff in enumerate(folds) for n in ff}
        pred = np.full(len(y), np.nan)
        for k in range(5):
            te = np.array([nf[a]==k and nf[b]==k for a,b in pairs])
            tr = np.array([nf[a]!=k and nf[b]!=k for a,b in pairs])
            if te.sum()<20 or y[tr].sum()<10 or y[te].sum()<5 or len(set(y[te]))<2: continue
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
            clf.fit(X[tr], y[tr]); pred[te]=clf.predict_proba(X[te])[:,1]
        m = ~np.isnan(pred)
        if m.sum()>30 and len(set(y[m]))>1:
            aucs.append(roc_auc_score(y[m], pred[m]))
    return (np.mean(aucs), np.std(aucs), len(aucs)) if aucs else (np.nan,np.nan,0)

def degree_matched_null(df, struct_col, contacting_only=True, nperm=2000):
    """Among (contacting) pairs, compare funconn-nonzero rate: struct-positive vs
    degree-matched struct-negative controls. Match on binned (out_syn,in_syn,cdeg_pre,cdeg_post)
    and contact_area bin. Returns dict with observed diff, null mean/CI, p, z."""
    d = df.copy()
    if contacting_only:
        d = d[d.contact==1].copy()
    # bins
    def qbin(x, nb=4):
        try: return pd.qcut(x, nb, labels=False, duplicates='drop')
        except Exception: return pd.cut(x, nb, labels=False)
    d['b_os']=qbin(d.out_syn); d['b_is']=qbin(d.in_syn)
    d['b_cp']=qbin(d.cdeg_pre); d['b_cq']=qbin(d.cdeg_post); d['b_ca']=qbin(d.contact_area)
    d['cell']=list(zip(d.b_os,d.b_is,d.b_cp,d.b_cq,d.b_ca))
    pos = d[d[struct_col]==1]
    neg = d[d[struct_col]==0]
    if len(pos)<15: return dict(struct=struct_col, n_pos=len(pos), status='too_few')
    obs_pos_rate = pos.fc_nonzero.mean()
    # build matched control pool: for each cell, list of neg rows
    from collections import defaultdict
    neg_by_cell = defaultdict(list)
    for _,r in neg.iterrows(): neg_by_cell[r['cell']].append(r['fc_nonzero'])
    # matched-control expected rate: for each pos row draw a random neg from same cell
    rng = np.random.default_rng(7)
    usable = pos[pos['cell'].isin(neg_by_cell.keys())]
    if len(usable)<15: return dict(struct=struct_col, n_pos=len(pos), status='no_matched_cells')
    obs_pos_rate_m = usable.fc_nonzero.mean()
    null=[]
    cells = usable['cell'].values
    for _ in range(nperm):
        draw = [neg_by_cell[c][rng.integers(len(neg_by_cell[c]))] for c in cells]
        null.append(np.mean(draw))
    null=np.array(null)
    diff = obs_pos_rate_m - null.mean()
    p = (1+np.sum(null>=obs_pos_rate_m))/(1+len(null))
    z = (obs_pos_rate_m - null.mean())/(null.std()+1e-12)
    return dict(struct=struct_col, contacting_only=contacting_only,
                n_pos=int(len(pos)), n_pos_matched=int(len(usable)),
                pos_rate=round(float(obs_pos_rate),4),
                pos_rate_matched=round(float(obs_pos_rate_m),4),
                null_rate_mean=round(float(null.mean()),4),
                null_ci_lo=round(float(np.percentile(null,2.5)),4),
                null_ci_hi=round(float(np.percentile(null,97.5)),4),
                diff=round(float(diff),4), z=round(float(z),3), p=round(float(p),4),
                status='ok')

def run(fc_kind):
    print(f"\n########## Test 1 on funconn={fc_kind} ##########")
    df, use = build_pair_table(fc_kind)
    print(f"usable neurons={len(use)}  pairs={len(df)}  funconn-nonzero={df.fc_nonzero.mean():.4f}")
    print(f"contacting pairs={df.contact.sum()}  synapse pairs={df.has_syn.sum()}  "
          f"strong_core1={df.strong_core1.sum()}")

    results = []
    # ---- A. Regression AUC: does structural scaffold add beyond degree+contact+distance? ----
    base_cols = ['contact','contact_area','out_syn','in_syn','cdeg_pre','cdeg_post']
    dist_cols = base_cols + ['gdist','connected']
    struct_cols = dist_cols + ['strong_core1','variable','preadult_persist','n_preadult']
    for label, cols in [('degree_only', ['out_syn','in_syn','cdeg_pre','cdeg_post']),
                        ('degree+contact', base_cols),
                        ('degree+contact+dist', dist_cols),
                        ('+structural_scaffold', struct_cols)]:
        m,s,n = ndcv_auc(df, cols, 'fc_nonzero')
        print(f"  [AUC fc_nonzero] {label:28s} = {m:.3f} ± {s:.3f}  (n_seeds={n})")
        results.append(dict(fc=fc_kind, test='regression_auc', model=label,
                            target='fc_nonzero', auc=round(m,4), auc_std=round(s,4)))
    # delta
    base_m = ndcv_auc(df, dist_cols, 'fc_nonzero')[0]
    full_m = ndcv_auc(df, struct_cols, 'fc_nonzero')[0]
    print(f"  >> dAUC structural over degree+contact+dist = {full_m-base_m:+.4f}")
    results.append(dict(fc=fc_kind, test='regression_auc', model='dAUC_struct_over_baseline',
                        target='fc_nonzero', auc=round(full_m-base_m,4), auc_std=np.nan))

    # ---- B. Degree+contact-matched null for each structural class (DECISIVE) ----
    for sc in ['strong_core1','strong_core2','strong_core4','variable','preadult_persist','has_syn']:
        r = degree_matched_null(df, sc, contacting_only=True)
        if r.get('status')=='ok':
            print(f"  [matched-null | contacting] {sc:16s} pos={r['n_pos_matched']:4d} "
                  f"rate={r['pos_rate_matched']:.3f} null={r['null_rate_mean']:.3f} "
                  f"diff={r['diff']:+.3f} z={r['z']:+.2f} p={r['p']:.4f}")
        else:
            print(f"  [matched-null] {sc}: {r.get('status')}")
        r['fc']=fc_kind; r['test']='degree_matched_null'; results.append(r)

    # ---- C. Raw functional-nonzero rate + mean|fc| by structural class (descriptive) ----
    d = df[df.contact==1]
    for name, mask in [
        ('strong_core1', d.strong_core1==1),
        ('variable_halo', (d.variable==1)),
        ('synapse_any', d.has_syn==1),
        ('contact_only_nosyn', d.has_syn==0),
    ]:
        sub = d[mask]
        if len(sub)==0: continue
        print(f"  [descriptive|contacting] {name:20s} n={len(sub):5d} "
              f"fc_nonzero={sub.fc_nonzero.mean():.3f} mean|fc|={sub.fc_abs.mean():.3f} "
              f"frac_neg={sub.fc_neg.mean():.3f}")
        results.append(dict(fc=fc_kind, test='descriptive_by_class', model=name,
                            n=len(sub), fc_nonzero_rate=round(float(sub.fc_nonzero.mean()),4),
                            mean_abs_fc=round(float(sub.fc_abs.mean()),4),
                            frac_neg=round(float(sub.fc_neg.mean()),4)))
    return results, df

if __name__ == "__main__":
    allres=[]
    for kind in ['leifer','wt']:
        res, df = run(kind)
        allres += res
    out = pd.DataFrame(allres)
    path = f"{OUT}/B1_functional_edgeclass_results.csv"
    out.to_csv(path, index=False)
    print(f"\nwrote {path}  ({len(out)} rows)")
