#!/usr/bin/env python3
"""B1 Test 2: WT vs unc-31 functional connectivity.

unc-31/CAPS is required for dense-core-vesicle (neuropeptide/monoamine) release.
The unc-31 funconn matrix is therefore a NEUROMODULATORY functional perturbation,
NOT a structural wiring change (anatomy is unchanged). Question: are the functional
connections that CHANGE most enriched for particular structural scaffold classes,
neurotransmitter identities, or hub/interneuron status? And is the strong-core
anatomical scaffold more functionally PRESERVED than the variable/halo?

Coverage differs sharply between preps (unc-31 has fewer stimulated neurons), so
all comparisons are restricted to pairs MEASURED IN BOTH preps.
"""
import os, sys, math
import numpy as np, pandas as pd
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))
from B1_common import load_funconn, funconn_matrix, load_contact, load_edge_table, load_nt_map, ROOT

OUT = f"{ROOT}/data/processed/mechanistic_perturbation"

def build():
    wt = load_funconn('wt'); un = load_funconn('unc31')
    wn, wix, Mw = funconn_matrix(wt)
    un_, uix, Mu = funconn_matrix(un)
    assert wn == un_, "neuron sets differ"
    neur = wn
    cneur, cix, A = load_contact()
    edge = load_edge_table()
    syn = {(r['pre'],r['post']):r for _,r in edge.iterrows()}
    ntmap = load_nt_map()   # full NT coverage for all neurons
    # measured masks (per prep)
    stim_w = set(n for n in neur if (Mw[wix[n]]!=0).any())
    rec_w  = set(n for n in neur if (Mw[:,wix[n]]!=0).any())
    stim_u = set(n for n in neur if (Mu[uix[n]]!=0).any())
    rec_u  = set(n for n in neur if (Mu[:,uix[n]]!=0).any())
    stim_both = stim_w & stim_u; rec_both = rec_w & rec_u
    print(f"stim WT={len(stim_w)} unc31={len(stim_u)} both={len(stim_both)}")
    print(f"rec  WT={len(rec_w)} unc31={len(rec_u)} both={len(rec_both)}")
    # synaptic + contact degree
    out_syn=defaultdict(int); in_syn=defaultdict(int)
    for (p,q),r in syn.items():
        if r['strong_core1']==1: out_syn[p]+=1; in_syn[q]+=1
    Ab=(A>0).astype(int); cdeg={n:(int(Ab[cix[n]].sum()) if n in cix else 0) for n in neur}
    recs=[]
    for i in neur:
        if i not in stim_both: continue
        for j in neur:
            if i==j or j not in rec_both: continue
            wv=Mw[wix[i],wix[j]]; uv=Mu[uix[i],uix[j]]
            ca=A[cix[i],cix[j]] if (i in cix and j in cix) else 0.0
            r=syn.get((i,j))
            recs.append(dict(pre=i,post=j,wt=wv,unc31=uv,
                wt_nz=int(wv!=0),unc_nz=int(uv!=0),
                delta=uv-wv, abs_delta=abs(uv-wv),
                lost=int(wv!=0 and uv==0), gained=int(wv==0 and uv!=0),
                contact=int(ca>0), contact_area=math.log1p(ca),
                out_syn=out_syn.get(i,0), in_syn=in_syn.get(j,0),
                cdeg_pre=cdeg.get(i,0), cdeg_post=cdeg.get(j,0),
                has_syn=int(r is not None),
                strong_core1=int(r['strong_core1']) if r is not None else 0,
                strong_core2=int(r['strong_core2']) if r is not None else 0,
                strong_core4=int(r['strong_core4']) if r is not None else 0,
                variable=int(r['variable']) if r is not None else 0,
                preadult_persist=int(r['n_preadult_groups']>=2) if r is not None else 0,
                pre_nt=ntmap.get(i,'Unk'), post_nt=ntmap.get(j,'Unk'),
                pre_mono=int(ntmap.get(i,'Unk')=='mono'), post_mono=int(ntmap.get(j,'Unk')=='mono'),
                pre_mod=(r['pre_mod'] if r is not None else 'na'),
                post_mod=(r['post_mod'] if r is not None else 'na')))
    return pd.DataFrame(recs)

def matched_loss_null(d, col, nperm=3000):
    """Among WT-present pairs, is loss-rate of `col`-positive different from
    degree/contact-matched `col`-negative controls?"""
    dd=d[d.wt_nz==1].copy()
    def qb(x,nb=3):
        try: return pd.qcut(x,nb,labels=False,duplicates='drop')
        except Exception: return pd.cut(x,nb,labels=False)
    dd['b_os']=qb(dd.out_syn); dd['b_is']=qb(dd.in_syn)
    dd['b_cp']=qb(dd.cdeg_pre); dd['b_cq']=qb(dd.cdeg_post); dd['b_ca']=qb(dd.contact_area)
    dd['cell']=list(zip(dd.b_os,dd.b_is,dd.b_cp,dd.b_cq,dd.b_ca))
    pos=dd[dd[col]==1]; neg=dd[dd[col]==0]
    if len(pos)<10: return dict(col=col,status='too_few',n_pos=int(len(pos)))
    nbc=defaultdict(list)
    for _,r in neg.iterrows(): nbc[r['cell']].append(r['lost'])
    use=pos[pos['cell'].isin(nbc)]
    if len(use)<10: return dict(col=col,status='no_cells',n_pos=int(len(pos)))
    obs=use.lost.mean(); rng=np.random.default_rng(3)
    cells=use['cell'].values
    null=np.array([np.mean([nbc[c][rng.integers(len(nbc[c]))] for c in cells]) for _ in range(nperm)])
    z=(obs-null.mean())/(null.std()+1e-12)
    p=(1+np.sum(null<=obs))/(1+len(null))  # test for LOWER loss (preservation)
    return dict(col=col,status='ok',n_pos=int(len(use)),
                loss_rate=round(float(obs),4),null_loss=round(float(null.mean()),4),
                null_lo=round(float(np.percentile(null,2.5)),4),
                null_hi=round(float(np.percentile(null,97.5)),4),
                z=round(float(z),3),p_preserved=round(float(p),4))

def run():
    d=build()
    print(f"\nmeasured-in-both pairs={len(d)}")
    print(f"WT-present={d.wt_nz.sum()} unc31-present={d.unc_nz.sum()}  "
          f"overall lost={d.lost.sum()} gained={d.gained.sum()}")
    res=[]
    wtp=d[d.wt_nz==1]
    global_loss=wtp.lost.mean()
    print(f"global loss rate among WT-present = {global_loss:.3f}")
    res.append(dict(test='global',metric='wt_present',value=int(d.wt_nz.sum())))
    res.append(dict(test='global',metric='unc_present',value=int(d.unc_nz.sum())))
    res.append(dict(test='global',metric='global_loss_rate',value=round(float(global_loss),4)))
    res.append(dict(test='global',metric='gained',value=int(d.gained.sum())))

    # ---- loss rate by structural class (raw) ----
    print("\n-- loss rate among WT-present pairs by class (raw) --")
    for name,mask in [('strong_core1',wtp.strong_core1==1),('strong_core2',wtp.strong_core2==1),
                      ('strong_core4',wtp.strong_core4==1),('variable_halo',wtp.variable==1),
                      ('preadult_persist',wtp.preadult_persist==1),
                      ('has_syn',wtp.has_syn==1),('contact_only',(wtp.contact==1)&(wtp.has_syn==0)),
                      ('no_contact',wtp.contact==0)]:
        s=wtp[mask]
        if len(s)==0: continue
        print(f"   {name:18s} n={len(s):4d} loss={s.lost.mean():.3f} mean_absdelta={s.abs_delta.mean():.3f}")
        res.append(dict(test='loss_by_class_raw',metric=name,n=int(len(s)),
                        loss_rate=round(float(s.lost.mean()),4),
                        mean_abs_delta=round(float(s.abs_delta.mean()),4)))

    # ---- degree/contact-matched preservation null (DECISIVE) ----
    print("\n-- degree/contact-matched: is class MORE PRESERVED (lower loss) than matched controls? --")
    for col in ['strong_core1','strong_core2','strong_core4','variable','preadult_persist','has_syn']:
        r=matched_loss_null(d,col)
        if r.get('status')=='ok':
            print(f"   {col:16s} n={r['n_pos']:4d} loss={r['loss_rate']:.3f} "
                  f"null={r['null_loss']:.3f} z={r['z']:+.2f} p_preserved={r['p_preserved']:.4f}")
        else:
            print(f"   {col}: {r.get('status')}")
        r['test']='matched_preservation_null'; res.append(r)

    # ---- NT enrichment: DCV/neuropeptide users (monoaminergic) should change more.
    # This is the BIOLOGICAL POSITIVE CONTROL: unc-31 kills DCV release, so functional
    # edges FROM monoaminergic/peptidergic neurons should be preferentially disrupted. ----
    print("\n-- loss rate by presynaptic NT (mono=DCV/monoamine users; full NT coverage) --")
    for nt in ['ACh','Glu','GABA','mono','Unk']:
        s=wtp[wtp.pre_nt==nt]
        if len(s)<8: continue
        print(f"   pre_nt={nt:5s} n={len(s):4d} loss={s.lost.mean():.3f} mean_absdelta={s.abs_delta.mean():.3f}")
        res.append(dict(test='loss_by_pre_nt',metric=nt,n=int(len(s)),
                        loss_rate=round(float(s.lost.mean()),4),
                        mean_abs_delta=round(float(s.abs_delta.mean()),4)))
    # mono vs non-mono formal test (matched on degree/contact)
    for side in ['pre_mono','post_mono']:
        rmono=matched_loss_null(d,side)
        if rmono.get('status')=='ok':
            rmono['p_more_changed']=round(1-rmono['p_preserved'],4)
            print(f"   >> {side} matched: loss={rmono['loss_rate']:.3f} null={rmono['null_loss']:.3f} "
                  f"z={rmono['z']:+.2f} p_more_lost={rmono['p_more_changed']:.4f}")
        else:
            print(f"   >> {side}: {rmono.get('status')} (n_pos={rmono.get('n_pos')})")
        rmono['test']=f'{side}_matched'; res.append(rmono)

    # ---- hub / modality ----
    print("\n-- change magnitude by presynaptic modality --")
    for mod in ['inter','sensory','motor','command','na']:
        s=wtp[wtp.pre_mod==mod]
        if len(s)<8: continue
        print(f"   pre_mod={mod:8s} n={len(s):4d} loss={s.lost.mean():.3f} mean_absdelta={s.abs_delta.mean():.3f}")
        res.append(dict(test='loss_by_pre_mod',metric=mod,n=int(len(s)),
                        loss_rate=round(float(s.lost.mean()),4),
                        mean_abs_delta=round(float(s.abs_delta.mean()),4)))
    # hub = high total degree endpoints
    d['tot_deg']=d.cdeg_pre+d.cdeg_post
    thr=d['tot_deg'].median()
    for name,mask in [('hub_pairs',wtp.cdeg_pre+wtp.cdeg_post>thr),('nonhub',wtp.cdeg_pre+wtp.cdeg_post<=thr)]:
        s=wtp[mask]
        print(f"   {name:10s} n={len(s):4d} loss={s.lost.mean():.3f}")
        res.append(dict(test='loss_by_hub',metric=name,n=int(len(s)),loss_rate=round(float(s.lost.mean()),4)))

    # ---- top changed pairs ----
    top=d.reindex(d.abs_delta.sort_values(ascending=False).index).head(25)
    top[['pre','post','wt','unc31','delta','strong_core1','variable','pre_nt','post_nt']].to_csv(
        f"{OUT}/B1_unc31_top_changed_pairs.csv",index=False)
    print(f"\nwrote top-changed pairs. top delta example:")
    print(top[['pre','post','wt','unc31','delta','pre_nt']].head(6).to_string(index=False))

    dfres=pd.DataFrame(res)
    dfres.to_csv(f"{OUT}/B1_randi_unc31_results.csv",index=False)
    print(f"\nwrote {OUT}/B1_randi_unc31_results.csv ({len(dfres)} rows)")
    return d, dfres

if __name__=="__main__":
    run()
