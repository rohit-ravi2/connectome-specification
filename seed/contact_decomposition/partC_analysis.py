"""Part C: is 'contact' local topology/neighborhood rather than pairwise area?"""
import pandas as pd, numpy as np, networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from itertools import combinations
import os, json
np.random.seed(0)

OUT='data/processed/contact_decomposition'
DOC='docs/contact_decomposition'
os.makedirs(DOC,exist_ok=True)

df=pd.read_csv(f'{OUT}/contact_decomposition_master.csv')
neurons=sorted(set(df.na)|set(df.nb))
print('pairs',len(df),'neurons',len(neurons))

# ---- build undirected contact graphs per era ----
def build_graph(col):
    G=nx.Graph(); G.add_nodes_from(neurons)
    sub=df[df[col]>0]
    for a,b,w in zip(sub.na,sub.nb,sub[col]):
        G.add_edge(a,b,weight=float(w))
    return G
G_L1=build_graph('c_L1'); G_ad=build_graph('c_adult')
print('L1 edges',G_L1.number_of_edges(),'adult edges',G_ad.number_of_edges())

# precompute per-node stats
def node_stats(G):
    deg=dict(G.degree())
    core=nx.core_number(G)
    clust=nx.clustering(G)
    tri=nx.triangles(G)
    adj={n:set(G.neighbors(n)) for n in G.nodes()}
    return deg,core,clust,tri,adj
L1s=node_stats(G_L1); ADs=node_stats(G_ad)

def topo_features(a,b,G,stats):
    deg,core,clust,tri,adj=stats
    Na,Nb=adj[a],adj[b]
    cn=Na&Nb
    ncn=len(cn)
    union=len(Na|Nb)
    jac=ncn/union if union else 0.0
    aa=sum(1.0/np.log(deg[x]) for x in cn if deg[x]>1)
    ra=sum(1.0/deg[x] for x in cn if deg[x]>0)
    da,dbn=deg[a],deg[b]
    # local density: density of subgraph induced by closed neighborhood union
    nodes=list((Na|Nb|{a,b}))
    sub=G.subgraph(nodes)
    m=sub.number_of_edges(); n=len(nodes)
    dens=2*m/(n*(n-1)) if n>1 else 0.0
    kcore_min=min(core[a],core[b]); kcore_max=max(core[a],core[b])
    return dict(cn=ncn, jac=jac, aa=aa, ra=ra, tri_a=tri[a], tri_b=tri[b],
                deg_src=da, deg_tgt=dbn, deg_prod=da*dbn, deg_min=min(da,dbn),
                clust_a=clust[a], clust_b=clust[b], local_dens=dens,
                kcore_min=kcore_min, kcore_max=kcore_max, twohop=ncn)

# ---- assemble feature table ----
rows=[]
for r in df.itertuples(index=False):
    a,b=r.na,r.nb
    d=dict(na=a,nb=b,
        T_first_L1=r.T_first_L1, T_preadult_persist=r.T_preadult_persist,
        adult_strong_core1=r.adult_strong_core1,
        # area features
        c_L1=r.c_L1, c_adult=r.c_adult, logc_L1=r.logc_L1, logc_adult=r.logc_adult,
        frac_src=r.frac_src, frac_tgt=r.frac_tgt, frac_src_L1=r.frac_src_L1)
    for x,v in topo_features(a,b,G_L1,L1s).items(): d['L1_'+x]=v
    for x,v in topo_features(a,b,G_ad,ADs).items(): d['AD_'+x]=v
    rows.append(d)
F=pd.DataFrame(rows)
print('feature table',F.shape)

# ---- neuron-disjoint CV AUC ----
K=5
rng=np.random.RandomState(42)
fold_of={n:rng.randint(K) for n in neurons}
F['fa']=F.na.map(fold_of); F['fb']=F.nb.map(fold_of)

AREA_L1=['logc_L1','c_L1','frac_src_L1']
AREA_AD=['logc_adult','c_adult','frac_src','frac_tgt']
TOPO_L1=['L1_cn','L1_jac','L1_aa','L1_ra','L1_tri_a','L1_tri_b','L1_deg_src','L1_deg_tgt','L1_deg_prod','L1_deg_min','L1_clust_a','L1_clust_b','L1_local_dens','L1_kcore_min','L1_kcore_max']
TOPO_AD=['AD_cn','AD_jac','AD_aa','AD_ra','AD_tri_a','AD_tri_b','AD_deg_src','AD_deg_tgt','AD_deg_prod','AD_deg_min','AD_clust_a','AD_clust_b','AD_local_dens','AD_kcore_min','AD_kcore_max']

def disjoint_cv_auc(feat_cols, target):
    """neuron-disjoint: test=both endpoints in fold k, train=both endpoints NOT in k."""
    oof_y=[]; oof_p=[]
    for k in range(K):
        te = F[(F.fa==k)&(F.fb==k)]
        tr = F[(F.fa!=k)&(F.fb!=k)]
        if te[target].nunique()<2 or tr[target].nunique()<2 or len(te)<5: continue
        Xtr=tr[feat_cols].values; Xte=te[feat_cols].values
        ytr=tr[target].values; yte=te[target].values
        sc=StandardScaler().fit(Xtr)
        clf=LogisticRegression(max_iter=2000,C=1.0,class_weight='balanced')
        clf.fit(sc.transform(Xtr),ytr)
        p=clf.predict_proba(sc.transform(Xte))[:,1]
        oof_y.append(yte); oof_p.append(p)
    y=np.concatenate(oof_y); p=np.concatenate(oof_p)
    return roc_auc_score(y,p), len(y), int(y.sum())

targets={'T_first_L1':('L1',AREA_L1,TOPO_L1),
         'T_preadult_persist':('L1',AREA_L1,TOPO_L1),
         'adult_strong_core1':('AD',AREA_AD,TOPO_AD)}

c1=[]
for t,(era,area,topo) in targets.items():
    a_auc,n,npos=disjoint_cv_auc(area,t)
    t_auc,_,_=disjoint_cv_auc(topo,t)
    at_auc,_,_=disjoint_cv_auc(area+topo,t)
    c1.append(dict(target=t,era=era,n_test=n,n_pos=npos,
                   auc_area=round(a_auc,4),auc_topo=round(t_auc,4),
                   auc_area_topo=round(at_auc,4),
                   delta_topo_over_area=round(at_auc-a_auc,4)))
    print(t,'area',round(a_auc,4),'topo',round(t_auc,4),'area+topo',round(at_auc,4),'delta',round(at_auc-a_auc,4))
C1=pd.DataFrame(c1)

# ---- topology-residual-beyond-area (residualize topo preds? use nested delta already) ----
# also: bootstrap CI on delta via fold reshuffle
def delta_ci(area,topo,t,nboot=200):
    deltas=[]
    for bi in range(nboot):
        r=np.random.RandomState(bi)
        fo={n:r.randint(K) for n in neurons}
        F['fa']=F.na.map(fo);F['fb']=F.nb.map(fo)
        try:
            a,_,_=disjoint_cv_auc(area,t); at,_,_=disjoint_cv_auc(area+topo,t)
            deltas.append(at-a)
        except Exception: pass
    F['fa']=F.na.map(fold_of);F['fb']=F.nb.map(fold_of)
    return np.percentile(deltas,[2.5,50,97.5])
cis={}
for t,(era,area,topo) in targets.items():
    lo,md,hi=delta_ci(area,topo,t,nboot=120)
    cis[t]=(round(lo,4),round(md,4),round(hi,4))
    print('delta CI',t,cis[t])
C1['delta_ci_lo']=[cis[t][0] for t in C1.target]
C1['delta_ci_med']=[cis[t][1] for t in C1.target]
C1['delta_ci_hi']=[cis[t][2] for t in C1.target]
C1.to_csv(f'{OUT}/partC_contact_topology_results.csv',index=False)
print('wrote C1')

F.drop(columns=['fa','fb']).to_csv(f'{OUT}/partC_features_full.csv',index=False)

# ================= C2: neighborhood-as-niche at fixed area =================
# stratify by adult contact-area quintile; within stratum test CN(shared nbhd) vs early synapse
from scipy.stats import mannwhitneyu, spearmanr
F2=F.copy()
F2['area_bin']=pd.qcut(F2.logc_L1.rank(method='first'),5,labels=False)
c2=[]
for t,gcol in [('T_first_L1','L1_cn'),('T_preadult_persist','L1_cn')]:
    for ab in range(5):
        s=F2[F2.area_bin==ab]
        pos=s[s[t]==1][gcol]; neg=s[s[t]==0][gcol]
        if len(pos)<5 or len(neg)<5: continue
        try: u,pv=mannwhitneyu(pos,neg,alternative='greater')
        except Exception: pv=np.nan
        c2.append(dict(target=t,area_bin=ab,n_pos=len(pos),n_neg=len(neg),
                       cn_pos_med=round(pos.median(),3),cn_neg_med=round(neg.median(),3),
                       mwu_p_greater=pv))
C2=pd.DataFrame(c2)
# partial correlation: CN vs target controlling for area (logistic w/ area then add CN)
def partial_addcn(t,area,cn):
    from sklearn.linear_model import LogisticRegression as LR
    X0=StandardScaler().fit_transform(F[area].values)
    X1=StandardScaler().fit_transform(F[area+[cn]].values)
    y=F[t].values
    a0=roc_auc_score(y,LR(max_iter=1000,class_weight='balanced').fit(X0,y).predict_proba(X0)[:,1])
    a1=roc_auc_score(y,LR(max_iter=1000,class_weight='balanced').fit(X1,y).predict_proba(X1)[:,1])
    return round(a0,4),round(a1,4),round(a1-a0,4)
c2b=[]
for t in ['T_first_L1','T_preadult_persist']:
    a0,a1,d=partial_addcn(t,AREA_L1,'L1_cn')
    c2b.append(dict(target=t,insample_auc_area=a0,insample_auc_area_cn=a1,delta=d))
C2b=pd.DataFrame(c2b)
C2.to_csv(f'{OUT}/partC_niche_stratified.csv',index=False)
print('C2 done'); print(C2); print(C2b)

# ================= C3: competition per source neuron =================
# use adult contact area as budget; synapse=adult_strong_core1 partners (directed na->nb)
c3=[]
# For each source na, look at its contact partners (nb) with c_adult>0 and check whether
# synapse partners are high contact-rank.
rank_records=[]
for src,grp in df[df.c_adult>0].groupby('na'):
    areas=grp.c_adult.values.astype(float)
    parts=grp.nb.values
    tot=areas.sum()
    if tot<=0 or len(grp)<3: continue
    p=areas/tot
    ent=-np.sum(p*np.log(p+1e-12))
    ent_norm=ent/np.log(len(p))
    gini=1-np.sum(p**2)  # simpson diversity; also compute gini
    srt=np.sort(p)[::-1]; gcoef=(np.sum((2*np.arange(1,len(p)+1)-len(p)-1)*np.sort(p)))/(len(p)*np.sum(p))
    # rank of contact area (1=largest)
    order=np.argsort(-areas); ranks=np.empty(len(areas),int); ranks[order]=np.arange(1,len(areas)+1)
    for nb,ar,rk,pp in zip(parts,areas,ranks,p):
        row=df[(df.na==src)&(df.nb==nb)]
        syn=int(row.adult_strong_core1.values[0]) if len(row) else 0
        rank_records.append(dict(src=src,tgt=nb,c_adult=ar,rank=rk,rank_frac=rk/len(areas),
                                 area_frac=pp,syn=syn,nparts=len(areas)))
    nsyn=sum(1 for nb in parts if int(df[(df.na==src)&(df.nb==nb)].adult_strong_core1.values[0])==1)
    c3.append(dict(src=src,n_contacts=len(p),budget=tot,entropy=ent,entropy_norm=ent_norm,
                   simpson=gini,gini=gcoef,n_syn_partners=nsyn,
                   syn_frac=nsyn/len(p)))
C3=pd.DataFrame(c3)
RR=pd.DataFrame(rank_records)
# do synapse partners have higher contact rank (lower rank number)?
syn_rank=RR[RR.syn==1].rank_frac; nsyn_rank=RR[RR.syn==0].rank_frac
u,pv=mannwhitneyu(syn_rank,nsyn_rank,alternative='less')
print('C3 rank test: syn partners rank_frac med',round(syn_rank.median(),3),
      'vs nonsyn',round(nsyn_rank.median(),3),'MWU p(syn<)',pv)
# is synapse formation competition: within source, is syn prob ~ area rank? logistic per-pair with area_frac
from scipy.stats import spearmanr as sp
sr=sp(RR.area_frac,RR.syn)
print('spearman area_frac vs syn', round(sr.statistic,3), 'p',sr.pvalue)
# top-heavy sources form more synapses?
sc_ent=sp(C3.entropy_norm,C3.syn_frac)
print('spearman entropy_norm vs syn_frac (per source)',round(sc_ent.statistic,3),'p',sc_ent.pvalue)
C3.to_csv(f'{OUT}/partC_competition_results.csv',index=False)
RR.to_csv(f'{OUT}/partC_competition_pairranks.csv',index=False)
c3summary=dict(syn_rankfrac_med=float(syn_rank.median()),nonsyn_rankfrac_med=float(nsyn_rank.median()),
               mwu_p_synlower=float(pv),spearman_areafrac_syn=float(sr.statistic),spearman_areafrac_syn_p=float(sr.pvalue),
               spearman_entropy_synfrac=float(sc_ent.statistic))

# ================= C4: residual graph =================
# regress adult synapse strength residual after area; does topology explain?
# use in-sample logistic of adult_strong_core1 on area -> residual = y - phat
from sklearn.linear_model import LogisticRegression as LR
Xa=StandardScaler().fit_transform(F[AREA_AD].values)
y=F['adult_strong_core1'].values
phat=LR(max_iter=1000,class_weight='balanced').fit(Xa,y).predict_proba(Xa)[:,1]
resid=y-phat
F4=F[['na','nb','adult_strong_core1']].copy()
F4['area_pred']=phat; F4['resid']=resid
F4['resid_sign']=np.where(resid>0.25,'pos',np.where(resid<-0.25,'neg','flat'))
# does topology correlate with residual?
topo_resid=[]
for f in TOPO_AD:
    r=sp(F[f],resid)
    topo_resid.append(dict(feature=f,spearman_vs_resid=round(r.statistic,4),p=r.pvalue))
TR=pd.DataFrame(topo_resid).sort_values('spearman_vs_resid',key=lambda s:s.abs(),ascending=False)
# can topology predict residual sign among contacting-but-uncertain pairs? AUC of topo on residual>0
mask=np.abs(resid)>0.1
auc_topo_resid=roc_auc_score((resid[mask]>0).astype(int),
    LR(max_iter=1000).fit(StandardScaler().fit_transform(F.loc[mask,TOPO_AD].values),(resid[mask]>0).astype(int)).predict_proba(StandardScaler().fit_transform(F.loc[mask,TOPO_AD].values))[:,1])
print('C4 topo->resid-sign in-sample AUC',round(auc_topo_resid,4))
for f in TOPO_AD[:3]: pass
print(TR.head(8).to_string())
for c in [f for f in TOPO_AD]:
    F4[c]=F[c].values
F4.to_csv(f'{OUT}/partC_residual_contact_graph.csv',index=False)

summary=dict(C1=c1,delta_ci=cis,C2b=c2b.__repr__(),C3=c3summary,
             C4_topo_resid_auc=round(auc_topo_resid,4),
             C4_top_topo_resid_corr=TR.head(6).to_dict('records'))
with open(f'{OUT}/partC_summary.json','w') as fh: json.dump(summary,fh,indent=2,default=str)
print('ALL DONE')
print(json.dumps({'C1':c1,'delta_ci':cis,'C3':c3summary,'C4_auc':round(auc_topo_resid,4)},indent=2,default=str))
