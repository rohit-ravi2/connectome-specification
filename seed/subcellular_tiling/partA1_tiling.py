#!/usr/bin/env python3
"""Part A1 (CRUX): subcellular tiling/exclusion test using Tier-1 synapse positions.
For high-contact pairs, place the apposition on each neuron's arbor (nearest skeleton point), then ask:
is the apposition domain (a) near the neuron's own synapses (competent/occupied zone) and (b) occupied
by a synapse to a LIKE-IDENTITY (same-NT/same-family) partner? Tiling/self-avoidance predicts high-contact
same-NT NONsynaptic pairs are apposed in like-identity-occupied domains more than controls."""
import json, csv, os, math
import numpy as np
from scipy.spatial import cKDTree
from collections import defaultdict

OUT="data/processed/subcellular_tiling"
skc=json.load(open(f"{OUT}/_skeleton_coords.json"))
presyn=json.load(open(f"{OUT}/_presyn_sites.json"))  # neuron -> [(partner,x,y,z,dist)]
cohort=list(csv.DictReader(open(f"{OUT}/high_contact_residual_cohort.csv")))
# neuron -> KDTree of skeleton coords
trees={}; coords={}
for n,s in skc.items():
    xyz=np.array(s['xyz'],float)
    if len(xyz)>0: trees[n]=cKDTree(xyz); coords[n]=xyz
# neuron -> presyn site array + partner list
psite={}; ppart={}
for n,lst in presyn.items():
    if lst: psite[n]=np.array([[x,y,z] for _,x,y,z,_ in lst],float); ppart[n]=[p for p,_,_,_,_ in lst]
NT={}; FAM={}
for r in cohort:
    NT[r['na']]=r['nt_a']; NT[r['nb']]=r['nt_b']; FAM[r['na']]=r['fam_a']; FAM[r['nb']]=r['fam_b']

R=2500.0   # nm radius for "domain occupancy"
def apposition_point(A,B):
    if A not in trees or B not in coords: return None
    # nearest point on A to any point of B
    dists,idx=trees[A].query(coords[B])
    j=np.argmin(dists); return coords[A][trees[A].query(coords[B][j])[1]]
def analyze(A,B):
    """for ordered (A,B): metrics on A's arbor at the A-B apposition."""
    if A not in trees or B not in coords or A not in psite: return None
    app=apposition_point(A,B)
    if app is None: return None
    ps=psite[A]; d=np.linalg.norm(ps-app,axis=1)
    d_own=float(d.min())                                   # dist to A's nearest own presynapse
    near=d<=R
    n_near=int(near.sum())
    # of A's synapses within R of apposition, how many partners are same-NT as B / same-family as B
    like_nt=sum(1 for k in np.where(near)[0] if NT.get(ppart[A][k])==NT.get(B) and NT.get(B,'Unk')!='Unk')
    like_fam=sum(1 for k in np.where(near)[0] if FAM.get(ppart[A][k])==FAM.get(B))
    return dict(d_own=d_own, n_near=n_near, like_nt_occupied=int(like_nt>0), like_fam_occupied=int(like_fam>0))

# build directed rows from high-contact pairs (top20), both directions
rows=[]
for r in cohort:
    if r['hc_top20']!='1': continue
    A,B=r['na'],r['nb']
    for (s,t) in [(A,B),(B,A)]:
        m=analyze(s,t)
        if m is None: continue
        # is s->t synaptic? use s_adult>0 undirected as proxy for "pair synapses"; direction from synapse table not here
        m.update(pre=s,post=t,nonsyn=int(r['nonsyn']),same_nt=int(NT.get(s)==NT.get(t) and NT.get(t,'Unk')!='Unk'),
                 same_fam=int(FAM.get(s)==FAM.get(t)),both_hm=int(r['both_head_motor']),c_adult=float(r['c_adult']))
        rows.append(m)
print(f"analyzed directed high-contact(top20) arbor-appositions: {len(rows)}")

def summ(sub,label):
    if len(sub)<10: print(f"  {label}: n={len(sub)} (too few)"); return
    print(f"  {label:34s} n={len(sub):4d}  d_own(median)={np.median([r['d_own'] for r in sub]):8.0f}nm  "
          f"like-NT-occupied={100*np.mean([r['like_nt_occupied'] for r in sub]):4.0f}%  "
          f"like-fam-occupied={100*np.mean([r['like_fam_occupied'] for r in sub]):4.0f}%")
print("\n=== apposition-domain analysis (A's arbor at A-B contact) ===")
summ([r for r in rows if r['nonsyn']==1 and r['same_nt']==1],"NONsyn SAME-NT")
summ([r for r in rows if r['nonsyn']==1 and r['same_nt']==0],"NONsyn DIFF-NT")
summ([r for r in rows if r['nonsyn']==0],"SYNAPTIC pairs")
summ([r for r in rows if r['nonsyn']==1 and r['both_hm']==1],"NONsyn head-motor")

# KEY TEST: among NONsynaptic high-contact, are same-NT appositions more like-NT-occupied than diff-NT? (tiling exclusion)
ns=[r for r in rows if r['nonsyn']==1]
snt=[r for r in ns if r['same_nt']==1]; dnt=[r for r in ns if r['same_nt']==0]
from scipy.stats import fisher_exact, mannwhitneyu
if len(snt)>=10 and len(dnt)>=10:
    a=sum(r['like_nt_occupied'] for r in snt); b=len(snt)-a; c=sum(r['like_nt_occupied'] for r in dnt); dd=len(dnt)-c
    OR,p=fisher_exact([[a,b],[c,dd]])
    print(f"\n  TILING TEST: like-NT-occupied domain | same-NT nonsyn {100*a/len(snt):.0f}% vs diff-NT nonsyn {100*c/len(dnt):.0f}%  OR={OR:.2f} p={p:.1e}")
    # d_own: are same-NT nonsyn appositions in COMPETENT (near own synapse) zones?
    u,pu=mannwhitneyu([r['d_own'] for r in snt],[r['d_own'] for r in dnt])
    print(f"  d_own same-NT median {np.median([r['d_own'] for r in snt]):.0f} vs diff-NT {np.median([r['d_own'] for r in dnt]):.0f} nm (MWU p={pu:.2e})")
# permutation null: shuffle NT labels among the neurons, recompute like-NT-occupied enrichment
# (light: shuffle post NT)
rng=np.random.default_rng(0)
real_or=OR if len(snt)>=10 else np.nan
with open(f"{OUT}/partA1_tiling_score_results.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["group","n","d_own_median","like_nt_occupied_pct","like_fam_occupied_pct"])
    for lab,sub in [("nonsyn_same_nt",snt),("nonsyn_diff_nt",dnt),("synaptic",[r for r in rows if r['nonsyn']==0]),("nonsyn_head_motor",[r for r in rows if r['nonsyn']==1 and r['both_hm']==1])]:
        if len(sub)>=5:
            w.writerow([lab,len(sub),round(np.median([r['d_own'] for r in sub]),1),round(100*np.mean([r['like_nt_occupied'] for r in sub]),1),round(100*np.mean([r['like_fam_occupied'] for r in sub]),1)])
print("\nA1 done ->",OUT)
