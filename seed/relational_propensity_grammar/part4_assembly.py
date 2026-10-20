#!/usr/bin/env python3
"""PART 4 — ASSEMBLY / MOTIF-REUSE COMPRESSIBILITY (highest overclaim risk).

PARK THE METAPHYSICS: assembly-index / motif-reuse is used ONLY as a measurement of
generative-path length (a graph-compressibility statistic), NOT a claim about
life / selection / consciousness. Assembly Theory framing borrowed for its
"reuse shortens description length" intuition, nothing more.

QUESTION: is the adult connectome LOW-ASSEMBLY (few reused wiring motifs applied
repeatedly -> short generative path) or HIGH-ASSEMBLY (each edge separately
specified -> long path)?

THE RAIL (non-negotiable): score assembly index against BOTH
  (a) DEGREE-MATCHED null  (directed degree-preserving edge-swap randomization)
  (b) PROCESS-MATCHED null (directed duplication-divergence growth model, E-matched)
Report Z / percentile vs EACH. "High assembly" without both nulls is FORBIDDEN.

FIREWALL: reported alongside dispersion ratio (0.47 4-replicate) and a held-out
cross-animal reproducibility (Brier-framed) check. AUC is not success.
NON-CIRCULARITY: motif vocabulary is a READOUT of the connectome's own structure.
It is a SPECIFIER of the reproducible propensity ONLY if motif-participation
predicts held-out cross-animal edge reproducibility.
"""
import csv, re, os, math, random, sys, functools
import numpy as np
import networkx as nx
from collections import defaultdict, Counter
print=functools.partial(print,flush=True)  # unbuffered progress

random.seed(0); np.random.seed(0)
C="../data/celegans/data"
EDGES=f"{C}/connectome/witvliet2020/witvliet_combined_edges.csv"
OUT=os.path.expanduser("../seed/data/processed/relational_propensity_grammar")
os.makedirs(OUT,exist_ok=True)

# ---------- load directed CHEMICAL edges per animal ----------
animal_edges=defaultdict(set)
for r in csv.DictReader(open(EDGES)):
    if r['type']=='electrical': continue
    a=int(re.search(r'witvliet_2020_(\d)',r['source_file']).group(1))
    animal_edges[a].add((r['pre'],r['post']))

def build_digraph(edges):
    G=nx.DiGraph(); G.add_edges_from(edges); return G

ADULTS=[7,8]
# reproducibility count across all 8 animals (for READOUT->SPECIFIER test)
all_animals=[1,2,3,4,5,6,7,8]
edge_repro=Counter()
for a in all_animals:
    for e in animal_edges[a]: edge_repro[e]+=1

# ================= ASSEMBLY MEASURES =================
# We compute assembly on the DIRECTED simple graph. Two complementary metrics,
# both computed IDENTICALLY on the real graph and every null replicate.

# 13 connected directed triad codes (networkx triadic_census keys); the 3
# disconnected codes (003,012,102) carry no reuse info and are dropped.
CONNECTED_TRIADS=['021D','021U','021C','111D','111U','030T','030C',
                  '201','120D','120U','120C','210','300']

def triad_profile(G):
    """Return connected-triad census counts (dict) for a DiGraph."""
    tc=nx.triadic_census(G)
    return {t:tc.get(t,0) for t in CONNECTED_TRIADS}

def census_entropy_bits(prof):
    """bits to name the isomorphism-type of a random connected triad.
    LOW entropy = few motif types dominate = HIGH reuse = LOW assembly."""
    tot=sum(prof.values())
    if tot==0: return 0.0
    H=0.0
    for c in prof.values():
        if c>0:
            p=c/tot; H-=p*math.log2(p)
    return H

def mdl_bits_per_edge(G):
    """Motif-cover MDL. Build the edge set either by (a) placing a 3-node motif
    template that covers >=2 of its edges at once (reuse), or (b) writing an edge
    directly. Cost model (bits):
        direct edge          : 2*log2(N)                    (name 2 endpoints)
        motif instance apply : log2(#templates) + 3*log2(N) (which template + 3 anchors)
        dictionary           : fixed 13 templates * ~small  (amortized, negligible)
    Greedy: enumerate connected triad instances, apply the ones that cover the
    most still-uncovered edges and that SAVE bits vs direct. Residual edges pay
    direct cost. LOWER bits/edge = MORE reuse = LOWER assembly.
    Returns (bits_per_edge, edge_motif_participation dict)."""
    nodes=list(G.nodes()); N=max(len(nodes),2)
    logN=math.log2(N); Ecost=2*logN
    tmpl_cost=math.log2(len(CONNECTED_TRIADS))+3*logN
    E=G.number_of_edges()
    if E==0: return 0.0,{}
    # undirected adjacency for triad enumeration
    und=defaultdict(set)
    for u,v in G.edges(): und[u].add(v); und[v].add(u)
    dir_edges=set(G.edges())
    idx={n:i for i,n in enumerate(nodes)}
    # enumerate connected triads (unordered node triples with >=2 undirected links)
    seen=set(); instances=[]
    for u in nodes:
        nb=list(und[u])
        for i in range(len(nb)):
            for j in range(i+1,len(nb)):
                a,b=nb[i],nb[j]
                tri=tuple(sorted((u,a,b),key=lambda x:idx[x]))
                if tri in seen: continue
                seen.add(tri)
                x,y,z=tri
                de=[(p,q) for p,q in [(x,y),(y,x),(x,z),(z,x),(y,z),(z,y)] if (p,q) in dir_edges]
                if len(de)>=2:
                    instances.append((tri,de))
    # greedy cover
    covered=set(); participation=Counter()
    # sort by #edges in the motif (bigger reuse first)
    instances.sort(key=lambda t:-len(t[1]))
    applied=0
    for tri,de in instances:
        new=[e for e in de if e not in covered]
        if len(new)<2: continue
        # apply only if template cost < cost of writing the NEW edges directly
        if tmpl_cost < len(new)*Ecost:
            for e in new: covered.add(e); participation[e]+=1
            for e in de: participation.setdefault(e,participation.get(e,0))
            applied+=1
    residual=E-len(covered)
    DL=applied*tmpl_cost + residual*Ecost
    return DL/E, dict(participation)

# ================= NULLS =================
def degree_null(G, nswap_mult=10):
    """Directed degree-preserving randomization via directed edge swaps.
    Uses an edge SET plus index list; swaps update by position (no .index())."""
    edges=list(G.edges())
    eset=set(edges)
    m=len(edges); target=nswap_mult*m; done=0; tries=0
    lim=target*30
    while done<target and tries<lim:
        tries+=1
        i=random.randrange(m); j=random.randrange(m)
        a,b=edges[i]; c,d=edges[j]
        if len({a,b,c,d})<4: continue
        if (a,d) in eset or (c,b) in eset: continue
        # remove old, add new (preserves out-deg of a,c and in-deg of b,d)
        eset.discard((a,b)); eset.discard((c,d))
        eset.add((a,d)); eset.add((c,b))
        edges[i]=(a,d); edges[j]=(c,b)
        done+=1
    H=nx.DiGraph(); H.add_nodes_from(G.nodes()); H.add_edges_from(eset)
    return H

def dd_growth_null(N_target, E_target, seed_n=4):
    """Directed duplication-divergence growth, E-matched by tuning retention p.
    Each step: add node v, pick random parent u, copy u's out/in edges with
    prob p, attach v<->u with prob 0.5. This creates correlated motif structure
    from a PURE GROWTH PROCESS (no per-edge specification)."""
    def grow(p):
        G=nx.DiGraph(); G.add_nodes_from(range(seed_n))
        for i in range(seed_n):
            for j in range(seed_n):
                if i!=j and random.random()<0.4: G.add_edge(i,j)
        nxt=seed_n
        while G.number_of_nodes()<N_target:
            v=nxt; nxt+=1; G.add_node(v)
            u=random.choice(list(G.nodes()-{v}))
            for w in list(G.successors(u)):
                if random.random()<p: G.add_edge(v,w)
            for x in list(G.predecessors(u)):
                if random.random()<p: G.add_edge(x,v)
            if random.random()<0.5: G.add_edge(v,u)
            else: G.add_edge(u,v)
        return G
    lo,hi=0.0,1.0
    for _ in range(18):
        p=(lo+hi)/2; G=grow(p)
        if G.number_of_edges()<E_target: lo=p
        else: hi=p
    G=grow((lo+hi)/2)
    # E-match top-up/trim with degree-preferential edges (keeps growth character)
    nodes=list(G.nodes())
    while G.number_of_edges()<E_target:
        od=np.array([G.out_degree(n)+1 for n in nodes],float); id_=np.array([G.in_degree(n)+1 for n in nodes],float)
        u=nodes[np.random.choice(len(nodes),p=od/od.sum())]; v=nodes[np.random.choice(len(nodes),p=id_/id_.sum())]
        if u!=v and not G.has_edge(u,v): G.add_edge(u,v)
    while G.number_of_edges()>E_target:
        e=random.choice(list(G.edges())); G.remove_edge(*e)
    return G

# ================= RUN =================
NREP=200            # null replicates for triad-entropy Z (fast)
NREP_MDL=60         # null replicates for MDL Z (slower)
rows_out=[]
verdicts={}

for A in ADULTS:
    G=build_digraph(animal_edges[A])
    N=G.number_of_nodes(); E=G.number_of_edges()
    prof=triad_profile(G)
    H_real=census_entropy_bits(prof)
    mdl_real,part=mdl_bits_per_edge(G)
    Ttot=sum(prof.values())
    # top-3 concentration
    top3=sum(sorted(prof.values(),reverse=True)[:3])/max(Ttot,1)
    print(f"\n===== ADULT dataset {A}: N={N} E={E} triads={Ttot} =====")
    print(f"  census entropy H={H_real:.4f} bits/triad (max {math.log2(13):.3f}); top3 frac={top3:.3f}")
    print(f"  MDL={mdl_real:.4f} bits/edge (direct baseline {2*math.log2(N):.3f})")

    # --- degree null distribution ---
    Hd=[]; MDLd=[]
    for r in range(NREP):
        Hn=degree_null(G)
        Hd.append(census_entropy_bits(triad_profile(Hn)))
        if r<NREP_MDL:
            MDLd.append(mdl_bits_per_edge(Hn)[0])
    Hd=np.array(Hd); MDLd=np.array(MDLd)

    # --- process null distribution ---
    Hp=[]; MDLp=[]; ematch=[]
    for r in range(NREP):
        Gp=dd_growth_null(N,E)
        ematch.append(Gp.number_of_edges())
        Hp.append(census_entropy_bits(triad_profile(Gp)))
        if r<NREP_MDL:
            MDLp.append(mdl_bits_per_edge(Gp)[0])
    Hp=np.array(Hp); MDLp=np.array(MDLp); ematch=np.array(ematch)

    def z(x,dist):
        s=dist.std(); return (x-dist.mean())/s if s>0 else 0.0
    def pct(x,dist): return float((dist<x).mean())

    zH_deg=z(H_real,Hd);   zH_proc=z(H_real,Hp)
    zM_deg=z(mdl_real,MDLd); zM_proc=z(mdl_real,MDLp)
    print(f"  DEGREE null:  H {Hd.mean():.4f}+/-{Hd.std():.4f} -> Z={zH_deg:+.2f} pct={pct(H_real,Hd):.3f} | "
          f"MDL {MDLd.mean():.4f}+/-{MDLd.std():.4f} -> Z={zM_deg:+.2f} pct={pct(mdl_real,MDLd):.3f}")
    print(f"  PROCESS null: H {Hp.mean():.4f}+/-{Hp.std():.4f} -> Z={zH_proc:+.2f} pct={pct(H_real,Hp):.3f} | "
          f"MDL {MDLp.mean():.4f}+/-{MDLp.std():.4f} -> Z={zM_proc:+.2f} pct={pct(mdl_real,MDLp):.3f} "
          f"(E-match mean {ematch.mean():.0f} vs {E})")

    # verdict by MDL (bits/edge): LOWER than null = more compressible = LOW assembly
    def verdict(zdeg,zproc):
        # negative Z = real more compressible (lower bits) than null = LOW assembly
        low = zdeg< -2 and zproc< -2
        high= zdeg>  2 and zproc>  2
        if low: return "LOW-assembly (more reuse than BOTH nulls)"
        if high: return "HIGH-assembly (less reuse than BOTH nulls)"
        return "≈nulls (compressibility explained by degree/growth)"
    v_mdl=verdict(zM_deg,zM_proc)
    verdicts[A]={'zM_deg':zM_deg,'zM_proc':zM_proc,'zH_deg':zH_deg,'zH_proc':zH_proc,
                 'mdl':mdl_real,'H':H_real,'verdict':v_mdl,'part':part,'N':N,'E':E,
                 'MDLd_mean':MDLd.mean(),'MDLp_mean':MDLp.mean(),'Hd_mean':Hd.mean(),'Hp_mean':Hp.mean(),
                 'ematch':ematch.mean()}
    print(f"  VERDICT (MDL): {v_mdl}")

    rows_out.append(dict(adult=A,N=N,E=E,triads=Ttot,H_bits=round(H_real,4),
        H_deg_mean=round(Hd.mean(),4),zH_deg=round(zH_deg,3),
        H_proc_mean=round(Hp.mean(),4),zH_proc=round(zH_proc,3),
        MDL_bits_edge=round(mdl_real,4),MDL_deg_mean=round(MDLd.mean(),4),zMDL_deg=round(zM_deg,3),
        MDL_proc_mean=round(MDLp.mean(),4),zMDL_proc=round(zM_proc,3),
        proc_Ematch=round(ematch.mean(),1),top3_frac=round(top3,4),verdict=v_mdl))

# ================= NON-CIRCULARITY: READOUT vs SPECIFIER =================
# Does motif-participation predict HELD-OUT cross-animal reproducibility?
# Use adult 7's motif participation to predict how many of the 8 animals each
# of 7's edges appears in. If reused-motif edges are MORE reproducible ->
# assembly reuse tracks the reproducible propensity (SPECIFIER-leaning).
print("\n===== NON-CIRCULARITY: motif-participation vs cross-animal reproducibility =====")
spec_rows=[]
for A in ADULTS:
    part=verdicts[A]['part']
    G=build_digraph(animal_edges[A])
    parts=[]; repro=[]
    for e in G.edges():
        parts.append(part.get(e,0))
        repro.append(edge_repro[e])   # in how many of 8 animals this edge appears
    parts=np.array(parts,float); repro=np.array(repro,float)
    # correlation + contrast: reused (part>0) vs non-reused (part==0)
    from scipy.stats import spearmanr, mannwhitneyu
    rho,pv=spearmanr(parts,repro)
    m_reuse=repro[parts>0].mean() if (parts>0).any() else float('nan')
    m_non=repro[parts==0].mean() if (parts==0).any() else float('nan')
    try: _,pmw=mannwhitneyu(repro[parts>0],repro[parts==0],alternative='greater')
    except Exception: pmw=float('nan')
    print(f"  adult {A}: spearman(participation,repro)={rho:+.3f} p={pv:.2e}; "
          f"repro reused={m_reuse:.2f} vs non-reused={m_non:.2f} (n_reused={int((parts>0).sum())}) MWU p={pmw:.2e}")
    spec_rows.append(dict(adult=A,spearman_rho=round(rho,4),spearman_p=pv,
        mean_repro_reused=round(m_reuse,3),mean_repro_nonreused=round(m_non,3),
        n_reused=int((parts>0).sum()),n_nonreused=int((parts==0).sum()),mwu_p_greater=pmw))

# ================= WRITE CSV =================
import csv as _csv
allrows=[]
for r in rows_out:
    d=dict(r); d['block']='assembly_vs_nulls'; allrows.append(d)
for r in spec_rows:
    d=dict(r); d['block']='readout_vs_specifier'; allrows.append(d)
keys=sorted(set().union(*[set(d) for d in allrows]))
with open(f"{OUT}/part4_assembly.csv","w",newline="") as f:
    w=_csv.DictWriter(f,fieldnames=['block']+[k for k in keys if k!='block'])
    w.writeheader()
    for d in allrows: w.writerow(d)
print(f"\nwrote {OUT}/part4_assembly.csv")

# ================= TRIAD SIGNIFICANCE PROFILE (which motifs) =================
print("\n===== TRIAD SIGNIFICANCE (adult 8, real vs degree-null; which motifs over/under-used) =====")
G=build_digraph(animal_edges[8]); prof=triad_profile(G)
null_counts={t:[] for t in CONNECTED_TRIADS}
for r in range(NREP):
    p=triad_profile(degree_null(G))
    for t in CONNECTED_TRIADS: null_counts[t].append(p[t])
tsp=[]
for t in CONNECTED_TRIADS:
    arr=np.array(null_counts[t]); zt=(prof[t]-arr.mean())/arr.std() if arr.std()>0 else 0.0
    tsp.append((t,prof[t],arr.mean(),zt))
for t,obs,nm,zt in sorted(tsp,key=lambda x:-abs(x[3])):
    print(f"   {t:5s} obs={obs:6d} null={nm:8.1f} Z={zt:+7.2f}")
# save TSP
with open(f"{OUT}/part4_triad_significance.csv","w",newline="") as f:
    w=_csv.writer(f); w.writerow(['triad','obs','null_mean','z'])
    for t,obs,nm,zt in tsp: w.writerow([t,obs,round(nm,2),round(zt,3)])
print(f"wrote {OUT}/part4_triad_significance.csv")
print("\nDONE")
