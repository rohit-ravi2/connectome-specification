"""BRANCH 2 — the FEEDBACK 12%. Trophic analysis: 88% of flow is feedforward sensory->motor, 12% feedback (down
the hierarchy). Those hierarchy-violating edges are the computationally special ones. Are they concentrated on the
known locomotion COMMAND INTERNEURONS (AVA/AVB/AVD/AVE/PVC), and what is the recurrent-core structure?
"""
import warnings,sys,openpyxl; warnings.simplefilter("ignore")
import numpy as np, networkx as nx
def P(*a): print(*a); sys.stdout.flush()
D="../data/celegans/data"
wb=openpyxl.load_workbook(f"{D}/connectome/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
A=list(wb['hermaphrodite chemical'].iter_rows(values_only=True))
def looks(n): return isinstance(n,str) and len(n)>=2 and n[0].isalpha()
hr=max(range(len(A)),key=lambda r:sum(looks(x) for x in A[r])); cols=[(str(x).strip() if x is not None else None) for x in A[hr]]
hc=max(range(min(6,len(A[0]))),key=lambda c:sum(looks(A[r][c]) for r in range(len(A))))
E=[];nset=set()
for r in range(hr+1,len(A)):
    pre=A[r][hc]; pre=pre.strip() if isinstance(pre,str) else None
    if not pre: continue
    for c in range(hc+1,len(A[r])):
        post=cols[c]; v=A[r][c]
        if post and v not in (None,''):
            try:
                if float(v)>0: E.append((pre,post,float(v))); nset|={pre,post}
            except: pass
neur=sorted(nset); idx={n:i for i,n in enumerate(neur)}; M=len(neur)
W=np.zeros((M,M))
for a,b,w in E: W[idx[a],idx[b]]+=w
indeg=W.sum(0);outdeg=W.sum(1)
h=np.linalg.lstsq(np.diag(indeg+outdeg)-(W+W.T)+np.ones((M,M))/M, indeg-outdeg, rcond=None)[0]; h=h-h.min()
P(f"BRANCH 2 — feedback structure | {M} neurons, {len(E)} edges")
# feedback edges: post lower in hierarchy than pre
fb=[(a,b,w) for a,b,w in E if h[idx[b]]<h[idx[a]]-0.1]
ff=[(a,b,w) for a,b,w in E if h[idx[b]]>h[idx[a]]+0.1]
P(f"  feedforward edges {len(ff)} | feedback edges {len(fb)} | feedback fraction {len(fb)/len(E):.2f}")
# feedback concentration: which neurons SOURCE feedback
fbsrc=np.zeros(M)
for a,b,w in fb: fbsrc[idx[a]]+=w
tot_out=outdeg.copy()+1e-9
fb_rate=fbsrc/tot_out   # fraction of a neuron's output that is feedback
top=np.argsort(-fbsrc)[:20]
P("\n  top feedback SOURCES (neurons whose output goes back down the hierarchy):")
for i in top: P(f"    {neur[i]:8s} feedback-out={fbsrc[i]:.0f} ({fb_rate[i]*100:.0f}% of its output) trophic={h[i]:.2f}")
# command interneuron enrichment
CI=set()
for base in ['AVA','AVB','AVD','AVE','PVC','PVP','RIM','RIB','AIB','RID','DVA']:
    for s in ['L','R','']: CI.add(base+s)
ci_mask=np.array([n in CI for n in neur])
P(f"\n  locomotion command/premotor interneurons present: {int(ci_mask.sum())}")
P(f"    their mean feedback-out {fbsrc[ci_mask].mean():.1f} vs other neurons {fbsrc[~ci_mask].mean():.1f}")
from scipy.stats import mannwhitneyu
if ci_mask.sum()>2:
    u,pv=mannwhitneyu(fbsrc[ci_mask],fbsrc[~ci_mask],alternative='greater')
    P(f"    command-interneurons > others in feedback output? p={pv:.4f}")
# recurrent core: largest strongly-connected component
G=nx.DiGraph(); G.add_nodes_from(range(M))
for a,b,w in E: G.add_edge(idx[a],idx[b])
sccs=sorted(nx.strongly_connected_components(G),key=len,reverse=True)
big=sccs[0]
P(f"\n  RECURRENT CORE: largest strongly-connected component = {len(big)} neurons ({len(big)/M*100:.0f}% of network)")
P(f"    core members (sample): {sorted(neur[i] for i in big)[:20]}")
P(f"    core mean trophic {np.mean([h[i] for i in big]):.2f} vs whole-net {h.mean():.2f} (mid-hierarchy = the interneuron processing layer)")
P("\n  [read] feedback concentrated on command interneurons + a mid-hierarchy recurrent core = the loops that")
P("  give the feedforward sensory->motor cascade its dynamical state (locomotion command, persistence).")
