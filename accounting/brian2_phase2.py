"""PHASE 2 (full) — conductance-based Brian2 GROW-AND-RUN. The generated network runs with real COBA-LIF spiking
dynamics (E/I from NT identity); spike-train co-activity feeds back on synapse stabilization. Leak-clean: spikes
come ONLY from the generated network; no real connectivity or activity in. Make-or-break: real spiking activity
must break the static ceiling AND beat shuffled-activity, or the dynamical hypothesis fails at biophysical scale.
"""
import warnings,sys,json,csv,re; warnings.simplefilter("ignore")
import numpy as np, openpyxl, pandas as pd
from collections import defaultdict
import brian2 as b2
from brian2 import mV, ms, Hz
b2.prefs.codegen.target='numpy'; b2.BrianLogger.suppress_name('resolution_conflict','method_choice')
def P(*a): print(*a); sys.stdout.flush()
RNG=np.random.default_rng(0); D="../data/celegans/data"
pj=json.load(open("../data/network_positions.json"))
pos={c["cell"]:np.array([c["x"],c["y"],c["z"]],float) for c in pj["cells"] if None not in (c.get("x"),c.get("y"),c.get("z"))}
cg=pd.read_csv(D+"/expression/cengen/thresholded/021821_liberal_threshold1.csv"); classes=list(cg.columns[3:]); cls_idx={c:k for k,c in enumerate(classes)}
def n2c(n):
    if n in cls_idx: return cls_idx[n]
    b=n
    for s in "LR":
        if b.endswith(s): b=b[:-1]; break
    if b in cls_idx: return cls_idx[b]
    for s in "DV":
        if b.endswith(s) and b[:-1] in cls_idx: return cls_idx[b[:-1]]
    return cls_idx.get(b,None)
wb0=openpyxl.load_workbook(f"{D}/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx",read_only=True,data_only=True)
ws=wb0['adult nerve ring neighbors']; dd=list(ws.iter_rows(values_only=True)); hdr=[h for h in dd[0][1:]]
ca=defaultdict(float)
for r in dd[1:]:
    rn=r[0]
    if rn is None: continue
    for jj,v in enumerate(r[1:]):
        cn=hdr[jj]
        if cn and v not in (None,'') and rn!=cn:
            try: ca[(min(rn,cn),max(rn,cn))]=max(ca[(min(rn,cn),max(rn,cn))],float(v))
            except: pass
ring=[n for n in hdr if n in pos and n2c(n) is not None]; M=len(ring); idx={n:i for i,n in enumerate(ring)}
pairs=[(ring[i],ring[j]) for i in range(M) for j in range(i+1,M)]; Np=len(pairs)
pi=np.array([idx[a] for a,b in pairs]); pj_=np.array([idx[b] for a,b in pairs])
area=np.array([ca.get((min(a,b),max(a,b)),0.0) for a,b in pairs]); geo=np.log1p(area); Eb=area>0
syn=defaultdict(int)
with open(f"{D}/connectome/witvliet2020/witvliet_combined_edges.csv") as f:
    for r in csv.DictReader(f):
        if r['type']!='chemical': continue
        m=re.search(r'witvliet_2020_(\d)',r['source_file'])
        if m and int(m.group(1)) in (7,8): syn[(min(r['pre'],r['post']),max(r['pre'],r['post']))]+=1
yW=np.array([1 if syn.get((min(a,b),max(a,b)),0)>=2 else 0 for a,b in pairs]); K=int(yW.sum())
wb=openpyxl.load_workbook(f"{D}/connectome/cook2019/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx",read_only=True,data_only=True)
A=list(wb['hermaphrodite chemical'].iter_rows(values_only=True)); nset=set(ring)
def nm(seq): return sum(1 for x in seq if isinstance(x,str) and x.strip() in nset)
hrow=int(np.argmax([nm(r) for r in A])); cols=[(str(x).strip() if x is not None else None) for x in A[hrow]]
hcol=int(np.argmax([nm([A[r][c] for r in range(len(A))]) for c in range(min(6,len(A[0])))]))
GC=np.zeros((M,M))
for r in range(hrow+1,len(A)):
    rn=A[r][hcol]; rn=rn.strip() if isinstance(rn,str) else None
    if rn not in idx: continue
    for c in range(hcol+1,len(A[r])):
        cn=cols[c]; v=A[r][c]
        if cn in idx and v not in (None,''):
            try:
                if float(v)>0: GC[idx[rn],idx[cn]]=1
            except: pass
yCk=np.array([1 if (GC[pi[k],pj_[k]] or GC[pj_[k],pi[k]]) else 0 for k in range(Np)])
# NT identity for E/I
t5=np.load("../data/tier5/connectome.npz",allow_pickle=True)
ntm={str(n):str(v) for n,v in zip(t5["names"],t5["nt_primary"])}
def bc(n):
    for s in "LRDV":
        if n.endswith(s): return n[:-1]
    return n
inh=np.array([str(ntm.get(n,ntm.get(bc(n),"?"))).startswith("GABA") for n in ring])
P(f"ring {M} | eligible {Eb.sum()} | Witvliet syn {K} | Cook syn {yCk.sum()} | inhibitory(GABA) neurons {inh.sum()}")

def f1(sel,y):
    tp=(sel&(y>0)).sum(); return 2*tp/max(1,sel.sum()+y.sum())

# ---- Brian2 COBA-LIF simulation of the GENERATED network -> spike co-activity ----
Vt,Vr,El,Ee,Ei=-50*mV,-60*mV,-60*mV,0*mV,-80*mV
taum,taue,taui=20*ms,5*ms,10*ms
eqs='''dv/dt=(El-v+ge*(Ee-v)+gi*(Ei-v))/taum : volt (unless refractory)
dge/dt=-ge/taue : 1
dge2/dt=-ge2/taue : 1
dgi/dt=-gi/taui : 1'''
def coactivity(selvec, shuffle=False):
    b2.start_scope()
    G=b2.NeuronGroup(M,eqs.replace('ge*(Ee-v)','(ge+ge2)*(Ee-v)'),threshold='v>Vt',reset='v=Vr',refractory=5*ms,method='euler')
    G.v='Vr+rand()*(Vt-Vr)'
    edges=np.where(selvec>0)[0]; src=[];tgt=[];exc=[]
    for e in edges:
        a,b=pi[e],pj_[e]
        for (s,t) in [(a,b),(b,a)]:
            src.append(s); tgt.append(t); exc.append(not inh[s])
    src=np.array(src);tgt=np.array(tgt);exc=np.array(exc)
    Se=b2.Synapses(G,G,on_pre='ge+=0.14');            # tuned for a realistic ~5-15 Hz regime
    if exc.sum()>0: Se.connect(i=src[exc],j=tgt[exc])
    Si=b2.Synapses(G,G,on_pre='gi+=1.2')
    if (~exc).sum()>0: Si.connect(i=src[~exc],j=tgt[~exc])
    Pin=b2.PoissonInput(G,'ge2',1,8*Hz,weight=0.9)    # spontaneous drive (network's own emergent activity)
    mon=b2.SpikeMonitor(G)
    b2.run(600*ms)
    # binned spike counts -> correlation
    tr=mon.spike_trains(); nb=60; B=np.zeros((M,nb))
    for n in range(M):
        st=np.array(tr[n]/ms)
        if len(st): B[n]=np.histogram(st,bins=nb,range=(0,600))[0]
    live=B.sum(1)>0
    C=np.zeros((M,M))
    if live.sum()>2:
        Cc=np.corrcoef(B[live]); C[np.ix_(np.where(live)[0],np.where(live)[0])]=np.nan_to_num(Cc)
    if shuffle:
        p=RNG.permutation(M); C=C[np.ix_(p,p)]
    return C, float(len(mon.t))/M/0.6   # C, mean firing rate Hz

def grow_and_run(beta, mode, T=3):
    prop=geo*Eb; sel=np.zeros(Np,bool); sel[np.argsort(-prop)[:K]]=True; rate=0
    for t in range(T):
        if beta>0:
            C,rate=coactivity(sel.astype(float), shuffle=(mode=='shuffle'))
            co=np.array([C[pi[k],pj_[k]] for k in range(Np)])
            prop=geo*Eb+beta*co*Eb
        newsel=np.zeros(Np,bool); newsel[np.argsort(-prop)[:K]]=True; sel=newsel
    return sel,rate

static,_=grow_and_run(0,'static')
P("\n"+"="*80); P("PHASE 2 (Brian2 conductance-based grow-and-run)"); P("="*80)
_,rt=coactivity(static.astype(float)); P(f"  [sanity: generated-network mean firing rate ~{rt:.1f} Hz]")
P(f"  STATIC ceiling: F1(Witvliet)={f1(static,yW):.3f}  F1(Cook)={f1(static,yCk):.3f}")
for beta in [1.0,2.0]:
    real=[grow_and_run(beta,'real') for _ in range(2)]; sh=[grow_and_run(beta,'shuffle') for _ in range(2)]
    fR=np.mean([f1(s,yW) for s,_ in real]); fRc=np.mean([f1(s,yCk) for s,_ in real]); fS=np.mean([f1(s,yW) for s,_ in sh])
    P(f"  beta={beta}: REAL-spiking F1(W)={fR:.3f} F1(Cook)={fRc:.3f} | SHUFFLED F1(W)={fS:.3f} | real-shuffle Δ={fR-fS:+.3f}")
    if beta==1.0: fR1,fS1,st1=fR,fS,f1(static,yW)
P("\n"+"="*80)
brk=fR1>st1+0.01; beats=fR1>fS1+0.01
if brk and beats: P("OUTCOME: real spiking activity BREAKS ceiling AND beats shuffled -> DYNAMICAL residual (biophysical).")
elif brk: P("OUTCOME: ceiling-break but shuffled matches -> capacity not activity. Dynamical hypothesis NOT supported.")
else: P("OUTCOME: real spiking activity does NOT break the static ceiling -> dynamical hypothesis fails at BIOPHYSICAL scale too.")
P("="*80)
