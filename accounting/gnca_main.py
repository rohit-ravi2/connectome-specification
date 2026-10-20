import sys, time, numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
dev='cuda' if torch.cuda.is_available() else 'cpu'
torch.set_float32_matmul_precision('high')

# ---------- data ----------
d=np.load(SC+"/deepfeats.npz",allow_pickle=True)
I,J=d['I'].astype(int),d['J'].astype(int); neurons=list(d['neurons'])
y7,y8=d['y7'],d['y8']; yrep=((y7==1)&(y8==1)).astype(int)
sub=np.load(SC+"/gnca_substrate.npz")
A=sub['A']; L1,L2,L3=sub['L1'],sub['L2'],sub['L3']
M=A.shape[0]; N=len(I)

# neuron-holdout folds (EXACT copy of scoring convention)
endpts=sorted(set([neurons[k] for k in list(I)+list(J)]))
nf={n:i%5 for i,n in enumerate(endpts)}
NI=np.array([nf[neurons[int(I[k])]] for k in range(N)])
NJ=np.array([nf[neurons[int(J[k])]] for k in range(N)])

# node init features: contact-degree(count), log sum-area, eigenvector centrality
Ab=(A>0).astype(float)
cdeg=Ab.sum(1)
sarea=np.log1p(A.sum(1))
ev,V=np.linalg.eigh(A); evc=np.abs(V[:,-1])   # principal eigenvector centrality
def z(x): return (x-x.mean())/(x.std()+1e-9)
Xnode=np.stack([z(cdeg),z(sarea),z(evc),np.ones(M)],1).astype(np.float32)  # F=4
F=Xnode.shape[1]

# row-normalized contact-area propagation matrix
rs=A.sum(1,keepdims=True); Wprop=(A/np.maximum(rs,1e-9)).astype(np.float32)

# edge feature per directed pair: log1p contact area
eij=np.log1p(A[I,J]).astype(np.float32)

# tensors
Xnode_t=torch.tensor(Xnode,device=dev)
Wprop_t=torch.tensor(Wprop,device=dev)
It=torch.tensor(I,device=dev); Jt=torch.tensor(J,device=dev)
eij_t=torch.tensor(eij,device=dev).unsqueeze(1)
yrep_t=torch.tensor(yrep.astype(np.float32),device=dev)
L1_t=torch.tensor(L1.astype(np.float32),device=dev)
L2_t=torch.tensor(L2.astype(np.float32),device=dev)
L3_t=torch.tensor(L3.astype(np.float32),device=dev)

# ---------- model ----------
class GNCA(nn.Module):
    def __init__(self,H=24,T=8,freeze_update=False):
        super().__init__()
        self.T=T; self.H=H
        self.enc=nn.Linear(F,H)
        # shared local update rule
        self.msg=nn.Linear(H,H)
        self.u1=nn.Linear(2*H,H); self.u2=nn.Linear(H,H)
        self.ln=nn.LayerNorm(H)
        # readout (directed): [h_i;h_j;e_ij]
        self.r1=nn.Linear(2*H+1,H); self.r2=nn.Linear(H,1)
        self.freeze_update=freeze_update
        if freeze_update:
            for p in list(self.msg.parameters())+list(self.u1.parameters())+list(self.u2.parameters()):
                p.requires_grad_(False)
    def step(self,h):
        m=torch.relu(self.msg(h))
        agg=Wprop_t@m
        delta=self.u2(torch.relu(self.u1(torch.cat([h,agg],1))))
        return self.ln(h+delta)
    def states(self):
        h=self.enc(Xnode_t)
        hs=[h]
        for _ in range(self.T):
            h=self.step(h); hs.append(h)
        return hs
    def readout(self,h):
        z=torch.cat([h[It],h[Jt],eij_t],1)
        return self.r2(torch.relu(self.r1(z))).squeeze(1)  # logits

def train_fold(f,H=24,T=8,traj=True,epochs=600,lr=5e-3,seed=0,freeze_update=False):
    torch.manual_seed(seed); np.random.seed(seed)
    tr=torch.tensor((NI!=f)&(NJ!=f),device=dev)
    te=(NI==f)&(NJ==f)
    m=GNCA(H,T,freeze_update).to(dev)
    opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=lr,weight_decay=1e-5)
    bce=nn.BCEWithLogitsLoss()
    # aux stage steps: map L1/L2/L3 to intermediate iterations
    aux_map={}
    if traj and T>=4:
        aux_map={ max(1,T//4):L1_t, max(2,T//2):L2_t, max(3,(3*T)//4):L3_t }
    for ep in range(epochs):
        m.train(); opt.zero_grad()
        hs=m.states()
        logit=m.readout(hs[-1])
        loss=bce(logit[tr],yrep_t[tr])
        if aux_map:
            for tt,tgt in aux_map.items():
                la=m.readout(hs[tt])
                loss=loss+0.3*bce(la[tr],tgt[tr])
        loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        p=torch.sigmoid(m.readout(m.states()[-1])).cpu().numpy()
    return p,te,m

def run_config(name,H=24,T=8,traj=True,epochs=600,seed=0,freeze_update=False,nseed=1):
    preds=np.full(N,np.nan); cov=np.zeros(N,bool)
    for f in range(5):
        acc=np.zeros(N)
        for s in range(nseed):
            p,te,_=train_fold(f,H,T,traj,epochs,seed=seed+s,freeze_update=freeze_update)
            acc+=p
        acc/=nseed
        preds[te]=acc[te]; cov[te]=True
    auc=roc_auc_score(yrep[cov],preds[cov])
    P(f"[{name}] T={T} traj={traj} freeze={freeze_update} nseed={nseed} -> neuron-holdout AUC={auc:.4f} cov={cov.sum()}")
    np.savez(f"{SC}/gnca_pred_{name}.npz",preds=preds,cov=cov,auc=auc)
    return preds,cov,auc

if __name__=="__main__":
    t0=time.time()
    P(f"device={dev} N={N} M={M} F={F}")
    cfgs=[
        ("T8_traj",   dict(T=8,traj=True)),
        ("T0",        dict(T=0,traj=False)),
        ("T1",        dict(T=1,traj=False)),
        ("T4_traj",   dict(T=4,traj=True)),
        ("T8_notraj", dict(T=8,traj=False)),
        ("T12_traj",  dict(T=12,traj=True)),
    ]
    for name,kw in cfgs:
        run_config(name,**kw); P(f"  elapsed {time.time()-t0:.0f}s")
    # random-rule ablation (3 seeds averaged)
    run_config("T8_randrule",T=8,traj=True,freeze_update=True,nseed=3)
    P(f"  elapsed {time.time()-t0:.0f}s")
    P("DONE")
