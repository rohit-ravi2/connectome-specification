import sys,time,numpy as np,torch,torch.nn as nn
from sklearn.metrics import roc_auc_score
def P(*a):print(*a);sys.stdout.flush()
SC="../work"
dev='cuda'
d=np.load(SC+"/deepfeats.npz",allow_pickle=True)
I,J=d['I'].astype(int),d['J'].astype(int);neurons=list(d['neurons'])
y7,y8=d['y7'],d['y8'];yrep=((y7==1)&(y8==1)).astype(int);N=len(I)
sub=np.load(SC+"/gnca_substrate.npz");A=sub['A'];L1,L2,L3=sub['L1'],sub['L2'],sub['L3']
M=A.shape[0]
endpts=sorted(set([neurons[k] for k in list(I)+list(J)]));nf={n:i%5 for i,n in enumerate(endpts)}
NI=np.array([nf[neurons[int(I[k])]] for k in range(N)]);NJ=np.array([nf[neurons[int(J[k])]] for k in range(N)])
Ab=(A>0).astype(float);cdeg=Ab.sum(1);sarea=np.log1p(A.sum(1))
ev,V=np.linalg.eigh(A);evc=np.abs(V[:,-1])
def z(x):return (x-x.mean())/(x.std()+1e-9)
Xnode=np.stack([z(cdeg),z(sarea),z(evc),np.ones(M)],1).astype(np.float32);F=Xnode.shape[1]
rs=A.sum(1,keepdims=True);Wprop=(A/np.maximum(rs,1e-9)).astype(np.float32)
eij=np.log1p(A[I,J]).astype(np.float32)
Xn=torch.tensor(Xnode,device=dev);Wp=torch.tensor(Wprop,device=dev)
It=torch.tensor(I,device=dev);Jt=torch.tensor(J,device=dev)
et=torch.tensor(eij,device=dev).unsqueeze(1)
yr=torch.tensor(yrep.astype(np.float32),device=dev)
L1t,L2t,L3t=[torch.tensor(x.astype(np.float32),device=dev) for x in (L1,L2,L3)]

class SteelGNCA(nn.Module):
    """persistent-seed re-injection + GRU-gated update + multi-scale readout."""
    def __init__(self,H=24,T=8):
        super().__init__();self.T=T;self.H=H
        self.enc=nn.Linear(F,H)
        self.msg=nn.Linear(H,H)
        self.gru=nn.GRUCell(2*H+H,H)   # input = [agg, seed], hidden=h
        self.ln=nn.LayerNorm(H)
        self.r1=nn.Linear(2*H+1,H);self.r2=nn.Linear(H,1)
    def states(self):
        seed=self.enc(Xn);h=seed;hs=[h]
        for _ in range(self.T):
            m=torch.relu(self.msg(h));agg=Wp@m
            h=self.ln(self.gru(torch.cat([h,agg,seed],1),h));hs.append(h)
        return hs
    def ro(self,h):
        return self.r2(torch.relu(self.r1(torch.cat([h[It],h[Jt],et],1)))).squeeze(1)

def train_fold(f,H=24,T=8,epochs=700,lr=5e-3,seed=0):
    torch.manual_seed(seed)
    tr=torch.tensor((NI!=f)&(NJ!=f),device=dev);te=(NI==f)&(NJ==f)
    m=SteelGNCA(H,T).to(dev)
    opt=torch.optim.Adam(m.parameters(),lr=lr,weight_decay=1e-5);bce=nn.BCEWithLogitsLoss()
    aux={max(1,T//4):L1t,max(1,T//2):L2t,max(1,(3*T)//4):L3t} if T>=4 else {}
    for ep in range(epochs):
        m.train();opt.zero_grad();hs=m.states()
        logit=m.ro(hs[-1]);loss=bce(logit[tr],yr[tr])
        for tt,tg in aux.items():
            loss=loss+0.3*bce(m.ro(hs[tt])[tr],tg[tr])
        loss.backward();opt.step()
    m.eval()
    with torch.no_grad():
        p=torch.sigmoid(m.ro(m.states()[-1])).cpu().numpy()
    return p,te

def run(name,H=24,T=8,epochs=700):
    preds=np.full(N,np.nan);cov=np.zeros(N,bool)
    for f in range(5):
        p,te=train_fold(f,H,T,epochs);preds[te]=p[te];cov[te]=True
    auc=roc_auc_score(yrep[cov],preds[cov])
    P(f"[{name}] steel T={T} -> AUC={auc:.4f}")
    np.savez(f"{SC}/gnca_pred_{name}.npz",preds=preds,cov=cov,auc=auc)

if __name__=="__main__":
    # NOTE r1 in_features must be 3*H+1 but final readout uses 2*H+1 -> fix: make two heads
    P("device",dev)
    for T in (2,4,8):
        run(f"steel_T{T}",T=T)
    P("DONE")
