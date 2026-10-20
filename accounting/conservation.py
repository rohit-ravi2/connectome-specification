import sys, numpy as np
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True)
I,J,y7,y8,neurons=z['I'],z['J'],z['y7'],z['y8'],list(z['neurons']); M=len(neurons); N=len(I)
var=(y7!=y8)  # variable/coin-flip edges (present in exactly one adult)
core=((y7==1)&(y8==1)); P(f"core edges (both)={core.sum()} | variable edges (one animal)={var.sum()}")
def degs(present):
    od=np.zeros(M); idg=np.zeros(M)
    for k in np.where(present)[0]: od[I[k]]+=1; idg[J[k]]+=1
    return od,idg
o7,i7=degs(y7==1); o8,i8=degs(y8==1)
# real per-neuron budget imbalance across the two animals
S_real=(np.abs(o7-o8)+np.abs(i7-i8)).sum()
P(f"per-neuron OUT-degree corr(animal7,animal8)={np.corrcoef(o7,o8)[0,1]:.3f} | IN corr={np.corrcoef(i7,i8)[0,1]:.3f}")
# NULL: reassign each VARIABLE edge to animal 7 or 8 at random (preserve core; preserve #var per animal)
vidx=np.where(var)[0]; v7=int((y7[var]==1).sum())
rng=np.random.default_rng(0); Snull=[]
oc,ic=degs(core)  # core contributes equally to both animals
for s in range(500):
    perm=rng.permutation(len(vidx)); a7=vidx[perm[:v7]]; a8=vidx[perm[v7:]]
    o7n=oc.copy();i7n=ic.copy();o8n=oc.copy();i8n=ic.copy()
    for k in a7: o7n[I[k]]+=1;i7n[J[k]]+=1
    for k in a8: o8n[I[k]]+=1;i8n[J[k]]+=1
    Snull.append((np.abs(o7n-o8n)+np.abs(i7n-i8n)).sum())
Snull=np.array(Snull)
P(f"[BUDGET CONSERVATION] real imbalance S={S_real:.0f} vs random-assignment null {Snull.mean():.0f}±{Snull.std():.0f}  z={(S_real-Snull.mean())/Snull.std():+.1f}")
P(f"  -> {'CONSERVED: variable edges COMPENSATE per-neuron (dithering/error-correction SUPPORTED)' if (S_real<Snull.mean()-2*Snull.std()) else ('ANTI-conserved' if S_real>Snull.mean()+2*Snull.std() else 'random (no compensation; genuine dice, dithering NOT supported)')}")
# per-neuron loss/gain compensation: does a neuron that LOSES an out-edge (7-only) also GAIN one (8-only)?
loss=np.zeros(M); gain=np.zeros(M)
for k in np.where(var & (y7==1))[0]: loss[I[k]]+=1
for k in np.where(var & (y8==1))[0]: gain[I[k]]+=1
mm=(loss+gain)>0
P(f"[COMPENSATION] corr(per-neuron out-losses, out-gains)={np.corrcoef(loss[mm],gain[mm])[0,1]:+.3f} (high+ => neurons swap partners keeping budget)")
# module/path-count preservation: 2-hop path counts per neuron, animal7 vs animal8
def A(present):
    Am=np.zeros((M,M)); 
    for k in np.where(present)[0]: Am[I[k],J[k]]=1
    return Am
A7=A(y7==1);A8=A(y8==1); p7=(A7@A7).sum(1); p8=(A8@A8).sum(1)
P(f"[PATH-COUNT preservation] per-neuron 2-hop out-path count corr(7,8)={np.corrcoef(p7,p8)[0,1]:.3f}")
