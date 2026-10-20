import sys, numpy as np
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/deepfeats.npz",allow_pickle=True); z2=np.load(f"{SC}/resid2.npz")
I,J,y7,y8,base,neurons=z['I'],z['J'],z['y7'],z['y8'],z['base'],list(z['neurons'])
N=len(I); yrep=((y7==1)&(y8==1)).astype(int); idx={n:i for i,n in enumerate(neurons)}; ns=set(neurons)
r1=z2['r1']; p1=yrep-r1
logarea=base[:,0]; cdeg_s=base[:,3]; cdeg_t=base[:,4]
t5=np.load("../data/tier5/connectome.npz",allow_pickle=True)
t5nt={str(n):str(v) for n,v in zip(t5["names"],t5["nt_primary"])}
def bcl(n): return n[:-1] if n[-1] in 'LR' else n
nt=[t5nt.get(neurons[i],t5nt.get(bcl(neurons[i]),"Unk")) for i in range(len(neurons))]
def mir(n): 
    m=n[:-1]+('R' if n[-1]=='L' else 'L') if n[-1] in 'LR' else None; return m if m in ns else None
SENS={'ADF','ADL','AFD','ASE','ASG','ASH','ASI','ASJ','ASK','AWA','AWB','AWC','BAG','URX','IL1','IL2','OLL','OLQ','CEP','ADE','URY','URB','AVM','ALM','PLM','PVD','FLP'}
def isS(n): c=bcl(n).rstrip('DV'); return any(c.startswith(s) for s in SENS)
sameNT=np.array([nt[I[k]]==nt[J[k]] for k in range(N)])
homolog=np.array([mir(neurons[I[k]])==neurons[J[k]] for k in range(N)])
bothsens=np.array([isS(neurons[I[k]]) and isS(neurons[J[k]]) for k in range(N)])
la_hi=logarea>np.quantile(logarea,0.75)
FN=(r1>0.3); FP=(r1<-0.3)  # FN=surprising CONNECTION(wires,model doubted); FP=surprising NON-connection(touches,model expected)
P(f"HARD CORE: {int((FN|FP).sum())} pairs | surprising CONNECTIONS (wire, model said no): {FN.sum()} | surprising NON-CONNECTIONS (touch, model said yes): {FP.sum()}")
def prof(mask,lab):
    P(f"\n[{lab}] n={mask.sum()}")
    P(f"   mean log-contact-area={logarea[mask].mean():.2f} (all={logarea.mean():.2f}) | high-contact-quartile frac={la_hi[mask].mean():.2f} (base 0.25)")
    P(f"   same-NT={sameNT[mask].mean():.2f} (all {sameNT.mean():.2f}) | homolog-pair={homolog[mask].mean():.3f} (all {homolog.mean():.3f}) | both-sensory={bothsens[mask].mean():.2f} (all {bothsens.mean():.2f})")
    P(f"   mean degree src={cdeg_s[mask].mean():.0f} tgt={cdeg_t[mask].mean():.0f} (all {cdeg_s.mean():.0f})")
    from collections import Counter
    ntp=Counter(tuple(sorted((nt[I[k]],nt[J[k]]))) for k in np.where(mask)[0]); 
    P(f"   top NT-pairs: {ntp.most_common(4)}")
    ex=np.where(mask)[0][:6]; P(f"   examples: {[(neurons[I[k]],neurons[J[k]]) for k in ex]}")
prof(FP,"SURPRISING NON-CONNECTIONS (touch-but-reproducibly-DONT-wire)")
prof(FN,"SURPRISING CONNECTIONS (reproducibly wire despite model doubt)")
