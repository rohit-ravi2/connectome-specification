import sys,numpy as np,pandas as pd
def P(*a): print(*a); sys.stdout.flush()
SC="../work"
z=np.load(f"{SC}/feats.npz",allow_pickle=True)
neurons=list(z['neurons']); idx={n:i for i,n in enumerate(neurons)}; M=len(neurons)
C=z['C'].astype(float); cdeg=z['cdeg']
nt_arr=z['nt_arr']; mod_arr=z['mod_arr']; birth_arr=z['birth_arr']
# ---- bilateral mirror map: swap trailing L<->R ----
def mirror(n):
    if n.endswith('L'): return n[:-1]+'R'
    if n.endswith('R'): return n[:-1]+'L'
    return None
mir=np.array([idx[mirror(n)] if (mirror(n) in idx) else -1 for n in neurons])
haspair=mir>=0
P(f"nodes with bilateral mirror: {haspair.sum()}/{M}")
# ---- residual: actual - mirror-symmetry prediction ----
R=np.zeros((M,M))
for i in range(M):
    for j in range(M):
        if mir[i]>=0 and mir[j]>=0:
            R[i,j]=C[i,j]-C[mir[i],mir[j]]
        else:
            R[i,j]=0.0
P(f"residual Frobenius {np.linalg.norm(R):.2f}  vs C {np.linalg.norm(C):.2f}  (frac {np.linalg.norm(R)/np.linalg.norm(C):.3f})")
# ---- SVD of residual, top-k ----
U,s,Vt=np.linalg.svd(R); K=6
P("top singular values:",s[:8].round(2))
Uk=U[:,:K]; Vk=Vt[:K].T   # node loadings: out-role (Uk), in-role (Vk)
# ---- varimax rotation ----
def varimax(Phi,gamma=1.0,q=100,tol=1e-6):
    Phi=Phi.copy(); p,k=Phi.shape; Rr=np.eye(k); d=0
    for _ in range(q):
        Lam=Phi@Rr
        u,sv,vt=np.linalg.svd(Phi.T@(Lam**3-(gamma/p)*Lam@np.diag(np.diag(Lam.T@Lam))))
        Rr=u@vt; dold=d; d=sv.sum()
        if dold!=0 and d/dold<1+tol: break
    return Phi@Rr
Ur=varimax(Uk); Vr=varimax(Vk)
# ---- candidate node variables ----
cand={}
for lev in ['ACh','Glu','GABA','mono','Unk']: cand[f'NT={lev}']=(nt_arr==lev).astype(float)
for lev in ['sensory','inter','motor','command']: cand[f'mod={lev}']=(mod_arr==lev).astype(float)
bt=birth_arr.copy(); bt[np.isnan(bt)]=np.nanmedian(bt); cand['birth_time']=bt
cand['contact_degree']=cdeg.astype(float)
cand['is_right']=np.array([1.0 if n.endswith('R') else 0.0 for n in neurons])
def corr(a,b):
    a=a-a.mean(); b=b-b.mean()
    d=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/d) if d>0 else 0.0
rows=[]
P("\n=== TASK 2: rotated residual axes vs nameable node variables (|corr|, top-3) ===")
for role,Rot in [('OUT',Ur),('IN',Vr)]:
    for c in range(K):
        ax=Rot[:,c]
        cc=sorted(((abs(corr(ax,v)),k,corr(ax,v)) for k,v in cand.items()),reverse=True)[:3]
        best=cc[0]
        rows.append(dict(role=role,axis=c+1,best_var=best[1],best_corr=best[2],
                         second_var=cc[1][1],second_corr=cc[1][2],third_var=cc[2][1],third_corr=cc[2][2]))
        P(f"  {role} axis{c+1}: "+"  ".join(f"{k}={v:+.2f}" for _,k,v in cc))
pd.DataFrame(rows).to_csv(f"{SC}/part1_residual_pca.csv",index=False)
# summary: max |corr| any axis to any nameable var
allc=[abs(r['best_corr']) for r in rows]
P(f"\nmax |corr| of any rotated residual axis to any nameable node variable = {max(allc):.3f}")
P("(|corr|>~0.4 => axis aligns with a nameable variable; else un-nameable low-rank residual)")
