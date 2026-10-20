"""E0 make-or-break: is there a compact generator SHORTER than the lineage trace, that predicts
held-out subtrees? Tests the pure 'inverse-lineage-generator' claim on the DIVISION/STRUCTURE scaffold.

Data: jtauber Sulston tree (structure) + Table-S4 early/late born (terminal fate). (No per-node
division TIMES on disk -> E0 tests STRUCTURE compression + terminal-fate rule-prediction, the
runnable core; timing-minutes noted as a data gap.)

Tests:
 (1) SUBTREE-SHARING MDL: compress tree by sharing identical subtree shapes (minimal DAG).
     compression ratio = total internal subtrees / distinct subtree shapes, BY SIZE, vs random-tree null.
     Real >> random => lineage has reusable repeated structure (generator signature).
 (2) HELD-OUT FOUNDER BLOCK: learn grammar on some founders, predict held-out founder subtree shapes.
 (3) TERMINAL FATE grammar: does founder+depth+local-branch predict held-out-subtree early/late born
     below explicit lookup?
"""
import json, warnings; warnings.simplefilter("ignore")
import numpy as np, pandas as pd
from collections import defaultdict, Counter
from pathlib import Path
RNG=np.random.default_rng(0)
OUT=Path("../data/celegans/data/processed/inverse_lineage"); OUT.mkdir(parents=True,exist_ok=True)
tree=json.loads(open("../lineage/jtauber_lineage/cells.json").read())

# ---- build node structure ----
children=defaultdict(list); parent={}; name_of=[]
def build(node,par):
    nm=node["name"]; parent[nm]=par; name_of.append(nm)
    for c in node.get("children",[]):
        children[nm].append(c["name"]); build(c,nm)
build(tree,None)
def is_leaf(n): return len(children[n])==0
def depth(n):
    d=0; x=n
    while parent[x] is not None: x=parent[x]; d+=1
    return d
allnodes=name_of
internal=[n for n in allnodes if not is_leaf(n)]
leaves=[n for n in allnodes if is_leaf(n)]
print(f"tree: {len(allnodes)} nodes | internal {len(internal)} | leaves {len(leaves)}")

# ---- subtree shape signatures (unordered) + size ----
shape={}; size={}
def compute(n):
    if is_leaf(n): shape[n]="L"; size[n]=1; return
    for c in children[n]:
        if c not in shape: compute(c)
    kids=sorted(shape[c] for c in children[n])
    shape[n]="("+",".join(kids)+")"; size[n]=1+sum(size[c] for c in children[n])
for n in allnodes:
    if n not in shape: compute(n)

# ---- (1) subtree-sharing MDL: distinct shapes vs total, by size ----
by_size_total=Counter(); by_size_shapes=defaultdict(set)
for n in internal:
    by_size_total[size[n]]+=1; by_size_shapes[size[n]].add(shape[n])
tot=len(internal); distinct=len(set(shape[n] for n in internal))
print(f"\n(1) SUBTREE-SHARING MDL: {tot} internal subtrees, {distinct} distinct shapes -> "
      f"overall compression ratio {tot/distinct:.2f}x")

# random-tree null: build random binary trees w/ same leaf-count as whole tree; same measure
def random_bintree(nleaves):
    # random growing binary tree
    ch=defaultdict(list); nodes=[0]; leaves_=[0]; nxt=1
    while len(leaves_)<nleaves:
        p=leaves_.pop(RNG.integers(len(leaves_)))
        a,b=nxt,nxt+1; nxt+=2; ch[p]=[a,b]; leaves_+= [a,b]
    sh={}; sz={}
    def cp(x):
        if not ch[x]: sh[x]="L"; sz[x]=1; return
        for c in ch[x]:
            if c not in sh: cp(c)
        sh[x]="("+",".join(sorted(sh[c] for c in ch[x]))+")"; sz[x]=1+sum(sz[c] for c in ch[x])
    cp(0)
    inter=[x for x in sh if ch[x]]
    return len(inter), len(set(sh[x] for x in inter))
rnd=[random_bintree(len(leaves)) for _ in range(20)]
rt=np.mean([t/d for t,d in rnd]); rt_sd=np.std([t/d for t,d in rnd])
print(f"    random-tree null compression ratio {rt:.2f}±{rt_sd:.2f}x  -> real/null = {(tot/distinct)/rt:.2f}x "
      f"({'MORE compressible than random' if tot/distinct>rt+3*rt_sd else 'NOT more than random'})")
print("    compression by subtree size (real distinct/total):")
for s in sorted(by_size_shapes):
    if by_size_total[s]>=5:
        print(f"      size {s:2d}: {by_size_total[s]:3d} subtrees, {len(by_size_shapes[s]):3d} distinct shapes "
              f"({by_size_total[s]/len(by_size_shapes[s]):.2f}x)")

# ---- (2) held-out founder block: predict subtree shapes in held-out founder ----
def founder(n):
    x=n
    while parent[x] not in (None,"P0","P1","EMS") and parent[x] is not None: x=parent[x]
    # founder = the node just under the early stem
    import re
    m=re.match(r'^([A-Z]+)',n); return m.group(1) if m else "?"
fnd={n:founder(n) for n in internal}
founders=[f for f,c in Counter(fnd.values()).items() if c>=20]
print(f"\n(2) HELD-OUT FOUNDER: shapes seen in a founder that ALSO occur in OTHER founders (reuse across lineage):")
shapes_by_f={f:set(shape[n] for n in internal if fnd[n]==f) for f in founders}
for f in founders:
    others=set().union(*[shapes_by_f[g] for g in founders if g!=f])
    reuse=len(shapes_by_f[f]&others)/len(shapes_by_f[f])
    print(f"    {f:4s}: {len(shapes_by_f[f]):3d} distinct shapes, {reuse:.2f} also appear in other founders")

# ---- (3) terminal fate grammar: predict early/late born on held-out subtrees ----
bt=pd.read_csv("../data/celegans/data/processed/developmental_rules/birthtime_earlylate.csv")
born={r.neuron:(1 if r.born=="early" else 0) for r in bt.itertuples()}
term=[l for l in leaves if l in born]
print(f"\n(3) TERMINAL FATE (early/late born) grammar, {len(term)} covered terminals:")
# feature: founder + depth + last-2 branch tokens (local rule); predict born; held-out by founder
import re
def feats(l):
    d=depth(l); f=founder(l); tok=re.sub(r'^[A-Z]+','',l)[-2:]
    return f,d,tok
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
X=[feats(l) for l in term]; yb=np.array([born[l] for l in term])
ff=[x[0] for x in X]; dd=np.array([x[1] for x in X]); tk=[x[2] for x in X]
enc=OneHotEncoder(sparse_output=False,handle_unknown='ignore')
Xf=np.column_stack([enc.fit_transform(np.array(list(zip(ff,tk)))),dd])
# held-out founder CV
from sklearn.metrics import roc_auc_score
aucs=[]
for f in founders:
    tr=[i for i in range(len(term)) if ff[i]!=f]; te=[i for i in range(len(term)) if ff[i]==f]
    if len(te)<8 or len(set(yb[te]))<2 or len(set(yb[tr]))<2: continue
    m=LogisticRegression(max_iter=1000,class_weight="balanced").fit(Xf[tr],yb[tr])
    aucs.append(roc_auc_score(yb[te],m.predict_proba(Xf[te])[:,1]))
base=max(yb.mean(),1-yb.mean())
print(f"    held-out-founder AUC(rule->early/late) = {np.mean(aucs):.3f} (base-rate {base:.3f}, n-founders {len(aucs)})")

print("\n"+"="*80)
comp_ok=(tot/distinct)>rt+3*rt_sd
print("E0 VERDICT:", "COMPRESSION > random (reusable structure exists)" if comp_ok else "NO compression beyond random",
      "| held-out-fate AUC {:.2f}".format(np.mean(aucs) if aucs else 0))
print("="*80)
pd.DataFrame([{"internal_subtrees":tot,"distinct_shapes":distinct,"compression_ratio":tot/distinct,
   "random_null_ratio":rt,"real_over_null":(tot/distinct)/rt,"heldout_fate_auc":np.mean(aucs) if aucs else np.nan}]
   ).to_csv(OUT/"E0_mdl_compression_results.csv",index=False)
print("saved",OUT/"E0_mdl_compression_results.csv")
