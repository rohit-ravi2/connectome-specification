"""Specificity IN BITS: H(reproducible connectome | contact geometry).
How much wiring information must the genome supply BEYOND physical contact?
Specification problem = which of the contacting pairs actually wire."""
import openpyxl, csv, re, math
from collections import defaultdict

CONTACT = "../data/celegans/data/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx"
EDGES = "../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv"

def h(p):  # binary entropy, bits
    if p <= 0 or p >= 1: return 0.0
    return -p*math.log2(p) - (1-p)*math.log2(1-p)

# contact
wb = openpyxl.load_workbook(CONTACT, read_only=True, data_only=True)
ws = wb['adult nerve ring neighbors']; data = list(ws.iter_rows(values_only=True))
header = list(data[0][1:]); contact = {}
for row in data[1:]:
    rn = row[0]
    if rn is None: continue
    for j, val in enumerate(row[1:]):
        if val in (None, ''): continue
        try: area = float(val)
        except (TypeError, ValueError): continue
        cn = header[j]
        if area > 0 and rn != cn:
            k = (min(rn,cn),max(rn,cn)); contact[k] = max(contact.get(k,0), area)
ring = set(header)

# reproducible connectome (both adults)
chem = defaultdict(set); elec = defaultdict(set)
with open(EDGES) as f:
    for r in csv.DictReader(f):
        ds = int(re.search(r'witvliet_2020_(\d)', r['source_file']).group(1))
        if ds not in (7,8): continue
        pre,post,typ = r['pre'],r['post'],r['type']
        (elec if typ=='electrical' else chem)[((min(pre,post),max(pre,post)) if typ=='electrical' else (pre,post))].add(ds)
repro = set()
for e,dss in chem.items():
    if len(dss)==2: repro.add((min(e[0],e[1]),max(e[0],e[1])))
for e,dss in elec.items():
    if len(dss)==2: repro.add(e)

pairs = sorted(contact, key=lambda p: contact[p])     # contacting pairs = the candidate edges
y = [1 if p in repro else 0 for p in pairs]
N = len(pairs); K = sum(y); pbar = K/N

# (0) base: no info beyond the base rate
base_bits = N * h(pbar)
# (1) conditional on contact area, via quantile bins (empirical p per bin)
nb = 25; area_bits = 0.0
for i in range(nb):
    lo, hi = i*N//nb, (i+1)*N//nb
    seg = y[lo:hi]; n = len(seg)
    if n: area_bits += n * h(sum(seg)/n)

info_from_area = base_bits - area_bits
print(f"specification problem: which of {N} contacting pairs form a reproducible synapse "
      f"({K} do, base rate {100*pbar:.1f}%)")
print(f"  H(connectome)                         = {base_bits:8.0f} bits  ({base_bits/N:.3f} bits/pair)")
print(f"  H(connectome | contact AREA)          = {area_bits:8.0f} bits  <- residual specificity")
print(f"  information contact-area supplies      = {info_from_area:8.0f} bits  "
      f"({100*info_from_area/base_bits:.0f}% of the total)")
print(f"  => molecular-address-code residual     = {area_bits:8.0f} bits")
print(f"     per nerve-ring neuron ({len(ring)})       = {area_bits/len(ring):8.2f} bits/neuron")

# reference points for the per-neuron address-code budget
print(f"\n  reference: log2(#neuron classes ~118) = {math.log2(118):.1f} bits/neuron")
print(f"            log2(182 ring neurons)       = {math.log2(182):.1f} bits/neuron  (unique-ID upper bound)")
