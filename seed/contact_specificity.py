"""Contact-vs-specificity decomposition of the reproducible connectome core.
Of nerve-ring pairs that physically contact, how many form a reproducible synapse?
P(synapse|contact) ~ 1 => contact-determined (Peters' rule); << 1 => molecular specificity."""
import openpyxl, csv, re
from collections import defaultdict

CONTACT = "../data/celegans/data/connectome/nerve_ring_neighbors/Adult and L4 nerve ring neighbors.xlsx"
EDGES = "../data/celegans/data/connectome/witvliet2020/witvliet_combined_edges.csv"

# ---- contact matrix (adult) ----
wb = openpyxl.load_workbook(CONTACT, read_only=True, data_only=True)
ws = wb['adult nerve ring neighbors']
data = list(ws.iter_rows(values_only=True))
header = [h for h in data[0][1:]]
contact = {}                      # undirected pair -> contact area
for row in data[1:]:
    rn = row[0]
    if rn is None: continue
    for j, val in enumerate(row[1:]):
        if val in (None, ''): continue
        cn = header[j]
        try: area = float(val)
        except (TypeError, ValueError): continue
        if area > 0 and rn != cn:
            k = (min(rn, cn), max(rn, cn))
            contact[k] = max(contact.get(k, 0), area)
ring = set(header)
print(f"nerve-ring neurons (contact matrix): {len(ring)}")
print(f"contacting pairs (area>0): {len(contact)}")

# ---- connectome: reproducible core (present in both adults 7,8) ----
chem = defaultdict(set); elec = defaultdict(set)
with open(EDGES) as f:
    for r in csv.DictReader(f):
        ds = int(re.search(r'witvliet_2020_(\d)', r['source_file']).group(1))
        if ds not in (7, 8): continue
        pre, post, typ = r['pre'], r['post'], r['type']
        if typ == 'electrical': elec[(min(pre,post),max(pre,post))].add(ds)
        else: chem[(pre,post)].add(ds)

repro = set()                     # undirected pairs with a reproducible (both-adult) connection
any_conn = set()
for (pre,post), dss in chem.items():
    any_conn.add((min(pre,post),max(pre,post)))
    if len(dss) == 2: repro.add((min(pre,post),max(pre,post)))
for pair, dss in elec.items():
    any_conn.add(pair)
    if len(dss) == 2: repro.add(pair)

# restrict everything to nerve-ring neurons
inring = lambda p: p[0] in ring and p[1] in ring
contacting = set(contact)                          # all within ring by construction
repro_ring = {p for p in repro if inring(p)}
any_ring   = {p for p in any_conn if inring(p)}

# ---- the decomposition ----
conn_and_contact = repro_ring & contacting
p_syn_given_contact = len(conn_and_contact) / len(contacting)
p_contact_given_syn = len(conn_and_contact) / len(repro_ring) if repro_ring else 0
noncontact_syn = repro_ring - contacting

print(f"\nreproducible connections among ring neurons: {len(repro_ring)}")
print(f"  NECESSITY  P(contact | reproducible synapse) = {100*p_contact_given_syn:.1f}%  "
      f"(reproducible synapses without recorded contact: {len(noncontact_syn)})")
print(f"  SUFFICIENCY P(reproducible synapse | contact) = {100*p_syn_given_contact:.1f}%  "
      f"<-- the specificity number")
print(f"  => specificity gap (contact present, no repro synapse): {100*(1-p_syn_given_contact):.1f}%")

print(f"\n  contrast P(ANY synapse | contact) = {100*len(any_ring & contacting)/len(contacting):.1f}%  "
      f"(incl. weak/variable)")

# ---- does contact AREA predict connection? (graded Peters' rule) ----
pairs_sorted = sorted(contacting, key=lambda p: contact[p])
nb = 6; m = len(pairs_sorted)
print("\n  P(reproducible synapse | contact) by contact-area sextile:")
for i in range(nb):
    grp = pairs_sorted[i*m//nb:(i+1)*m//nb]
    areas = [contact[p] for p in grp]
    frac = 100*sum(p in repro_ring for p in grp)/len(grp)
    print(f"    area {min(areas):7.0f}-{max(areas):7.0f} ({len(grp):4d} pairs): {frac:5.1f}% connect")
