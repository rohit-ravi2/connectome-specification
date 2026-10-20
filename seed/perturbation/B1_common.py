#!/usr/bin/env python3
"""Part B1 shared loaders: Randi functional connectivity + structural scaffold.

Functional connectivity (Randi/Leifer 2023) = measured neural RESPONSE coupling
(stimulate src -> response in dst). NOT an anatomical synapse. Signed (+ excit / - inhib).
"""
import os, math
import numpy as np, pandas as pd
import openpyxl

ROOT = os.path.expanduser("../seed")
EXT  = f"{ROOT}/data/external/mechanistic_frontier"
SSD  = "../data/celegans/data/randi2023"
EDGE_CSV = f"{ROOT}/data/processed/mechanistic_perturbation/dev_edge_trajectory_table.csv"
CONTACT_XLSX = f"{EXT}/cook2019_adult/Adult and L4 nerve ring neighbors.xlsx"

def load_funconn(kind):
    """kind in {'leifer','wt','unc31','wildcp'} -> DataFrame(src,dst,val)."""
    f = {
        'leifer': f"{EXT}/randi2023_funconn/leifer_funconn_wildtype_2022-04-01.tsv",
        'wt':     f"{SSD}/randi_funconn_wildty.tsv",
        'wildcp': f"{SSD}/randi_funconn_wildcp.tsv",
        'unc31':  f"{SSD}/randi_funconn_unc31.tsv",
    }[kind]
    d = pd.read_csv(f, sep='\t', header=None, names=['src','dst','val'])
    d['val'] = d['val'].astype(float)
    return d

def funconn_matrix(df):
    """Return (neurons list, N x N signed matrix M[i,j] = response in j to stim in i)."""
    neurons = sorted(set(df.src) | set(df.dst))
    idx = {n:i for i,n in enumerate(neurons)}
    M = np.zeros((len(neurons), len(neurons)))
    for s,d,v in df.itertuples(index=False):
        M[idx[s], idx[d]] = v
    return neurons, idx, M

def load_contact():
    """Adult nerve-ring contact area matrix (symmetric). -> (neurons, idx, A)."""
    wb = openpyxl.load_workbook(CONTACT_XLSX, read_only=True, data_only=True)
    ws = wb['adult nerve ring neighbors']
    rows = list(ws.iter_rows(values_only=True))
    header = [h for h in rows[0][1:]]
    neurons = [h for h in header if h]
    ix = {n:i for i,n in enumerate(header) if n}
    A = np.zeros((len(header), len(header)))
    for row in rows[1:]:
        rn = row[0]
        if rn in ix:
            for j,v in enumerate(row[1:]):
                if v not in (None,''):
                    try: A[ix[rn], j] = float(v)
                    except: pass
    A = np.maximum(A, A.T)
    keep = [n for n in neurons]
    return keep, ix, A

def load_edge_table():
    return pd.read_csv(EDGE_CSV)

NPZ = "../data/tier5/connectome.npz"
def load_nt_map():
    """neuron -> NT class {ACh,Glu,GABA,mono,Unk} for ALL neurons (not just synapse-bearing)."""
    d = np.load(NPZ, allow_pickle=True)
    nm = [str(x) for x in d['names']]; ntp = [str(x) for x in d['nt_primary']]
    def cls(x):
        for k,v in [("Acetyl","ACh"),("ACh","ACh"),("Glu","Glu"),("GABA","GABA"),
                    ("Dopamine","mono"),("Serotonin","mono"),("5HT","mono"),
                    ("Octopamine","mono"),("Tyramine","mono")]:
            if k in x: return v
        return "Unk"
    return {a:cls(b) for a,b in zip(nm,ntp)}

if __name__ == "__main__":
    for k in ['leifer','wt','unc31']:
        d = load_funconn(k)
        print(k, "pairs", len(d), "neurons", d.src.nunique(),
              "nonzero", (d.val!=0).mean().round(4), "neg", (d.val<0).mean().round(4))
    neurons, ix, A = load_contact()
    print("contact neurons", len([n for n in neurons if n]))
