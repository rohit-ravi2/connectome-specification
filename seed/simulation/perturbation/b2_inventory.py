"""Part B2 Test 1: simulation resource inventory + c302 feasibility check."""
import os, shutil, importlib, subprocess, glob
import pandas as pd

SIM = "../data/celegans/simulation"
OUT = "../seed/data/processed/mechanistic_perturbation"
os.makedirs(OUT, exist_ok=True)

rows = []
def add(resource, kind, path, present, detail):
    rows.append(dict(resource=resource, kind=kind, path=path, present=present, detail=detail))

# --- On-disk NeuroML / c302 assets ---
def count(pat):
    return len(glob.glob(pat))

add("c302 python scripts", "code", f"{SIM}/c302_code", count(f"{SIM}/c302_code/*.py") > 0,
    f"{count(f'{SIM}/c302_code/*.py')} .py files (c302.py, c302_Full.py, parameter builders)")
add("cell NeuroML morphologies", "data", f"{SIM}/cells", count(f"{SIM}/cells/*.cell.nml") > 0,
    f"{count(f'{SIM}/cells/*.cell.nml')} .cell.nml files")
add("channel NeuroML", "data", f"{SIM}/channels", count(f"{SIM}/channels/*.nml") > 0,
    f"{count(f'{SIM}/channels/*.nml')} channel files (LeakConductance, Generic_GJ only -> passive/iaf)")
add("synapse NeuroML", "data", f"{SIM}/synapses", count(f"{SIM}/synapses/*.nml") > 0,
    f"{count(f'{SIM}/synapses/*.synapse.nml')} synapse mechanism files (ACh,GABA,Glu,mono...)")
add("network NeuroML", "data", f"{SIM}/networks", count(f"{SIM}/networks/*.net.nml") > 0,
    f"{count(f'{SIM}/networks/*.net.nml')} full .net.nml networks (variants A/B/C x Full/Syns/...)")
add("parsed network connections", "data", f"{SIM}/parsed/out_network_connections.csv",
    os.path.exists(f"{SIM}/parsed/out_network_connections.csv"), "9562 rows connection table (pre/post indices)")
add("upstream c302 repo", "code", f"{SIM}/upstream/c302", os.path.isdir(f"{SIM}/upstream/c302"),
    "full c302 source incl setup.py, examples")
add("upstream BAAIWorm", "code", f"{SIM}/upstream/BAAIWorm", os.path.isdir(f"{SIM}/upstream/BAAIWorm"),
    "BAAIWorm embodied model source")

# --- Runtime engines needed for a real simulation ---
for mod, label in [("neuron","NEURON simulator (pyNEURON)"),
                   ("pyneuroml","pyNeuroML (jnml wrapper)"),
                   ("neuroml","libNeuroML (model IO)"),
                   ("c302","c302 package"),
                   ("lems","PyLEMS")]:
    try:
        importlib.import_module(mod)
        ok = True; detail = "importable in ~/venvs/ds"
    except Exception as e:
        ok = False; detail = f"NOT importable: {type(e).__name__}"
    add(label, "engine", "~/venvs/ds", ok, detail)

for exe, label in [("jnml","jNeuroML CLI"),("nrniv","NEURON nrniv"),("java","Java JRE")]:
    p = shutil.which(exe)
    add(label, "engine", p or "-", p is not None, "found on PATH" if p else "NOT on PATH")

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/B2_simulation_resource_inventory.csv", index=False)
print(df.to_string())

engines = df[df.kind == "engine"]
sim_feasible = engines[engines.resource.isin(
    ["NEURON simulator (pyNEURON)","pyNeuroML (jnml wrapper)","jNeuroML CLI"])].present.any()
print("\nFULL c302/NeuroML RUN FEASIBLE IN THIS ENV:", bool(sim_feasible))
