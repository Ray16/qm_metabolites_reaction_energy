"""Centralized paths so scripts never hard-code relative locations."""
import os

PKG = os.path.dirname(os.path.abspath(__file__))          # .../gnn_dgf/gnn
ROOT = os.path.dirname(PKG)                                # .../gnn_dgf
THERMO = os.path.dirname(os.path.dirname(ROOT))            # .../thermodynamic_calc
PIPE = os.path.join(THERMO, "pipeline")
RESULTS = os.path.join(THERMO, "results")
FIGURES = os.path.join(THERMO, "figures")
DB = os.path.join(os.path.dirname(THERMO), "ModelSEEDDatabase", "Biochemistry")
ARTIFACTS = os.path.join(ROOT, "artifacts")
LOGS = os.path.join(ROOT, "logs")

os.makedirs(ARTIFACTS, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)


def artifact(name):
    return os.path.join(ARTIFACTS, name)
