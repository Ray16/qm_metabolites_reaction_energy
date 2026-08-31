"""Measure the two proposed anchors (acyl-phosphate, carboxy-phosphate) against real TECRDB members
BEFORE wiring: find members structurally, report err/bias/std/sign, and verify the new gates do NOT
collide with the existing anchor subclasses (adenylylate/thioester/phosphagen/phosphatase). No wiring."""
import json, os
import numpy as np
from collections import defaultdict
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metag.routing import anchor as A

HERE = os.path.dirname(__file__)
CAL = {r["rid"]: r for r in json.load(open(os.path.join(HERE, "..", "metag", "data", "sigma_class_calibrated.json")))["per_reaction"]}
RX = json.load(open(os.path.join(HERE, "..", "..", "experiments", "qm_mlip_solvation", "scripts", "reactions_tecrdb_all.json")))

_CO2 = Chem.MolFromSmarts("O=C=O")
_BICARB = Chem.MolFromSmarts("[OX2H1,OX1-]C(=O)[OX2H1,OX1-]")   # carbonic acid / bicarbonate
_ATP_POP = A._PYRO
_CARBOXYL = Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]")


def mols(species):
    out = []
    for name, (coeff, q, smi) in species.items():
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        out.append((int(coeff), name, smi, m))
    return out


def net(ms, patt):
    return sum(c * len(m.GetSubstructMatches(patt)) for c, _, _, m in ms)


def is_co2_or_bicarb_consumed(ms):
    return any(c < 0 and (m.HasSubstructMatch(_CO2) or (m.GetNumHeavyAtoms() <= 4 and m.HasSubstructMatch(_BICARB)))
              for c, _, _, m in ms)


def classify_new(ms):
    """Return 'acylP' / 'carboxyP' / None for the two PROPOSED classes, using gates designed NOT to
    overlap the existing anchor subclasses."""
    ams = [(c, smi, m) for c, _, smi, m in ms]   # anchor.py helpers expect (coeff, smi, mol)
    ppi = A._produces_ppi(ams)
    net_ma = net(ms, A._MIXEDANHYDRIDE)
    net_thio = net(ms, A._THIOESTER)
    has_pop = any(m.HasSubstructMatch(_ATP_POP) for _, _, _, m in ms)
    # acyl-phosphate: a C(=O)-O-P mixed anhydride created/destroyed, NO free PPi (excludes adenylylate),
    # NO thioester change (excludes acyl-CoA thioester class).
    if net_ma != 0 and ppi <= 0 and net_thio == 0:
        return "acylP"
    # carboxy-phosphate (biotin/ATP carboxylase): CO2/bicarbonate consumed + ATP P-O-P consumed +
    # a carboxyl group net created. The mixed anhydride is an intermediate (not in net eqn), so gate on
    # the net transformation instead.
    if is_co2_or_bicarb_consumed(ms) and net(ms, _ATP_POP) < 0 and net(ms, _CARBOXYL) > 0:
        return "carboxyP"
    return None


rows = defaultdict(list)
collide = defaultdict(list)
for rid, rx in RX.items():
    if rid not in CAL:
        continue
    ms = mols(rx["species"])
    if ms is None:
        continue
    newc = classify_new(ms)
    if newc is None:
        continue
    # does the EXISTING anchor router already claim this reaction?
    existing = A.subclass(rx["species"])
    err = CAL[rid]["err"]
    rows[newc].append((rid, err, CAL[rid]["class"], rx.get("EC", ""), rx["note"][:55]))
    if existing is not None:
        collide[newc].append((rid, existing))

for cls in ("acylP", "carboxyP"):
    r = rows[cls]
    print(f"\n{'='*90}\n{cls}: {len(r)} TECRDB members")
    if not r:
        print("  NONE found")
        continue
    errs = np.array([e for _, e, *_ in r])
    print(f"  bias(mean err) = {errs.mean():+.1f}   std = {errs.std():.1f}   "
          f"sign-consistency = {max((errs>0).mean(),(errs<0).mean())*100:.0f}%   "
          f"|median| = {abs(np.median(errs)):.1f}")
    for rid, e, cl, ec, note in sorted(r, key=lambda x: x[1]):
        print(f"    {rid}  err={e:+6.1f}  [{cl:22s}] EC {ec:10s} {note}")
    if collide[cls]:
        print(f"  !! GATE COLLISION with existing anchor subclass:")
        for rid, ex in collide[cls]:
            print(f"     {rid} already -> {ex}")
    else:
        print(f"  OK: no collision with existing anchor subclasses.")
