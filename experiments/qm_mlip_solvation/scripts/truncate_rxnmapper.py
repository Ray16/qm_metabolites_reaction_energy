"""RXNMapper-based truncation: same reaction-center -> grow -> cap machinery as truncate.py, but the
atom-atom map comes from RXNMapper (a reaction transformer) instead of MCS. Built to EMPIRICALLY
compare against MCS: run both, find where they differ, let the ΔG-vs-experiment decide which wins.

RXNMapper maps are precomputed (rxnfp env, old python) into artifacts/rxnmapper_maps.json as
{rid: {"mapped": "<mapped_rxn>", "conf": c}}. Here (uma env) we parse the mapped reactant/product,
recover the reactant->product atom correspondence from the shared atom-map numbers, and truncate.
Same balance + n_H+ + consistency guards as truncate_v2 (radius-sensitivity validates downstream)."""
import os, json
from collections import Counter
from rdkit import Chem
import truncate as T

_MAPS = None


def _load_maps():
    global _MAPS
    if _MAPS is None:
        p = os.path.join(os.path.dirname(__file__), "..", "artifacts", "rxnmapper_maps.json")
        _MAPS = json.load(open(p)) if os.path.exists(p) else {}
    return _MAPS


def _mol_and_mapnums(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None, None
    mapnum = {a.GetIdx(): a.GetAtomMapNum() for a in m.GetAtoms()}
    return m, mapnum


def build_truncated_reaction_rxnmapper(rid, species_dict, radius=2, min_conf=0.0):
    """-> (new_species, n_Hplus) or None. Uses the precomputed RXNMapper map for `rid`."""
    rec = _load_maps().get(rid)
    if not rec or not rec.get("mapped"):
        return None
    if rec.get("conf") is not None and rec["conf"] < min_conf:
        return None
    try:
        rsmi, psmi = rec["mapped"].split(">>")
    except ValueError:
        return None
    superR, mapR = _mol_and_mapnums(rsmi)
    superP, mapP = _mol_and_mapnums(psmi)
    if superR is None or superP is None:
        return None
    full_nH = T._full_nHplus(species_dict)
    if full_nH is None:
        return None
    # reactant_idx -> product_idx via shared non-zero atom-map number
    p_by_num = {n: i for i, n in mapP.items() if n}
    amap = {i: p_by_num[n] for i, n in mapR.items() if n and n in p_by_num}
    if not amap:
        return None
    inv = {v: k for k, v in amap.items()}
    cR = T.reaction_center(superR, amap, superP)
    cP = T.reaction_center(superP, inv, superR)
    keepR = T.grow(superR, cR, radius)
    keepP = T.grow(superP, cP, radius) | {amap[i] for i in keepR if i in amap}
    keepR = keepR | {inv[j] for j in keepP if j in inv}

    def per_frag(mol, keep_global):
        """Split the combined mapped mol into FRAGMENTS (= species), truncate+cap each kept fragment."""
        frags = Chem.GetMolFrags(mol, asMols=False)          # tuples of atom idx per fragment
        caps = []
        removed = Counter()
        for atoms in frags:
            keep_local_global = [a for a in atoms if a in keep_global]
            if not keep_local_global:
                # a small reactant (water/CO2/NH3, <=4 heavy atoms) never needs truncation and must
                # NOT drop the reaction -- keep it WHOLE; only a LARGE unkept fragment is ill-posed.
                heavy = sum(1 for a in atoms if mol.GetAtomWithIdx(a).GetAtomicNum() > 1)
                if heavy <= 4:
                    keep_local_global = list(atoms)
                else:
                    return None, None
            # build a submol-preserving cap: reuse truncate_species on the fragment's own SMILES
            # (strip map numbers first so chemistry is clean)
            fm = Chem.RWMol(Chem.PathToSubmol(mol, [b.GetIdx() for b in mol.GetBonds()
                                                    if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms]))
            # simpler + robust: canonical SMILES of the fragment, map global keep -> local indices
            idx_map = {g: l for l, g in enumerate(atoms)}
            frag_atoms = list(atoms)
            frag_mol = Chem.MolFragmentToSmiles(mol, atomsToUse=frag_atoms, canonical=False)
            fmol = Chem.MolFromSmiles(frag_mol)
            if fmol is None:
                return None, None
            # strip atom-map numbers for clean chemistry
            for a in fmol.GetAtoms():
                a.SetAtomMapNum(0)
            keep_local = {idx_map[g] for g in keep_local_global}
            cap, rem = T.truncate_species(Chem.MolToSmiles(fmol), keep_local)
            cm = Chem.MolFromSmiles(cap)
            if cm is None:
                return None, None
            caps.append(cap)
            for fr in rem:
                for piece in fr.split("."):
                    pm = Chem.MolFromSmiles(piece)
                    removed[Chem.MolToSmiles(pm) if pm else piece] += 1
        return caps, removed

    r_caps, rem_r = per_frag(superR, keepR)
    p_caps, rem_p = per_frag(superP, keepP)
    if r_caps is None or p_caps is None:
        return None
    if rem_r != rem_p:                                       # consistency guard
        return None

    def q_h(smi):
        m = Chem.MolFromSmiles(smi)
        return (Chem.GetFormalCharge(m), sum(a.GetTotalNumHs() for a in m.GetAtoms())) if m else (None, None)
    new = {}
    Hr = Hp = qr = qp = 0
    for i, cap in enumerate(r_caps):
        q, h = q_h(cap)
        if q is None:
            return None
        new[f"R{i}_t"] = [-1, int(q), cap]; Hr += h; qr += q
    for i, cap in enumerate(p_caps):
        q, h = q_h(cap)
        if q is None:
            return None
        new[f"P{i}_t"] = [1, int(q), cap]; Hp += h; qp += q
    nH = Hr - Hp
    if nH != qr - qp or nH != full_nH:
        return None
    return new, int(nH)
