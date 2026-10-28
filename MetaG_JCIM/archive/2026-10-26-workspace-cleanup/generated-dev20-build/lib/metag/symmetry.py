"""Geometry-based molecular symmetry for the ideal-gas rotational term: linearity (drop 5 vs 6
external modes) and the external rotational symmetry number sigma (order of the proper-rotation
subgroup of the point group). Self-contained (numpy only) so it runs in the UMA env.

Why: the RRHO thermal term needs the correct sigma (rotational entropy = ... - R ln sigma) and the
correct count of external modes (5 for a linear molecule, 6 for a nonlinear one). Hard-coding
sigma=1/nonlinear biases every reaction whose net species include a symmetric molecule (water sigma=2,
appears many times -> a fixed ~1.7 kJ/mol error that does NOT cancel when water is created/destroyed)
and mis-treats linear species (CO2/O2/H2/N2) that are common in ModelSEED.

sigma is the EXTERNAL symmetry number (proper rotations only) -- it deliberately does NOT count internal
rotations (a freely rotating methyl is not an external symmetry), so this is geometry-based, not a graph
automorphism count.
"""
import numpy as np

# minimal atomic masses (amu) for the elements that appear in metabolism; extend as needed.
_MASS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "P": 30.974, "S": 32.06,
         "F": 18.998, "Cl": 35.45, "Br": 79.904, "I": 126.90, "Na": 22.990, "K": 39.098,
         "Mg": 24.305, "Ca": 40.078, "Fe": 55.845, "Zn": 65.38, "Mn": 54.938, "Co": 58.933,
         "Ni": 58.693, "Cu": 63.546, "Se": 78.971, "B": 10.811, "Si": 28.085}


def _com_frame(symbols, coords):
    m = np.array([_MASS.get(s, 12.0) for s in symbols])
    x = np.asarray(coords, float)
    com = (m[:, None] * x).sum(0) / m.sum()
    return x - com, m


def is_linear(symbols, coords, tol=0.05):
    """True if all atoms are collinear (linear molecule / diatomic). tol in Angstrom."""
    x, m = _com_frame(symbols, coords)
    if len(x) <= 1:
        return False                                    # monatomic handled separately
    if len(x) == 2:
        return True
    # inertia tensor: linear => one principal moment ~ 0 and the other two ~ equal
    I = np.zeros((3, 3))
    for mi, ri in zip(m, x):
        I += mi * (np.dot(ri, ri) * np.eye(3) - np.outer(ri, ri))
    w = np.sort(np.linalg.eigvalsh(I))                  # ascending
    # smallest ~0 relative to the largest, and the top two comparable
    return w[0] < 1e-3 * max(w[2], 1e-9) or w[0] < 0.5


def _rotation_matrix(axis, angle):
    a = axis / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = a
    return np.array([
        [c + x*x*(1-c),   x*y*(1-c)-z*s, x*z*(1-c)+y*s],
        [y*x*(1-c)+z*s,   c + y*y*(1-c), y*z*(1-c)-x*s],
        [z*x*(1-c)-y*s,   z*y*(1-c)+x*s, c + z*z*(1-c)],
    ])


def _maps_onto_self(x, symbols, R, tol):
    xr = x @ R.T
    used = [False] * len(x)
    for p, s in zip(xr, symbols):
        hit = -1
        for j, (q, s2) in enumerate(zip(x, symbols)):
            if not used[j] and s == s2 and np.dot(p - q, p - q) < tol * tol:
                hit = j
                break
        if hit < 0:
            return False
        used[hit] = True
    return True


def symmetry_number(symbols, coords, tol=0.15):
    """External rotational symmetry number sigma from the geometry (proper rotations only).

    Linear molecules: sigma=2 if it has an inversion centre (homonuclear/centrosymmetric, e.g. O2, CO2,
    H2, N2), else 1 (heteronuclear, e.g. CO, HCN). Nonlinear: count distinct proper rotations that map the
    molecule onto itself, over a candidate-axis set built from the geometry (principal axes, atom
    directions, and atom-pair sums/differences -- covers Cn, Dn, and the Td/Oh axes).
    """
    x, m = _com_frame(symbols, coords)
    n = len(x)
    if n <= 1:
        return 1
    if is_linear(symbols, coords):
        # centrosymmetric linear (X...X about the centre) -> sigma 2, else 1
        centro = _maps_onto_self(x, symbols, -np.eye(3), tol)  # inversion
        return 2 if centro else 1

    # candidate rotation axes
    axes = []
    I = np.zeros((3, 3))
    for mi, ri in zip(m, x):
        I += mi * (np.dot(ri, ri) * np.eye(3) - np.outer(ri, ri))
    axes.extend(np.linalg.eigh(I)[1].T)                 # principal axes
    for ri in x:
        if np.linalg.norm(ri) > 0.1:
            axes.append(ri)
    for i in range(n):
        for j in range(i + 1, n):
            for v in (x[i] + x[j], x[i] - x[j]):
                if np.linalg.norm(v) > 0.1:
                    axes.append(v)
    # dedup axis directions
    uniq = []
    for a in axes:
        a = a / np.linalg.norm(a)
        if not any(abs(abs(np.dot(a, b)) - 1.0) < 1e-2 for b in uniq):
            uniq.append(a)

    rots = [np.eye(3)]
    def _seen(R):
        return any(np.allclose(R, S, atol=1e-2) for S in rots)
    for a in uniq:
        for order in (2, 3, 4, 5, 6):
            for k in range(1, order):
                R = _rotation_matrix(a, 2 * np.pi * k / order)
                if not _seen(R) and _maps_onto_self(x, symbols, R, tol):
                    rots.append(R)
    return len(rots)


_SIGMA_CACHE = {}


def species_sigma_from_smiles(smi):
    """Rotational symmetry number sigma from a SMILES (embed a 3D conformer, then count proper rotations).
    Cached. Rigid small species (water=2, ammonia=3, CO2/O2/H2/N2=2) are unambiguous; floppy species may
    read 1 on an asymmetric instantaneous conformer (conservative). Returns 1 on any failure."""
    if smi in _SIGMA_CACHE:
        return _SIGMA_CACHE[smi]
    sig = 1
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        if AllChem.EmbedMolecule(m, randomSeed=1) == 0:
            try:
                AllChem.MMFFOptimizeMolecule(m)
            except Exception:
                pass
            c = m.GetConformer()
            sym = [a.GetSymbol() for a in m.GetAtoms()]
            crd = [[c.GetAtomPosition(i).x, c.GetAtomPosition(i).y, c.GetAtomPosition(i).z]
                   for i in range(m.GetNumAtoms())]
            sig = symmetry_number(sym, crd)
    except Exception:
        sig = 1
    _SIGMA_CACHE[smi] = sig
    return sig


def thermal_sigma_delta(species, T=298.15):
    """Analytic ΔG shift (kJ/mol) of the RRHO symmetry-number fix (old hard-coded sigma=1 -> sigma_true):
    Δ(ΔG) = RT * Σ coeff * ln(sigma_species). EXACT and geometry-independent (the rotational partition
    function ∝ 1/sigma), so it applies the thermal fix to a cached pre-fix ΔG WITHOUT re-running QM.
    Nonzero only for reactions with a net symmetric species (net water/ammonia/...). For TECRDB there are
    no net LINEAR species, so this is the complete effect of the fix. `species` = {name:[coeff,q,smi]}."""
    import math
    RT = 8.314e-3 * T
    d = 0.0
    for coeff, q, smi in species.values():
        sg = species_sigma_from_smiles(smi)
        if sg > 1:
            d += coeff * RT * math.log(sg)
    return d


def geometry_and_sigma(symbols, coords):
    """Return (geometry_str, sigma, n_drop) for ase.thermochemistry.IdealGasThermo, where n_drop is the
    number of external (translation+rotation) modes to remove from the sorted |frequency| list:
    monatomic 3 (3 trans), linear 5 (3 trans + 2 rot), nonlinear 6 (3 trans + 3 rot)."""
    n = len(symbols)
    if n == 1:
        return "monatomic", 1, 3
    if is_linear(symbols, coords):
        return "linear", symmetry_number(symbols, coords), 5
    return "nonlinear", symmetry_number(symbols, coords), 6
