"""Lightweight 3D geometry container and XYZ I/O.

A single immutable `Geometry` type is passed between the conformer generator and
the QM backends, decoupling them from RDKit/ASE specifics.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem


@dataclass(frozen=True)
class Geometry:
    """Cartesian geometry in Angstrom with associated charge/multiplicity."""

    symbols: tuple[str, ...]
    coords: tuple[tuple[float, float, float], ...]   # Angstrom
    charge: int
    multiplicity: int

    def __post_init__(self) -> None:
        if len(self.symbols) != len(self.coords):
            raise ValueError("symbols and coords length mismatch")

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)

    def to_xyz_block(self, comment: str = "") -> str:
        """Return a standard .xyz file body (with header lines)."""
        lines = [str(self.n_atoms), comment]
        for sym, (x, y, z) in zip(self.symbols, self.coords):
            lines.append(f"{sym:<3s} {x:>18.10f} {y:>18.10f} {z:>18.10f}")
        return "\n".join(lines) + "\n"

    def to_orca_coords(self) -> str:
        """Return just the atom lines for an ORCA `* xyz` block."""
        return "\n".join(
            f"{sym:<3s} {x:>18.10f} {y:>18.10f} {z:>18.10f}"
            for sym, (x, y, z) in zip(self.symbols, self.coords)
        )


def from_rdkit_conformer(mol: Chem.Mol, conf_id: int, charge: int, mult: int) -> Geometry:
    conf = mol.GetConformer(conf_id)
    symbols = tuple(a.GetSymbol() for a in mol.GetAtoms())
    coords = tuple(
        (p.x, p.y, p.z)
        for p in (conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms()))
    )
    return Geometry(symbols=symbols, coords=coords, charge=charge, multiplicity=mult)


def read_xyz(path: str, charge: int, mult: int) -> Geometry:
    """Read a (possibly multi-line-commented) XYZ file's first structure."""
    with open(path) as fh:
        lines = fh.read().splitlines()
    n = int(lines[0].split()[0])
    symbols: list[str] = []
    coords: list[tuple[float, float, float]] = []
    for line in lines[2 : 2 + n]:
        parts = line.split()
        symbols.append(parts[0])
        coords.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return Geometry(tuple(symbols), tuple(coords), charge=charge, multiplicity=mult)


def write_xyz(geom: Geometry, path: str, comment: str = "") -> None:
    with open(path, "w") as fh:
        fh.write(geom.to_xyz_block(comment))
