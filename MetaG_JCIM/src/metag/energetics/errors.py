"""Failures shared by species evaluation and reaction orchestration."""


class SpeciesRearranged(Exception):
    """Every relaxed conformer of a species changed covalent connectivity (e.g. zwitterion proton transfer):
    the sampled relaxations could not retain the requested state. This does not prove the absence of a
    physical minimum. Never scored or cached under its SMILES; score_reaction may retry a permitted
    neutral-reference route with declared protonation terms, or returns no estimate."""
    def __init__(self, name, smi, n):
        super().__init__(f"{name} ({smi}): bonding changed during relaxation in all {n} conformers")
        self.smi = smi


