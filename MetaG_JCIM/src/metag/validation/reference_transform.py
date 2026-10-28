"""Local ModelSEED matching and condition transforms; no reaction-energy fitting."""
import csv
import glob
import os
import re
from collections import Counter

PROTON, WATER = "cpd00067", "cpd00001"
R_KJ = 8.314462618e-3
KEGG_RE = re.compile(r"C\d{5}")
STD = dict(p_h=7.0, ionic_strength=0.0, p_mg=14.0)

def kegg_to_modelseed(db, ambiguity="exclude"):
    """Resolve unique compound aliases; alphabetical order is not chemical evidence.

    ``historical_first`` is available only for explicitly labelled legacy replay.
    """
    if ambiguity not in ("exclude", "historical_first"):
        raise ValueError("unknown compound mapping ambiguity policy")
    path = os.path.join(db, "Biochemistry", "Aliases", "Unique_ModelSEED_Compound_Aliases.txt")
    candidates = {}
    with open(path) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["Source"] != "KEGG":
                continue
            kegg, seed = row["External ID"].strip(), row["ModelSEED ID"].strip()
            if KEGG_RE.fullmatch(kegg):
                candidates.setdefault(kegg, set()).add(seed)
    return {kegg: min(seeds) for kegg, seeds in candidates.items()
            if len(seeds) == 1 or ambiguity == "historical_first"}


def parse_equation(text):
    if "=" not in text:
        return Counter(), False
    left, right = text.split("=", 1)
    stoich = Counter()
    for side, sign in ((left, -1), (right, 1)):
        for term in side.split("+"):
            term = term.strip()
            if not term:
                continue
            m = re.fullmatch(r"(?:(\d+(?:\.\d+)?)\s+)?(?:kegg:)?(C\d{5})", term)
            if not m:
                return Counter(), False
            coefficient = float(m.group(1) or 1)
            if coefficient <= 0:
                return Counter(), False
            stoich[m.group(2)] += sign * coefficient
    return stoich, bool(stoich)


def load_modelseed_reactions(db):
    reactions = {}
    for path in sorted(glob.glob(os.path.join(db, "Biochemistry", "reaction_??.tsv"))):
        with open(path) as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                if row.get("is_obsolete", "0") not in ("0", "", "False"):
                    continue
                st = {}
                for part in (row.get("stoichiometry") or "").split(";"):
                    f = part.split(":")
                    if len(f) >= 2:
                        try:
                            st[f[1]] = st.get(f[1], 0.0) + float(f[0])
                        except ValueError:
                            pass
                if st:
                    reactions[row["id"]] = st
    return reactions


def signature(st, drop):
    return tuple(sorted((c, round(v, 6)) for c, v in st.items() if c not in drop and abs(v) > 1e-9))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


class Transformer:
    def __init__(self, ms_reactions):
        from equilibrator_api import ComponentContribution, Q_
        self.cc, self.Q_, self.ms = ComponentContribution(), Q_, ms_reactions
        self._rxn, self._memo = {}, {}

    def provenance(self):
        from importlib.metadata import version
        from pathlib import Path
        import hashlib
        database = Path(self.cc.ccache.engine.url.database)
        digest = hashlib.sha256()
        with database.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        return {'method': 'dissociation_transform_difference_only',
                'equilibrator_api': version('equilibrator-api'),
                'equilibrator_cache': version('equilibrator-cache'),
                'compound_database_sha256': digest.hexdigest()}

    def reaction(self, rid):
        if rid not in self._rxn:
            st = self.ms[rid]
            lhs = [f"{abs(v):g} seed:{c}" for c, v in st.items() if v < 0 and c != PROTON]
            rhs = [f"{abs(v):g} seed:{c}" for c, v in st.items() if v > 0 and c != PROTON]
            r = self.cc.parse_reaction_formula(" + ".join(lhs) + " = " + " + ".join(rhs))
            self._rxn[rid] = r if (r.is_balanced() and r.can_be_transformed()) else None
        return self._rxn[rid]

    def dgp(self, rid, p_h, ionic_strength, p_mg, T):
        key = (rid, p_h, ionic_strength, p_mg, T)
        if key not in self._memo:
            cc, Q_ = self.cc, self.Q_
            cc.p_h = Q_(p_h); cc.ionic_strength = Q_(f"{ionic_strength}M")
            cc.p_mg = Q_(p_mg); cc.temperature = Q_(f"{T}K")
            # Use only the dissociation/ionic-strength transform. No fitted
            # reaction formation energy or Component Contribution prediction enters.
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                value = self.reaction(rid).transform(p_h=cc.p_h, ionic_strength=cc.ionic_strength,
                                                    p_mg=cc.p_mg, temperature=cc.temperature)
            self._memo[key] = float(value.m_as("kJ/mol"))
        return self._memo[key]

    def correction(self, rid, p_h, ionic_strength, p_mg, T):
        """ΔG'°(standard) - ΔG'°(observed conditions), same T: the Legendre-transform difference."""
        return (self.dgp(rid, STD["p_h"], STD["ionic_strength"], STD["p_mg"], T)
                - self.dgp(rid, p_h, ionic_strength, p_mg, T))

