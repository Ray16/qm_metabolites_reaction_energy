#!/usr/bin/env python
"""One reaction per ROW: the localized reaction scheme (A + B <=> C + D) on the left, its
experiment / dGPredictor / pipeline bars on the right, same row. 8 unique reactions (the 2 reversal
duplicates dropped). Localized cores = what the pipeline computes. dpi 300, no baked-in captions.
"""
from __future__ import annotations
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D

HERE = os.path.dirname(os.path.abspath(__file__))
THERMO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(THERMO, "experiments", "qm_mlip_solvation", "scripts"))
from cofactor_truncate import cofactor_ring
from truncate import mcs_atom_map, reaction_center

HL = (0.95, 0.75, 0.12)                                     # gold highlight for the transformation

# Curated transformation motifs: the specific bond/atoms that form or break in each reaction TYPE.
# Reliable (no fragile MCS): a small SMARTS per motif marks exactly the reacting center, symmetric on
# both sides. Order matters only for which fires; a molecule may match several.
_MOTIFS = [
    "[#16X2H1,#16X1H0-]",                    # thiol / thiolate S  (redox S-H)
    "[#16X2]-[#16X2]",                       # disulfide S-S
    "[C,c]([#7X3,n])=,:[C,c]",               # nicotinamide C4 / dihydro C=C-N region (approx redox ring)
    "[CX4;R]([OX2;R])[OX2][#6]",             # anomeric C with ring-O and exocyclic O-glycoside
    "[CX4;R]([OX2;R])[NX3,n]",               # anomeric C with ring-O and N-glycoside
    "[PX4](=O)([OX2])[OX2][PX4]",            # phosphoanhydride P-O-P
    "[#16X2][CX3]=O",                        # thioester S-C(=O)
]
_MOTIF_MOLS = [Chem.MolFromSmarts(s) for s in _MOTIFS]


def _changed(mol, partners=None):
    """Atoms of the reacting center in `mol`, matched by curated transformation SMARTS (reliable +
    symmetric), NOT by fragile pairwise MCS. `partners` kept for signature compatibility."""
    hit = set()
    for pat in _MOTIF_MOLS:
        if pat is None:
            continue
        for m in mol.GetSubstructMatches(pat):
            hit.update(m)
    return hit

DECK = os.path.join(THERMO, "experiments", "qm_mlip_solvation", "figures", "deck_top10_comparison.png")
RXNS = [("rxn00086", "redox"), ("rxn00070", "redox"),
        ("rxn00605", "glycosyl"), ("rxn01713", "glycosyl"),
        ("rxn01834", "glyoxalase"), ("rxn00579", "glycosyl"),
        ("rxn01675", "nucleotidyl"), ("rxn01005", "nucleotidyl")]
RXN_DB = json.load(open(os.path.join(THERMO, "experiments", "qm_mlip_solvation",
                                     "scripts", "reactions_tecrdb_all.json")))
H = 300                                                     # common scheme-element height (px)


def _trim(im):
    from PIL import ImageChops
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    diff = ImageChops.difference(Image.alpha_composite(bg, im).convert("RGB"),
                                 Image.new("RGB", im.size, (255, 255, 255))).getbbox()
    return im.crop(diff) if diff else im


def _mol(m, highlight=None):
    if isinstance(m, str):
        m = Chem.MolFromSmiles(m)
    if m is None:
        return Image.new("RGBA", (H, H), (255, 255, 255, 0))
    AllChem.Compute2DCoords(m)
    hl = list(highlight or [])
    hb = [b.GetIdx() for b in m.GetBonds()
          if b.GetBeginAtomIdx() in hl and b.GetEndAtomIdx() in hl]
    d = rdMolDraw2D.MolDraw2DCairo(560, 460)
    o = d.drawOptions()
    o.bondLineWidth = 3; o.minFontSize = 26; o.maxFontSize = 34; o.padding = 0.06
    rdMolDraw2D.PrepareAndDrawMolecule(d, m, highlightAtoms=hl, highlightBonds=hb,
                                       highlightAtomColors={a: HL for a in hl},
                                       highlightBondColors={b: HL for b in hb})
    d.FinishDrawing()
    img = _trim(Image.open(BytesIO(d.GetDrawingText())).convert("RGBA"))
    w = max(1, int(img.width * H / img.height))
    return img.resize((w, H), Image.LANCZOS)


def _plus():
    im = Image.new("RGBA", (70, H), (255, 255, 255, 0)); d = ImageDraw.Draw(im)
    c = H // 2
    d.line([(20, c), (50, c)], fill=(40, 40, 40, 255), width=6)
    d.line([(35, c - 15), (35, c + 15)], fill=(40, 40, 40, 255), width=6)
    return im


def _arrow():
    im = Image.new("RGBA", (120, H), (255, 255, 255, 0)); d = ImageDraw.Draw(im)
    c = H // 2
    d.line([(20, c - 9), (100, c - 9)], fill=(40, 40, 40, 255), width=5)      # top -> right
    d.polygon([(100, c - 17), (100, c - 1), (112, c - 9)], fill=(40, 40, 40, 255))
    d.line([(20, c + 9), (100, c + 9)], fill=(40, 40, 40, 255), width=5)      # bottom <- left
    d.polygon([(20, c + 1), (20, c + 17), (8, c + 9)], fill=(40, 40, 40, 255))
    return im


def scheme_row(rid):
    sp = cofactor_ring({n: tuple(v) for n, v in RXN_DB[rid]["species"].items()})
    react = [s for c, q, s in sp.values() if c < 0 for _ in range(abs(int(c)))]
    prod = [s for c, q, s in sp.values() if c > 0 for _ in range(abs(int(c)))]
    rmol = [Chem.MolFromSmiles(s) for s in react]
    pmol = [Chem.MolFromSmiles(s) for s in prod]
    rhl = [_changed(m, [p for p in pmol if p]) if m else set() for m in rmol]   # transformation atoms
    phl = [_changed(m, [r for r in rmol if r]) if m else set() for m in pmol]
    parts = []
    for k, m in enumerate(rmol):
        if k: parts.append(_plus())
        parts.append(_mol(m, rhl[k]))
    parts.append(_arrow())
    for k, m in enumerate(pmol):
        if k: parts.append(_plus())
        parts.append(_mol(m, phl[k]))
    W = sum(p.width for p in parts)
    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 0)); x = 0
    for p in parts:
        canvas.paste(p, (x, 0), p); x += p.width
    return canvas


def main():
    exp = {r: v["dG_kJ"] for r, v in json.load(open(f"{HERE}/tecrdb_full_experiment.json")).items()}
    rtr = json.load(open(f"{THERMO}/results/eq/dgpredictor_retrained_full.json"))
    cur = {k: v for k, v in json.load(open(f"{HERE}/current_pipeline_top10.json")).items()
           if not k.startswith("_")}
    ids = [r for r, _ in RXNS]
    E = np.array([exp[r] for r in ids])
    # top-to-bottom within each row group: experiment, UMA pipeline, dGPredictor
    series = [("TECRDB (experiment)", E, "#4C4C4C"),
              ("UMA pipeline (pH-0 + cofactor cores + truncation)", np.array([cur[r] for r in ids]), "#2A9D8F"),
              ("dGPredictor (retrained-ModelSEED)", np.array([rtr[r]["dG_kJ"] for r in ids]), "#D1495B")]

    n = len(ids)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(18, 10), width_ratios=[1.08, 1.0])
    yrow = np.arange(n)[::-1]                               # first reaction at top
    h = 0.24
    for i, (label, vals, color) in enumerate(series):
        off = (1 - i) * h                                  # i=0 exp -> +h (top), dGP -> -h (bottom)
        leg = label if label.startswith("TECRDB") else f"{label}  (MAE {np.mean(np.abs(vals - E)):.0f})"
        axR.barh(yrow + off, vals, h, color=color, edgecolor="white", linewidth=0.5, label=leg)
    axR.axvline(0, color="black", lw=0.8)
    axR.set_yticks([]); axR.set_ylim(-0.6, n - 0.4)
    xmin = min(min(v) for _, v, _ in series); xmax = max(max(v) for _, v, _ in series)
    axR.set_xlim(xmin - 6, xmax + 6)                       # tight padding -> less gap to the schemes
    axR.set_xlabel(r"$\Delta_r G'^{\circ}$ (kJ/mol)")
    axR.spines[["top", "right", "left"]].set_visible(False)
    axR.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), fontsize=10, ncol=1)

    axL.set_xlim(0, 1); axL.set_ylim(-0.6, n - 0.4); axL.axis("off")
    for i, (rid, cat) in enumerate(RXNS):
        y = yrow[i]
        im = scheme_row(rid)
        zoom = min(0.34, 980.0 / im.width * 0.34)          # fit width; consistent, legible height
        ab = AnnotationBbox(OffsetImage(im, zoom=zoom), (0.0, y - 0.02),
                            frameon=False, xycoords=("axes fraction", "data"), box_alignment=(0.0, 0.5))
        axL.add_artist(ab)
        axL.text(0.0, y + 0.40, f"{rid}  ({cat})", fontsize=10, color="#555", va="bottom", ha="left")
    fig.subplots_adjust(wspace=0.0, left=0.01, right=0.995, top=0.93, bottom=0.06)
    os.makedirs(os.path.dirname(DECK), exist_ok=True)
    fig.savefig(DECK, dpi=300, bbox_inches="tight")
    print(f"wrote {DECK}")


if __name__ == "__main__":
    main()
