#!/usr/bin/env python
"""Grouped bar chart on the dGPredictor-disagreement reactions, with each reaction's LOCALIZED-core
scheme drawn above its bar (reactants over products -- two lines). 8 unique reactions (the 2 reversal
duplicates are dropped; they carry no new chemistry). Localized cores = what the pipeline computes
(cofactor cores shrink NAD/GSH); substrate reactions shown as-is. dpi 300, no baked-in captions.
"""
from __future__ import annotations
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

HERE = os.path.dirname(os.path.abspath(__file__))
THERMO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(THERMO, "experiments", "qm_mlip_solvation", "scripts"))
from cofactor_truncate import cofactor_ring

DECK = os.path.join(THERMO, "experiments", "qm_mlip_solvation", "figures", "deck_top10_comparison.png")

RXNS = [("rxn00086", "redox"), ("rxn00070", "redox"),
        ("rxn00605", "glycosyl"), ("rxn01713", "glycosyl"),
        ("rxn01834", "glyoxalase"), ("rxn00579", "glycosyl"),
        ("rxn01675", "nucleotidyl"), ("rxn01005", "nucleotidyl")]

RXN_DB = json.load(open(os.path.join(THERMO, "experiments", "qm_mlip_solvation",
                                     "scripts", "reactions_tecrdb_all.json")))


def _trim(im):
    """Crop transparent/white margins to reduce intra-scheme whitespace."""
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    diff = Image.alpha_composite(bg, im).convert("RGB")
    from PIL import ImageChops
    bbox = ImageChops.difference(diff, Image.new("RGB", im.size, (255, 255, 255))).getbbox()
    return im.crop(bbox) if bbox else im


def _row(smis, w_per=420, h=360):
    """Render a horizontal row of molecules (localized cores) at high resolution + clean options."""
    from rdkit.Chem import AllChem
    mols = []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            AllChem.Compute2DCoords(m)
            mols.append(m)
    if not mols:
        return Image.new("RGBA", (w_per, h), (255, 255, 255, 0))
    opts = Draw.rdMolDraw2D.MolDrawOptions()
    opts.bondLineWidth = 2
    opts.minFontSize = 22
    opts.maxFontSize = 30
    opts.padding = 0.06
    img = Draw.MolsToGridImage(mols, molsPerRow=len(mols), subImgSize=(w_per, h),
                               useSVG=False, drawOptions=opts).convert("RGBA")
    return _trim(img)


def scheme_image(rid):
    """Two-line localized scheme: reactant cores (top) over product cores (bottom)."""
    sp = {n: tuple(v) for n, v in RXN_DB[rid]["species"].items()}
    sp = cofactor_ring(sp)                                  # shrink NAD/GSH; no-op otherwise
    react = [s for c, q, s in sp.values() if c < 0 for _ in range(abs(int(c)))]
    prod = [s for c, q, s in sp.values() if c > 0 for _ in range(abs(int(c)))]
    top = _row(react); bot = _row(prod)
    W = max(top.width, bot.width)
    gap = 26
    canvas = Image.new("RGBA", (W, top.height + bot.height + gap), (255, 255, 255, 0))
    canvas.paste(top, ((W - top.width) // 2, 0), top)
    canvas.paste(bot, ((W - bot.width) // 2, top.height + gap), bot)
    # a downward arrow between the two lines
    from PIL import ImageDraw
    dr = ImageDraw.Draw(canvas)
    cx, y0, y1 = W // 2, top.height + 3, top.height + gap - 3
    dr.line([(cx, y0), (cx, y1)], fill=(60, 60, 60, 255), width=3)
    dr.polygon([(cx - 6, y1 - 7), (cx + 6, y1 - 7), (cx, y1)], fill=(60, 60, 60, 255))
    return canvas


def main():
    exp = {r: v["dG_kJ"] for r, v in json.load(open(f"{HERE}/tecrdb_full_experiment.json")).items()}
    rtr = json.load(open(f"{THERMO}/results/eq/dgpredictor_retrained_full.json"))
    cur = {k: v for k, v in json.load(open(f"{HERE}/current_pipeline_top10.json")).items()
           if not k.startswith("_")}
    ids = [r for r, _ in RXNS]
    E = np.array([exp[r] for r in ids])
    series = [
        ("TECRDB (experiment)", E, "#4C4C4C"),
        ("dGPredictor (retrained-ModelSEED)", np.array([rtr[r]["dG_kJ"] for r in ids]), "#D1495B"),
        ("UMA pipeline (pH-0 + cofactor cores + truncation)", np.array([cur[r] for r in ids]), "#2A9D8F"),
    ]

    fig, ax = plt.subplots(figsize=(16, 8.4))
    x = np.arange(len(ids)); width = 0.8 / len(series); off = (len(series) - 1) / 2.0
    for i, (label, vals, color) in enumerate(series):
        leg = label if label.startswith("TECRDB") else f"{label}   (subset MAE {np.mean(np.abs(vals - E)):.0f})"
        ax.bar(x + (i - off) * width, vals, width, label=leg, color=color, edgecolor="white", linewidth=0.5)
    # reserve an empty band at the bottom for the reaction schemes, so each scheme sits RIGHT
    # ABOVE the x-axis line (bars above the band; x-axis line + labels below the band)
    hi_bar = max(max(v) for _, v, _ in series)
    lo_bar = min(min(v) for _, v, _ in series)
    band_top = lo_bar - 10                                  # just below the deepest bar
    band_h = 78
    band_bot = band_top - band_h
    band_ctr = (band_top + band_bot) / 2
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylim(band_bot - 2, hi_bar + 14)
    ax.set_ylabel(r"$\Delta_r G'^{\circ}$ (kJ/mol)")
    ax.set_yticks([t for t in range(-50, 101, 25)])         # label the data range, not the band
    ax.set_xticks(x); ax.set_xticklabels([f"{r}\n{c}" for r, c in RXNS], fontsize=10)
    ax.set_xlim(-0.5, len(ids) - 0.5)
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["bottom"].set_position(("data", band_bot))    # x-axis line + labels BELOW the schemes
    ax.spines["left"].set_bounds(-50, 100)                  # don't extend the y-spine into the band
    # localized reaction scheme in the band, right above the x-axis line, under each column
    for xi, (rid, _) in enumerate(RXNS):
        im = scheme_image(rid)
        zoom = 0.16 if im.width < 720 else 0.11
        ab = AnnotationBbox(OffsetImage(im, zoom=zoom), (xi, band_ctr),
                            frameon=False, xycoords="data", box_alignment=(0.5, 0.5))
        ax.add_artist(ab)
    os.makedirs(os.path.dirname(DECK), exist_ok=True)
    fig.savefig(DECK, dpi=300, bbox_inches="tight")
    print(f"wrote {DECK}")


if __name__ == "__main__":
    main()
