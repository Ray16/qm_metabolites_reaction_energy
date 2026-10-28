"""Figure-led project talk: render preserved evidence, without new calculations."""
from pathlib import Path
import csv
import hashlib
import json
import re
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
MANIFEST = Path(__file__).resolve().parent / "presentation_results.json"
TEAL, INK, GRAY, PURPLE = "#087F8C", "#173640", "#899AA1", "#9467A6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 19,
                     "axes.labelsize": 19, "axes.titlesize": 21,
                     "xtick.labelsize": 18, "ytick.labelsize": 18,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "text.color": INK, "axes.labelcolor": INK})


def save(fig, name):
    fig.savefig(OUT / name, dpi=250, bbox_inches="tight", pad_inches=.18,
                facecolor="white")
    plt.close(fig)


def main():
    manifest = json.loads(MANIFEST.read_text())
    def source(key, field="source"):
        return ROOT / manifest[key][field]
    def figure(key):
        return manifest["figures"][key]
    def save_result(fig, key):
        if key in manifest["render_figures"]:
            save(fig, figure(key))
        else:
            plt.close(fig)
            if not (OUT / figure(key)).is_file():
                raise FileNotFoundError(f"Replacement figure is missing: {figure(key)}")
    def read_tsv(path):
        with path.open() as stream:
            return list(csv.DictReader(stream, delimiter="\t"))
    macros = {}
    def number(name, value, digits=None):
        macros[name] = (f"{value:.{digits}f}" if digits is not None
                        else f"{int(value):,}".replace(",", "{,}"))
    def escape(text):
        return re.sub(r"[\\&%$#_{}~^]", lambda m: {
            "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
            "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
            "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}[m[0]], text)
    for key, name in [("accuracy", "AccuracyNote"), ("comparison", "ComparisonNote"),
                      ("matched", "MatchedNote"), ("applicability", "ApplicabilityNote"),
                      ("components", "ComponentNote")]:
        macros[name] = escape(manifest[key]["note"])
        if "snapshot" in manifest[key]:
            macros[name.replace("Note", "Snapshot")] = escape(manifest[key]["snapshot"])
        macros[name.replace("Note", "Source")] = r"\path{" + manifest[key]["source"] + "}"
    macros.update(manifest["figures"])

    implementation = json.loads(source("implementation").read_text())
    test_log = source("implementation", "tests_source").read_text()
    passed = re.search(r"(\d+) passed, (\d+) warnings", test_log)
    if passed is None or not implementation["full_suite"].startswith(passed[1] + " passed"):
        raise ValueError("Implementation status and test record disagree")
    number("ImplementationTests", int(passed[1]))
    number("ProtectedArtifacts", implementation["protected_artifacts_verified"])
    if implementation["overall_accuracy_improvement_established"] or implementation["publication_ready"]:
        raise ValueError("Review slide claims before adopting new accuracy/publication status")
    for relative, expected in implementation["track_artifacts_sha256"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Evidence changed since integration: {relative}")
    network = json.loads(source("network").read_text())["release_20261001c"]["summary"]
    number("NetworkCycles", network["n_independent_cycles_audited"])
    number("NetworkViolations", network["n_cycle_basis_violations"])
    aqueous = json.loads(source("aqueous_ensemble").read_text())
    keto = aqueous["complete_vertical_common_sets"]["ip_keto_anion"]["models"]
    number("SourceKetoCoverage", 100*keto["UMA_ALPB"]["selected_fraction_in_whole_expected_set"], 2)
    number("TargetKetoCoverage", 100*keto["DFT_SMD"]["selected_fraction_in_whole_expected_set"], 2)
    if aqueous["normal_aqueous_deltaG"] is not None or aqueous["validated_CI"] is not None:
        raise ValueError("Review ensemble slide before adopting new aqueous results")

    records = [json.loads(p.read_text()) for p in sorted(source("accuracy").glob("*.json"))]
    valid = [r for r in records if r.get("dG") is not None and r.get("exp") is not None]
    if not valid:
        raise ValueError("No scored accuracy records with reference energies")
    x = np.array([r["exp"] for r in valid], dtype=float)
    y = np.array([r["dG"] for r in valid], dtype=float)
    if not np.isfinite(np.r_[x, y]).all():
        raise ValueError("Nonfinite accuracy energy")
    number("AccuracyN", len(records)); number("AccuracyScored", len(valid))
    number("AccuracyMAE", np.abs(y-x).mean(), 2)
    number("AccuracyMedian", np.median(np.abs(y-x)), 2)
    number("AccuracyRMSE", np.sqrt(np.mean((y-x)**2)), 2)
    fig, ax = plt.subplots(figsize=(5.8, 5.1), layout="constrained")
    low, high = min(x.min(), y.min())-5, max(x.max(), y.max())+5
    ax.plot([low, high], [low, high], "--", color=GRAY, lw=1.2)
    ax.scatter(x, y, s=36, color=TEAL, alpha=.66, edgecolors="none")
    ax.set(xlim=(low, high), ylim=(low, high),
           xlabel="Reference (kJ/mol)", ylabel="MetaG (kJ/mol)")
    save_result(fig, "AccuracyFigure")

    comparison = read_tsv(source("comparison"))
    wanted = [("MetaG_cached_FD", "MetaG", TEAL),
              ("dGbyG_eQ_bridge", "dGbyG + bridge", PURPLE),
              ("dGPredictor_published", "dGPredictor", "#0072B2"),
              ("eQuilibrator_CC", "eQuilibrator CC", "#CF8A22"),
              ("eQuilibrator_GC_only", "GC-only", GRAY)]
    selected = [next(r for r in comparison if r["method"] == method)
                for method, _, _ in wanted]
    if len({r["n_common"] for r in selected}) != 1 or len({r["n_input"] for r in selected}) != 1:
        raise ValueError("Shared comparison cohorts differ")
    number("SharedN", int(selected[0]["n_common"]))
    number("SharedMAE", float(selected[0]["MAE_archived_common_kJ"]), 2)
    strict = read_tsv(source("comparison", "strict_source"))
    strict_metag = next(r for r in strict if r["method"] == "MetaG_cached_FD" and r["subset"] == "common_set")
    number("StrictMAE", float(strict_metag["MAE_kJ"]), 2)
    fig, axes = plt.subplots(1, 2, figsize=(12.7, 3.6), layout="constrained",
                             gridspec_kw={"wspace": .1})
    labels = [label for _, label, _ in wanted]
    colors = [c for _, _, c in wanted]
    errors = [float(r["MAE_archived_common_kJ"]) for r in selected]
    counts = [int(r["n_predicted"]) for r in selected]
    n_input = int(selected[0]["n_input"])
    for ax, values, limit, title, xlabel in [
            (axes[0], errors, max(errors)*1.22,
             f"Shared targets (n = {selected[0]['n_common']})", "MAE (kJ/mol)"),
            (axes[1], counts, n_input*1.20,
             f"All inputs (n = {n_input})", "Predictions returned")]:
        ax.barh(range(len(values)), values, color=colors, height=.6)
        ax.set_yticks(range(len(values)), labels)
        ax.invert_yaxis(); ax.set_xlim(0, limit)
        ax.set_title(title, pad=16); ax.set_xlabel(xlabel)
        for i, v in enumerate(values):
            ax.text(v+limit*.018, i, f"{v:.2f}" if ax is axes[0] else f"{v}/{n_input}", va="center", fontsize=18)
    save_result(fig, "ComparisonFigure")

    # Reconstruct the exact common cohort behind the comparison bars. Archived
    # labels are preserved; strict/current references describe a different set.
    prediction_rows = read_tsv(source("comparison", "predictions_source"))
    archived = json.loads(source("comparison", "references_source").read_text())
    references = {rid: float(statistics.median(rx["exp"]))
                  for rid, rx in archived.items() if rx.get("exp")}
    method_rows, valid_ids = {}, {}
    declared_conditions = set()
    for method, _, _ in wanted:
        method_rows[method] = {}
        for row in prediction_rows:
            if row["method"] != method:
                continue
            rid = row["reaction"]
            if rid in method_rows[method]:
                raise ValueError(f"Duplicate prediction: {method}/{rid}")
            method_rows[method][rid] = row
            declared_conditions.add(tuple(float(row[k]) for k in ("T_K", "pH", "I_M", "pMg")))
        valid_ids[method] = {rid for rid, row in method_rows[method].items()
                             if row["status"] == "ok" and np.isfinite(float(row["dG_kJ"]))}
    if declared_conditions != {(298.15, 7.0, 0.0, 14.0)}:
        raise ValueError("Archived prediction conditions differ; review comparison")
    panels = [set(rows) for rows in method_rows.values()]
    if not all(panel == panels[0] for panel in panels) or len(panels[0]) != n_input:
        raise ValueError("Prediction input panels disagree with comparison metrics")
    shared_ids = sorted(set(references).intersection(*valid_ids.values()))
    if len(shared_ids) != int(selected[0]["n_common"]):
        raise ValueError("Scatter and bar comparison cohorts differ")
    shared_ref = np.array([references[rid] for rid in shared_ids])
    shared_predictions, scatter_metrics = {}, {}
    for (method, label, color), metric in zip(wanted, selected):
        prediction = np.array([float(method_rows[method][rid]["dG_kJ"]) for rid in shared_ids])
        residual = prediction - shared_ref
        mae = float(np.abs(residual).mean())
        if (not np.isfinite(shared_ref).all()
                or not np.isclose(mae, float(metric["MAE_archived_common_kJ"]), rtol=0, atol=1e-10)
                or len(valid_ids[method]) != int(metric["n_predicted"])):
            raise ValueError(f"Scatter metrics disagree with archived summary: {method}")
        shared_predictions[method] = prediction
        scatter_metrics[method] = {"label": label, "n": len(shared_ids), "MAE_kJ": mae,
                                   "RMSE_kJ": float(np.sqrt(np.mean(residual**2)))}
    overlap = sum(method_rows["dGbyG_eQ_bridge"][rid]["training_overlap"] == "exact_normalized_reaction"
                  for rid in shared_ids)
    number("SharedTrainingOverlap", overlap)
    low = min(shared_ref.min(), *(p.min() for p in shared_predictions.values()))
    high = max(shared_ref.max(), *(p.max() for p in shared_predictions.values()))
    limits = (np.floor(low/10)*10-5, np.ceil(high/10)*10+5)
    for indices, figure_key in [([0, 1, 2], "LearnedScatterFigure"),
                                ([0, 3, 4], "GroupScatterFigure")]:
        fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.3), layout="constrained")
        for ax, index in zip(axes, indices):
            method, label, color = wanted[index]
            ax.plot(limits, limits, "--", color=GRAY, lw=1.2, zorder=0)
            ax.scatter(shared_ref, shared_predictions[method], s=22, color=color,
                       alpha=.62, edgecolors="none", rasterized=True)
            ax.set(xlim=limits, ylim=limits, aspect="equal",
                   xlabel="Archived reference (kJ/mol)")
            ax.set_title("MetaG (FD replay)" if index == 0 else label, fontsize=18, pad=12)
            ax.tick_params(labelsize=14)
            ax.xaxis.label.set_size(15)
            m = scatter_metrics[method]
            ax.text(.045, .965, f"MAE {m['MAE_kJ']:.2f}\nRMSE {m['RMSE_kJ']:.2f}",
                    transform=ax.transAxes, va="top", fontsize=14,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .86})
        axes[0].set_ylabel("Prediction (kJ/mol)", fontsize=15)
        save_result(fig, figure_key)
    scatter_report = {"scope": "Historical development/pretrained comparison against archived labels; not held-out accuracy",
                      "common_reactions": shared_ids, "n_common": len(shared_ids),
                      "dGbyG_shared_training_matches": overlap, "axis_limits_kJ": list(limits),
                      "methods": scatter_metrics,
                      "sources_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                         for path in [source("comparison"), source("comparison", "predictions_source"),
                                                      source("comparison", "references_source")]}}
    (OUT.parent / "presentation_scatter_metrics.json").write_text(json.dumps(scatter_report, indent=2) + "\n")

    with source("matched").open() as f:
        pairs = list(csv.DictReader(f, delimiter="\t"))
    ref = np.array([float(r["reference"]) for r in pairs])
    predictions = [np.array([float(r[k]) for r in pairs]) for k in ("metag", "dgbyg")]
    number("MatchedN", len(ref))
    number("MatchedMetaGMAE", np.abs(predictions[0]-ref).mean(), 2)
    number("MatchedBaselineMAE", np.abs(predictions[1]-ref).mean(), 2)
    low = min(ref.min(), *(p.min() for p in predictions)) - 6
    high = max(ref.max(), *(p.max() for p in predictions)) + 6
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 3.7), layout="constrained")
    for ax, pred, title, color in zip(axes, predictions,
            ["MetaG", "dGbyG (out-of-fold)"], [TEAL, PURPLE]):
        ax.plot([low, high], [low, high], "--", color=GRAY, lw=1.3)
        ax.scatter(ref, pred, s=40, color=color, alpha=.7, edgecolors="none")
        ax.set(xlim=(low, high), ylim=(low, high),
               xlabel="Reference (kJ/mol)", ylabel="Prediction (kJ/mol)")
        ax.set_title(title, pad=14)
        ax.text(.05, .94, f"MAE {np.abs(pred-ref).mean():.2f}",
                transform=ax.transAxes, va="top", fontsize=22,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9})
    save_result(fig, "MatchedFigure")

    audit = json.loads(source("components").read_text())
    methods = ["UMA_ALPB", "DFT_ALPB", "UMA_SMD", "DFT_SMD"]
    values = [audit["gaps_Z_enol_minus_keto"][m]["full_kj"] for m in methods]
    number("ElectronicChange", abs(values[1]-values[0]), 2)
    number("SolventChange", abs(values[3]-values[1]), 2)
    fig, ax = plt.subplots(figsize=(10.4, 2.65), layout="constrained")
    ax.axhline(0, color=GRAY, lw=1.2)
    ax.bar(range(4), values, width=.55, color=[GRAY, GRAY, TEAL, TEAL])
    for i, v in enumerate(values):
        ax.text(i, v + (.6 if v >= 0 else -.6), f"{v:+.2f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=22, fontweight="bold")
    ax.set_xticks(range(4), ["UMA\nALPB", "DFT\nALPB", "UMA\nSMD*", "DFT\nSMD"])
    ax.set(ylim=(-18, 15), ylabel="Z-enol − keto (kJ/mol)")
    ax.text(1.0, 11, "Keto lower", ha="center", fontsize=19, color=TEAL)
    ax.text(2.6, -12, "Z-enol lower", ha="center", fontsize=19, color=GRAY)
    save_result(fig, "ComponentFigure")

    landscape = json.loads(source("landscape").read_text())
    number("LiveReactions", landscape["n_live"])
    number("MissingEnergies", landscape["n_balanced_no_estimate"])
    scope = json.loads(source("structural_scope").read_text())
    number("UsableMissing", sum(r["eligible"] and not r["has_deltag"] for r in scope["modelseed"]))
    domain = json.loads(source("domain").read_text())
    values = [100*domain[k]["T4"]/sum(domain[k].values()) for k in
              ["openTECR panel (where errors are measured)", "ModelSEED balanced (all)"]]
    fig, ax = plt.subplots(figsize=(11.8, 3.0), layout="constrained")
    ax.barh([0, 1], values, color=[GRAY, TEAL], height=.5)
    ax.set_yticks([0, 1], ["openTECR benchmark", "Balanced ModelSEED"])
    ax.invert_yaxis(); ax.set_xlim(0, 60)
    ax.set_xlabel("Reactions containing a low-similarity unseen compound (%)")
    for i, v in enumerate(values):
        ax.text(v+1, i, f"{v:.1f}%", va="center", fontsize=24, fontweight="bold")
    save_result(fig, "DomainFigure")
    application = json.loads(source("applicability").read_text())
    number("ApplicationN", application["n_input"])
    number("ApplicationScored", application["n_estimates"])
    number("ApplicationFailed", application["n_failures"])
    if application["n_input"] != application["n_estimates"] + application["n_failures"]:
        raise ValueError("Applicability report has incomplete outcomes")
    tex = "% Generated from presentation_results.json and its local evidence.\n"
    tex += "".join(f"\\newcommand{{\\{name}}}{{{value}}}\n" for name, value in macros.items())
    (OUT.parent / "presentation_numbers.tex").write_text(tex)
    dependencies = []
    for key, config in manifest.items():
        if not isinstance(config, dict):
            continue
        for field, value in config.items():
            if field == "source" or field.endswith("_source"):
                path = ROOT / value
                dependencies.append("../" + value)
                if path.is_dir():
                    dependencies.extend("../" + str(p.relative_to(ROOT)) for p in sorted(path.glob("*.json")))
    (OUT.parent / "presentation_sources.mk").write_text(
        "# Generated evidence dependencies.\npresentation_numbers.tex: " + " ".join(dependencies) + "\n")
    print(f"Rendered {len(manifest['render_figures'])} selected figures and {len(macros)} result macros.")


if __name__ == "__main__":
    main()
