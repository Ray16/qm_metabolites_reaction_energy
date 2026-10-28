"""Runtime switch defaults and the legacy environment adapter (no QM imports)."""
import os

UMA_RUNTIME_PROFILE_DEFAULT = "nitro"


def uma_runtime_profile(value=None):
    """Resolve the declared calculator profile before model construction."""
    value = os.environ.get("UMA_RUNTIME_PROFILE", UMA_RUNTIME_PROFILE_DEFAULT) if value is None else value
    if not isinstance(value, str):
        raise ValueError("UMA_RUNTIME_PROFILE must be a string")
    value = value.strip().lower()
    if value not in {"nitro", "official_fp32", "legacy"}:
        raise ValueError("UMA_RUNTIME_PROFILE must be nitro, official_fp32 or legacy")
    return value


# One table of production switch defaults; uncertainty applicability is checked separately. Every
# switch is read through _flag(), so the default lives here only and effective_config() can report it.
FLAG_DEFAULTS = {
    # Protonation controls share this table with their runtime readers.
    "PKA_ENV": True, "FREE_PPI_PKA": True, "ANHYDRIDE_PKA": False,
    "CARBOXYL_PAIRS": False, "POLYACID_PKA": False,
    "ARYLAMINE_NONBASIC": True, "PH0_REDOX_PROTON": True,
    # Exact source ladders passed own/shared-cache retained-coverage checks.
    # Neutral ring-proton pooling failed the whole-panel gate; keep it experimental.
    "SOURCE_PKA_REGISTRY": True, "RING_PROTON_ENSEMBLE": False,
    "COFACTOR_RING": True, "COA_CORE": False,
    # NTP_CORE ADOPTED 2026-10-01: the nucleoside of a spectator NTP/NDP/NMP is capped to a methyl
    # polyphosphate (isodesmic, like COFACTOR_RING). TECRDB 10.35 -> 10.18 (adenylate kinase -26 -> +4,
    # nucleoside-diphosphate kinase exactly isodesmic).
    "NTP_CORE": True,
    "AUTO_TRUNCATE": True, "ROUTE_FULL": True, "TRUNC_V2": False, "TRUNC_VALIDATE": False,
    "PH0_AUTO": True, "PH0_BASES": True, "ZWITTERION_PH0": True, "NEUTRAL_QM": False,
    # PH0_ISOMERASE ADOPTED 2026-10-01: isomerizations also take the pH-0 route. The gate dated from the
    # COSMO baseline; under ALPB charged sugar-phosphate ring isomers do not cancel (G6P isomerase -25,
    # mannose-6-P isomerase -34) while their neutral forms do: isomerase MAE 8.4 -> 5.6, TECRDB 10.08 -> 9.64.
    "PH0_ISOMERASE": True,
    "CARBONYL_HYDRATION_ALL": True,    # every aldehyde/ketone except alpha-keto acids (validated vs K_hyd, ALPB)
    # RING_CHAIN NOT ADOPTED 2026-10-06 (routing/ring_chain.py; off): keto/aldehyde + intramolecular amide N-H ->
    # fold the 5/6-ring hydroxy-lactam forms into the species G and keep such species out of truncation. The
    # neutral QM ring closure of 2-oxoglutaramate is right (-13.3 vs exp -14.4 kJ), but on the pH-0 route the
    # ring's alpha-hydroxy-alpha-amido carboxyl gets the lactate-class pKa 3.8 vs 1.8 for the open alpha-keto
    # acid, cancelling the ring stabilisation (rxn00415 err +14.9 -> +16.9). No measured pKa exists for the
    # ring acid and the result swings ~10 kJ over plausible values (2.2-3.8): adopting needs that pKa first.
    "RING_CHAIN": False,
    "HYDRATION_CAL": True,             # K_hyd-calibrated ΔG_hyd (independent data; only with CARBONYL_HYDRATION_ALL)
    # 2026-10-01 physics revision (analysis/sweep_20261001/NOTES.md): ALPB solvation + the experimental
    # liquid-water reference replace the COSMO-era compensations. The class anchors and the hydro-lyase
    # water patch were absorbing COSMO's missing H-bond term; with ALPB + exact water they are not used
    # (TECRDB 364: COSMO+anchors MAE 12.01 -> ALPB, no anchors 10.73 -> + hydration 10.35).
    # ZWITTERION_PH0: zwitterions are not gas-phase minima (proton transfer on relaxation) -> neutral route.
    "STD_STATE_1M": True, "ALDEHYDE_HYDRATION": True, "ANCHOR_CORRECT": False, "SMD_SOLV": False,
    "WATER_REF_HYDROLYASE": False, "WATER_REF_EXP": True,
    # UNDER A/B (default = the validated old behaviour; flip only after the subset A/B shows an improvement):
    "TRUNC_SPECTATOR_CATIONS": False,   # a cation removed WITH the spectator is not a "mangled" cation
    "TRUNC_MAXANION_RETRY": False,      # retry truncation on max-anion forms (protonation-consistent spectators)
    # ADOPTED 2026-09-30 (correctness): the old rule cut C-O/P-O bonds and H-capped them, producing invalid
    # cores (aspartate->isopropylamine, phosphoester->free H3PO4, P(III)); this cuts only single C-C bonds at
    # an sp3 kept carbon. A/B on its affected set: dG_raw MAE 13.1->12.5, RMS 17.1->15.3; where it looks worse
    # it exposes a real error the broken core had cancelled by luck. Set 0 only to reproduce the old cores.
    "TRUNC_FG_CUTS": True,
    # ADOPTED 2026-09-30: radius 2 is not a converged local model for reactions at an anomeric
    # sugar carbon: it can amputate ribose substituents (including a conserved 5'-phosphate)
    # that remain electrostatically coupled to the reactive center. Radius 3 was identical to
    # radius 4/full on the completed validation panel; severe MAE 47.11->29.64 (n=6; 5 improved,
    # 0 worsened), while four controls were unchanged. Set 0 only to reproduce radius-2 results.
    "TRUNC_ANOMERIC_RADIUS": True,
}


def flag(name):
    """Pipeline switch: env NAME (0/off/false/no disables) overriding FLAG_DEFAULTS[name]. The defaults are
    the production configuration; set NAME=0/1 only for an ablation/baseline run."""
    v = os.environ.get(name)
    if v is None:
        return FLAG_DEFAULTS[name]
    return v.strip().lower() not in ("", "0", "off", "false", "no")

