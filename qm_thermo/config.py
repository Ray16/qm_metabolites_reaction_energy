"""Central configuration for the QM thermodynamics pipeline.

All tunable parameters (paths, physical conditions, QM levels, parallel layout)
live here so the rest of the package contains no hard-coded constants. Values can
be overridden via environment variables where it is useful for cluster/HPC runs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
THERMO_DIR = os.path.dirname(PKG_DIR)                       # thermodynamic_calc/
PROJECT_DIR = os.path.dirname(THERMO_DIR)                   # repo root
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")

# Inputs / outputs
CENTRAL_METABOLITES_JSON = os.path.join(
    THERMO_DIR, "central_metabolites_in_opentecr.json"
)
RESULTS_DIR = os.path.join(THERMO_DIR, "results")           # cached per-compound G
COMPOUND_CACHE_DIR = os.path.join(RESULTS_DIR, "compounds")
REACTION_RESULTS_DIR = os.path.join(RESULTS_DIR, "reactions")
BENCHMARK_DIR = os.path.join(RESULTS_DIR, "benchmark")
# Persisted tier-1 optimised geometries + per-conformer energy records, so a
# higher-level single point can be added later without re-optimising.
GEOMETRY_DIR = os.path.join(RESULTS_DIR, "geometries")
CONFORMER_DIR = os.path.join(RESULTS_DIR, "conformers")
# Append-only per-ORCA-job timing log (one row per job: OPT+FREQ, SP, ...).
TIMINGS_CSV = os.path.join(RESULTS_DIR, "timings.csv")

# Reference tables from the ModelSEED submodule (existing group-contribution methods)
MSEED_THERMO_DIR = os.path.join(
    PROJECT_DIR, "ModelSEEDDatabase", "Biochemistry", "Thermodynamics"
)
EQUILIBRATOR_RXN_TBL = os.path.join(
    MSEED_THERMO_DIR, "eQuilibrator", "MetaNetX_Reaction_Energies.tbl"
)
EQUILIBRATOR_CPD_TBL = os.path.join(
    MSEED_THERMO_DIR, "eQuilibrator", "MetaNetX_Compound_Energies.tbl"
)


# ---------------------------------------------------------------------------
# External executables
# ---------------------------------------------------------------------------
# ORCA: full path is required (ORCA uses argv[0] to locate its MPI helpers).
ORCA_ROOT = os.environ.get(
    "ORCA_ROOT",
    "/nfs/lambda_stor_01/homes/rzhu/orca_6_1_1_linux_x86-64_shared_openmpi418_nodmrg",
)
ORCA_BIN = os.path.join(ORCA_ROOT, "orca")

# OpenMPI 4.1.8 (must match the ORCA build) installed locally from source.
OPENMPI_ROOT = os.environ.get(
    "OPENMPI_ROOT", "/nfs/lambda_stor_01/homes/rzhu/openmpi-4.1.8-install"
)

# xtb for the cheap conformer-screening tier. Resolved from the `xtb` conda env.
XTB_BIN = os.environ.get(
    "XTB_BIN", "/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/xtb/bin/xtb"
)

# Scratch must be LOCAL disk: the NFS home is ~full and QM scratch is large/IO-heavy.
SCRATCH_ROOT = os.environ.get("QM_SCRATCH", "/tmp/qm_thermo_scratch")


# ---------------------------------------------------------------------------
# Physical conditions (match openTECR / eQuilibrator conventions)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Conditions:
    """Aqueous standard conditions for the transformed Gibbs energy."""

    temperature_K: float = 298.15
    pH: float = 7.0
    ionic_strength_M: float = 0.25      # eQuilibrator default; openTECR varies
    pMg: float = 3.0                    # used only if Mg2+ speciation is enabled

    # Aqueous-proton reference free energy on the QM total-energy scale, kJ/mol.
    # ModelSEED reactions balance H with explicit H+, which we fold into a
    # per-hydrogen reference so balanced reactions cancel exactly. The theoretical
    # value is G_gas(H+) + dG_solv(H+) ~ -1122 kJ/mol; it carries a ~10-20 kJ/mol
    # ambiguity (proton solvation / SMD reference), so it is calibrated against
    # experimental openTECR data -- see benchmark.calibrate_proton_reference.
    proton_reference_kJ: float = -1122.8

    # Standard-state conversion for the gas-phase term, kJ/mol per species.
    # xtb's --ohess RRHO gives G_gas at 1 atm (its translational entropy matches
    # Sackur-Tetrode at 1 atm exactly), but xtb's dGsolv reference state is
    # "1 M gas/solution". Composing them without RT*ln(24.46) leaves every
    # species short by this amount. It cancels only when a reaction conserves
    # solute count, so it hides in isomerisations and bites every hydrolysis,
    # condensation or redox step with dn != 0.
    gas_1atm_to_1M_kJ: float = 7.93

    # Physical constants
    R_kJ: float = 8.314462618e-3        # kJ / mol / K
    F_kJ_per_V: float = 96.48533212     # Faraday constant, kJ / mol / V


DEFAULT_CONDITIONS = Conditions()


# ---------------------------------------------------------------------------
# Conformer search
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ConformerSettings:
    n_confs: int = 30                   # RDKit ETKDG seeds
    rmsd_prune: float = 0.5             # Angstrom, post-embedding dedup
    mmff_max_iters: int = 2000
    # After xtb optimisation, keep conformers within this window of the minimum
    # and at most `max_qm_confs` of them go on to DFT.
    energy_window_kJ: float = 12.0
    # Default 1: on the 83-metabolite benchmark, Boltzmann-averaging up to 3
    # conformers shifted the reaction MAE by only 0.10 kJ/mol (the per-compound
    # ensemble lowering averaged -0.84 kJ/mol and largely cancels across balanced
    # reactions) while ~1.8x-ing total DFT cost. Raise this for genuinely flexible
    # species (sugar-phosphates, CoA, nucleotides) where conformational entropy
    # changes between reactant/product sides -- but also raise the ETKDG seed
    # count then, since 30 seeds undersamples those.
    max_qm_confs: int = 1
    random_seed: int = 0xF00D


DEFAULT_CONFORMERS = ConformerSettings()


# ---------------------------------------------------------------------------
# QM levels of theory
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QMLevel:
    """A single ORCA level-of-theory specification.

    The pipeline runs, per conformer:
      1. geometry optimisation + frequencies at `opt_keywords` (gas or implicit),
      2. an optional higher single point at `sp_keywords`,
    both in SMD water unless overridden.
    """

    # Tier 1: geometry + frequencies (thermal/entropy) at an affordable composite.
    opt_keywords: str = "r2SCAN-3c TightSCF OPT FREQ"
    # Tier 2: higher-level single-point electronic energy on the tier-1 geometry.
    # wB97M-V (range-separated hybrid meta-GGA + VV10) / def2-TZVPD is among the
    # best reaction-energy functionals and should fix the O2/redox failures of the
    # meta-GGA tier. Set to None to skip the second tier.
    sp_keywords: str | None = "wB97M-V def2-TZVPD def2/J RIJCOSX VeryTightSCF"
    solvent: str = "water"              # SMD solvent; "" disables solvation
    use_smd: bool = True                # SMD (True) vs CPCM (False)


DEFAULT_QM_LEVEL = QMLevel()


# ---------------------------------------------------------------------------
# Parallel execution layout
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParallelSettings:
    # Cores per ORCA job. User requirement: >= 16.
    orca_nprocs: int = 16
    # Concurrent ORCA jobs. 80-core node => up to 5 jobs * 16 cores = full node.
    max_concurrent_jobs: int = 5
    # Max memory per core hint passed to ORCA (%maxcore, in MB).
    maxcore_mb: int = 3000
    # Threads for the cheap xtb screening stage.
    xtb_threads: int = 8


DEFAULT_PARALLEL = ParallelSettings()


def ensure_dirs() -> None:
    """Create all output/scratch directories (idempotent)."""
    for d in (
        RESULTS_DIR,
        COMPOUND_CACHE_DIR,
        REACTION_RESULTS_DIR,
        BENCHMARK_DIR,
        GEOMETRY_DIR,
        CONFORMER_DIR,
        FIGURES_DIR,
        SCRATCH_ROOT,
    ):
        os.makedirs(d, exist_ok=True)
