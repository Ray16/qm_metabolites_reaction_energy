"""Quantum-chemistry backends.

`QMBackend` is the abstract interface the pipeline depends on. `ORCABackend` is the
production DFT backend (opt + freq + SMD water -> aqueous Gibbs energy). A thin
`XTBBackend` is provided for a fast tier / smoke tests. Keeping this behind one
interface is what makes the engine pluggable per the project plan.
"""

from __future__ import annotations

import abc
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass

from . import config
from .geometry import Geometry, read_xyz, write_xyz

HARTREE_TO_KJ = 2625.499639


@dataclass(frozen=True)
class QMResult:
    """Outcome of a single-conformer QM free-energy calculation (atomic units)."""

    gibbs_hartree: float          # final aqueous Gibbs free energy G(aq)
    electronic_hartree: float     # final single-point electronic energy (in solvent)
    enthalpy_hartree: float | None
    entropy_term_hartree: float | None   # T*S contribution
    geometry: Geometry            # optimised geometry
    method: str                   # level-of-theory label
    converged: bool

    @property
    def gibbs_kJ(self) -> float:
        return self.gibbs_hartree * HARTREE_TO_KJ


class QMError(RuntimeError):
    pass


class QMBackend(abc.ABC):
    """Compute an aqueous standard Gibbs free energy for one geometry."""

    @abc.abstractmethod
    def compute_gibbs(self, geom: Geometry, label: str, workdir: str) -> QMResult:
        ...


# ---------------------------------------------------------------------------
# ORCA backend
# ---------------------------------------------------------------------------
class ORCABackend(QMBackend):
    def __init__(
        self,
        level: config.QMLevel = config.DEFAULT_QM_LEVEL,
        parallel: config.ParallelSettings = config.DEFAULT_PARALLEL,
        orca_bin: str = config.ORCA_BIN,
        openmpi_root: str = config.OPENMPI_ROOT,
    ) -> None:
        self.level = level
        self.parallel = parallel
        self.orca_bin = orca_bin
        self.openmpi_root = openmpi_root
        # Per-job timing accumulated over this backend's lifetime. One ORCABackend
        # is created per compound (see compute._worker), so summing job_times gives
        # that compound's total ORCA wall time.
        self.job_times: list[dict] = []
        if not os.path.isfile(orca_bin):
            raise QMError(f"ORCA binary not found: {orca_bin}")

    @property
    def total_wall_s(self) -> float:
        return sum(j["wall_s"] for j in self.job_times)

    # -- environment -------------------------------------------------------
    def _env(self) -> dict:
        """ORCA needs its own libs and the matching OpenMPI on PATH/LD_LIBRARY_PATH."""
        env = dict(os.environ)
        orca_root = os.path.dirname(self.orca_bin)
        mpi_bin = os.path.join(self.openmpi_root, "bin")
        mpi_lib = os.path.join(self.openmpi_root, "lib")
        env["PATH"] = os.pathsep.join(
            [mpi_bin, orca_root, os.path.join(orca_root, "lib"), env.get("PATH", "")]
        )
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [mpi_lib, os.path.join(orca_root, "lib"), orca_root,
             env.get("LD_LIBRARY_PATH", "")]
        )
        # Let OpenMPI run as root/oversubscribe-tolerant on a shared node.
        env["OMPI_MCD_rmaps_base_oversubscribe"] = "1"
        return env

    # -- input generation --------------------------------------------------
    def _build_input(self, geom: Geometry, keywords: str) -> str:
        lines = [f"! {keywords}"]
        if self.parallel.orca_nprocs > 1:
            lines.append(f"%pal nprocs {self.parallel.orca_nprocs} end")
        lines.append(f"%maxcore {self.parallel.maxcore_mb}")
        if self.level.solvent:
            if self.level.use_smd:
                lines.append(
                    "%cpcm\n"
                    "  smd true\n"
                    f'  SMDsolvent "{self.level.solvent}"\n'
                    "end"
                )
            else:
                lines.append(f'%cpcm epsilon 80.4 refrac 1.33 end')
        lines.append(f"* xyz {geom.charge} {geom.multiplicity}")
        lines.append(geom.to_orca_coords())
        lines.append("*")
        return "\n".join(lines) + "\n"

    # -- run + parse -------------------------------------------------------
    def _run_orca(
        self, geom: Geometry, keywords: str, label: str, workdir: str
    ) -> str:
        """Write an ORCA input, run it, and return the output text.

        ORCA must be invoked by ABSOLUTE path for MPI startup to work.
        """
        os.makedirs(workdir, exist_ok=True)
        inp_path = os.path.join(workdir, f"{label}.inp")
        out_path = os.path.join(workdir, f"{label}.out")
        with open(inp_path, "w") as fh:
            fh.write(self._build_input(geom, keywords))
        t0 = time.monotonic()
        with open(out_path, "w") as out:
            proc = subprocess.run(
                [self.orca_bin, f"{label}.inp"],
                cwd=workdir, env=self._env(),
                stdout=out, stderr=subprocess.STDOUT,
            )
        wall_s = time.monotonic() - t0
        text = _read(out_path)

        # Record timing (wall clock + ORCA's own self-reported run time) and log it.
        orca_s = _parse_orca_runtime(text)
        ok = "ORCA TERMINATED NORMALLY" in text
        stage = _classify_stage(keywords)
        self.job_times.append({
            "label": label, "stage": stage, "wall_s": round(wall_s, 1),
            "orca_s": None if orca_s is None else round(orca_s, 1), "ok": ok,
        })
        orca_str = "   n/a" if orca_s is None else f"{orca_s:7.1f}s"
        print(
            f"[orca] {label:20s} {stage:12s} wall={wall_s:7.1f}s  "
            f"orca={orca_str}  {'OK' if ok else 'FAILED'}",
            flush=True,
        )

        if not ok:
            raise QMError(
                f"ORCA did not terminate normally for {label} "
                f"(rc={proc.returncode}); see {out_path}"
            )
        return text

    def compute_gibbs(self, geom: Geometry, label: str, workdir: str) -> QMResult:
        text = self._run_orca(geom, self.level.opt_keywords, label, workdir)
        return self._parse(text, geom, workdir, label)

    def optimize_geometry(
        self, geom: Geometry, keywords: str, label: str, workdir: str
    ) -> Geometry:
        """Geometry optimisation only (no frequencies); return the optimised geometry.

        `keywords` must request OPT. Used for geometry-sensitivity studies.
        """
        self._run_orca(geom, keywords, label, workdir)
        opt_xyz = os.path.join(workdir, f"{label}.xyz")
        if not os.path.isfile(opt_xyz):
            raise QMError(f"no optimised geometry written for {label}")
        return read_xyz(opt_xyz, geom.charge, geom.multiplicity)

    def single_point_energy(
        self, geom: Geometry, keywords: str, label: str, workdir: str
    ) -> float:
        """Higher-level single-point electronic energy (in solvent), in Hartree.

        Run on an already-optimised geometry to refine the electronic energy
        without repeating the (expensive) optimisation/frequency step.
        """
        text = self._run_orca(geom, keywords, label, workdir)
        energy = _grep_float(
            text, r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", last=True
        )
        if energy is None:
            raise QMError(f"no single-point energy in ORCA output for {label}")
        return energy

    def _parse(self, text: str, geom: Geometry, workdir: str, label: str) -> QMResult:
        gibbs = _grep_float(text, r"Final Gibbs free energy\s*\.*\s*(-?\d+\.\d+)")
        if gibbs is None:
            raise QMError(
                f"no Gibbs free energy in ORCA output for {label} "
                "(did FREQ run? imaginary modes?)"
            )
        electronic = _grep_float(
            text, r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", last=True
        )
        enthalpy = _grep_float(text, r"Total Enthalpy\s*\.*\s*(-?\d+\.\d+)")
        entropy = _grep_float(text, r"Final entropy term\s*\.*\s*(-?\d+\.\d+)")

        # Optimised geometry is written to <label>.xyz by ORCA.
        opt_xyz = os.path.join(workdir, f"{label}.xyz")
        opt_geom = (
            read_xyz(opt_xyz, geom.charge, geom.multiplicity)
            if os.path.isfile(opt_xyz)
            else geom
        )
        n_imag = len(re.findall(r"\*\*\*imaginary mode\*\*\*", text))
        return QMResult(
            gibbs_hartree=gibbs,
            electronic_hartree=electronic if electronic is not None else float("nan"),
            enthalpy_hartree=enthalpy,
            entropy_term_hartree=entropy,
            geometry=opt_geom,
            method=self.level.opt_keywords,
            converged=(n_imag == 0),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read(path: str) -> str:
    with open(path, errors="replace") as fh:
        return fh.read()


def _grep_float(text: str, pattern: str, *, last: bool = False) -> float | None:
    matches = re.findall(pattern, text)
    if not matches:
        return None
    return float(matches[-1] if last else matches[0])


def _classify_stage(keywords: str) -> str:
    """Label a job by its ORCA keywords for the timing log/CSV. The production
    tier-1 runs optimisation+frequencies as ONE job -> 'OPT+FREQ' (one number);
    the tier-2 refinement is 'SP'."""
    toks = {t.upper() for t in keywords.split()}
    opt, freq = "OPT" in toks, "FREQ" in toks
    if opt and freq:
        return "OPT+FREQ"
    if opt:
        return "OPT"
    if freq:
        return "FREQ"
    return "SP"


def _parse_orca_runtime(text: str) -> float | None:
    """ORCA's self-reported wall time, in seconds, from the footer line:
    'TOTAL RUN TIME: 0 days 0 hours 17 minutes 43 seconds 695 msec'.
    Returns None if absent (e.g. the run crashed before writing it)."""
    m = re.search(
        r"TOTAL RUN TIME:\s+(\d+)\s+days?\s+(\d+)\s+hours?\s+(\d+)\s+minutes?\s+"
        r"(\d+)\s+seconds?\s+(\d+)\s+msec",
        text,
    )
    if not m:
        return None
    d, h, mi, s, ms = (int(x) for x in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + s + ms / 1000.0
