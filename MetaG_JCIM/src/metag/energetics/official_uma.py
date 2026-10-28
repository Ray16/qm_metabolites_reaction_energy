"""Official UMA semantics with Nitro-preferred optimization and FP32 finals.

The default calculator uses Nitro only for supported single-system optimization
calls. Energies entering a partition, convergence verification and force-derived
Hessians use the unmerged official general FP32 evaluator. Actual execution is
counted separately from the requested profile. Model IDs and composition rules
are never remapped for legacy reproduction.
"""
from collections import Counter
from dataclasses import asdict
from contextlib import contextmanager, nullcontext
import gc
from pathlib import Path
import random

import numpy as np
import torch

from metag.energetics.provenance import content_hash, file_hash


def sdk_sources():
    """Bind semantic code as well as the SDK's distribution version."""
    from fairchem.core.models.uma import escn_md, escn_moe
    from fairchem.core.units.mlip_unit import predict, utils
    from fairchem.core.units.mlip_unit.api import inference
    return {module.__name__: file_hash(Path(module.__file__))
            for module in (escn_md, escn_moe, predict, inference, utils)}


class OfficialUMACalculator:
    _metag_official_calculator = True

    def __init__(self, model, *, profile="nitro", device="cuda"):
        from fairchem.core import pretrained_mlip
        from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
        if "nitro" not in InferenceSettings.__dataclass_fields__:
            raise RuntimeError(
                "The Nitro/official profile requires the pinned newer FairChem SDK. "
                "Use its CUDA12.6 runtime on V100 or CUDA13 runtime on supported GPUs; "
                "UMA_RUNTIME_PROFILE=legacy is for the frozen FairChem2.21 environment only.")
        if profile not in {"nitro", "official_fp32"}:
            raise ValueError("official profile must be 'nitro' or 'official_fp32'")
        if not torch.cuda.is_available():
            raise RuntimeError("The QM calculator requires a usable CUDA runtime")
        capability = tuple(torch.cuda.get_device_capability(device))
        if capability == (7, 0) and "sm_70" not in torch._C._cuda_getArchFlags():
            raise RuntimeError("This Torch wheel lacks V100 kernels; use the CUDA12.6 V100 profile")
        self._sdk = pretrained_mlip
        self._device = device
        if model not in pretrained_mlip._MODEL_CKPTS.checkpoints:
            raise ValueError("MetaG requires a registered molecular model with explicit reference energies")
        self._resolved = pretrained_mlip.resolve_checkpoint(model)
        if self._resolved.atom_refs is None:
            raise ValueError("The molecular calculator requires the registered atomic reference energies")
        self._checkpoint = Path(self._resolved.path)
        self._checkpoint_hash = file_hash(self._checkpoint)
        self._reference_hash = content_hash({"atom_refs": self._resolved.atom_refs,
                                             "form_elem_refs": self._resolved.form_elem_refs})
        self._reference_settings = InferenceSettings(
            nitro=False, tf32=False, activation_checkpointing=True,
            merge_mole=False, compile=False, execution_mode="general",
            internal_graph_gen_version=2, use_quaternion_wigner=True)
        self._reference = self._load(self._reference_settings, seed=41)
        if "omol" not in self._reference.dataset_to_tasks:
            raise ValueError("The selected checkpoint does not provide the required omol task")
        backbone = self._reference.model.module.backbone
        identity = {"model_id": getattr(backbone, "model_id", None),
                    "model_version": getattr(backbone, "model_version", None)}
        if model == "uma-s-1p2p1" and identity["model_id"] != "UMA-S-1.2.1":
            raise RuntimeError("The registered checkpoint did not retain its official model identity")
        self._last_unit = self._reference
        self._nitro = None
        self._nitro_key = None
        self._counts = Counter()
        self._fallbacks = Counter()
        self._nitro_available = (profile == "nitro" and 8 <= capability[0] <= 12
                                 and model == "uma-s-1p2p1")
        self.metag_model = model
        self.metag_runtime_profile = {
            "name": "official-nitro-fp32-final-v1" if profile == "nitro" else "official-fp32-v1",
            "requested_calculator": profile,
            "optimization": "nitro_preferred" if profile == "nitro" else "official_fp32",
            "final_energies": "official_general_fp32",
            "endpoint_convergence": "official_fp32; refinement within the remaining step budget",
            "hessian": "official_general_fp32_force_autograd",
            "batch_fallback": "official_general_fp32",
            "hardware_capability": list(capability),
            "nitro_eligible_here": self._nitro_available,
            "model_semantics": "official SDK; no model-ID or composition override",
            "secondary_unit_rng": "SDK seed41 with caller Python/NumPy/Torch RNG states restored",
            "backbone_identity": identity,
            "sdk_source_sha256": sdk_sources(),
            "reference_settings": {
                key: str(value) if isinstance(value, torch.dtype)
                else sorted(value) if isinstance(value, (set, frozenset)) else value
                for key, value in asdict(self._reference_settings).items()},
        }
        self.metag_provenance = {
            "checkpoint_sha256": self._checkpoint_hash,
            "references_sha256": self._reference_hash,
            "inference_sha256": content_hash(self.metag_runtime_profile),
        }

    def _load(self, settings, *, seed=None):
        # Path loading must receive the named registry's references explicitly.
        if file_hash(self._checkpoint) != self._checkpoint_hash:
            raise RuntimeError("UMA checkpoint changed during calculator use")
        # This SDK calls set_seed unconditionally, despite older API docs
        # describing None as optional. Preserve the sampler stream when a
        # secondary, composition-specific calculator is constructed.
        context = self._preserve_rng() if seed is None else nullcontext()
        with context:
            unit = self._sdk.load_predict_unit(
                self._checkpoint, inference_settings=settings, device=self._device,
                atom_refs=self._resolved.atom_refs, form_elem_refs=self._resolved.form_elem_refs,
                seed=41 if seed is None else seed)
        if file_hash(self._checkpoint) != self._checkpoint_hash:
            raise RuntimeError("UMA checkpoint changed while loading")
        return unit

    @contextmanager
    def _preserve_rng(self):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        devices = ([torch.cuda.current_device()]
                   if torch.device(self._device).type == "cuda" else [])
        with torch.random.fork_rng(devices=devices):
            try:
                yield
            finally:
                random.setstate(python_state)
                np.random.set_state(numpy_state)

    @property
    def inference_settings(self):
        """Settings of the engine that actually served the most recent call."""
        return self._last_unit.inference_settings

    @property
    def model(self):
        return self._reference.model

    def predict_precise(self, batch):
        result = self._reference.predict(self._initial_input(self._reference, batch))
        settings = self._reference.inference_settings
        if settings.nitro or settings.tf32 or settings.merge_mole or settings.compile:
            raise RuntimeError("Official FP32 evaluator changed its declared numerical settings")
        self._last_unit = self._reference
        self._counts["official_fp32_calls"] += 1
        return result

    @staticmethod
    def _initial_input(unit, batch):
        # MOLE/Nitro preparation occurs before the SDK moves the model to CUDA.
        # Initialize from CPU data, as in its public ASE calculator, even when
        # our existing batching layer has already transferred the input.
        if not getattr(unit, "lazy_model_intialized", True):
            return batch.clone().cpu()
        return batch

    @staticmethod
    def precision_context():
        from fairchem.core.units.mlip_unit.utils import tf32_context_manager
        return tf32_context_manager(False)

    def predict(self, batch):
        """Default Nitro-preferred call; known unsupported requests use FP32."""
        if not self._nitro_available:
            self._fallbacks["profile_or_hardware"] += 1
            return self.predict_precise(batch)
        if batch.natoms.numel() != 1:
            self._fallbacks["multiple_systems"] += 1
            return self.predict_precise(batch)
        key = (tuple(batch.atomic_numbers.detach().cpu().tolist()),
               tuple(batch.charge.detach().cpu().reshape(-1).tolist()),
               tuple(batch.spin.detach().cpu().reshape(-1).tolist()))
        if self._nitro_key != key:
            self._nitro = None
            gc.collect()
            self._nitro = self._load("nitro")
            self._nitro_key = key
            self._counts["nitro_initializations"] += 1
        if not self._nitro.inference_settings.nitro:
            self._fallbacks["vendor_disabled_nitro"] += 1
            return self.predict_precise(batch)
        result = self._nitro.predict(self._initial_input(self._nitro, batch))
        if not self._nitro.inference_settings.nitro:
            # A vendor fallback may retain TF32/compile settings. Discard its
            # result and use the declared general FP32 fallback instead.
            self._fallbacks["vendor_disabled_nitro"] += 1
            return self.predict_precise(batch)
        self._last_unit = self._nitro
        self._counts["actual_nitro_calls"] += 1
        return result

    def record(self, name, count=1):
        self._counts[name] += count

    def execution_snapshot(self):
        return {"counts": dict(self._counts), "fallbacks": dict(self._fallbacks)}

    def execution_since(self, before):
        def delta(after, prior):
            return {k: v - prior.get(k, 0) for k, v in after.items() if v != prior.get(k, 0)}
        return {"profile": self.metag_runtime_profile,
                "counts": delta(self._counts, before["counts"]),
                "fallbacks": delta(self._fallbacks, before["fallbacks"]),
                "scope": "Actual calls in this scoring invocation, including nested validations; cache hits need no new inference."}


def trace_backend(function):
    """Attach actual mode counts without changing the scientific config key."""
    from functools import wraps

    @wraps(function)
    def wrapped(pu, *args, **kwargs):
        if getattr(pu, "_metag_official_calculator", False) is not True:
            return function(pu, *args, **kwargs)
        before = pu.execution_snapshot()
        result = function(pu, *args, **kwargs)
        result["backend_execution"] = pu.execution_since(before)
        return result
    return wrapped
