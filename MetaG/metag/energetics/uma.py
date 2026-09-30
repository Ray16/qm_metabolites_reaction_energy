"""Batched UMA relaxation — relax MANY structures in ONE forward pass per step.

The Step 1-4 scripts optimized conformers one-at-a-time with ASE BFGS (hundreds of
tiny GPU calls, GPU mostly idle). This relaxes all structures TOGETHER: one
`pu.predict(batch)` per FIRE step returns forces for every structure at once, so
100 conformers cost ~the same wall-time as 1. Mandatory for the database-scale run
(~50k reactions); identical energies to sequential (verify_batched_relax.py).

API:
  pu = load_uma()
  relaxed, energies_eV = batched_fire(pu, atoms_list, fmax=0.03, steps=300)
      atoms_list: list[ase.Atoms], each with .info={"charge":int,"spin":int}

FIRE mirrors ASE's unit-mass dynamics (dt=0.1, dtmax=1.0, Nmin=5, finc=1.1,
fdec=0.5, astart=0.1, fa=0.99); per-structure dt/alpha/state, converged structures
freeze. All state on GPU; graphs rebuilt each step from current positions.
"""
import numpy as np
import torch
from ase import Atoms
from fairchem.core import pretrained_mlip
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch

DEV = "cuda"


def _ensure_registered(model):
    """uma-s-1p2p1 ships in the facebook/UMA HF repo (checkpoints/uma-s-1p2p1.pt) but is NOT in
    fairchem 2.21.0's name registry. Register it by cloning the uma-s-1p2 entry (same repo + same
    reference yamls -- only the .pt weights differ) with the patched checkpoint filename, so
    get_predict_unit resolves it WITHOUT a fairchem upgrade. No-op if already known."""
    from fairchem.core.calculate.pretrained_mlip import _MODEL_CKPTS
    if model in _MODEL_CKPTS.checkpoints:
        return
    known_patch = {"uma-s-1p2p1": ("uma-s-1p2", "uma-s-1p2p1.pt")}
    if model not in known_patch:
        raise KeyError(f"{model} not in fairchem registry and no manual registration recipe known")
    base_name, filename = known_patch[model]
    base = _MODEL_CKPTS.checkpoints[base_name]
    _MODEL_CKPTS.checkpoints[model] = base.__class__(**{**base.__dict__, "filename": filename})


def load_uma(model="uma-s-1p2p1"):
    _ensure_registered(model)
    pu = pretrained_mlip.get_predict_unit(model, device=DEV)
    pu.metag_model = model                  # provenance: the pipeline keys its species cache on this
    return pu


def _predict(pu, atoms_list):
    """One batched forward pass -> (energy_eV (N,), forces_eV_A (total,3), batch_idx)."""
    datas = []
    for a in atoms_list:
        d = AtomicData.from_ase(a, task_name="omol", r_edges=False,
                                r_data_keys=["spin", "charge"],   # <-- carry per-structure charge/spin
                                r_energy=False, r_forces=False, r_stress=False)
        for k in ("energy", "forces", "stress"):
            if k in d:
                del d[k]
        datas.append(d)
    batch = atomicdata_list_to_batch(datas).to(DEV)
    pred = pu.predict(batch)
    # float64 from here on: UMA's total energy is float32 (the ~1e5 eV element references are added in the
    # model's precision), so it resolves only ~0.4-1.5 kJ for ATP/NAD/CoA-size species. Keeping it float32
    # through eV->kJ and the E+ΔGsolv sums would round again; conformers.UniqueMinima widens its energy
    # tolerance to >= 2 float32 steps of the species' own energy for the same reason.
    E = pred["energy"].detach().double().view(-1)         # (N,)
    F = pred["forces"].detach().float()                    # (total,3)
    return E, F, batch.batch.to(F.device).long()


import os as _os
_ENERGY_CHUNK = int(_os.environ.get("UMA_ENERGY_CHUNK", "256"))


_FIRE_CHUNK = int(_os.environ.get("UMA_FIRE_CHUNK", "0"))   # 0 = one batch (default); >0 = chunk force eval


def _predict_chunked(pu, atoms_list, chunk):
    """Same return as _predict, but evaluate forces in chunks of `chunk` structures
    and stitch back into a single (E, F, batch_idx) — result-preserving memory lever
    (forces are independent per structure). chunk<=0 -> single batch (identity)."""
    if chunk is None or chunk <= 0 or len(atoms_list) <= chunk:
        return _predict(pu, atoms_list)
    Es, Fs, bis, offset = [], [], [], 0
    for i in range(0, len(atoms_list), chunk):
        sub = atoms_list[i:i + chunk]
        E, F, bi = _predict(pu, sub)
        Es.append(E); Fs.append(F); bis.append(bi + offset)
        offset += len(sub)
    return torch.cat(Es), torch.cat(Fs), torch.cat(bis)


def batched_energies(pu, atoms_list, chunk=None):
    """Batched single-point energies (eV) for ranking; no relaxation. Chunked to
    bound GPU memory when the pool is large. `chunk` defaults to UMA_ENERGY_CHUNK
    (result-preserving memory lever for big molecules on small GPUs)."""
    if chunk is None:
        chunk = _ENERGY_CHUNK
    out = []
    for i in range(0, len(atoms_list), chunk):
        E, _, _ = _predict(pu, atoms_list[i:i + chunk])
        out.append(E.cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def batched_fire(pu, atoms_list, fmax=0.05, steps=300, maxstep=0.2, stop_frac=1.0,
                 straggler_fmax=None, return_converged=False, dt0=0.1, dtmax=1.0,
                 Nmin=5, finc=1.1, fdec=0.5, astart=0.1, fa=0.99,
                 verbose=False, log_every=25, label="", fix_mask=None):
    """Relax all atoms_list simultaneously. Returns (relaxed_atoms, energies_eV np).
    Straggler-robust: stops early once `stop_frac` of structures are below `fmax`
    AND every remaining structure is below `straggler_fmax` (default 2*fmax) — so a
    single slow conformer can't drag the whole batch to the step cap (energies of
    near-converged stragglers are fine to <~1 kJ). verbose prints per-step progress.

    fix_mask: optional per-atom bool over the FLAT concatenated atom list (length =
    Σ len(atoms_list)), True = FROZEN. Frozen atoms have their force zeroed, so they
    never move and convergence is judged on the FREE atoms only. Used by matched /
    frozen-scaffold relaxation: the conserved spectator is pinned at a shared geometry
    (identical across the reactant/product pair) so it cancels exactly in ΔE, and only
    the transformation region relaxes."""
    straggler_fmax = straggler_fmax if straggler_fmax is not None else 2.0 * fmax
    import time as _time
    _t0 = _time.time()
    N = len(atoms_list)
    nat = torch.tensor([len(a) for a in atoms_list], device=DEV)
    # flat positions (total,3); per-atom structure index
    pos = torch.tensor(np.concatenate([a.get_positions() for a in atoms_list]),
                       dtype=torch.float32, device=DEV)
    # frozen-atom free-mask (1.0 = free, 0.0 = frozen); multiplies forces so frozen
    # atoms feel zero force -> zero velocity -> never move, and don't gate convergence.
    freemask = None
    if fix_mask is not None:
        fm = torch.as_tensor(np.asarray(fix_mask, dtype=bool), device=DEV)
        assert fm.shape[0] == pos.shape[0], "fix_mask length must equal total atom count"
        freemask = (~fm).float().unsqueeze(1)                # (total,1)
    bidx = torch.repeat_interleave(torch.arange(N, device=DEV), nat)
    v = torch.zeros_like(pos)
    dt = torch.full((N,), dt0, device=DEV)
    alpha = torch.full((N,), astart, device=DEV)
    Npos = torch.zeros(N, dtype=torch.long, device=DEV)
    done = torch.zeros(N, dtype=torch.bool, device=DEV)
    E_last = torch.zeros(N, device=DEV)

    def scat(x):  # per-structure sum of per-atom scalar x
        return torch.zeros(N, device=DEV).scatter_add_(0, bidx, x)

    for _step in range(steps):
        # write current positions back into the Atoms, predict forces (batched)
        off = 0
        for i, a in enumerate(atoms_list):
            n = int(nat[i]); a.set_positions(pos[off:off + n].detach().cpu().numpy()); off += n
        E, F, bi = _predict_chunked(pu, atoms_list, _FIRE_CHUNK)
        E_last = E
        if freemask is not None:
            F = F * freemask                                 # frozen atoms: zero force -> never move / never gate convergence
        # per-structure max force
        fnorm = F.norm(dim=1)
        fmax_s = torch.zeros(N, device=DEV).scatter_reduce_(0, bi, fnorm, reduce="amax",
                                                            include_self=False)
        done = fmax_s < fmax
        frac = float(done.float().mean())
        worst = float(fmax_s.max())
        early = (frac >= stop_frac) and (worst < straggler_fmax)
        if verbose and (_step % log_every == 0 or bool(done.all()) or early):
            print(f"    [relax{(' '+label) if label else ''}] step {_step:4d}  "
                  f"converged {int(done.sum()):3d}/{N}  worst fmax {worst:.3f}  "
                  f"{_time.time()-_t0:5.1f}s", flush=True)
        if bool(done.all()) or early:
            break
        active_atom = ~done[bi]                              # mask atoms of unconverged structures
        # FIRE mixing (per structure)
        P = scat((F * v).sum(1))                             # power
        vn = scat((v * v).sum(1)).sqrt()
        fn = scat((F * F).sum(1)).sqrt().clamp(min=1e-12)
        pos_pow = (P > 0) & ~done
        # v = (1-a) v + a |v|/|f| * F   for structures with P>0
        mix = alpha[bi].unsqueeze(1)
        scale = (vn / fn)[bi].unsqueeze(1)
        v_mixed = (1 - mix) * v + mix * scale * F
        v = torch.where(pos_pow[bi].unsqueeze(1), v_mixed, v)
        # adapt dt/alpha for P>0
        Npos = torch.where(pos_pow, Npos + 1, torch.zeros_like(Npos))
        grow = pos_pow & (Npos > Nmin)
        dt = torch.where(grow, (dt * finc).clamp(max=dtmax), dt)
        alpha = torch.where(grow, alpha * fa, alpha)
        # reset for P<=0
        neg = (P <= 0) & ~done
        v = torch.where(neg[bi].unsqueeze(1), torch.zeros_like(v), v)
        dt = torch.where(neg, dt * fdec, dt)
        alpha = torch.where(neg, torch.full_like(alpha, astart), alpha)
        # MD step (unit mass): v += dt F ; dr = dt v, capped at maxstep per atom
        step_dt = torch.where(done, torch.zeros_like(dt), dt)[bi].unsqueeze(1)
        v = v + step_dt * F
        dr = step_dt * v
        drn = dr.norm(dim=1, keepdim=True).clamp(min=1e-12)
        dr = torch.where(drn > maxstep, dr * (maxstep / drn), dr)   # ASE FIRE maxstep cap
        pos = pos + dr

    # write final positions
    off = 0
    for i, a in enumerate(atoms_list):
        n = int(nat[i]); a.set_positions(pos[off:off + n].detach().cpu().numpy()); off += n
    converged = done.detach().cpu().numpy()   # per-structure: reached fmax
    if return_converged:
        return atoms_list, E_last.detach().cpu().numpy(), converged
    return atoms_list, E_last.detach().cpu().numpy()
