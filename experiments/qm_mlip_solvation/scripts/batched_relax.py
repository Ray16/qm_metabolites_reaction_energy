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


def load_uma(model="uma-s-1p2"):
    return pretrained_mlip.get_predict_unit(model, device=DEV)


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
    E = pred["energy"].detach().float().view(-1)          # (N,)
    F = pred["forces"].detach().float()                    # (total,3)
    return E, F, batch.batch.to(F.device).long()


def batched_fire(pu, atoms_list, fmax=0.03, steps=300, maxstep=0.2,
                 dt0=0.1, dtmax=1.0, Nmin=5, finc=1.1, fdec=0.5, astart=0.1, fa=0.99,
                 verbose=False, log_every=25, label=""):
    """Relax all atoms_list simultaneously. Returns (relaxed_atoms, energies_eV np).
    verbose=True prints per-step progress (#converged, worst residual force)."""
    import time as _time
    _t0 = _time.time()
    N = len(atoms_list)
    nat = torch.tensor([len(a) for a in atoms_list], device=DEV)
    # flat positions (total,3); per-atom structure index
    pos = torch.tensor(np.concatenate([a.get_positions() for a in atoms_list]),
                       dtype=torch.float32, device=DEV)
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
        E, F, bi = _predict(pu, atoms_list)
        E_last = E
        # per-structure max force
        fnorm = F.norm(dim=1)
        fmax_s = torch.zeros(N, device=DEV).scatter_reduce_(0, bi, fnorm, reduce="amax",
                                                            include_self=False)
        done = fmax_s < fmax
        if verbose and (_step % log_every == 0 or bool(done.all())):
            worst = float(fmax_s.max())
            print(f"    [relax{(' '+label) if label else ''}] step {_step:4d}  "
                  f"converged {int(done.sum()):3d}/{N}  worst fmax {worst:.3f}  "
                  f"{_time.time()-_t0:5.1f}s", flush=True)
        if bool(done.all()):
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
    return atoms_list, E_last.detach().cpu().numpy()
