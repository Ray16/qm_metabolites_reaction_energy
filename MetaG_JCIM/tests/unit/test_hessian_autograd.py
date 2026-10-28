"""UMA_HESSIAN=autograd: exact Hessian back-end for uma_gibbs_corr (CPU-only, analytic toy potentials)."""
import numpy as np
import pytest
import torch

from metag.energetics import thermal as TH

# bent triatomic (water-like) at its minimum of the toy potential below
_POS0 = np.array([[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]])
_SYMS = ["O", "H", "H"]
_PAIRS = [(0, 1, 30.0), (0, 2, 30.0), (1, 2, 4.0)]                 # eV/Å² springs


def _energy(pos_t):
    """Anharmonic pair potential, minimum at _POS0 (eV, positions Å)."""
    e = pos_t.new_zeros(())
    p0 = torch.as_tensor(_POS0, dtype=pos_t.dtype)
    for i, j, k in _PAIRS:
        r = (pos_t[i] - pos_t[j]).norm(); r0 = (p0[i] - p0[j]).norm()
        e = e + 0.5 * k * (r - r0) ** 2 + 0.3 * k * (r - r0) ** 3
    return e


def _exact_H(pos):
    p = torch.tensor(pos, dtype=torch.float64)
    return torch.autograd.functional.hessian(lambda x: _energy(x.view(-1, 3)), p.reshape(-1)).numpy()


def _fake_predict(pu, atoms_list):
    """Stand-in for metag.energetics.uma._predict: analytic E/F, same return contract."""
    Es, Fs, bis = [], [], []
    for k, a in enumerate(atoms_list):
        p = torch.tensor(a.get_positions(), dtype=torch.float64, requires_grad=True)
        e = _energy(p); g, = torch.autograd.grad(e, p)
        Es.append(e.detach().view(1)); Fs.append(-g); bis.append(torch.full((len(a),), k))
    return torch.cat(Es), torch.cat(Fs), torch.cat(bis)


def test_hessian_method_flag(monkeypatch):
    monkeypatch.delenv("UMA_HESSIAN", raising=False)
    assert TH.hessian_method() == "autograd"                        # production default (2026-10-06)
    monkeypatch.setenv("UMA_HESSIAN", "fd")
    assert TH.hessian_method() == "fd"
    monkeypatch.setenv("UMA_HESSIAN", "bogus")
    with pytest.raises(ValueError):
        TH.hessian_method()


@pytest.mark.parametrize("rows", [1, 2, 4, 9, 50])               # incl. non-divisible and > 3N
def test_batched_vjp_hessian_matches_exact(rows):
    rng = np.random.default_rng(0)
    pos = _POS0 + 0.05 * rng.standard_normal(_POS0.shape)         # off-minimum: anharmonic terms matter
    p = torch.tensor(pos, dtype=torch.float64, requires_grad=True)
    e = _energy(p)
    g, = torch.autograd.grad(e, p, create_graph=True)
    H = TH._hessian_from_forces((-g).reshape(-1), p, rows)
    np.testing.assert_allclose(H, _exact_H(pos), atol=1e-10)


def test_oom_halves_row_chunk_and_preserves_result(monkeypatch):
    """A CUDA OOM on a big VJP chunk retries with half the rows; the Hessian is unchanged."""
    real_grad = torch.autograd.grad
    seen = []

    def grad(outputs, inputs, grad_outputs=None, is_grads_batched=False, **kw):
        if is_grads_batched:
            seen.append(grad_outputs.shape[0])
            if grad_outputs.shape[0] > 2:
                raise torch.cuda.OutOfMemoryError("fake OOM")
        return real_grad(outputs, inputs, grad_outputs=grad_outputs, is_grads_batched=is_grads_batched, **kw)
    p = torch.tensor(_POS0, dtype=torch.float64, requires_grad=True)
    g, = real_grad(_energy(p), p, create_graph=True)
    monkeypatch.setattr(torch.autograd, "grad", grad)
    H = TH._hessian_from_forces((-g).reshape(-1), p, 8)
    np.testing.assert_allclose(H, _exact_H(_POS0), atol=1e-10)
    assert seen[:3] == [8, 4, 2] and max(seen[3:]) <= 2            # halves until it fits, then stays small


def test_force_graph_context_patches_and_restores():
    escn = pytest.importorskip("fairchem.core.models.uma.escn_md")
    orig = (escn.compute_forces, escn.compute_forces_and_stress)
    p = torch.tensor(_POS0, dtype=torch.float64, requires_grad=True)
    with TH._ForceGraph() as fg:
        F = escn.compute_forces(_energy(p).view(1), p, training=False)   # what the head calls at inference
        assert fg.pos is p and F.requires_grad                          # differentiable despite training=False
    assert (escn.compute_forces, escn.compute_forces_and_stress) == orig
    H = TH._hessian_from_forces(F.reshape(-1), p, 4)
    np.testing.assert_allclose(H, _exact_H(_POS0), atol=1e-10)


def test_gibbs_corr_autograd_equals_fd_limit(monkeypatch):
    """Both back-ends feed the SAME downstream thermochemistry: with the exact Hessian vs a fine
    finite-difference one the Gcorr agrees to << 0.01 kJ/mol, and the FD error shrinks as delta^2."""
    monkeypatch.setattr(TH, "_predict", _fake_predict)
    monkeypatch.setattr(TH, "uma_hessian_autograd", lambda pu, atoms, rows=None: _exact_H(atoms.get_positions()))
    g_ad = TH.uma_gibbs_corr(None, _SYMS, _POS0, 0, hessian="autograd")
    g_fd = TH.uma_gibbs_corr(None, _SYMS, _POS0, 0, hessian="fd", delta=0.005)
    assert abs(g_ad - g_fd) < 1e-2
    from ase import Atoms
    base = Atoms(_SYMS, _POS0, info={"charge": 0, "spin": 1})
    err = [np.abs(TH.fd_hessian(None, base, delta=d) - _exact_H(_POS0)).max() for d in (0.02, 0.01)]
    assert 3.0 < err[0] / err[1] < 5.0                                 # central difference: O(delta^2)


def test_env_flag_routes_uma_gibbs_corr(monkeypatch):
    calls = []
    monkeypatch.setattr(TH, "_predict", _fake_predict)
    monkeypatch.setattr(TH, "uma_hessian_autograd",
                        lambda pu, atoms, rows=None: calls.append("ad") or _exact_H(atoms.get_positions()))
    monkeypatch.setenv("UMA_HESSIAN", "autograd")
    TH.uma_gibbs_corr(None, _SYMS, _POS0, 0)
    monkeypatch.setenv("UMA_HESSIAN", "fd")
    TH.uma_gibbs_corr(None, _SYMS, _POS0, 0)
    assert calls == ["ad"]


def test_default_hessian_is_in_cache_keys_and_config():
    """The autograd default must key species caches and the config fingerprint separately from legacy fd
    results, so fd-era caches and calibrations can never be silently reused (fd keeps the old keys)."""
    from metag import pipeline as P
    if TH.hessian_method() != "autograd":
        pytest.skip("UMA_HESSIAN overridden in the environment")
    assert P._IMPLICIT_SETTINGS.get("hessian") == "autograd"
    assert P._explicit_settings().get("hessian") == "autograd"
    assert P.effective_config().get("hessian") == "autograd"


def test_hessian_memo_reuses_identical_geometry_only(monkeypatch):
    """Same geometry/charge/spin/method -> one Hessian, identical Gcorr; any change -> recomputed."""
    class PU:                                                       # predict-unit stand-in that can carry the memo
        pass
    calls = []
    monkeypatch.setattr(TH, "_predict", _fake_predict)
    monkeypatch.setattr(TH, "uma_hessian_autograd",
                        lambda pu, atoms, rows=None: calls.append(1) or _exact_H(atoms.get_positions()))
    pu = PU()
    g1 = TH.uma_gibbs_corr(pu, _SYMS, _POS0, 0, hessian="autograd")
    g2 = TH.uma_gibbs_corr(pu, _SYMS, _POS0, 0, hessian="autograd", imag_as_soft=True)
    assert len(calls) == 1 and g1 == pytest.approx(g2)
    TH.uma_gibbs_corr(pu, _SYMS, _POS0 + 1e-6, 0, hessian="autograd")     # moved geometry
    TH.uma_gibbs_corr(PU(), _SYMS, _POS0, 0, hessian="autograd")         # different model/predict unit
    assert len(calls) == 3
