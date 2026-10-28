"""Optimizer outputs must describe the returned geometry, including at the cap."""
import numpy as np
import pytest
import torch
from ase import Atoms
from metag.energetics import uma


@pytest.fixture
def harmonic(monkeypatch):
    monkeypatch.setattr(uma, 'DEV', 'cpu')
    calls=[]
    def predict(pu, atoms_list, chunk):
        calls.append([at.positions.copy() for at in atoms_list])
        energy=torch.tensor([.5*np.square(at.positions).sum() for at in atoms_list],dtype=torch.float64)
        force=torch.tensor(np.concatenate([-at.positions for at in atoms_list]),dtype=torch.float32)
        batch=torch.cat([torch.full((len(at),),i,dtype=torch.long) for i,at in enumerate(atoms_list)])
        return energy,force,batch
    monkeypatch.setattr(uma, '_predict_chunked', predict)
    return calls


def test_step_cap_returns_energy_and_convergence_at_final_position(harmonic):
    atom=Atoms('H',positions=[[1.,0,0]])
    relaxed,energy,converged=uma.batched_fire(None,[atom],steps=1,fmax=.999,return_converged=True)
    assert relaxed[0].positions[0,0] < 1.
    assert energy[0] == pytest.approx(.5*np.square(relaxed[0].positions).sum(),abs=1e-12)
    assert converged.tolist()==[True]  # final displacement crossed the threshold
    assert len(harmonic)==2


def test_already_converged_geometry_does_not_get_a_redundant_evaluation(harmonic):
    atom=Atoms('H',positions=[[0.,0,0]])
    _,energy,converged=uma.batched_fire(None,[atom],steps=4,return_converged=True)
    assert energy[0]==0 and converged.tolist()==[True]
    assert len(harmonic)==1


def test_capped_endpoint_uses_total_solvent_force_and_refreshes_all(harmonic):
    masks=[]
    def extra(atoms,done):
        masks.append(done.copy())
        return .5*np.concatenate([at.positions for at in atoms])
    atoms=[Atoms('H',positions=[[0.,0,0]]),Atoms('H',positions=[[1.,0,0]])]
    relaxed,energy,converged=uma.batched_fire(None,atoms,steps=1,fmax=.4995,
                                             extra_forces=extra,return_converged=True)
    assert converged.tolist()==[True,True]
    assert masks[-1].tolist()==[False,False]
    np.testing.assert_allclose(energy,[.5*np.square(at.positions).sum() for at in relaxed])


def test_zero_steps_still_evaluates_input_instead_of_returning_zero_energy(harmonic):
    atom=Atoms('H',positions=[[1.,0,0]])
    _,energy,converged=uma.batched_fire(None,[atom],steps=0,return_converged=True)
    assert energy.tolist()==[.5] and converged.tolist()==[False]
    assert len(harmonic)==1
