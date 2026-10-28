"""A withheld minimum must leave a reproducible, JSON-serializable geometry."""
import json
import numpy as np
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem

from metag import pipeline as p
from metag.energetics.conformers import UniqueMinima, bond_graph


def test_unresolved_minimum_retains_initial_and_attempted_geometries(monkeypatch):
    molecule=Chem.AddHs(Chem.MolFromSmiles('CCO'))
    assert AllChem.EmbedMolecule(molecule,randomSeed=17)==0
    positions=molecule.GetConformer().GetPositions()
    atoms=Atoms([a.GetSymbol() for a in molecule.GetAtoms()],positions=positions)
    minima=UniqueMinima(template=molecule)
    minima.add(atoms,-100.,-120.)
    def correction(*args,**kwargs):
        return None, {'n_imag':1,'max_imag_cm':100.,'_imag_vecs':[]}
    monkeypatch.setattr(p,'uma_gibbs_corr',correction)
    monkeypatch.setattr(p,'batched_fire',lambda pu,ats,**kw:(ats,np.array([-100.]),np.array([False])))
    monkeypatch.setattr(p,'THERMAL_RETRY_STEPS',p.THERMAL_RELAX_STEPS)
    diagnostics={}
    result=p._validated_thermal_minima(None,minima,{},molecule,bond_graph(molecule),0,1,
                                       'test',lambda *_:None,diagnostics=diagnostics)
    assert result is None
    failure=diagnostics['failure_geometry']
    assert failure['reference_smiles']=='CCO'
    assert failure['charge']==0 and failure['spin_multiplicity']==1
    np.testing.assert_array_equal(failure['initial_positions_angstrom'],positions)
    assert len(failure['thermal_trials'])==2
    assert failure['optimization_trials'][0]['converged'] is False
    assert failure['optimization_trials'][0]['identity_preserved'] is True
    assert failure['initial_electronic_kj']==-100.
    assert failure['initial_vertical_kj']==-120.
    json.dumps(diagnostics,allow_nan=False)
