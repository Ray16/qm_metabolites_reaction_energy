"""Supplied-Hessian thermochemistry must work without the MLIP installation."""
import os
from pathlib import Path
import subprocess
import sys


def test_supplied_hessian_and_solvation_import_without_torch_or_fairchem():
    code='''
import importlib.abc,sys
class NoMLIP(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','fairchem'}:
            raise AssertionError('Optional MLIP imported: '+fullname)
sys.meta_path.insert(0,NoMLIP())
import numpy as np
from ase import Atoms
from metag.energetics.thermal import gibbs_corr_from_hessian,xtb_dgsolv
atoms=Atoms('OH2',positions=[[0,0,0],[.96,0,0],[-.24,.93,0]])
correction,info=gibbs_corr_from_hessian(atoms,np.eye(9),0.,'nonlinear',2,return_info=True)
assert np.isfinite(correction) and info['n_imag']==0
assert 'torch' not in sys.modules and 'fairchem' not in sys.modules
'''
    source=Path(__file__).resolve().parents[2]/'src'
    proc=subprocess.run([sys.executable,'-c',code],env={**os.environ,'PYTHONPATH':str(source),'QRRHO':'0'},
                        capture_output=True,text=True,timeout=20)
    assert proc.returncode==0,proc.stderr
