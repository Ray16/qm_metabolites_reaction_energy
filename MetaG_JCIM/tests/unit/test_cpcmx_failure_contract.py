"""Partial CPCM-X output is not evidence of a successful solvent calculation."""
from pathlib import Path
from types import SimpleNamespace
import pytest
from metag.energetics import thermal


@pytest.mark.parametrize('returncode,energy,expected', [
    (0, '-1.000000E-02', -26.25499639),
    (1, '-1.000000E-02', None),
    (0, '1.000000E+999', None),
    (0, None, None),
])
def test_cpcmx_requires_successful_exit_and_finite_complete_result(monkeypatch, returncode, energy, expected):
    def producer(command, *, cwd, **kwargs):
        assert command[command.index('--uhf') + 1] == '2'
        if energy is not None:
            Path(cwd, 'fort.6').write_text('solvation free energy (dG_solv): ' + energy + '\n')
        return SimpleNamespace(returncode=returncode)
    monkeypatch.setattr(thermal.subprocess, 'run', producer)
    actual = thermal.dgsolv(['O', 'O'], [[0, 0, 0], [0, 0, 1.2]], 0, 'cpcmx', mult=3)
    assert actual == pytest.approx(expected) if expected is not None else actual is None
