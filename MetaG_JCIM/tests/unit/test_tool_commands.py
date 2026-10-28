"""Maintained tools dispatch without importing the energy scorer or loading its data."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys

from metag.cli import main


def test_command_preflight_and_direct_preflight_write_identical_reports(monkeypatch, tmp_path):
    from metag import reactions
    from metag.tools.preflight import main as direct
    def forbidden():
        raise AssertionError('Tool dispatch must not load the bundled scoring panel')
    monkeypatch.setattr(reactions, 'load_reactions', forbidden)
    source = tmp_path / 'input.json'
    source.write_text(json.dumps({'test': {'species': {'a': [-1, 0, 'CCO'], 'b': [1, 0, 'COC']}}}))
    first, second = tmp_path / 'first.json', tmp_path / 'second.json'
    with contextlib.redirect_stdout(io.StringIO()):
        main(['preflight', str(source), '-o', str(first)])
        direct([str(source), '-o', str(second)])
    assert json.loads(first.read_text()) == json.loads(second.read_text())


def test_every_maintained_command_help_and_legacy_score_help_are_QM_free():
    source = Path(__file__).resolve().parents[2] / 'src'
    script = '''
import contextlib, importlib.abc, io, sys
sys.path.insert(0, sys.argv[1])
class NoQM(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'ase','torch','fairchem'}:
            raise AssertionError('Optional QM imported: '+fullname)
sys.meta_path.insert(0, NoQM())
from metag.cli import main
from metag.tools import COMMANDS
for command in COMMANDS:
    with contextlib.redirect_stdout(io.StringIO()):
        try: main([command,'--help'])
        except SystemExit as exc: assert exc.code == 0
for args in (['--help'], ['score','--help'], ['tools']):
    with contextlib.redirect_stdout(io.StringIO()):
        try: main(args)
        except SystemExit as exc: assert exc.code == 0
assert not any(name in sys.modules for name in ('ase','torch','fairchem','metag.pipeline'))
'''
    subprocess.run([sys.executable, '-I', '-c', script, str(source)], check=True, capture_output=True, text=True)
