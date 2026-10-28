from concurrent.futures import ThreadPoolExecutor
import json

import pytest

from metag.execution import TaskClaim, atomic_write_json


def test_exactly_one_worker_acquires_a_task(tmp_path):
    def acquire(_):
        return TaskClaim.acquire(tmp_path, "species-42")

    with ThreadPoolExecutor(max_workers=16) as workers:
        claims = list(workers.map(acquire, range(64)))

    assert sum(claim.acquired for claim in claims) == 1
    owner = next(claim for claim in claims if claim.acquired)
    owner.release()
    assert TaskClaim.acquire(tmp_path, "species-42").acquired


def test_unowned_claim_cannot_release_another_worker(tmp_path):
    owner = TaskClaim.acquire(tmp_path, "reaction-7")
    contender = TaskClaim.acquire(tmp_path, "reaction-7")

    contender.release()
    assert owner.path.is_dir()
    owner.release()
    assert not owner.path.exists()


def test_atomic_json_write_publishes_complete_record(tmp_path):
    destination = tmp_path / "nested" / "result.json"
    value = {"reaction": "rxn00001", "dG": -31.3, "routes": {"ph0": True}}

    atomic_write_json(destination, value)

    assert json.loads(destination.read_text()) == value
    assert not list(destination.parent.glob("*.tmp"))


def test_atomic_json_write_cleans_up_after_serialization_error(tmp_path):
    destination = tmp_path / "result.json"
    with pytest.raises(TypeError):
        atomic_write_json(destination, {"invalid": object()})

    assert not destination.exists()
    assert not list(tmp_path.glob("*.tmp"))

