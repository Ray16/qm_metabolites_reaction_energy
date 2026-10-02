import hashlib
import json
from pathlib import Path

import metag
from metag import pipeline


FIXTURES = Path(__file__).with_name("fixtures")


def test_frozen_release_metadata_is_internally_consistent():
    release = json.loads((FIXTURES / "frozen_2026-10-01c.json").read_text())
    assert release["release"] == pipeline.effective_config()["physics"]
    assert release["tecrdb"]["estimated"] == release["tecrdb"]["attempted"]
    assert release["modelseed"]["estimated"] + release["modelseed"]["explicit_failures"] == 300


def test_calibration_artifact_matches_frozen_hash():
    release = json.loads((FIXTURES / "frozen_2026-10-01c.json").read_text())
    calibration = Path(metag.__file__).with_name("data") / "sigma_class_calibrated.json"
    digest = hashlib.sha256(calibration.read_bytes()).hexdigest()
    assert digest == release["artifact_hashes"]["sigma_class_calibrated.json"]

