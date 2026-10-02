import pytest

from metag import pipeline
from metag.reactions import ReactionInput, SpeciesRecord, validate_reaction


def test_reaction_input_normalizes_valid_legacy_mapping():
    reaction = ReactionInput.from_mapping(
        {
            "note": "ethanol oxidation",
            "n_Hplus": 1,
            "species": {
                "ethanol": [-1, 0, "CCO"],
                "acetaldehyde": [1, 0, "CC=O"],
                "hydrogen": [1, 0, "[HH]"],
            },
        }
    )

    assert reaction.n_hplus == 1.0
    assert reaction.species["ethanol"] == SpeciesRecord(-1.0, 0, "CCO")


def test_reaction_input_is_immutable():
    reaction = ReactionInput.from_mapping(
        {"species": {"acetate": [-1, -1, "CC(=O)[O-]"]}}
    )
    with pytest.raises(Exception):
        reaction.n_hplus = 2
    with pytest.raises(TypeError):
        reaction.species["acetate"] = SpeciesRecord(-2.0, -1, "CC(=O)[O-]")


def test_validator_rejects_non_mapping_input():
    with pytest.raises(ValueError, match="mapping"):
        validate_reaction([])


def test_router_rejects_malformed_input_at_its_public_boundary():
    with pytest.raises(ValueError, match="unparseable SMILES"):
        pipeline.route_reaction(
            {"species": {"bad": [-1, 0, "not a smiles"]}},
            log=lambda *_: None,
        )
