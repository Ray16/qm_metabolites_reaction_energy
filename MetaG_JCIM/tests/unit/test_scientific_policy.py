from pathlib import Path

from metag.validation.scientific_policy import (
    audit_point_routing_source,
    modules_importing_anchor,
)


SOURCE = Path(__file__).parents[2] / "src" / "metag"


def test_point_routing_has_no_dataset_id_or_annotation_conditionals():
    paths = [SOURCE / "pipeline.py", *sorted((SOURCE / "routing").glob("*.py"))]
    violations = [
        violation
        for path in paths
        for violation in audit_point_routing_source(path)
    ]
    assert violations == []


def test_anchor_dependency_inventory_cannot_grow_silently():
    paths = [SOURCE / "pipeline.py", *sorted((SOURCE / "routing").glob("*.py"))]
    coupled = {
        Path(path).relative_to(SOURCE).as_posix()
        for path in modules_importing_anchor(paths)
    }
    # Only the disabled legacy correction path in pipeline.py may import the anchor offsets;
    # detection lives in routing/reaction_families.py and never implies correction.
    assert coupled == {"pipeline.py"}


def test_policy_audit_detects_mapping_annotation_and_reaction_id(tmp_path):
    source = tmp_path / "bad_router.py"
    source.write_text(
        'def route(reaction, reaction_id):\n'
        '    if reaction.get("note") == "kinase":\n'
        '        return 1\n'
        '    if reaction_id == "rxn00123":\n'
        '        return 2\n'
    )
    kinds = {item.kind for item in audit_point_routing_source(source)}
    assert kinds == {"annotation_condition", "reaction_id_condition"}
