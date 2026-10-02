"""Static checks that keep benchmark knowledge out of point routing."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable


_REACTION_ID = re.compile(r"\brxn\d+\b", re.IGNORECASE)


@dataclass(frozen=True)
class PolicyViolation:
    path: str
    line: int
    kind: str
    detail: str


def _names(node: ast.AST) -> Iterable[str]:
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            yield child.id
        elif isinstance(child, ast.Attribute):
            yield child.attr


def _string_literals(node: ast.AST) -> Iterable[str]:
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            yield child.value.lower()


def audit_point_routing_source(path: str | Path) -> list[PolicyViolation]:
    """Find dataset identifiers or annotation fields in routing decisions.

    Comments and docstrings are intentionally ignored. The audit targets
    executable conditional expressions, where benchmark-specific knowledge
    could alter a point-estimate route.
    """
    source_path = Path(path)
    source = source_path.read_text()
    tree = ast.parse(source, filename=str(source_path))
    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.If, ast.IfExp, ast.While, ast.Match)):
            continue
        expression = node.test if hasattr(node, "test") else node.subject
        text = ast.get_source_segment(source, expression) or ""
        if _REACTION_ID.search(text):
            violations.append(
                PolicyViolation(
                    str(source_path),
                    node.lineno,
                    "reaction_id_condition",
                    text.strip(),
                )
            )
        annotation_names = {"note", "enzyme", "enzyme_name", "ec", "ec_number"}
        used = annotation_names.intersection(
            {name.lower() for name in _names(expression)}
            | set(_string_literals(expression))
        )
        if used:
            violations.append(
                PolicyViolation(
                    str(source_path),
                    node.lineno,
                    "annotation_condition",
                    ", ".join(sorted(used)),
                )
            )
    return violations


def modules_importing_anchor(paths: Iterable[str | Path]) -> list[str]:
    """Return point-routing modules coupled to the legacy anchor module."""
    coupled = []
    for path in paths:
        source_path = Path(path)
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = {alias.name for alias in node.names}
                if module == "metag.routing.anchor" or (
                    module == "metag.routing" and "anchor" in names
                ):
                    coupled.append(str(source_path))
                    break
            if isinstance(node, ast.Import) and any(
                alias.name == "metag.routing.anchor" for alias in node.names
            ):
                coupled.append(str(source_path))
                break
    return sorted(coupled)
