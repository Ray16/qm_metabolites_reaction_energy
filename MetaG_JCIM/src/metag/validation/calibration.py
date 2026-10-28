"""Helpers for reading historical calibration artifacts."""


def reaction_classes(per_reaction):
    """Return ``reaction_id -> class`` across supported artifact schemas."""
    rows = per_reaction.items() if isinstance(per_reaction, dict) else enumerate(per_reaction)
    result = {}
    for key, row in rows:
        if isinstance(row, dict):
            reaction_id = row.get("rid") or row.get("reaction")
            if reaction_id is None and isinstance(key, str):
                reaction_id = key
            reaction_class = row.get("class")
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            reaction_id, reaction_class = row[:2]
        else:
            continue
        if reaction_id is not None and reaction_class is not None:
            result[str(reaction_id)] = str(reaction_class)
    return result

