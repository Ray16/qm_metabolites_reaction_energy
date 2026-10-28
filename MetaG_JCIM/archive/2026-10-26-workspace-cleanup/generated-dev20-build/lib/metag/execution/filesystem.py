"""Filesystem operations for restart-safe, multi-worker execution."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any


def atomic_write_json(path: str | Path, value: Any) -> None:
    """Publish JSON with an atomic replace on the destination filesystem."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


@dataclass
class TaskClaim:
    """An exclusive task claim represented by an atomically created directory."""

    path: Path
    acquired: bool = False

    @classmethod
    def acquire(cls, claim_root: str | Path, task_id: str) -> TaskClaim:
        root = Path(claim_root)
        root.mkdir(parents=True, exist_ok=True)
        path = root / task_id
        try:
            path.mkdir()
        except FileExistsError:
            return cls(path=path, acquired=False)
        return cls(path=path, acquired=True)

    def release(self) -> None:
        """Release an owned, empty claim. Completed workers may retain claims."""
        if not self.acquired:
            return
        try:
            self.path.rmdir()
        except FileNotFoundError:
            pass
        self.acquired = False

    def __enter__(self) -> TaskClaim:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()

