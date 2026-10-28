"""Execution primitives shared by local and distributed workers."""

from .filesystem import TaskClaim, atomic_write_json

__all__ = ["TaskClaim", "atomic_write_json"]

