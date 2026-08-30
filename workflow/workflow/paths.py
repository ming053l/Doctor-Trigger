"""Filesystem boundary checks for the isolated workflow namespace."""

from __future__ import annotations

import os
from pathlib import Path


# Derive the boundary from this installed/copied workflow so a verified
# release under a shared server path remains runnable without writing back to
# the original author's home directory.
WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
READ_ONLY_PREFIXES = tuple(
    Path(value).resolve()
    for value in os.environ.get("DT_READONLY_PREFIXES", "").split(":")
    if value
)


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def require_output_path(path: str | Path) -> Path:
    """Resolve and accept only paths inside the versioned workflow root."""
    resolved = Path(path).expanduser().resolve()
    if not is_relative_to(resolved, WORKFLOW_ROOT):
        raise ValueError(f"Output must stay under {WORKFLOW_ROOT}: {resolved}")
    for prefix in READ_ONLY_PREFIXES:
        if is_relative_to(resolved, prefix):
            raise ValueError(f"Refusing write into read-only source tree: {resolved}")
    return resolved


def require_input_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved
