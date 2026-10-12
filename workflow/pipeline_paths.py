"""Where the pipeline reads and writes.

Every stage writes only inside this directory (the write boundary). Directories listed in
``$DT_READONLY_PREFIXES`` (colon-separated) are refused as outputs even when they lie inside it.
The external inputs (the three source manifests and the SimSwap checkout) come from config.yaml
through ``doctor_trigger.config``.
"""

from __future__ import annotations

import os
from pathlib import Path

from doctor_trigger import config

WORKFLOW_ROOT = Path(__file__).resolve().parent
READ_ONLY_PREFIXES = tuple(
    Path(value).resolve()
    for value in os.environ.get("DT_READONLY_PREFIXES", "").split(":")
    if value
)

SOURCE_MANIFEST_KEYS = {
    "FF++": "source_manifests.ffpp",
    "DFDC": "source_manifests.dfdc",
    "CelebDF": "source_manifests.celebdf",
}
"""Dataset name (as written in every manifest) -> config key of its source manifest."""


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def require_output_path(path: str | Path) -> Path:
    """Resolve an output path and accept it only inside the write boundary."""
    resolved = Path(path).expanduser().resolve()
    if not is_relative_to(resolved, WORKFLOW_ROOT):
        raise ValueError(f"Output must stay under {WORKFLOW_ROOT}: {resolved}")
    for prefix in READ_ONLY_PREFIXES:
        if is_relative_to(resolved, prefix):
            raise ValueError(f"Refusing write into read-only source tree: {resolved}")
    return resolved


def require_input_path(path: str | Path) -> Path:
    """Resolve an input path; raise if it does not exist."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def source_manifests() -> dict[str, Path]:
    """Dataset -> source manifest (``split,class,video_id,frame_path``) from config.yaml."""
    return {dataset: Path(config.path(key)) for dataset, key in SOURCE_MANIFEST_KEYS.items()}


def simswap_root() -> Path:
    """The SimSwap checkout (code and weights) from config.yaml."""
    return Path(config.path("simswap_root"))
