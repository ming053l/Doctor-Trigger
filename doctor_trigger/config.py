"""Machine-local paths: the one reader of ``config.yaml``.

``config.yaml`` (copy ``config.example.yaml`` and fill it in; it is gitignored) holds every path
on this machine: the finished pipeline root, outputs, checkpoints, peers, engines. Values may refer
to other top-level string values and to ``{repo_root}`` with ``{placeholder}`` syntax, e.g.
``cases_csv: "{canonical_root}/common/cases.csv"``. Nested keys are addressed with dots:
``path("checkpoints.image_detector_aug")``.

Placeholders are filled from the top-level string values plus ``repo_root``, repeatedly, until
nothing changes (a value may name another value that itself has placeholders). Relative paths are
returned as written, so they resolve against the caller's working directory.

The config file is ``<repo_root>/config.yaml`` unless the environment variable
``DOCTOR_TRIGGER_CONFIG`` names another file. Nothing is read at import time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
"""The repository checkout that contains this package; the value of ``{repo_root}``."""

CONFIG_ENV = "DOCTOR_TRIGGER_CONFIG"

DEFAULTS: dict[str, str] = {
    # The query manifest the experiments read. workflow/run_workflow.sh currently writes
    # <workflow>/manifests_queries.csv while experiments read this location; the path is named
    # here once so callers ask the config instead of hard-coding either name.
    "queries_csv": "{canonical_root}/manifests/queries.csv",
}
"""Keys that may be absent from config.yaml, with the value used when they are."""

ENVIRONMENT_KEYS: dict[str, str] = {
    "DT_CANON_ROOT": "canonical_root",
    "DT_MODEL_SNAPSHOT": "model_snapshot",
    "DT_PEERS_ROOT": "peers_root",
    "DT_CKPTS_ROOT": "ckpts_root",
    "DT_IMAGE_DETECTOR": "checkpoints.image_detector",
    "DT_IMAGE_DETECTOR_AUG": "checkpoints.image_detector_aug",
    "DT_ENGINE_CACHE": "engine_cache",
    "DT_SIMSWAP_ROOT": "simswap_root",
    "DT_OUTPUTS": "outputs_root",
    "DT_MANIFEST_FFPP": "source_manifests.ffpp",
    "DT_MANIFEST_DFDC": "source_manifests.dfdc",
    "DT_MANIFEST_CELEBDF": "source_manifests.celebdf",
    "DT_LEGACY_TRIGGER_GLOB": "legacy_trigger_glob",
    "DT_RECORD_MODEL_RUNS": "record_model_runs",
}
"""``DT_*`` environment variable -> config key, for scripts that still read paths from the
environment (workflow/run_workflow.sh, peer subprocesses, experiments until they migrate)."""

_MAX_FILL_ROUNDS = 32


def fill_placeholders(template: str, values: Mapping[str, Any]) -> str:
    """Fill ``{name}`` placeholders from ``values`` until the string stops changing.

    Raises ``KeyError`` for a placeholder that names no value.
    """
    for _ in range(_MAX_FILL_ROUNDS):
        if "{" not in template:
            return template
        filled = template.format(**values)
        if filled == template:
            return template
        template = filled
    raise ValueError(f"placeholders do not resolve after {_MAX_FILL_ROUNDS} rounds: {template}")


@dataclass(frozen=True)
class Config:
    """A parsed config.yaml. ``raw`` is the YAML mapping as written (placeholders unfilled)."""

    file: Path
    raw: dict[str, Any]
    repo_root: Path = REPO_ROOT
    defaults: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULTS))

    def placeholder_values(self) -> dict[str, str]:
        """Values available to ``{placeholder}`` filling: top-level strings plus ``repo_root``."""
        values = {k: v for k, v in self.defaults.items() if k not in self.raw}
        values.update({k: v for k, v in self.raw.items() if isinstance(v, str)})
        values["repo_root"] = str(self.repo_root)
        return values

    def value(self, key: str) -> Any:
        """The raw value at a dotted key (no placeholder filling)."""
        node: Any = self.raw
        parts = key.split(".")
        for i, part in enumerate(parts):
            if not isinstance(node, Mapping) or part not in node:
                if i == 0 and len(parts) == 1 and key in self.defaults:
                    return self.defaults[key]
                raise KeyError(f"{key!r} is not set in {self.file}")
            node = node[part]
        return node

    def path(self, key: str) -> str:
        """The value at a dotted key with every ``{placeholder}`` filled (a string path)."""
        value = self.value(key)
        if isinstance(value, str):
            return fill_placeholders(value, self.placeholder_values())
        return value

    def environment(self) -> dict[str, str]:
        """``DT_*`` variable -> resolved path, for every entry of ``ENVIRONMENT_KEYS``."""
        return {env: self.path(key) for env, key in ENVIRONMENT_KEYS.items()}


def config_file() -> Path:
    """The config file in use: ``$DOCTOR_TRIGGER_CONFIG`` or ``<repo_root>/config.yaml``."""
    return Path(os.environ.get(CONFIG_ENV) or REPO_ROOT / "config.yaml")


def load_config(file: str | Path | None = None) -> Config:
    """Parse a config file (default: ``config_file()``)."""
    file = Path(file) if file is not None else config_file()
    if not file.is_file():
        raise FileNotFoundError(
            f"{file} not found: copy {REPO_ROOT / 'config.example.yaml'} to it and fill in the paths "
            f"(or point {CONFIG_ENV} at a config file)")
    with open(file, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return Config(file=file, raw=raw)


@lru_cache(maxsize=None)
def _cached(file: str) -> Config:
    return load_config(file)


def current() -> Config:
    """The config for this process, read once per config file."""
    return _cached(str(config_file()))


def path(key: str) -> str:
    """Resolved path for a dotted config key, from the current config file."""
    return current().path(key)


def export_environment(config: Config | None = None) -> None:
    """Set every ``DT_*`` variable that is not already set, for subprocesses that read paths from it."""
    for env, value in (config or current()).environment().items():
        os.environ.setdefault(env, value)
