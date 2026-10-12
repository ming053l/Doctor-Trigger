"""Compatibility shim over ``doctor_trigger.config``; new code imports that module instead.

Kept for scripts that still do ``from dt.paths import P`` (to be deleted in refactor phase 3).
As before, the config is read at import from ``<this repository>/config.yaml`` (or the file named
by ``$DOCTOR_TRIGGER_CONFIG``) and ``{repo_root}`` is this repository.
"""
import os
from pathlib import Path

from doctor_trigger import config as _config

_ROOT = Path(__file__).resolve().parent.parent
_FILE = Path(os.environ.get(_config.CONFIG_ENV) or _ROOT / "config.yaml")
_CONFIG = _config.Config(file=_FILE, raw=_config.load_config(_FILE).raw, repo_root=_ROOT)
_CFG = _CONFIG.raw
_ENV_MAP = _config.ENVIRONMENT_KEYS


def P(key):
    """Resolved path for a dotted config key (``doctor_trigger.config.Config.path``)."""
    return _CONFIG.path(key)


def export_env():
    """Set every DT_* variable that is not already set (``doctor_trigger.config.export_environment``)."""
    _config.export_environment(_CONFIG)
