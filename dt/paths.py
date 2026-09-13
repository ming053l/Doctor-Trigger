"""Single source of truth for machine-local paths.

Usage:  from dt.paths import P
        P("cases_csv")            -> resolved absolute path (str)
        P("checkpoints.image_detector") -> nested keys via dots
Every hardcoded absolute path in an experiment script is a bug; route it through here.
"""
import os
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(_ROOT, "config.yaml")) as f:
    _CFG = yaml.safe_load(f)


def _resolve(value, depth=0):
    if depth > 8:
        raise ValueError("path template recursion")
    while isinstance(value, str) and "{" in value:
        new = value.format(**{**{k: v for k, v in _CFG.items() if isinstance(v, str)}, "repo_root": _ROOT})
        if new == value:
            break
        value = new
        depth += 1
    return value


def P(key):
    node = _CFG
    for part in key.split("."):
        node = node[part]
    return _resolve(node)


_ENV_MAP = {
    "DT_CANON_ROOT": "canonical_root",
    "DT_MODEL_SNAPSHOT": "model_snapshot",
    "DT_PEERS_ROOT": "peers_root",
    "DT_CKPTS_ROOT": "ckpts_root",
    "DT_LEGACY_OUTPUTS": "legacy_outputs",
    "DT_IMAGE_DETECTOR": "checkpoints.image_detector",
    "DT_IMAGE_DETECTOR_AUG": "checkpoints.image_detector_aug",
    "DT_IMAGE_DETECTOR_LEGACY": "checkpoints.image_detector_legacy",
    "DT_ENGINE_CACHE": "engine_cache",
    "DT_SIMSWAP_ROOT": "simswap_root",
    "DT_OUTPUTS": "outputs_root",
    "DT_MANIFEST_FFPP": "source_manifests.ffpp",
    "DT_MANIFEST_DFDC": "source_manifests.dfdc",
    "DT_MANIFEST_CELEBDF": "source_manifests.celebdf",
    "DT_PAPER_REPO": "paper_repo",
    "DT_LEGACY_TRIGGER_GLOB": "legacy_trigger_glob",
    "DT_RECORD_MODEL_RUNS": "record_model_runs",
}


def export_env():
    """Inject every config path as a DT_* env var (the vendored scripts read only these)."""
    for env, key in _ENV_MAP.items():
        os.environ.setdefault(env, P(key))
    os.environ.setdefault("IMAGE_DETECTOR_CKPT", os.path.join(P("checkpoints.image_detector"), "ffpp", "best_val_auc.pt"))
