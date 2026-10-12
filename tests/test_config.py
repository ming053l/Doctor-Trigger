"""doctor_trigger.config against dt/paths.py, the reader it replaces.

dt/paths.py reads ``<its repo>/config.yaml`` at import, so each reference config is placed in a
scratch copy of the repository layout (``tmp/dt/paths.py`` + ``tmp/config.yaml``) and the new
reader is pointed at the same file with the same ``repo_root``. Every key must resolve to the
identical string, and the DT_* environment map must match ``export_env``.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

import pytest

from doctor_trigger import config

REPO = Path(__file__).resolve().parents[1]
REFERENCE_CONFIGS = [
    REPO / "config.example.yaml",
    Path("/ssd2/ming/DT_repo_public/config.yaml"),
    REPO / "config.yaml",
]


def import_by_path(name: str, file: Path):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def leaf_keys(node, prefix=""):
    for key, value in node.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            yield from leaf_keys(value, dotted + ".")
        else:
            yield dotted


@pytest.mark.parametrize("source", REFERENCE_CONFIGS, ids=lambda p: str(p))
def test_paths_identical_to_dt_paths(tmp_path, source, monkeypatch):
    if not source.exists():
        pytest.skip(f"{source} not present")
    (tmp_path / "dt").mkdir()
    shutil.copy(REPO / "dt/paths.py", tmp_path / "dt/paths.py")
    shutil.copy(source, tmp_path / "config.yaml")
    old = import_by_path(f"old_dt_paths_{abs(hash(str(source)))}", tmp_path / "dt/paths.py")

    new = config.Config(file=tmp_path / "config.yaml", raw=config.load_config(tmp_path / "config.yaml").raw,
                        repo_root=tmp_path)
    keys = list(leaf_keys(old._CFG))
    assert keys
    for key in keys:
        assert new.path(key) == old.P(key), key

    for name in config.ENVIRONMENT_KEYS:
        monkeypatch.delenv(name, raising=False)
    old.export_env()
    assert new.environment() == {name: os.environ[name] for name in old._ENV_MAP}
    assert config.ENVIRONMENT_KEYS == old._ENV_MAP


def test_default_queries_csv(tmp_path):
    file = tmp_path / "config.yaml"
    file.write_text("canonical_root: /data/root\n")
    cfg = config.load_config(file)
    assert cfg.path("queries_csv") == "/data/root/manifests/queries.csv"
    file.write_text("canonical_root: /data/root\nqueries_csv: '{canonical_root}/manifests_queries.csv'\n")
    assert config.load_config(file).path("queries_csv") == "/data/root/manifests_queries.csv"


def test_missing_key_and_file(tmp_path):
    file = tmp_path / "config.yaml"
    file.write_text("canonical_root: /x\n")
    with pytest.raises(KeyError):
        config.load_config(file).path("checkpoints.image_detector")
    with pytest.raises(FileNotFoundError):
        config.load_config(tmp_path / "absent.yaml")


def test_env_override_and_module_path(tmp_path, monkeypatch):
    file = tmp_path / "other.yaml"
    file.write_text("canonical_root: /elsewhere\ncases_csv: '{canonical_root}/common/cases.csv'\n")
    monkeypatch.setenv(config.CONFIG_ENV, str(file))
    assert config.config_file() == file
    assert config.path("cases_csv") == "/elsewhere/common/cases.csv"
    assert config.current().path("canonical_root") == "/elsewhere"
