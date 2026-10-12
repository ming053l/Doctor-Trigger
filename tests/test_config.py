"""doctor_trigger.config: reading config.yaml, filling {placeholders}, dotted keys, DT_* export."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from doctor_trigger import config

REPO = Path(__file__).resolve().parents[1]


def write(tmp_path: Path, text: str, name: str = "config.yaml") -> Path:
    file = tmp_path / name
    file.write_text(text)
    return file


def test_placeholders_fill_through_chains_and_repo_root(tmp_path):
    file = write(tmp_path, (
        "canonical_root: '{repo_root}/../canonical'\n"
        "cases_csv: '{canonical_root}/common/cases.csv'\n"
        "features_csv: '{cases_csv}.features'\n"
    ))
    cfg = config.Config(file=file, raw=config.load_config(file).raw, repo_root=Path("/r"))
    assert cfg.path("canonical_root") == "/r/../canonical"
    assert cfg.path("cases_csv") == "/r/../canonical/common/cases.csv"
    assert cfg.path("features_csv") == "/r/../canonical/common/cases.csv.features"
    assert cfg.value("cases_csv") == "{canonical_root}/common/cases.csv"


def test_default_repo_root_is_this_checkout(tmp_path):
    cfg = config.load_config(write(tmp_path, "outputs_root: '{repo_root}/outputs'\n"))
    assert cfg.path("outputs_root") == f"{REPO}/outputs"


def test_nested_keys_with_dots_and_non_string_values(tmp_path):
    cfg = config.load_config(write(tmp_path, (
        "ckpts_root: /ckpts\n"
        "checkpoints:\n"
        "  image_detector: '{ckpts_root}/plain'\n"
        "workers: 8\n"
    )))
    assert cfg.path("checkpoints.image_detector") == "/ckpts/plain"
    assert cfg.path("workers") == 8
    assert cfg.value("checkpoints") == {"image_detector": "{ckpts_root}/plain"}
    with pytest.raises(KeyError):
        cfg.path("checkpoints.image_detector_aug")
    with pytest.raises(KeyError):
        cfg.path("ckpts_root.anything")


def test_nested_values_are_not_placeholders(tmp_path):
    cfg = config.load_config(write(tmp_path, "group:\n  inner: /x\nuses: '{inner}/y'\n"))
    with pytest.raises(KeyError):
        cfg.path("uses")


def test_relative_paths_are_returned_as_written(tmp_path):
    cfg = config.load_config(write(tmp_path, "outputs_root: outputs/results\n"))
    assert cfg.path("outputs_root") == "outputs/results"


def test_placeholder_cycle_raises(tmp_path):
    cfg = config.load_config(write(tmp_path, "a: '{b}x'\nb: '{a}y'\n"))
    with pytest.raises(ValueError):
        cfg.path("a")


def test_fill_placeholders_unknown_name():
    with pytest.raises(KeyError):
        config.fill_placeholders("{missing}/x", {"present": "/p"})
    assert config.fill_placeholders("/plain/path", {}) == "/plain/path"


def test_default_queries_csv(tmp_path):
    file = write(tmp_path, "canonical_root: /data/root\n")
    assert config.load_config(file).path("queries_csv") == "/data/root/manifests/queries.csv"
    file.write_text("canonical_root: /data/root\nqueries_csv: '{canonical_root}/manifests_queries.csv'\n")
    assert config.load_config(file).path("queries_csv") == "/data/root/manifests_queries.csv"


def test_missing_file_names_the_example(tmp_path):
    with pytest.raises(FileNotFoundError, match="config.example.yaml"):
        config.load_config(tmp_path / "absent.yaml")


def test_empty_file_is_an_empty_config(tmp_path):
    cfg = config.load_config(write(tmp_path, ""))
    assert cfg.raw == {}
    with pytest.raises(KeyError):
        cfg.path("canonical_root")


def test_example_config_resolves_every_key_and_marks_every_environment_key():
    example = REPO / "config.example.yaml"
    cfg = config.load_config(example)
    environment = cfg.environment()
    assert set(environment) == set(config.ENVIRONMENT_KEYS)
    for name, value in environment.items():
        assert "{" not in value, name
    marked = set(re.findall(r"\[(DT_[A-Z_]+)\]", example.read_text()))
    assert marked == set(config.ENVIRONMENT_KEYS)


def test_environment_maps_each_variable_to_its_key(tmp_path):
    nested: dict[str, list[str]] = {}
    top: list[str] = []
    for key in config.ENVIRONMENT_KEYS.values():
        head, _, tail = key.partition(".")
        if tail:
            nested.setdefault(head, []).append(f"  {tail}: /v/{key}")
        else:
            top.append(f"{head}: /v/{key}")
    text = "\n".join(top + [f"{head}:\n" + "\n".join(items) for head, items in nested.items()]) + "\n"
    cfg = config.load_config(write(tmp_path, text))
    assert cfg.environment() == {name: f"/v/{key}" for name, key in config.ENVIRONMENT_KEYS.items()}


def test_export_environment_keeps_variables_already_set(tmp_path, monkeypatch):
    for name in config.ENVIRONMENT_KEYS:
        monkeypatch.delenv(name, raising=False)
    cfg = config.load_config(REPO / "config.example.yaml")
    monkeypatch.setenv("DT_OUTPUTS", "/already/set")
    config.export_environment(cfg)
    assert os.environ["DT_OUTPUTS"] == "/already/set"
    assert os.environ["DT_CANON_ROOT"] == cfg.path("canonical_root")


def test_environment_variable_selects_the_config_file(tmp_path, monkeypatch):
    file = write(tmp_path, "canonical_root: /elsewhere\ncases_csv: '{canonical_root}/common/cases.csv'\n", "other.yaml")
    monkeypatch.setenv(config.CONFIG_ENV, str(file))
    assert config.config_file() == file
    assert config.path("cases_csv") == "/elsewhere/common/cases.csv"
    assert config.current().path("canonical_root") == "/elsewhere"
    monkeypatch.delenv(config.CONFIG_ENV)
    assert config.config_file() == REPO / "config.yaml"
