"""``python -m doctor_trigger``: registry commands (list, show, status, reproduce) and method commands."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from doctor_trigger import cli, config

REPO = Path(__file__).resolve().parents[1]


def run_module(*args: str, config_file: Path | None = None) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("DT_") and k != config.CONFIG_ENV}
    env["PYTHONPATH"] = str(REPO)
    if config_file is not None:
        env[config.CONFIG_ENV] = str(config_file)
    return subprocess.run([sys.executable, "-m", "doctor_trigger", *args], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=300)


@pytest.fixture(scope="module")
def registry() -> dict:
    with open(REPO / "experiments/registry.yaml") as handle:
        return yaml.safe_load(handle)


@pytest.fixture
def cfg(tmp_path) -> config.Config:
    file = tmp_path / "config.yaml"
    file.write_text("root: /data\nouts: '{root}/outs'\nckpts_root: /ck\n"
                    "checkpoints:\n  image_detector: '{ckpts_root}/plain'\n  image_detector_aug: '{ckpts_root}/aug'\n")
    return config.load_config(file)


# ----------------------------------------------------------------------------- list / show / status
def test_list_prints_every_entry_with_its_paper_artifact(registry):
    result = run_module("list")
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert [line.split()[0] for line in lines] == list(registry)
    for line, (key, entry) in zip(lines, registry.items()):
        assert line[len(key):].strip() == str(entry.get("paper", "")).strip(), key


def test_show_fills_config_placeholders(cfg, capsys):
    registry = {"probe": {"paper": "Table 9", "cwd": "{repo_root}/experiments", "cmd": "python x.py --out {outs}",
                          "output": "{outs}/x.csv", "note": "slot {unknown} stays"}}
    assert cli.cmd_show(registry, "probe", cfg) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        "paper   Table 9",
        f"cwd     {config.REPO_ROOT}/experiments",
        "cmd     python x.py --out /data/outs",
        "output  /data/outs/x.csv",
        "note    slot {unknown} stays",
    ]


def test_show_every_registry_entry(registry, tmp_path):
    example = REPO / "config.example.yaml"
    for experiment_id in list(registry)[:: max(1, len(registry) // 8)]:
        result = run_module("show", experiment_id, config_file=example)
        assert result.returncode == 0, (experiment_id, result.stderr)
        fields = [line.split()[0] for line in result.stdout.splitlines() if line.strip()]
        assert fields and set(fields) <= set(cli.SHOW_FIELDS), experiment_id
        assert "{repo_root}" not in result.stdout, experiment_id


def test_status_reports_present_missing_and_patterns(cfg, tmp_path, capsys):
    present = tmp_path / "present.csv"
    present.write_text("x\n")
    registry = {
        "here": {"output": str(present)},
        "gone": {"output": str(tmp_path / "absent.csv")},
        "many": {"output": "{outs}/{dataset}.csv"},
        "none": {"cmd": "true"},
    }
    assert cli.cmd_status(registry, cfg) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("here") and " OK " in lines[0] and lines[0].endswith(str(present))
    assert lines[1].startswith("gone") and "MISSING" in lines[1]
    assert lines[2].startswith("many") and "(pattern) {outs}/{dataset}.csv" in lines[2]  # a pattern with an unknown slot stays unfilled


def test_fill_template_rounds_and_extra_values():
    assert cli.fill_template("{a}/{b}", {"a": "{c}", "c": "/c"}, {"b": "x"}) == "/c/x"
    assert cli.fill_template("{a}/{missing}", {"a": "/a"}) == "{a}/{missing}"


# ----------------------------------------------------------------------------- reproduce
def test_reproduce_unfilled_slot(cfg, capsys):
    registry = {"slotted": {"cwd": "/", "cmd": "echo {dataset}"}}
    assert cli.cmd_reproduce(registry, "slotted", [], cfg) == 2
    assert "unfilled slot in: echo {dataset}" in capsys.readouterr().out


def test_reproduce_documentation_only_entry(cfg, capsys):
    registry = {"doc": {"doc": "see {outs}/README"}}
    assert cli.cmd_reproduce(registry, "doc", [], cfg) == 0
    assert capsys.readouterr().out == "'doc' is documentation-only:\n  see /data/outs/README\n"


def test_reproduce_runs_registered_command(tmp_path, monkeypatch):
    for name in config.ENVIRONMENT_KEYS:
        monkeypatch.delenv(name, raising=False)
    cfg = config.load_config(REPO / "config.example.yaml")
    marker = tmp_path / "env.txt"
    registry = {"probe": {"cwd": str(tmp_path), "cmd": f"{sys.executable} -c \"import os,sys; "
                          f"open('{marker}','w').write(os.environ['DT_CANON_ROOT']+'|'+os.environ['IMAGE_DETECTOR_CKPT']"
                          f"+'|'+os.getcwd()+'|'+sys.argv[1])\" {{word}}"}}
    assert cli.cmd_reproduce(registry, "probe", ["word=hello", "head=ffpp"], cfg) == 0
    checkpoint = os.path.join(cfg.path("checkpoints.image_detector_aug"), "ffpp", "best_val_auc.pt")
    assert marker.read_text() == f"{cfg.path('canonical_root')}|{checkpoint}|{tmp_path}|hello"


def test_reproduce_environment(monkeypatch):
    for name in config.ENVIRONMENT_KEYS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DT_OUTPUTS", "/kept")
    cfg = config.load_config(REPO / "config.example.yaml")
    env = cli.reproduce_environment(cfg, {"plain_head": "dfdc"})
    expected = cfg.environment()
    expected["DT_OUTPUTS"] = "/kept"
    assert {k: env[k] for k in config.ENVIRONMENT_KEYS} == expected
    assert env["IMAGE_DETECTOR_CKPT"] == os.path.join(cfg.path("checkpoints.image_detector"), "dfdc", "best_val_auc.pt")
    assert env["PATH"].startswith(os.path.dirname(sys.executable) + os.pathsep)
    assert "IMAGE_DETECTOR_CKPT" not in cli.reproduce_environment(cfg, {})
    monkeypatch.setenv("PYTHONPATH", "/elsewhere")
    assert cli.reproduce_environment(cfg, {})["PYTHONPATH"] == f"{config.REPO_ROOT}{os.pathsep}/elsewhere"
    monkeypatch.delenv("PYTHONPATH")
    assert cli.reproduce_environment(cfg, {})["PYTHONPATH"] == str(config.REPO_ROOT)


def test_reproduce_imports_the_package_without_installing(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    cfg = config.load_config(REPO / "config.example.yaml")
    marker = tmp_path / "module.txt"
    registry = {"probe": {"cwd": str(tmp_path), "cmd": f"{sys.executable} -c \"import doctor_trigger; "
                          f"open('{marker}','w').write(doctor_trigger.__file__)\""}}
    assert cli.cmd_reproduce(registry, "probe", [], cfg) == 0
    assert marker.read_text() == str(REPO / "doctor_trigger" / "__init__.py")


# ----------------------------------------------------------------------------- method commands
def test_enroll_verify_calibrate_commands(tmp_path):
    from PIL import Image

    from doctor_trigger.record import load_record

    image = REPO / "examples/data/office.png"
    enrolled = run_module("enroll", str(image), "--sample-id", "office", "--out", str(tmp_path))
    assert enrolled.returncode == 0, enrolled.stderr
    assert enrolled.stdout.startswith("alpha=")
    released_file, record_file = tmp_path / "office_released.png", tmp_path / "office_record_PRIVATE.npz"
    released = np.asarray(Image.open(released_file))
    record = load_record(record_file)
    assert released.shape[:2] == record.template_phase.shape

    verified = run_module("verify", str(released_file), str(record_file), "--tau-c", "0.5", "--tau-g", "0.5")
    assert verified.returncode == 0, verified.stderr
    readings, status = verified.stdout.splitlines()
    assert readings.startswith("H_B=") and " C_B=" in readings and " G_s=" in readings
    assert status == "status=Confirmed"

    other = Image.open(image).convert("RGB").resize((record.shape[1] // 2, record.shape[0] // 2))
    other.save(tmp_path / "small.png")
    unavailable = run_module("verify", str(tmp_path / "small.png"), str(record_file))
    assert unavailable.stdout.startswith("status=Unavailable")

    scores = tmp_path / "scores.csv"
    scores.write_text("C\n" + "\n".join(str(v) for v in np.linspace(0, 1, 101)) + "\n")
    calibrated = run_module("calibrate", str(scores), "--column", "C")
    assert calibrated.returncode == 0, calibrated.stderr
    assert calibrated.stdout.startswith("tau=") and "column=C frr=0.01 n=101" in calibrated.stdout
