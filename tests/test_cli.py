"""``python -m doctor_trigger`` against ``python run.py``, the entry point it replaces.

``list``, ``show <id>`` (for every registry entry) and ``status`` must print exactly what run.py
prints; ``reproduce`` must report the same command and exit code and run with the same
environment. Both read the repository's config.yaml.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from doctor_trigger import cli, config

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (REPO / "config.yaml").exists(), reason="needs config.yaml")


def run(*args: str, module: bool) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "doctor_trigger", *args] if module else [sys.executable, "run.py", *args]
    env = {k: v for k, v in os.environ.items() if not k.startswith("DT_") and k != config.CONFIG_ENV}
    return subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=120)


@pytest.fixture(scope="module")
def registry() -> dict:
    with open(REPO / "experiments/registry.yaml") as handle:
        return yaml.safe_load(handle)


def test_list_identical(registry):
    old, new = run("list", module=False), run("list", module=True)
    assert old.returncode == new.returncode == 0
    assert new.stdout == old.stdout
    listed = [line.split()[0] for line in new.stdout.splitlines()]
    assert listed == list(registry)


def test_show_identical_for_every_entry(registry):
    for experiment_id in registry:
        old, new = run("show", experiment_id, module=False), run("show", experiment_id, module=True)
        assert old.returncode == new.returncode == 0, experiment_id
        assert new.stdout == old.stdout, experiment_id
        assert new.stdout.strip(), experiment_id


def test_status_identical():
    strip_age = lambda text: re.sub(r"OK +[0-9.]+h old", "OK <age>", text)  # noqa: E731
    old, new = run("status", module=False), run("status", module=True)
    assert old.returncode == new.returncode == 0
    assert strip_age(new.stdout) == strip_age(old.stdout)


def test_reproduce_unfilled_slot_identical(registry):
    slotted = [k for k, e in registry.items() if "cmd" in e and re.search(r"\{(dataset|engine|method)\}", e["cmd"])]
    assert slotted
    for experiment_id in slotted[:3]:
        old, new = run("run", experiment_id, module=False), run("reproduce", experiment_id, module=True)
        assert old.returncode == new.returncode == 2
        assert new.stdout == old.stdout


def test_reproduce_runs_registered_command(tmp_path, monkeypatch):
    marker = tmp_path / "env.txt"
    registry = {"probe": {"cwd": str(tmp_path), "cmd": f"{sys.executable} -c \"import os,sys; "
                          f"open('{marker}','w').write(os.environ['DT_CANON_ROOT']+'|'+os.environ['IMAGE_DETECTOR_CKPT']"
                          f"+'|'+os.getcwd()+'|'+sys.argv[1])\" {{word}}"}}
    for name in config.ENVIRONMENT_KEYS:
        monkeypatch.delenv(name, raising=False)
    cfg = config.load_config(REPO / "config.yaml")
    assert cli.cmd_reproduce(registry, "probe", ["word=hello", "head=ffpp"], cfg) == 0
    expected_ckpt = os.path.join(cfg.path("checkpoints.image_detector_aug"), "ffpp", "best_val_auc.pt")
    assert marker.read_text() == f"{cfg.path('canonical_root')}|{expected_ckpt}|{tmp_path}|hello"


def test_reproduce_environment_matches_run_py(monkeypatch):
    for name in config.ENVIRONMENT_KEYS:
        monkeypatch.delenv(name, raising=False)
    code = ("import os, sys, json; sys.argv=['run.py']; sys.path.insert(0, '.');"
            "from dt.paths import export_env, P; export_env();"
            "print(json.dumps({k: v for k, v in os.environ.items() if k.startswith('DT_')}))")
    env = {k: v for k, v in os.environ.items() if not k.startswith("DT_")}
    old = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True, check=True)
    import json
    new = cli.reproduce_environment(config.load_config(REPO / "config.yaml"), {"plain_head": "dfdc"})
    assert {k: v for k, v in new.items() if k.startswith("DT_")} == json.loads(old.stdout)
    assert new["IMAGE_DETECTOR_CKPT"].endswith(os.path.join("dfdc", "best_val_auc.pt"))
    assert new["PATH"].startswith(os.path.dirname(sys.executable) + os.pathsep)


def test_enroll_command(tmp_path):
    pytest.importorskip("doctor_trigger.enrollment")
    from doctor_trigger.record import load_record

    image = REPO / "examples/data/office.png"
    result = subprocess.run([sys.executable, "-m", "doctor_trigger", "enroll", str(image), "--sample-id", "office",
                             "--out", str(tmp_path)], cwd=REPO, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    assert "alpha=" in result.stdout
    from PIL import Image
    released = np.asarray(Image.open(tmp_path / "office_released.png"))
    record = load_record(tmp_path / "office_record_PRIVATE.npz")
    assert released.shape[:2] == record.template_phase.shape


def test_calibrate_command(tmp_path):
    pytest.importorskip("doctor_trigger.calibration")
    scores = tmp_path / "scores.csv"
    scores.write_text("C\n" + "\n".join(str(v) for v in np.linspace(0, 1, 101)) + "\n")
    result = subprocess.run([sys.executable, "-m", "doctor_trigger", "calibrate", str(scores), "--column", "C"],
                            cwd=REPO, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("tau=")
