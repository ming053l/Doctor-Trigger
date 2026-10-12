"""The single command line: experiments from the registry, and the method on one image.

    python -m doctor_trigger list                    # every experiment and the paper artifact it feeds
    python -m doctor_trigger show ID                 # resolved command, cwd, outputs
    python -m doctor_trigger reproduce ID [k=v ...]  # run it (template slots filled from k=v)
    python -m doctor_trigger status                  # which outputs exist, with their age
    python -m doctor_trigger enroll IMAGE --sample-id ID --out DIR [--dataset FF++]
    python -m doctor_trigger verify IMAGE RECORD [--tau-c X --tau-g Y]
    python -m doctor_trigger calibrate SCORES.csv --column C [--frr 0.01]

``experiments/registry.yaml`` maps each experiment id to its paper artifact (``paper``), working
directory (``cwd``), shell command (``cmd``), outputs and notes; ``{placeholders}`` are filled from
config.yaml (``doctor_trigger.config``) and from ``k=v`` arguments. ``reproduce`` behaves exactly
like the earlier ``run.py run``: it exports the ``DT_*`` path variables, puts this Python first on
PATH, maps ``head=<tag>`` / ``plain_head=<tag>`` to ``IMAGE_DETECTOR_CKPT`` and runs the command
through the shell in its working directory. ``run`` is accepted as another name for ``reproduce``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from doctor_trigger import config

REGISTRY_FILE = config.REPO_ROOT / "experiments" / "registry.yaml"
SHOW_FIELDS = ("paper", "impl", "cwd", "cmd", "doc", "output", "note")
_FILL_ROUNDS = 6


def load_registry(file: str | Path = REGISTRY_FILE) -> dict[str, dict[str, Any]]:
    """Experiment id -> registry entry, in file order."""
    with open(file, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def fill_template(template: str, values: Mapping[str, Any], extra: Mapping[str, str] | None = None) -> str:
    """Fill registry ``{slots}`` from config values and ``k=v`` arguments.

    Best effort, as run.py did: up to six rounds of ``str.format``; if any slot is unknown the
    template is returned with the rounds done so far (so ``show`` prints unfilled slots verbatim).
    """
    merged = {**values, **(extra or {})}
    out = template
    for _ in range(_FILL_ROUNDS):
        try:
            out = out.format(**merged)
        except (KeyError, IndexError):
            break
        if "{" not in out:
            break
    return out


# ----------------------------------------------------------------------------- registry commands
def cmd_list(registry: Mapping[str, Mapping[str, Any]]) -> int:
    width = max(len(k) for k in registry)
    for key, entry in registry.items():
        print(f"{key:<{width}}  {entry.get('paper', '')}")
    return 0


def cmd_show(registry: Mapping[str, Mapping[str, Any]], experiment_id: str, cfg: config.Config) -> int:
    entry = registry[experiment_id]
    values = cfg.placeholder_values()
    for key in SHOW_FIELDS:
        if key in entry:
            print(f"{key:7s} {fill_template(str(entry[key]), values)}")
    return 0


def reproduce_environment(cfg: config.Config, extra: Mapping[str, str]) -> dict[str, str]:
    """The environment a registry command runs in (DT_* paths, PATH, detector checkpoint)."""
    env = dict(os.environ)
    for name, value in cfg.environment().items():
        env.setdefault(name, value)
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    if "head" in extra:            # deployed image-only detector (augmentation recipe)
        env["IMAGE_DETECTOR_CKPT"] = os.path.join(cfg.path("checkpoints.image_detector_aug"), extra["head"],
                                                  "best_val_auc.pt")
    if "plain_head" in extra:      # same architecture without augmentation (appendix E.2)
        env["IMAGE_DETECTOR_CKPT"] = os.path.join(cfg.path("checkpoints.image_detector"), extra["plain_head"],
                                                  "best_val_auc.pt")
    return env


def cmd_reproduce(registry: Mapping[str, Mapping[str, Any]], experiment_id: str, assignments: Sequence[str],
                  cfg: config.Config) -> int:
    entry = registry[experiment_id]
    values = cfg.placeholder_values()
    if "cmd" not in entry:
        print(f"'{experiment_id}' is documentation-only:\n  {fill_template(entry.get('doc', ''), values)}")
        return 0
    extra = dict(item.split("=", 1) for item in assignments)
    cwd, cmd = fill_template(entry["cwd"], values), fill_template(entry["cmd"], values, extra)
    if "{" in cmd:
        print(f"unfilled slot in: {cmd}\n  pass k=v for each {{slot}}")
        return 2
    env = reproduce_environment(cfg, extra)
    print(f"[{experiment_id}] cwd={cwd}\n[{experiment_id}] {cmd}", flush=True)
    return subprocess.call(cmd, shell=True, cwd=cwd, env=env)


def cmd_status(registry: Mapping[str, Mapping[str, Any]], cfg: config.Config) -> int:
    values = cfg.placeholder_values()
    for key, entry in registry.items():
        output = entry.get("output")
        if not output:
            continue
        file = fill_template(output, values)
        if any(c in file for c in "{},"):
            print(f"{key:24s} (pattern) {file}")
            continue
        if os.path.exists(file):
            age = (time.time() - os.path.getmtime(file)) / 3600
            print(f"{key:24s} OK   {age:6.1f}h old  {file}")
        else:
            print(f"{key:24s} MISSING       {file}")
    return 0


# ----------------------------------------------------------------------------- method commands
# These import the method modules inside the function, so the registry commands work without them.
def cmd_enroll(image: str, sample_id: str, out_dir: str, master_seed: int, dataset: str) -> int:
    """Enroll one original crop I: write the released image I' and the private record R."""
    import numpy as np
    from PIL import Image

    from doctor_trigger.enrollment import EnrollmentSettings, alpha_ladder, enroll
    from doctor_trigger.record import UnitId, save_record

    original = np.asarray(Image.open(image).convert("RGB"))
    result = enroll(original, UnitId(sample_id), alpha_ladder(dataset), EnrollmentSettings(master_seed=master_seed))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    released_file, record_file = out / f"{sample_id}_released.png", out / f"{sample_id}_record_PRIVATE.npz"
    Image.fromarray(result.released).save(released_file)
    save_record(record_file, result.record)
    print(f"alpha={result.alpha} PSNR={result.psnr:.2f} dB SSIM={result.ssim:.4f}")
    print(f"released: {released_file}")
    print(f"record (keep private): {record_file}")
    return 0


def cmd_verify(image: str, record_file: str, tau_c: float | None, tau_g: float | None) -> int:
    """Read H, C_B and G_s of a query J against a record R; with thresholds, print the status."""
    import numpy as np
    from PIL import Image

    from doctor_trigger.readings import read_unit
    from doctor_trigger.record import load_record

    reading = read_unit(np.asarray(Image.open(image).convert("RGB")), load_record(record_file))
    print(f"H={reading.H:.4f} C_B={reading.C_B:.4f} G_s={reading.G_s:.4f}")
    if tau_c is not None and tau_g is not None:
        from doctor_trigger.decision import unit_status

        print(f"status={unit_status(reading, {'tau_C': tau_c, 'tau_G': tau_g}).value}")
    return 0


def cmd_calibrate(scores_csv: str, column: str, frr: float) -> int:
    """Lower-tail threshold of a benign score column at a target benign false-rejection rate."""
    import pandas as pd

    from doctor_trigger.calibration import lower_tail_threshold

    scores = pd.read_csv(scores_csv)[column].to_numpy(dtype=float)
    print(f"tau={lower_tail_threshold(scores, frr=frr)!r} column={column} frr={frr} n={len(scores)}")
    return 0


# ----------------------------------------------------------------------------- entry point
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="doctor-trigger", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="every experiment and the paper artifact it feeds")
    show = sub.add_parser("show", help="resolved command, cwd and outputs of one experiment")
    show.add_argument("experiment_id")
    for name in ("reproduce", "run"):
        rep = sub.add_parser(name, help="run one registered experiment" if name == "reproduce" else "same as reproduce")
        rep.add_argument("experiment_id")
        rep.add_argument("assignments", nargs="*", metavar="k=v", help="values for {slots} in the command")
    sub.add_parser("status", help="which registered outputs exist")
    enroll = sub.add_parser("enroll", help="enroll one image: released image + private record")
    enroll.add_argument("image")
    enroll.add_argument("--sample-id", required=True)
    enroll.add_argument("--out", required=True)
    enroll.add_argument("--master-seed", type=int, default=17)
    enroll.add_argument("--dataset", default="FF++", help="selects the alpha ladder (DFDC has its own)")
    verify = sub.add_parser("verify", help="read H, C_B, G_s of a query against a record")
    verify.add_argument("image")
    verify.add_argument("record")
    verify.add_argument("--tau-c", type=float)
    verify.add_argument("--tau-g", type=float)
    calibrate = sub.add_parser("calibrate", help="threshold at a target benign false-rejection rate")
    calibrate.add_argument("scores_csv")
    calibrate.add_argument("--column", required=True)
    calibrate.add_argument("--frr", type=float, default=0.01)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "enroll":
        return cmd_enroll(args.image, args.sample_id, args.out, args.master_seed, args.dataset)
    if args.command == "verify":
        return cmd_verify(args.image, args.record, args.tau_c, args.tau_g)
    if args.command == "calibrate":
        return cmd_calibrate(args.scores_csv, args.column, args.frr)
    registry = load_registry()
    if args.command == "list":
        return cmd_list(registry)
    cfg = config.current()
    if args.command == "show":
        return cmd_show(registry, args.experiment_id, cfg)
    if args.command == "status":
        return cmd_status(registry, cfg)
    return cmd_reproduce(registry, args.experiment_id, args.assignments, cfg)
