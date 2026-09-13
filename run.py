#!/usr/bin/env python
"""Unified experiment entry point.

  python run.py list                 # every experiment and the paper artifact it feeds
  python run.py show <id>            # resolved command, cwd, outputs
  python run.py run <id> [k=v ...]   # execute (template slots filled from k=v)
  python run.py status               # which outputs exist, with mtimes

All machine paths come from config.yaml; experiments/registry.yaml maps
paper artifacts to commands. Never run a result-producing script outside
this entry point unless you are debugging it.
"""
import os
import subprocess
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from dt.paths import P, _CFG, export_env  # noqa: E402

export_env()

with open(os.path.join(ROOT, "experiments", "registry.yaml")) as f:
    REG = yaml.safe_load(f)

_FLAT = {k: v for k, v in _CFG.items() if isinstance(v, str)}
_FLAT["repo_root"] = ROOT


def fill(tpl, extra=None):
    out = tpl
    for _ in range(6):
        try:
            out = out.format(**{**_FLAT, **(extra or {})})
        except (KeyError, IndexError):
            break
        if "{" not in out:
            break
    return out


def cmd_list():
    w = max(len(k) for k in REG)
    for k, e in REG.items():
        print(f"{k:<{w}}  {e.get('paper', '')}")


def cmd_show(eid):
    e = REG[eid]
    for key in ("paper", "impl", "cwd", "cmd", "doc", "output", "note"):
        if key in e:
            print(f"{key:7s} {fill(str(e[key]))}")


def cmd_run(eid, kv):
    e = REG[eid]
    if "cmd" not in e:
        print(f"'{eid}' is documentation-only:\n  {fill(e.get('doc', ''))}")
        return 0
    extra = dict(x.split("=", 1) for x in kv)
    cwd, cmd = fill(e["cwd"]), fill(e["cmd"], extra)
    if "{" in cmd:
        print(f"unfilled slot in: {cmd}\n  pass k=v for each {{slot}}")
        return 2
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    if "head" in extra:            # deployed detector (augmentation recipe)
        env["IMAGE_DETECTOR_CKPT"] = os.path.join(P("checkpoints.image_detector_aug"), extra["head"], "best_val_auc.pt")
    if "plain_head" in extra:      # same architecture without augmentation (appendix E.2)
        env["IMAGE_DETECTOR_CKPT"] = os.path.join(P("checkpoints.image_detector"), extra["plain_head"], "best_val_auc.pt")
    print(f"[{eid}] cwd={cwd}\n[{eid}] {cmd}")
    return subprocess.call(cmd, shell=True, cwd=cwd, env=env)


def cmd_status():
    for k, e in REG.items():
        out = e.get("output")
        if not out:
            continue
        path = fill(out)
        if any(c in path for c in "{},"):
            print(f"{k:24s} (pattern) {path}")
            continue
        if os.path.exists(path):
            age = (time.time() - os.path.getmtime(path)) / 3600
            print(f"{k:24s} OK   {age:6.1f}h old  {path}")
        else:
            print(f"{k:24s} MISSING       {path}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("list", "show", "run", "status"):
        print(__doc__)
        sys.exit(1)
    a = sys.argv[1]
    if a == "list":
        cmd_list()
    elif a == "show":
        cmd_show(sys.argv[2])
    elif a == "status":
        cmd_status()
    else:
        sys.exit(cmd_run(sys.argv[2], sys.argv[3:]))
