#!/usr/bin/env python
"""Unified experiment entry point: a shim for ``python -m doctor_trigger`` (doctor_trigger/cli.py).

  python run.py list                 # every experiment and the paper artifact it feeds
  python run.py show <id>            # resolved command, cwd, outputs
  python run.py run <id> [k=v ...]   # execute (template slots filled from k=v); same as `reproduce`
  python run.py status               # which outputs exist, with mtimes

The method commands (enroll, verify, calibrate) are forwarded too. All machine paths come from
config.yaml; experiments/registry.yaml maps paper artifacts to commands.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from doctor_trigger.cli import main  # noqa: E402

COMMANDS = ("list", "show", "run", "reproduce", "status", "enroll", "verify", "calibrate")

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    sys.exit(main(sys.argv[1:]))
