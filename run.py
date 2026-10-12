#!/usr/bin/env python
"""Same as ``python -m doctor_trigger`` (doctor_trigger/cli.py); ``run`` is ``reproduce``.

  python run.py list | show ID | status
  python run.py run ID [k=v ...]
  python run.py enroll | verify | calibrate ...
"""
import io
import sys

from doctor_trigger.cli import main

COMMANDS = ("list", "show", "run", "reproduce", "status", "enroll", "verify", "calibrate")

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    if sys.stdout.isatty():
        sys.exit(main(sys.argv[1:]))
    # Redirected to a file or pipe, run.py's own lines (the "[ID] cwd=" banner) are written after
    # the experiment's output, as they always were, so saved logs keep their layout.
    redirected, sys.stdout = sys.stdout, io.StringIO()
    try:
        code = main(sys.argv[1:])
    finally:
        own_lines, sys.stdout = sys.stdout.getvalue(), redirected
        sys.stdout.write(own_lines)
    sys.exit(code)
