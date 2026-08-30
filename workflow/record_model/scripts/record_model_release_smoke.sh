#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pack_root="$(cd "$script_dir/.." && pwd)"
cd "$pack_root"

python -m pytest -q tests/test_trigger_conditioned_features.py
