#!/bin/bash
# Run sensitivity_release.py jobs one after another on one GPU (paths come from config.yaml),
# hiding the SimSwap/InsightFace loading messages.
# Usage: sensitivity_launch.sh GPU "FF++|alpha:0.02" "DFDC|band:0.05-0.45" ...
cd "$(dirname "$0")"
export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
GPU=$1; shift
for job in "$@"; do
  ds=${job%%|*}; spec=${job##*|}
  echo "=== [gpu$GPU] $ds $spec"
  CUDA_VISIBLE_DEVICES=$GPU python sensitivity_release.py "$ds" "$spec" 2>&1 | grep -vE "Warning|warn|providers|find model|det-size|input mean|-------|loading|Network|Set "
done
echo "GPU${GPU} sensitivity jobs done"
