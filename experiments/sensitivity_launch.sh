#!/bin/bash
: "${DT_CANON_ROOT:?set DT_CANON_ROOT}"
: "${DT_PEERS_ROOT:?set DT_PEERS_ROOT}"
cd "$(dirname "$0")"
GPU=$1; shift  # host launcher: adjust env below per machine
for job in "$@"; do
  ds=${job%%|*}; spec=${job##*|}
  echo "=== [gpu$GPU] $ds $spec"
  CUDA_VISIBLE_DEVICES=$GPU python sensitivity_release.py "$ds" "$spec" 2>&1 | grep -vE "Warning|warn|providers|find model|det-size|input mean|-------|loading|Network|Set "
done
echo GPU${GPU}_SENS_DONE
