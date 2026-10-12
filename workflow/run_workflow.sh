#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(dirname "$ROOT")
PYTHON_BIN=${PYTHON_BIN:-python}
MODE=run
GPU_ID=""

usage() {
  printf '%s\n' \
    "Usage: ./run_workflow.sh [--gpu ID] [--python PATH] [--verify-only]" \
    "" \
    "Activate the Python/CUDA environment first (see ENVIRONMENT.md). Paths to the source" \
    "manifests and SimSwap come from <repo>/config.yaml. A stage whose output already exists is" \
    "reused; the integrity gate, comparison interface, receipt and acceptance check always run."
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU_ID=${2:?--gpu requires an ID}
      shift 2
      ;;
    --python)
      PYTHON_BIN=${2:?--python requires a path}
      shift 2
      ;;
    --verify-only)
      MODE=verify
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "$ROOT/scripts" || ! -f "$ROOT/pipeline_paths.py" || ! -d "$REPO_ROOT/doctor_trigger" ]]; then
  printf 'Incomplete workflow root: %s\n' "$ROOT" >&2
  exit 2
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  printf 'Python executable not found: %s\n' "$PYTHON_BIN" >&2
  exit 2
fi

# The scripts import doctor_trigger and workflow.* from the repository root.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$MODE" == verify ]]; then
  exec "$PYTHON_BIN" "$ROOT/scripts/verify_workflow.py" --full
fi

if [[ -n "$GPU_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
elif [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi
export PYTHONUNBUFFERED=1

STATE_DIR="$ROOT/state"
LOG_DIR="$ROOT/logs"
MARKER_DIR="$STATE_DIR/markers"
mkdir -p "$STATE_DIR" "$LOG_DIR" "$MARKER_DIR"
exec > >(tee -a "$LOG_DIR/runner.log") 2>&1

RUNNER_PID=$$
CURRENT_STEP=preflight
FINISHED=0
printf '%s\n' "$RUNNER_PID" > "$STATE_DIR/runner.pid"

write_state() {
  local status=$1
  local timestamp
  timestamp=$(date --iso-8601=seconds)
  printf '{"status":"%s","current_step":"%s","runner_pid":%s,"updated_at":"%s","workflow_root":"%s"}\n' \
    "$status" "$CURRENT_STEP" "$RUNNER_PID" "$timestamp" "$ROOT" > "$STATE_DIR/status.json.tmp"
  mv "$STATE_DIR/status.json.tmp" "$STATE_DIR/status.json"
}

mark_done() {
  local disposition=${1:-generated}
  local timestamp
  timestamp=$(date --iso-8601=seconds)
  printf '{"status":"complete","step":"%s","disposition":"%s","completed_at":"%s"}\n' \
    "$CURRENT_STEP" "$disposition" "$timestamp" > "$MARKER_DIR/$CURRENT_STEP.json"
}

run_if_missing() {
  local step=$1
  local sentinel=$2
  shift 2
  CURRENT_STEP=$step
  write_state running
  if [[ -f "$sentinel" ]]; then
    printf '[reuse] %s: %s\n' "$step" "$sentinel"
    mark_done reused
  else
    "$@"
    [[ -f "$sentinel" ]] || { printf 'Stage %s did not create %s\n' "$step" "$sentinel" >&2; return 2; }
    mark_done generated
  fi
}

heartbeat() {
  while kill -0 "$RUNNER_PID" 2>/dev/null; do
    date --iso-8601=seconds > "$STATE_DIR/heartbeat"
    sleep 30
  done
}

on_exit() {
  local code=$?
  trap - EXIT INT TERM
  if [[ -n "${HEARTBEAT_PID:-}" ]]; then
    kill "$HEARTBEAT_PID" 2>/dev/null || true
  fi
  if [[ "$FINISHED" -eq 0 ]]; then
    write_state failed
  fi
  exit "$code"
}
trap on_exit EXIT INT TERM
heartbeat &
HEARTBEAT_PID=$!

write_state running
"$PYTHON_BIN" "$ROOT/scripts/preflight.py" --require-gpu --json "$ROOT/reports/preflight.json"
mark_done generated

run_if_missing base_manifest "$ROOT/manifests/base_samples.csv" \
  "$PYTHON_BIN" "$ROOT/scripts/build_base_manifest.py" --output "$ROOT/manifests/base_samples.csv"

run_if_missing alignment "$ROOT/manifests/alignment.csv" \
  "$PYTHON_BIN" "$ROOT/scripts/align_faces.py" \
    --manifest "$ROOT/manifests/base_samples.csv" \
    --output-root "$ROOT/artifacts/aligned" \
    --output-manifest "$ROOT/manifests/alignment.csv" \
    --quarantine "$ROOT/manifests/alignment_quarantine.csv" \
    --skip-existing

run_if_missing enrollment "$ROOT/manifests/enrollments.csv" \
  "$PYTHON_BIN" "$ROOT/scripts/enroll.py" \
    --alignment-manifest "$ROOT/manifests/alignment.csv" \
    --released-root "$ROOT/artifacts/released" \
    --record-root "$ROOT/private/templates" \
    --output-manifest "$ROOT/manifests/enrollments.csv" \
    --quarantine "$ROOT/manifests/enrollment_quarantine.csv" \
    --skip-existing

run_if_missing queries "$ROOT/manifests/queries.csv" \
  "$PYTHON_BIN" "$ROOT/scripts/build_queries.py" \
    --enrollments "$ROOT/manifests/enrollments.csv" \
    --alignment-manifest "$ROOT/manifests/alignment.csv" \
    --output-root "$ROOT/artifacts/queries" \
    --output-manifest "$ROOT/manifests/queries.csv" \
    --quarantine "$ROOT/manifests/query_quarantine.csv" \
    --batch-size 4 \
    --resume-existing

run_if_missing calibrated_features "$ROOT/features/calibrated_features.csv" \
  "$PYTHON_BIN" "$ROOT/scripts/extract_features.py" \
    --queries "$ROOT/manifests/queries.csv" \
    --output "$ROOT/features/calibrated_features.csv"

CURRENT_STEP=integrity_gate
write_state running
"$PYTHON_BIN" "$ROOT/scripts/integrity_gate.py" \
  --base "$ROOT/manifests/base_samples.csv" \
  --alignment "$ROOT/manifests/alignment.csv" \
  --alignment-quarantine "$ROOT/manifests/alignment_quarantine.csv" \
  --enrollments "$ROOT/manifests/enrollments.csv" \
  --enrollment-quarantine "$ROOT/manifests/enrollment_quarantine.csv" \
  --queries "$ROOT/manifests/queries.csv" \
  --query-quarantine "$ROOT/manifests/query_quarantine.csv" \
  --features "$ROOT/features/calibrated_features.csv" \
  --output-json "$ROOT/reports/calibrated_integrity.json" \
  --output-md "$ROOT/reports/calibrated_integrity.md"
mark_done generated

CURRENT_STEP=common_interface
write_state running
"$PYTHON_BIN" "$ROOT/scripts/export_common_interface.py" \
  --queries "$ROOT/manifests/queries.csv" \
  --enrollments "$ROOT/manifests/enrollments.csv" \
  --alignment-manifest "$ROOT/manifests/alignment.csv" \
  --output-dir "$ROOT/common"
mark_done generated

METRICS="$ROOT/results/record_model/metrics/record_model_release_calibrated_ffpp_to_all_metrics.csv"
run_if_missing train_eval "$METRICS" \
  "$PYTHON_BIN" "$ROOT/record_model/train_record_model.py" --config "$ROOT/configs/record_model.yaml"

CURRENT_STEP=final_receipt
write_state running
"$PYTHON_BIN" "$ROOT/scripts/finalize_receipt.py" \
  --base "$ROOT/manifests/base_samples.csv" \
  --alignment "$ROOT/manifests/alignment.csv" \
  --alignment-quarantine "$ROOT/manifests/alignment_quarantine.csv" \
  --enrollments "$ROOT/manifests/enrollments.csv" \
  --enrollment-quarantine "$ROOT/manifests/enrollment_quarantine.csv" \
  --queries "$ROOT/manifests/queries.csv" \
  --query-quarantine "$ROOT/manifests/query_quarantine.csv" \
  --features "$ROOT/features/calibrated_features.csv" \
  --integrity "$ROOT/reports/calibrated_integrity.json" \
  --metrics "$METRICS" \
  --common-cases "$ROOT/common/cases.csv" \
  --peer-contract "$ROOT/common/peer_score_contract.json" \
  --workflow-contract "$ROOT/common/workflow_contract.json" \
  --output-json "$ROOT/FINAL_RECEIPT.json" \
  --output-md "$ROOT/FINAL_RECEIPT.md"
mark_done generated

CURRENT_STEP=acceptance
write_state running
"$PYTHON_BIN" "$ROOT/scripts/verify_workflow.py"
mark_done generated

CURRENT_STEP=complete
FINISHED=1
write_state complete
printf '%s\n' 'Workflow completed.'
