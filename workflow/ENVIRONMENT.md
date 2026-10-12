# Pipeline environment

The pipeline uses whatever Python environment is active; it never activates one itself.

## Setup

1. Create a Python 3.10 environment. Install a PyTorch 2.5 build that matches your driver and
   CUDA runtime, then `pip install -r workflow/requirements-workflow.txt` and, from the repository
   root, `pip install -e .` (the `doctor_trigger` package). The versions of the paper's run are
   in `requirements-workflow.txt`; a finished run records its own in `FINAL_RECEIPT.json`.
2. Copy `config.example.yaml` to `config.yaml` at the repository root and set
   `source_manifests.{ffpp,dfdc,celebdf}` (stage-0 manifests, see `WORKFLOW.md`) and
   `simswap_root` (a SimSwap checkout with its weights). The pipeline reads both as read-only
   inputs.

## Preflight

```bash
PYTHONPATH=. python workflow/scripts/preflight.py                # packages and input files
PYTHONPATH=. python workflow/scripts/preflight.py --require-gpu  # also CUDA for PyTorch and ONNX Runtime
```

SimSwap runs on `cuda:0` inside the process; pick the physical GPU with `CUDA_VISIBLE_DEVICES` or
the runner's `--gpu` option.

## Run

```bash
workflow/run_workflow.sh --gpu 0
workflow/run_workflow.sh --verify-only     # read-only check of a finished run, no GPU
```
