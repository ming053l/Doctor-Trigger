# Environment contract

The Python/CUDA environment is deliberately not bundled in this release. A
virtual environment that inherits a particular user's Conda libraries is not
portable. The workflow instead uses whichever Python environment the caller
has activated.

## Setup

Create and activate a Python 3.10 environment. Install a PyTorch 2.5 build that
matches the server driver/CUDA runtime, then install `requirements-workflow.txt`.
The exact versions used for the completed release are also recorded in
`FINAL_RECEIPT.json`.

Do not copy another user's virtual environment. The source dataset manifests
and SimSwap weights are read-only external inputs; point the `DT_MANIFEST_*`
and `DT_SIMSWAP_ROOT` environment variables at your local copies.

## Preflight

From the workflow root:

```bash
python scripts/preflight.py
python scripts/preflight.py --require-gpu
```

The first command validates packages and external files. The second also
requires PyTorch CUDA and ONNX Runtime's CUDA provider. SimSwap uses `cuda:0`
inside the process, so select a physical GPU through `CUDA_VISIBLE_DEVICES` or
the runner's `--gpu` option.

## Run

```bash
./run_workflow.sh --gpu 0
```

The runner never activates a hard-coded environment or sets a user-specific
library path. Existing canonical artifacts are reused, then the integrity gate,
common comparison interface, final receipt, and acceptance checks are rebuilt.

Independent read-only verification does not need a GPU:

```bash
python scripts/verify_workflow.py
python scripts/verify_workflow.py --full
```
