#!/usr/bin/env python3
"""Check packages, GPU and external inputs before the pipeline starts; exit 2 on any failure."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

from workflow.pipeline_paths import WORKFLOW_ROOT, simswap_root, source_manifests

MODULES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "Pillow": "PIL",
    "PyYAML": "yaml",
    "OpenCV": "cv2",
    "scikit-image": "skimage",
    "scikit-learn": "sklearn",
    "PyTorch": "torch",
    "torchvision": "torchvision",
    "onnxruntime-gpu": "onnxruntime",
    "insightface": "insightface",
}

DETECTOR_FILES = {
    "SCRFD detector": WORKFLOW_ROOT / "assets/detectors/scrfd_10g_bnkps.onnx",
    "YuNet detector": WORKFLOW_ROOT / "assets/detectors/face_detection_yunet_2023mar.onnx",
}

SIMSWAP_FILES = {
    "SimSwap generator": "checkpoints/people/latest_net_G.pth",
    "SimSwap ArcFace": "arcface_model/arcface_checkpoint.tar",
    "SimSwap detector": "insightface_func/models/antelope/scrfd_10g_bnkps.onnx",
}
"""Files the query stage needs, relative to the SimSwap checkout."""


def external_inputs(failures: list[str]) -> tuple[dict[str, Path], dict[str, Path]]:
    """Source manifests and required files; a config.yaml problem is recorded as a failure."""
    files = dict(DETECTOR_FILES)
    try:
        manifests = source_manifests()
        simswap = simswap_root()
    except (FileNotFoundError, KeyError) as exc:
        failures.append(f"config.yaml: {exc}")
        return {}, files
    files.update({label: simswap / relative for label, relative in SIMSWAP_FILES.items()})
    return manifests, files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-gpu", action="store_true", help="fail unless PyTorch and ONNX Runtime see a CUDA device")
    parser.add_argument("--json", type=Path, help="also write the report here (inside the workflow directory)")
    args = parser.parse_args()

    failures: list[str] = []
    versions: dict[str, str] = {}
    loaded: dict[str, object] = {}
    for package, module_name in MODULES.items():
        try:
            module = importlib.import_module(module_name)
            loaded[module_name] = module
            versions[package] = str(getattr(module, "__version__", "installed"))
        except Exception as exc:  # pylint: disable=broad-except
            failures.append(f"missing/import-failed package {package}: {type(exc).__name__}: {exc}")

    manifests, required_files = external_inputs(failures)
    for dataset, path in manifests.items():
        if not path.is_file() or not os.access(path, os.R_OK):
            failures.append(f"{dataset} source manifest is missing or unreadable: {path}")
    for label, path in required_files.items():
        if not path.is_file() or not os.access(path, os.R_OK):
            failures.append(f"{label} is missing or unreadable: {path}")

    gpu: dict[str, object] = {"required": bool(args.require_gpu)}
    torch = loaded.get("torch")
    if torch is not None:
        cuda_available = bool(torch.cuda.is_available())
        gpu["torch_cuda_available"] = cuda_available
        gpu["torch_device_count"] = int(torch.cuda.device_count()) if cuda_available else 0
        gpu["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if args.require_gpu and not cuda_available:
            failures.append("PyTorch cannot access CUDA in the current process")
    onnxruntime = loaded.get("onnxruntime")
    if onnxruntime is not None:
        providers = list(onnxruntime.get_available_providers())
        gpu["onnxruntime_providers"] = providers
        if args.require_gpu and "CUDAExecutionProvider" not in providers:
            failures.append("ONNX Runtime does not expose CUDAExecutionProvider")

    report = {
        "status": "PASS" if not failures else "FAIL",
        "workflow_root": str(WORKFLOW_ROOT),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "versions": versions,
        "gpu": gpu,
        "source_manifests": {name: str(path) for name, path in manifests.items()},
        "failures": failures,
    }
    if args.json:
        output = args.json.expanduser().resolve()
        try:
            output.relative_to(WORKFLOW_ROOT)
        except ValueError as exc:
            raise ValueError(f"preflight report must stay under {WORKFLOW_ROOT}: {output}") from exc
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
