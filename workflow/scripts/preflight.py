#!/usr/bin/env python3
"""Fail-fast environment and external-input checks for the canonical workflow."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]

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

SOURCE_MANIFESTS = {
    "FF++": Path(os.environ.get("DT_MANIFEST_FFPP", "")),
    "DFDC": Path(os.environ.get("DT_MANIFEST_DFDC", "")),
    "CelebDF": Path(os.environ.get("DT_MANIFEST_CELEBDF", "")),
}

REQUIRED_FILES = {
    "SCRFD detector": ROOT / "assets/detectors/scrfd_10g_bnkps.onnx",
    "YuNet detector": ROOT / "assets/detectors/face_detection_yunet_2023mar.onnx",
    "SimSwap generator": Path(os.environ.get("DT_SIMSWAP_ROOT", "")) / "checkpoints/people/latest_net_G.pth",
    "SimSwap ArcFace": Path(os.environ.get("DT_SIMSWAP_ROOT", "")) / "arcface_model/arcface_checkpoint.tar",
    "SimSwap detector": Path(os.environ.get("DT_SIMSWAP_ROOT", "")) / "insightface_func/models/antelope/scrfd_10g_bnkps.onnx",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-gpu", action="store_true", help="Fail unless PyTorch and ONNX Runtime expose CUDA.")
    parser.add_argument("--json", type=Path, help="Optional report path inside the workflow root.")
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

    for dataset, path in SOURCE_MANIFESTS.items():
        if not path.is_file() or not os.access(path, os.R_OK):
            failures.append(f"{dataset} source manifest is missing or unreadable: {path}")
    for label, path in REQUIRED_FILES.items():
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
        "workflow_root": str(ROOT),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "versions": versions,
        "gpu": gpu,
        "source_manifests": {name: str(path) for name, path in SOURCE_MANIFESTS.items()},
        "failures": failures,
    }
    if args.json:
        output = args.json.expanduser().resolve()
        try:
            output.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise ValueError(f"preflight report must stay under {ROOT}: {output}") from exc
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
