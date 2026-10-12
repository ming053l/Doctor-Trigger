"""SimSwap wrapper for single-image face swaps (the attack engine of the experiments).

Used through experiments/shared/swap_engines.py. Its ONNX Runtime provider order prefers TensorRT;
the pipeline's copy (workflow/repo/src/simswap_engine.py) prefers CUDA. Each caller keeps the order
its results were produced with."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

def _totensor(array_rgb: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(array_rgb).permute(2, 0, 1).contiguous()
    return tensor.float().div(255.0)


def _patch_onnxruntime_providers() -> None:
    """Ensure onnxruntime uses CUDA providers by default when available.

    Newer onnxruntime builds require explicitly passing providers to
    `InferenceSession` when CUDA/TensorRT are enabled.
    """
    try:
        import onnxruntime  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return

    available = onnxruntime.get_available_providers()  # type: ignore[attr-defined]
    preferred = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    default_providers = [p for p in preferred if p in available] or ["CPUExecutionProvider"]

    original_session = onnxruntime.InferenceSession  # type: ignore[attr-defined]

    class PatchedSession(original_session):  # type: ignore[valid-type]
        def __init__(self, *args, providers=None, **kwargs):
            super().__init__(*args, providers=providers or default_providers, **kwargs)

    onnxruntime.InferenceSession = PatchedSession  # type: ignore[attr-defined]


@dataclass
class SimSwapConfig:
    simswap_dir: Path
    device: str = "cuda:0"
    crop_size: int = 224
    det_thresh: float = 0.6
    det_size: Tuple[int, int] = (640, 640)
    model_name: str = "people"


class SimSwapEngine:
    """Reuse the SimSwap model + detector for repeated swaps."""

    def __init__(self, cfg: SimSwapConfig) -> None:
        self.cfg = cfg
        self.simswap_dir = cfg.simswap_dir.resolve()
        if not self.simswap_dir.exists():
            raise FileNotFoundError(f"SimSwap directory not found: {self.simswap_dir}")

        _patch_onnxruntime_providers()

        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        if str(self.simswap_dir) not in sys.path:
            sys.path.insert(0, str(self.simswap_dir))

        from models.models import create_model  # type: ignore
        from options.test_options import TestOptions  # type: ignore
        from util.reverse2original import reverse2wholeimage  # type: ignore
        from insightface_func.face_detect_crop_single import Face_detect_crop  # type: ignore

        self._reverse2wholeimage = reverse2wholeimage

        transformer_Arcface = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self._arcface_transform = transformer_Arcface

        detector_root = self.simswap_dir / "insightface_func" / "models"
        self.detector = Face_detect_crop(name="antelope", root=str(detector_root))
        mode = "ffhq" if cfg.crop_size == 512 else "None"
        self.detector.prepare(ctx_id=0, det_thresh=cfg.det_thresh, det_size=cfg.det_size, mode=mode)

        checkpoints_dir = self.simswap_dir / "checkpoints"
        arc_path = self.simswap_dir / "arcface_model" / "arcface_checkpoint.tar"
        if not arc_path.exists():
            raise FileNotFoundError(f"SimSwap ArcFace checkpoint missing: {arc_path}")

        options = TestOptions()
        options.initialize()
        opt = options.parser.parse_args(
            [
                "--name",
                cfg.model_name,
                "--checkpoints_dir",
                str(checkpoints_dir),
                "--Arc_path",
                str(arc_path),
                "--crop_size",
                str(cfg.crop_size),
                "--no_simswaplogo",
            ]
        )
        opt.isTrain = False
        str_ids = str(opt.gpu_ids).split(",")
        opt.gpu_ids = []
        for str_id in str_ids:
            try:
                idx = int(str_id)
            except ValueError:
                continue
            if idx >= 0:
                opt.gpu_ids.append(idx)
        if opt.gpu_ids and self.device.type == "cuda":
            torch.cuda.set_device(opt.gpu_ids[0])
        torch.nn.Module.dump_patches = True
        self.model = create_model(opt)
        self.model.eval()

    def _latent_id(self, source_bgr: np.ndarray) -> torch.Tensor:
        crop_size = int(self.cfg.crop_size)
        aligned, _ = self.detector.get(source_bgr, crop_size)
        if aligned is None:
            raise RuntimeError("SimSwap: no face detected in source image")
        face_bgr = aligned[0]
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        img_a = self._arcface_transform(face_pil)
        img_id = img_a.view(1, img_a.shape[0], img_a.shape[1], img_a.shape[2]).to(self.device)
        with torch.no_grad():
            img_id_downsample = F.interpolate(img_id, size=(112, 112))
            latend_id = self.model.netArc(img_id_downsample)
            latend_id = F.normalize(latend_id, p=2, dim=1)
        return latend_id

    def swap(self, source_path: Path, target_path: Path, output_path: Path) -> None:
        crop_size = int(self.cfg.crop_size)
        source_bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if source_bgr is None:
            raise FileNotFoundError(f"SimSwap: failed to read source image: {source_path}")
        target_bgr = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
        if target_bgr is None:
            raise FileNotFoundError(f"SimSwap: failed to read target image: {target_path}")

        latend_id = self._latent_id(source_bgr)

        target_align_crop_list, mats = self.detector.get(target_bgr, crop_size)
        if target_align_crop_list is None or mats is None:
            raise RuntimeError("SimSwap: no face detected in target image")

        swap_result_list = []
        b_align_crop_tensor_list = []
        with torch.no_grad():
            for b_align_crop in target_align_crop_list:
                b_rgb = cv2.cvtColor(b_align_crop, cv2.COLOR_BGR2RGB)
                b_tensor = _totensor(b_rgb)[None, ...].to(self.device)
                swap_result = self.model(None, b_tensor, latend_id, None, True)[0]
                swap_result_list.append(swap_result)
                b_align_crop_tensor_list.append(b_tensor)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._reverse2wholeimage(
            b_align_crop_tensor_list,
            swap_result_list,
            mats,
            crop_size,
            target_bgr,
            None,
            str(output_path),
            True,
            pasring_model=None,
            norm=None,
            use_mask=False,
        )
