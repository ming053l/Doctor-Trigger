"""Face re-rendering operators M used as attacks: ``engine(target, source) -> image``.

Each entry of ``ENGINES`` builds an engine on a device and returns a function from a target PIL
image (the released face to re-render) and a source PIL image (the identity or style donor) to
a PIL image of the target's size. The wrappers hide each engine's own preprocessing so that
experiments can swap one for another:

* ``simswap``   SimSwap identity swap, the deployed query engine
  (``experiments/_snapshot/src/simswap_engine.py``; its ONNX provider order prefers TensorRT,
  unlike ``workflow/repo/src/simswap_engine.py``, which is kept as is per caller)
* ``stargan``   StarGAN-v2 reference-guided synthesis (target re-rendered in the source's style)
* ``uniface``   UniFace identity swap (authors' released ``500000.pt``)
* ``inswapper`` InsightFace inswapper_128 (the roop-family swap)
* ``sdinpaint`` SD-1.5 inpainting over the detected face box, no identity donor

Paths come from ``doctor_trigger.config``: ``engines_root`` (one directory per third-party engine
with its released checkpoints), ``simswap_root``, ``model_snapshot``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image, ImageDraw

from doctor_trigger import config

Engine = Callable[[Image.Image, Image.Image], Image.Image]


def _engines_root() -> str:
    return config.path("engines_root")


def _load_snapshot_simswap():
    file = Path(config.path("model_snapshot")) / "src" / "simswap_engine.py"
    spec = importlib.util.spec_from_file_location("doctor_trigger_snapshot_simswap_engine", file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def simswap(dev: str = "cuda:0") -> Engine:
    """SimSwap at crop size 224, detection threshold 0.3; output resized to the target size."""
    simswap_root = config.path("simswap_root")
    sys.path.insert(0, simswap_root)  # SimSwap's own modules import from its repository root
    engine_module = _load_snapshot_simswap()
    eng = engine_module.SimSwapEngine(engine_module.SimSwapConfig(
        simswap_dir=Path(simswap_root), device=dev, crop_size=224, det_thresh=0.3))

    def run(target_pil: Image.Image, source_pil: Image.Image) -> Image.Image:
        # a private directory per call: concurrent workers must not share scratch paths
        with tempfile.TemporaryDirectory(prefix="eng_") as d:
            tp, sp, op = f"{d}/t.png", f"{d}/s.png", f"{d}/o.png"
            target_pil.save(tp)
            source_pil.save(sp)
            eng.swap(Path(sp), Path(tp), Path(op))
            return Image.open(op).convert("RGB").resize(target_pil.size)
    return run


def stargan(dev: str = "cuda:0") -> Engine:
    """Reference-guided synthesis: the target face is re-rendered in the source's style."""
    root = f"{_engines_root()}/stargan-v2"
    sys.path.insert(0, root)
    cwd = os.getcwd()
    os.chdir(root)
    try:
        from munch import Munch
        from core.model import build_model
        args = Munch(img_size=256, style_dim=64, latent_dim=16, num_domains=2,
                     hidden_dim=512, w_hpf=1, wing_path="expr/checkpoints/wing.ckpt",
                     lm_path="expr/checkpoints/celeba_lm_mean.npz", checkpoint_dir="expr/checkpoints/celeba_hq")
        _, nets_ema = build_model(args)
        ck = torch.load("expr/checkpoints/celeba_hq/100000_nets_ema.ckpt", map_location="cpu",
                        weights_only=False)
        for name, module in nets_ema.items():
            if name in ck:
                sd = ck[name]
                tgt = module.module if hasattr(module, "module") else module
                tgt.load_state_dict({k.replace("module.", "", 1): v for k, v in sd.items()},
                                    strict=False)   # hpf.filter is a rebuilt buffer
            module.to(dev).eval()
    finally:
        os.chdir(cwd)

    def to_input(p: Image.Image) -> torch.Tensor:
        a = np.asarray(p.convert("RGB").resize((256, 256)), np.float32) / 127.5 - 1.0
        return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(dev)

    @torch.no_grad()
    def run(target_pil: Image.Image, source_pil: Image.Image) -> Image.Image:
        x_src, x_ref = to_input(target_pil), to_input(source_pil)
        y_ref = torch.tensor([1], device=dev)                    # female domain of celeba_hq
        s_ref = nets_ema.style_encoder(x_ref, y_ref)
        masks = nets_ema.fan.get_heatmap(x_src) if hasattr(nets_ema, "fan") else None
        x_fake = nets_ema.generator(x_src, s_ref, masks=masks)
        a = ((x_fake[0].clamp(-1, 1).permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).astype("uint8")
        return Image.fromarray(a).resize(target_pil.size)
    return run


def uniface(dev: str = "cuda:0") -> Engine:
    """Identity swap from the authors' released swap checkpoint (500000.pt)."""
    root = f"{_engines_root()}/UniFace"
    sys.path.insert(0, root)
    cwd = os.getcwd()
    os.chdir(root)
    try:
        from training.model import Generator_globalatt_return_32 as Generator
        from training.model import Encoder_return_32 as Encoder
        ck = torch.load("session/swap/checkpoints/500000.pt", map_location="cpu", weights_only=False)
        a = ck["train_args"]
        g = Generator(a.size, a.latent_channel_size, a.latent_spatial_size, lr_mul=a.lr_mul,
                      channel_multiplier=a.channel_multiplier, normalize_mode=a.normalize_mode,
                      small_generator=a.small_generator)
        e = Encoder(a.size, a.latent_channel_size, a.latent_spatial_size,
                    channel_multiplier=a.channel_multiplier)
        g.load_state_dict(ck["g_ema"])
        e.load_state_dict(ck["e_ema"])
        g.to(dev).eval()
        e.to(dev).eval()
        size = a.size
    finally:
        os.chdir(cwd)

    def to_input(p: Image.Image) -> torch.Tensor:
        arr = np.asarray(p.convert("RGB").resize((size, size)), np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(dev)

    @torch.no_grad()
    def run(target_pil: Image.Image, source_pil: Image.Image) -> Image.Image:
        trg, src = to_input(target_pil), to_input(source_pil)
        w, w_feat = e(torch.cat([trg, src], dim=0))
        w_feat_tgt = [torch.chunk(f, 2, dim=0)[0] for f in w_feat][::-1]
        trg_w, src_w = torch.chunk(w, 2, dim=0)
        fake = g([trg_w, src_w, w_feat_tgt])
        arr = ((fake[0].clamp(-1, 1).permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).astype("uint8")
        return Image.fromarray(arr).resize(target_pil.size)
    return run


def inswapper(dev: str = "cuda:0") -> Engine:
    """InsightFace inswapper_128 (the roop-family swap), authors' released ONNX."""
    import insightface
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(320, 320), det_thresh=0.3)
    sw = insightface.model_zoo.get_model(
        f"{_engines_root()}/inswapper/inswapper_128.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    def run(target_pil: Image.Image, source_pil: Image.Image) -> Image.Image:
        tgt = np.asarray(target_pil.convert("RGB"))[:, :, ::-1].copy()
        src = np.asarray(source_pil.convert("RGB"))[:, :, ::-1].copy()
        tf = app.get(tgt)
        sf = app.get(src)
        if not tf or not sf:
            raise ValueError("no face detected")
        out = sw.get(tgt, tf[0], sf[0], paste_back=True)
        return Image.fromarray(out[:, :, ::-1])
    return run


def sdinpaint(dev: str = "cuda:0") -> Engine:
    """Diffusion-based face resynthesis: SD-1.5 inpainting over the detected face region.

    No identity donor is used (``source_pil`` is ignored); the face is re-rendered by the
    diffusion prior, the M_d class of the problem setup. Fixed generator seed 7, 25 steps.
    """
    from diffusers import StableDiffusionInpaintPipeline
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(320, 320), det_thresh=0.3)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        f"{_engines_root()}/sd_inpaint", torch_dtype=torch.float16, safety_checker=None,
        requires_safety_checker=False).to(dev)
    pipe.set_progress_bar_config(disable=True)

    def run(target_pil: Image.Image, source_pil: Image.Image | None = None) -> Image.Image:
        im = target_pil.convert("RGB")
        arr = np.asarray(im)[:, :, ::-1].copy()
        faces = app.get(arr)
        if not faces:
            raise ValueError("no face detected")
        x1, y1, x2, y2 = faces[0].bbox.astype(int)
        mask = Image.new("L", im.size, 0)
        ImageDraw.Draw(mask).rectangle([max(0, x1), max(0, y1), min(im.width, x2), min(im.height, y2)], fill=255)
        out = pipe(prompt="a high quality photo of a person's face",
                   image=im.resize((512, 512)), mask_image=mask.resize((512, 512)),
                   num_inference_steps=25, guidance_scale=7.5,
                   generator=torch.Generator(dev).manual_seed(7)).images[0]
        return out.resize(im.size)
    return run


ENGINES: dict[str, Callable[..., Engine]] = {
    "simswap": simswap, "stargan": stargan, "uniface": uniface, "inswapper": inswapper, "sdinpaint": sdinpaint,
}
