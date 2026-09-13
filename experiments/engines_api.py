import os as _dtos
def _dt(env, rest=""):
    v = _dtos.environ.get(env)
    assert v, f"{env} unset: run through run.py, or export it (see config.example.yaml)"
    return v + rest
"""Uniform manipulation-engine wrappers: engine(target_pil, source_pil) -> pil.

SimSwap comes from the deployed engine; StarGAN-v2 and UniFace are loaded from
the authors' released checkpoints. Each wrapper hides the engine's own
preprocessing so peer_robust.py can swap one for another.
"""
import os, sys, tempfile
import numpy as np
import torch
from PIL import Image
from pathlib import Path

ENG = _dt("DT_PEERS_ROOT", "/engines")


def simswap(dev="cuda:0"):
    sys.path.insert(0, _dt("DT_SIMSWAP_ROOT"))
    sys.path.insert(0, _dt("DT_MODEL_SNAPSHOT"))
    from src.simswap_engine import SimSwapConfig, SimSwapEngine
    eng = SimSwapEngine(SimSwapConfig(simswap_dir=Path(_dt("DT_SIMSWAP_ROOT")),
                                      device=dev, crop_size=224, det_thresh=0.3))
    def run(target_pil, source_pil):
        # a private directory per call: concurrent workers must not share scratch paths
        with tempfile.TemporaryDirectory(prefix="eng_") as d:
            tp, sp, op = f"{d}/t.png", f"{d}/s.png", f"{d}/o.png"
            target_pil.save(tp); source_pil.save(sp)
            eng.swap(Path(sp), Path(tp), Path(op))
            return Image.open(op).convert("RGB").resize(target_pil.size)
    return run


def stargan(dev="cuda:0"):
    """Reference-guided synthesis: the target face is re-rendered in the source's style."""
    root = f"{ENG}/stargan-v2"
    sys.path.insert(0, root)
    cwd = os.getcwd(); os.chdir(root)
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

    def _t(p):
        a = np.asarray(p.convert("RGB").resize((256, 256)), np.float32) / 127.5 - 1.0
        return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(dev)

    @torch.no_grad()
    def run(target_pil, source_pil):
        x_src, x_ref = _t(target_pil), _t(source_pil)
        y_ref = torch.tensor([1], device=dev)                    # female domain of celeba_hq
        s_ref = nets_ema.style_encoder(x_ref, y_ref)
        masks = nets_ema.fan.get_heatmap(x_src) if hasattr(nets_ema, "fan") else None
        x_fake = nets_ema.generator(x_src, s_ref, masks=masks)
        a = ((x_fake[0].clamp(-1, 1).permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).astype("uint8")
        return Image.fromarray(a).resize(target_pil.size)
    return run


def uniface(dev="cuda:0"):
    """Identity swap from the authors' released swap checkpoint (500000.pt)."""
    root = f"{ENG}/UniFace"
    sys.path.insert(0, root)
    cwd = os.getcwd(); os.chdir(root)
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
        g.load_state_dict(ck["g_ema"]); e.load_state_dict(ck["e_ema"])
        g.to(dev).eval(); e.to(dev).eval()
        size = a.size
    finally:
        os.chdir(cwd)

    def _t(p):
        arr = np.asarray(p.convert("RGB").resize((size, size)), np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(dev)

    @torch.no_grad()
    def run(target_pil, source_pil):
        trg, src = _t(target_pil), _t(source_pil)
        w, w_feat = e(torch.cat([trg, src], dim=0))
        w_feat_tgt = [torch.chunk(f, 2, dim=0)[0] for f in w_feat][::-1]
        trg_w, src_w = torch.chunk(w, 2, dim=0)
        fake = g([trg_w, src_w, w_feat_tgt])
        arr = ((fake[0].clamp(-1, 1).permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255).astype("uint8")
        return Image.fromarray(arr).resize(target_pil.size)
    return run




def inswapper(dev="cuda:0"):
    """InsightFace inswapper_128 (the roop-family swap), authors' released ONNX."""
    import insightface
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(320, 320), det_thresh=0.3)
    sw = insightface.model_zoo.get_model(
        f"{ENG}/inswapper/inswapper_128.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    def run(target_pil, source_pil):
        import numpy as _np
        tgt = _np.asarray(target_pil.convert("RGB"))[:, :, ::-1].copy()
        src = _np.asarray(source_pil.convert("RGB"))[:, :, ::-1].copy()
        tf = app.get(tgt); sf = app.get(src)
        if not tf or not sf:
            raise ValueError("no face detected")
        out = sw.get(tgt, tf[0], sf[0], paste_back=True)
        return Image.fromarray(out[:, :, ::-1])
    return run



def sdinpaint(dev="cuda:0"):
    """Diffusion-based face resynthesis: SD-1.5 inpainting over the detected face
    region. No identity donor is used; the face is re-rendered by the diffusion
    prior, which is the M_d class the problem setup declares."""
    import torch as _torch, numpy as _np
    from diffusers import StableDiffusionInpaintPipeline
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(320, 320), det_thresh=0.3)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        f"{ENG}/sd_inpaint", torch_dtype=_torch.float16, safety_checker=None,
        requires_safety_checker=False).to(dev)
    pipe.set_progress_bar_config(disable=True)

    def run(target_pil, source_pil=None):
        im = target_pil.convert("RGB")
        arr = _np.asarray(im)[:, :, ::-1].copy()
        faces = app.get(arr)
        if not faces:
            raise ValueError("no face detected")
        x1, y1, x2, y2 = faces[0].bbox.astype(int)
        mask = Image.new("L", im.size, 0)
        from PIL import ImageDraw
        ImageDraw.Draw(mask).rectangle([max(0, x1), max(0, y1), min(im.width, x2), min(im.height, y2)], fill=255)
        out = pipe(prompt="a high quality photo of a person's face",
                   image=im.resize((512, 512)), mask_image=mask.resize((512, 512)),
                   num_inference_steps=25, guidance_scale=7.5,
                   generator=_torch.Generator(dev).manual_seed(7)).images[0]
        return out.resize(im.size)
    return run

ENGINES = {"simswap": simswap, "stargan": stargan, "uniface": uniface, "inswapper": inswapper, "sdinpaint": sdinpaint}
