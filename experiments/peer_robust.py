import os as _dtos
def _dt(env, rest=""):
    v = _dtos.environ.get(env)
    assert v, f"{env} unset: run through run.py, or export it (see config.example.yaml)"
    return v + rest
"""Peer robustness on the same axis as Table 2.

For each peer: embed its mark, make a swapped copy once, then apply the *same*
benign distortions used for our heads to both the genuine and the swapped query
and ask whether the peer's native signal still separates them (detection AUC).

Usage: python peer_robust.py <method> <ds>
"""
import sys, os, glob, tempfile
import numpy as np
import torch
sys.path.insert(0, _dt("DT_SIMSWAP_ROOT"))
sys.path.insert(0, _dt("DT_MODEL_SNAPSHOT"))
from PIL import Image
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score
from pathlib import Path
from scripts.benign_channels import transform_image
sys.path.insert(0, _dt("DT_PEERS_ROOT"))
from engines_api import ENGINES

METHOD = sys.argv[1] if len(sys.argv) > 1 else "facesigns"
DS = sys.argv[2] if len(sys.argv) > 2 else "ffpp"
ENGINE = sys.argv[3] if len(sys.argv) > 3 else "simswap"
N = 60
dev = "cuda"
PERTS = ["clean", "jpeg_q90", "jpeg_q70", "jpeg_q50", "blur_k7",
         "x264_crf30", "x264_crf35", "x264_crf40", "scale_0.5"]
FAM = {"JPEG quality": ["jpeg_q90", "jpeg_q70", "jpeg_q50"],
       "Gaussian blur": ["blur_k7"],
       "Video codec": ["x264_crf30", "x264_crf35", "x264_crf40"],
       "Pixelation": ["scale_0.5"]}


def load():
    """returns embed(pil)->pil , score(pil)->float  (higher = more likely manipulated)"""
    if METHOD == "facesigns":
        D = _dt("DT_PEERS_ROOT", "/FaceSigns"); sys.path.insert(0, D)
        enc = torch.jit.load(f"{D}/encoder_model.pth").eval().to(dev)
        dec = torch.jit.load(f"{D}/decoder_model.pth").eval().to(dev)
        tf = T.Compose([T.Resize((256, 256)), T.ToTensor()])
        sec = torch.from_numpy((np.random.RandomState(0).rand(1, 128) > 0.5).astype(np.float32)).to(dev)
        def embed(p):
            with torch.no_grad():
                wm, _ = enc(tf(p).unsqueeze(0).to(dev), sec)
            return Image.fromarray((wm[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))
        def score(p):
            with torch.no_grad():
                db, _ = dec(tf(p).unsqueeze(0).to(dev))
            return 1.0 - float(((db > 0.5).float() == sec).float().mean())
        return embed, score
    if METHOD == "sepmark":
        sys.path.insert(0, _dt("DT_PEERS_ROOT", "/SepMark"))
        from network.Dual_Mark import Network
        ML = 128
        net = Network(ML, ["Identity()"], ["Identity()"], torch.device(dev), 1, 2e-4, 0.5, "se", "se", [1,10,10,10,0.1])
        _ck = os.environ.get("SEPMARK_CKPT") or _dt("DT_PEERS_ROOT", "/SepMark/pretrained/results/"
                          "Dual_watermark_256_128_0.1_0.0002_0.5_se_se_1_10_10_10_0.1_2023_04_18_16_29_54/models/EC_90.pth")
        net.load_model_ed(_ck)
        net.encoder_decoder.eval()
        nm = T.Compose([T.Resize((256,256)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
        msg = torch.from_numpy(np.random.RandomState(0).choice([-0.1, 0.1], (1, ML)).astype("float32")).to(dev)
        def embed(p):
            with torch.no_grad():
                e = net.encoder_decoder.module.encoder(nm(p).unsqueeze(0).to(dev), msg)
            return Image.fromarray((((e[0].clamp(-1,1)+1)/2).permute(1,2,0).cpu().numpy()*255).astype("uint8"))
        def score(p):
            x = nm(p).unsqueeze(0).to(dev)
            with torch.no_grad():
                dC = net.encoder_decoder.module.decoder_C(x)
                dRF = net.encoder_decoder.module.decoder_RF(x)
            return float((dC.gt(0) != dRF.gt(0)).float().mean())   # tracer/detector mismatch
        return embed, score
    if METHOD == "stegastamp":
        sys.path.insert(0, _dt("DT_PEERS_ROOT", "/StegaStamp_pytorch"))
        import model
        base = DS[:-5] if DS.endswith("tight") else DS
        canon = _dt("DT_CKPTS_ROOT") + f"/stegastamp_canon/{base}"
        pd = _dt("DT_CKPTS_ROOT") + f"/stegastamp_pd/{base}"
        ck = canon if os.path.exists(f"{canon}/encoder.pth") else (
            pd if os.path.exists(f"{pd}/encoder.pth") else _dt("DT_CKPTS_ROOT", "/stegastamp/celebdf"))
        enc = torch.load(f"{ck}/encoder.pth", map_location=dev).eval()
        dec = torch.load(f"{ck}/decoder.pth", map_location=dev).eval()
        tf = T.Compose([T.Resize((400, 400)), T.ToTensor()])
        sec = torch.from_numpy((np.random.RandomState(0).rand(1, 100) > 0.5).astype(np.float32)).to(dev)
        def embed(p):
            x = tf(p).unsqueeze(0).to(dev)
            with torch.no_grad():
                wm = (x + enc((sec, x))).clamp(0, 1)
            return Image.fromarray((wm[0].permute(1,2,0).cpu().numpy()*255).astype("uint8"))
        def score(p):
            with torch.no_grad():
                d = dec(tf(p).unsqueeze(0).to(dev))
            return 1.0 - float(((d > 0.5).float() == sec).float().mean())
        return embed, score
    if METHOD == "lidmark":
        sys.path.insert(0, _dt("DT_PEERS_ROOT", "/LIDMark"))
        from model.lidmark import LIDMarkEncoder, FHD
        IMG, WM = 256, 152
        base = DS[:-5] if DS.endswith("tight") else DS
        ck = _dt("DT_CKPTS_ROOT") + f"/lidmark_canon/{base}/weights/checkpoints_distortions"
        if not os.path.isdir(ck):
            ck = _dt("DT_CKPTS_ROOT") + f"/lidmark/{base}/weights/checkpoints_distortions"
        cks = sorted(glob.glob(f"{ck}/checkpoint_epoch_*.pth"),
                     key=lambda q: int("".join(filter(str.isdigit, os.path.basename(q)))))
        enc = LIDMarkEncoder(128, 64, 3, WM).to(dev).eval(); dec = FHD(128, 64, 1, WM).to(dev).eval()
        sd = torch.load(cks[-1], map_location=dev, weights_only=False)["model_state_dict"]
        enc.load_state_dict({k[8:]: v for k, v in sd.items() if k.startswith("encoder.")})
        dec.load_state_dict({k[8:]: v for k, v in sd.items() if k.startswith("decoder.")})
        pool = f"lidpool_canon_{base}" if "lidmark_canon" in ck else f"lidpool_{base}"
        wmf = sorted(glob.glob(_dt("DT_PEERS_ROOT") + f"/{pool}/wm/128/**/*.npy", recursive=True))
        WMV = torch.from_numpy(np.load(wmf[0]).astype(np.float32)).unsqueeze(0).to(dev)
        gt_lm = WMV[:, :136]
        def _t(p):
            a = np.asarray(p.resize((128, 128)), np.float32) / 127.5 - 1.0
            return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(dev)
        def embed(p):
            with torch.no_grad(): wm = enc(_t(p), WMV)
            a = ((wm[0].permute(1,2,0).cpu().numpy()*0.5+0.5).clip(0,1)*255).astype("uint8")
            return Image.fromarray(a).resize((256, 256))
        def score(p):                      # landmark AED: high = structure moved
            with torch.no_grad(): lm, _ = dec(_t(p))
            d = (lm.view(-1,68,2) - gt_lm.view(-1,68,2)) * 128
            return float(torch.norm(d, p=2, dim=2).mean())
        return embed, score
    if METHOD == "kadnet":
        sys.path.insert(0, _dt("DT_PEERS_ROOT", "/KAD-Net"))
        from network.KAD_Net import Network
        ML = 128
        net = Network("FD", ML, ["Identity()"], ["Identity()"], torch.device(dev), 1, 2e-4, 0.5, "se", "se", [1,10,10,10,0.1])
        _ck = os.environ.get("KADNET_CKPT") or _dt("DT_PEERS_ROOT", "/KAD-Net/results/FD/256/models/EC_100.pth")
        net.load_model_ed(_ck)
        net.encoder_decoder.eval()
        nm = T.Compose([T.Resize((256,256)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
        msg = torch.from_numpy(np.random.RandomState(0).choice([-0.1, 0.1], (1, ML)).astype("float32")).to(dev)
        def embed(p):
            with torch.no_grad(): e = net.encoder_decoder.module.encoder(nm(p).unsqueeze(0).to(dev), msg)
            return Image.fromarray((((e[0].clamp(-1,1)+1)/2).permute(1,2,0).cpu().numpy()*255).astype("uint8"))
        def score(p):   # FD branch: one decoder, robust to benign / fragile to swaps
            x = nm(p).unsqueeze(0).to(dev)
            with torch.no_grad():
                d = net.encoder_decoder.module.decoder_RF(x)
            return float(net.decoded_message_error_rate_batch(msg, d))
        return embed, score
    if METHOD == "fractal":
        # proto-scale reproduction of FractalForensics, trained per dataset here;
        # its native signal is the encrypted-message bit-error from the decoder
        sys.path.insert(0, _dt("DT_MODEL_SNAPSHOT"))
        from src.fractalforensics_reimpl import (FractalEncoder, FractalDecoder,
            generate_fractal_batch, make_message_bits, xor_encrypt_bits, bit_error_rate)
        IMG, MSG = 128, 128
        base = DS[:-5] if DS.endswith("tight") else DS
        ck = torch.load(_dt("DT_CKPTS_ROOT") + f"/fractal/{base}/best.pt",
                        map_location=dev, weights_only=False)
        enc = FractalEncoder(IMG, MSG).to(dev).eval(); dec = FractalDecoder(MSG).to(dev).eval()
        enc.load_state_dict(ck["enc"]); dec.load_state_dict(ck["dec"])
        KEY = 20260828
        fmap = generate_fractal_batch([KEY], IMG, IMG, torch.device(dev))
        bits = make_message_bits([KEY], MSG)
        tb = torch.from_numpy(np.stack([xor_encrypt_bits(bits[0], KEY)])).float().to(dev)
        tf = T.Compose([T.Resize((IMG, IMG)), T.ToTensor()])
        def embed(p):
            with torch.no_grad(): wm, _ = enc(tf(p).unsqueeze(0).to(dev), tb, fmap)
            a = (wm[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            return Image.fromarray(a).resize((256, 256))
        def score(p):
            with torch.no_grad(): m, _ = dec(tf(p).unsqueeze(0).to(dev))
            return float(bit_error_rate(m, tb).mean())
        return embed, score
    if METHOD == "wam":
        # Watermark Anything (Meta), authors' released wam_mit checkpoint.
        # Robust localized watermarking: its native signal is payload bit-error.
        W = _dt("DT_PEERS_ROOT", "/WAM")
        sys.path.insert(0, W); sys.path.insert(0, f"{W}/notebooks")
        cwd = os.getcwd(); os.chdir(W)
        try:
            from notebooks.inference_utils import load_model_from_checkpoint
            from watermark_anything.data.metrics import msg_predict_inference
            wam = load_model_from_checkpoint(f"{W}/checkpoints/params.json",
                                             f"{W}/checkpoints/wam_mit.pth").to(dev).eval()
        finally:
            os.chdir(cwd)
        import torch.nn.functional as _F
        tf = T.Compose([T.Resize((256, 256)), T.ToTensor()])
        torch.manual_seed(0)
        MSG = wam.get_random_msg(1).to(dev)
        def embed(p):
            x = tf(p).unsqueeze(0).to(dev)
            with torch.no_grad(): out = wam.embed(x, MSG)
            w = out["imgs_w"] if isinstance(out, dict) else out
            return Image.fromarray((w[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))
        def score(p):
            with torch.no_grad(): pr = wam.detect(tf(p).unsqueeze(0).to(dev))["preds"]
            m = _F.sigmoid(pr[:, 0]); b = pr[:, 1:]
            pred = msg_predict_inference(b, m).cpu().float()
            return 1.0 - float((pred == MSG.cpu().float()).float().mean())
        return embed, score
    if METHOD == "vine":
        # VINE-R (Lu et al.), authors' released encoder/decoder from HuggingFace.
        # Robust watermarking built on a diffusion prior; native signal is bit-error.
        V = _dt("DT_PEERS_ROOT", "/VINE")
        sys.path.insert(0, V); sys.path.insert(0, f"{V}/vine/src")
        cwd = os.getcwd(); os.chdir(f"{V}/vine/src")
        try:
            from vine_turbo import VINE_Turbo
            from stega_encoder_decoder import CustomConvNeXt
            enc = VINE_Turbo.from_pretrained(f"{V}/ckpt/VINE-R-Enc").to(dev).eval()
            dec = CustomConvNeXt.from_pretrained(f"{V}/ckpt/VINE-R-Dec").to(dev).eval()
        finally:
            os.chdir(cwd)
        tf = T.Compose([T.Resize((256, 256), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor()])
        MSG = torch.tensor(np.random.RandomState(0).randint(0, 2, 100),
                           dtype=torch.float).unsqueeze(0).to(dev)
        def embed(p):
            x = (2.0 * tf(p).unsqueeze(0).to(dev)) - 1.0
            with torch.no_grad(): w = enc(x, MSG)
            a = torch.clamp(w * 0.5 + 0.5, 0, 1)
            return Image.fromarray((a[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))
        def score(p):
            with torch.no_grad(): pr = dec(tf(p).unsqueeze(0).to(dev))
            bits = torch.round(pr[0]).to(MSG.dtype)
            return 1.0 - float((bits == MSG[0]).float().mean())
        return embed, score
    raise SystemExit(f"unknown method {METHOD}")


if __name__ == "__main__":
    embed, score = load()
    swap = ENGINES[ENGINE]()
    if ENGINE != "simswap":
        PERTS = ["clean"]; FAM = {}
    files = sorted(glob.glob(_dt("DT_PEERS_ROOT") + f"/heldout_{DS}/*"))
    pairs = []
    for i, f in enumerate(files):
        if len(pairs) >= N:
            break
        try:
            wm = embed(Image.open(f).convert("RGB"))
            sw = swap(wm, Image.open(files[(i + 7) % len(files)]).convert("RGB"))
            pairs.append((wm, sw.resize(wm.size)))
        except Exception:
            continue

    res = {}
    for p in PERTS:
        ys, ss = [], []
        for k, (b, sw) in enumerate(pairs):
            try:
                ss.append(score(transform_image(b, p, seed=(k+17)*1009))); ys.append(0)
                ss.append(score(transform_image(sw, p, seed=(k+17)*1009))); ys.append(1)
            except Exception:
                continue
        try: res[p] = roc_auc_score(ys, ss)
        except Exception: res[p] = float("nan")

    bs = [score(b) for b, _ in pairs]; ss_ = [score(sw) for _, sw in pairs]
    print(f"== {METHOD} / {DS} / {ENGINE}  n_pairs={len(pairs)}  AUC={res.get('clean', float('nan')):.4f}"
          f"  | benign score {np.mean(bs):.4f}  swapped score {np.mean(ss_):.4f}"
          f"  | direction {'no signal' if abs(np.mean(ss_)-np.mean(bs)) < 1e-9 else ('OK' if np.mean(ss_) > np.mean(bs) else 'INVERTED')}")
    for fam, ps in FAM.items():
        v = [res[x] for x in ps if np.isfinite(res.get(x, np.nan))]
        if v:
            print(f"   {fam:15s} min={min(v):.4f} mean={np.mean(v):.4f} max={max(v):.4f}")
