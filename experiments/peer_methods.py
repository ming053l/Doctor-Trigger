"""Peer watermarking methods behind the Table 2 and Table 16 peer rows: embed and score per method.

``load_peer(method, dataset)`` returns the peer's own embedding and its native verification score,
loaded from the peer's repository (config key ``peers_root``) and from checkpoints trained on the
evaluation split (config key ``ckpts_root``). Used by peer_detection.py and deployment_cost_peers.py.
Methods: facesigns, sepmark, stegastamp, lidmark, kadnet, fractal (FractalForensics
re-implementation in experiments/_snapshot), wam, vine.
"""
import glob
import os
import sys

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from doctor_trigger import config

PEERS_ROOT = config.path("peers_root")
CKPTS_ROOT = config.path("ckpts_root")


def load_peer(method, dataset, device="cuda"):
    """Peer ``method`` with its checkpoint for ``dataset`` (ffpp | dfdc | celebdf).

    Returns ``embed(pil) -> pil`` (the peer's marked release) and ``score(pil) -> float`` (the
    peer's native signal, higher = more likely manipulated).
    """
    if method == "facesigns":
        D = PEERS_ROOT + "/FaceSigns"; sys.path.insert(0, D)
        enc = torch.jit.load(f"{D}/encoder_model.pth").eval().to(device)
        dec = torch.jit.load(f"{D}/decoder_model.pth").eval().to(device)
        tf = T.Compose([T.Resize((256, 256)), T.ToTensor()])
        sec = torch.from_numpy((np.random.RandomState(0).rand(1, 128) > 0.5).astype(np.float32)).to(device)
        def embed(p):
            with torch.no_grad():
                wm, _ = enc(tf(p).unsqueeze(0).to(device), sec)
            return Image.fromarray((wm[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))
        def score(p):
            with torch.no_grad():
                db, _ = dec(tf(p).unsqueeze(0).to(device))
            return 1.0 - float(((db > 0.5).float() == sec).float().mean())
        return embed, score
    if method == "sepmark":
        sys.path.insert(0, PEERS_ROOT + "/SepMark")
        from network.Dual_Mark import Network
        ML = 128
        net = Network(ML, ["Identity()"], ["Identity()"], torch.device(device), 1, 2e-4, 0.5, "se", "se", [1,10,10,10,0.1])
        _ck = os.environ.get("SEPMARK_CKPT") or (PEERS_ROOT + "/SepMark/pretrained/results/"
                          "Dual_watermark_256_128_0.1_0.0002_0.5_se_se_1_10_10_10_0.1_2023_04_18_16_29_54/models/EC_90.pth")
        net.load_model_ed(_ck)
        net.encoder_decoder.eval()
        nm = T.Compose([T.Resize((256,256)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
        msg = torch.from_numpy(np.random.RandomState(0).choice([-0.1, 0.1], (1, ML)).astype("float32")).to(device)
        def embed(p):
            with torch.no_grad():
                e = net.encoder_decoder.module.encoder(nm(p).unsqueeze(0).to(device), msg)
            return Image.fromarray((((e[0].clamp(-1,1)+1)/2).permute(1,2,0).cpu().numpy()*255).astype("uint8"))
        def score(p):
            x = nm(p).unsqueeze(0).to(device)
            with torch.no_grad():
                dC = net.encoder_decoder.module.decoder_C(x)
                dRF = net.encoder_decoder.module.decoder_RF(x)
            return float((dC.gt(0) != dRF.gt(0)).float().mean())   # tracer/detector mismatch
        return embed, score
    if method == "stegastamp":
        sys.path.insert(0, PEERS_ROOT + "/StegaStamp_pytorch")
        import model  # noqa: F401  (StegaStamp classes, needed to unpickle the checkpoints)
        canon = CKPTS_ROOT + f"/stegastamp_canon/{dataset}"
        pd = CKPTS_ROOT + f"/stegastamp_pd/{dataset}"
        ck = canon if os.path.exists(f"{canon}/encoder.pth") else (
            pd if os.path.exists(f"{pd}/encoder.pth") else CKPTS_ROOT + "/stegastamp/celebdf")
        enc = torch.load(f"{ck}/encoder.pth", map_location=device).eval()
        dec = torch.load(f"{ck}/decoder.pth", map_location=device).eval()
        tf = T.Compose([T.Resize((400, 400)), T.ToTensor()])
        sec = torch.from_numpy((np.random.RandomState(0).rand(1, 100) > 0.5).astype(np.float32)).to(device)
        def embed(p):
            x = tf(p).unsqueeze(0).to(device)
            with torch.no_grad():
                wm = (x + enc((sec, x))).clamp(0, 1)
            return Image.fromarray((wm[0].permute(1,2,0).cpu().numpy()*255).astype("uint8"))
        def score(p):
            with torch.no_grad():
                d = dec(tf(p).unsqueeze(0).to(device))
            return 1.0 - float(((d > 0.5).float() == sec).float().mean())
        return embed, score
    if method == "lidmark":
        sys.path.insert(0, PEERS_ROOT + "/LIDMark")
        from model.lidmark import LIDMarkEncoder, FHD
        IMG, WM = 256, 152
        ck = CKPTS_ROOT + f"/lidmark_canon/{dataset}/weights/checkpoints_distortions"
        if not os.path.isdir(ck):
            ck = CKPTS_ROOT + f"/lidmark/{dataset}/weights/checkpoints_distortions"
        cks = sorted(glob.glob(f"{ck}/checkpoint_epoch_*.pth"),
                     key=lambda q: int("".join(filter(str.isdigit, os.path.basename(q)))))
        enc = LIDMarkEncoder(128, 64, 3, WM).to(device).eval(); dec = FHD(128, 64, 1, WM).to(device).eval()
        sd = torch.load(cks[-1], map_location=device, weights_only=False)["model_state_dict"]
        enc.load_state_dict({k[8:]: v for k, v in sd.items() if k.startswith("encoder.")})
        dec.load_state_dict({k[8:]: v for k, v in sd.items() if k.startswith("decoder.")})
        pool = f"lidpool_canon_{dataset}" if "lidmark_canon" in ck else f"lidpool_{dataset}"
        wmf = sorted(glob.glob(PEERS_ROOT + f"/{pool}/wm/128/**/*.npy", recursive=True))
        WMV = torch.from_numpy(np.load(wmf[0]).astype(np.float32)).unsqueeze(0).to(device)
        gt_lm = WMV[:, :136]
        def _t(p):
            a = np.asarray(p.resize((128, 128)), np.float32) / 127.5 - 1.0
            return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device)
        def embed(p):
            with torch.no_grad(): wm = enc(_t(p), WMV)
            a = ((wm[0].permute(1,2,0).cpu().numpy()*0.5+0.5).clip(0,1)*255).astype("uint8")
            return Image.fromarray(a).resize((256, 256))
        def score(p):                      # landmark AED: high = structure moved
            with torch.no_grad(): lm, _ = dec(_t(p))
            d = (lm.view(-1,68,2) - gt_lm.view(-1,68,2)) * 128
            return float(torch.norm(d, p=2, dim=2).mean())
        return embed, score
    if method == "kadnet":
        sys.path.insert(0, PEERS_ROOT + "/KAD-Net")
        from network.KAD_Net import Network
        ML = 128
        net = Network("FD", ML, ["Identity()"], ["Identity()"], torch.device(device), 1, 2e-4, 0.5, "se", "se", [1,10,10,10,0.1])
        _ck = os.environ.get("KADNET_CKPT") or PEERS_ROOT + "/KAD-Net/results/FD/256/models/EC_100.pth"
        net.load_model_ed(_ck)
        net.encoder_decoder.eval()
        nm = T.Compose([T.Resize((256,256)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
        msg = torch.from_numpy(np.random.RandomState(0).choice([-0.1, 0.1], (1, ML)).astype("float32")).to(device)
        def embed(p):
            with torch.no_grad(): e = net.encoder_decoder.module.encoder(nm(p).unsqueeze(0).to(device), msg)
            return Image.fromarray((((e[0].clamp(-1,1)+1)/2).permute(1,2,0).cpu().numpy()*255).astype("uint8"))
        def score(p):   # FD branch: one decoder, robust to benign / fragile to swaps
            x = nm(p).unsqueeze(0).to(device)
            with torch.no_grad():
                d = net.encoder_decoder.module.decoder_RF(x)
            return float(net.decoded_message_error_rate_batch(msg, d))
        return embed, score
    if method == "fractal":
        # proto-scale reproduction of FractalForensics, trained per dataset here;
        # its native signal is the encrypted-message bit-error from the decoder
        from experiments._snapshot.src.fractalforensics_reimpl import (FractalEncoder, FractalDecoder,
            generate_fractal_batch, make_message_bits, xor_encrypt_bits, bit_error_rate)
        IMG, MSG = 128, 128
        ck = torch.load(CKPTS_ROOT + f"/fractal/{dataset}/best.pt",
                        map_location=device, weights_only=False)
        enc = FractalEncoder(IMG, MSG).to(device).eval(); dec = FractalDecoder(MSG).to(device).eval()
        enc.load_state_dict(ck["enc"]); dec.load_state_dict(ck["dec"])
        KEY = 20260828
        fmap = generate_fractal_batch([KEY], IMG, IMG, torch.device(device))
        bits = make_message_bits([KEY], MSG)
        tb = torch.from_numpy(np.stack([xor_encrypt_bits(bits[0], KEY)])).float().to(device)
        tf = T.Compose([T.Resize((IMG, IMG)), T.ToTensor()])
        def embed(p):
            with torch.no_grad(): wm, _ = enc(tf(p).unsqueeze(0).to(device), tb, fmap)
            a = (wm[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            return Image.fromarray(a).resize((256, 256))
        def score(p):
            with torch.no_grad(): m, _ = dec(tf(p).unsqueeze(0).to(device))
            return float(bit_error_rate(m, tb).mean())
        return embed, score
    if method == "wam":
        # Watermark Anything (Meta), authors' released wam_mit checkpoint.
        # Robust localized watermarking: its native signal is payload bit-error.
        W = PEERS_ROOT + "/WAM"
        sys.path.insert(0, W); sys.path.insert(0, f"{W}/notebooks")
        cwd = os.getcwd(); os.chdir(W)
        try:
            from notebooks.inference_utils import load_model_from_checkpoint
            from watermark_anything.data.metrics import msg_predict_inference
            wam = load_model_from_checkpoint(f"{W}/checkpoints/params.json",
                                             f"{W}/checkpoints/wam_mit.pth").to(device).eval()
        finally:
            os.chdir(cwd)
        import torch.nn.functional as _F
        tf = T.Compose([T.Resize((256, 256)), T.ToTensor()])
        torch.manual_seed(0)
        MSG = wam.get_random_msg(1).to(device)
        def embed(p):
            x = tf(p).unsqueeze(0).to(device)
            with torch.no_grad(): out = wam.embed(x, MSG)
            w = out["imgs_w"] if isinstance(out, dict) else out
            return Image.fromarray((w[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))
        def score(p):
            with torch.no_grad(): pr = wam.detect(tf(p).unsqueeze(0).to(device))["preds"]
            m = _F.sigmoid(pr[:, 0]); b = pr[:, 1:]
            pred = msg_predict_inference(b, m).cpu().float()
            return 1.0 - float((pred == MSG.cpu().float()).float().mean())
        return embed, score
    if method == "vine":
        # VINE-R (Lu et al.), authors' released encoder/decoder from HuggingFace.
        # Robust watermarking built on a diffusion prior; native signal is bit-error.
        V = PEERS_ROOT + "/VINE"
        sys.path.insert(0, V); sys.path.insert(0, f"{V}/vine/src")
        cwd = os.getcwd(); os.chdir(f"{V}/vine/src")
        try:
            from vine_turbo import VINE_Turbo
            from stega_encoder_decoder import CustomConvNeXt
            enc = VINE_Turbo.from_pretrained(f"{V}/ckpt/VINE-R-Enc").to(device).eval()
            dec = CustomConvNeXt.from_pretrained(f"{V}/ckpt/VINE-R-Dec").to(device).eval()
        finally:
            os.chdir(cwd)
        tf = T.Compose([T.Resize((256, 256), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor()])
        MSG = torch.tensor(np.random.RandomState(0).randint(0, 2, 100),
                           dtype=torch.float).unsqueeze(0).to(device)
        def embed(p):
            x = (2.0 * tf(p).unsqueeze(0).to(device)) - 1.0
            with torch.no_grad(): w = enc(x, MSG)
            a = torch.clamp(w * 0.5 + 0.5, 0, 1)
            return Image.fromarray((a[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))
        def score(p):
            with torch.no_grad(): pr = dec(tf(p).unsqueeze(0).to(device))
            bits = torch.round(pr[0]).to(MSG.dtype)
            return 1.0 - float((bits == MSG[0]).float().mean())
        return embed, score
    raise ValueError(f"unknown peer method {method!r}")
