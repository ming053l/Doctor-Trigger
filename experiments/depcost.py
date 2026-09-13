"""Deployment cost per method, using the same loaders the result tables use."""
import os as _dtos
def _dt(env, rest=""):
    v = _dtos.environ.get(env)
    assert v, f"{env} unset: run through run.py, or export it (see config.example.yaml)"
    return v + rest
import sys, os, time, glob, json
import numpy as np, torch
from PIL import Image
sys.path.insert(0, _dt("DT_PEERS_ROOT"))

METHOD = sys.argv[1]
sys.argv = ["x", METHOD, "ffpp"]
import peer_robust

files = sorted(glob.glob(_dt("DT_PEERS_ROOT", "/heldout_ffpp/*")))[:20]
imgs = [Image.open(f).convert("RGB").resize((256, 256)) for f in files]

CKPT = {
 "facesigns":  [_dt("DT_PEERS_ROOT", "/FaceSigns/encoder_model.pth"),
                _dt("DT_PEERS_ROOT", "/FaceSigns/decoder_model.pth")],
 "sepmark":    [_dt("DT_PEERS_ROOT", "/SepMark/pretrained/results/Dual_watermark_256_128_0.1_0.0002_0.5_se_se_1_10_10_10_0.1_2023_04_18_16_29_54/models/EC_90.pth")],
 "stegastamp": [_dt("DT_CKPTS_ROOT", "/stegastamp_pd/ffpp/encoder.pth"),
                _dt("DT_CKPTS_ROOT", "/stegastamp_pd/ffpp/decoder.pth")],
 "kadnet":     [_dt("DT_PEERS_ROOT", "/KAD-Net/results/FD/256/models/EC_100.pth")],
 "lidmark":    [_dt("DT_CKPTS_ROOT", "/lidmark/ffpp/weights/checkpoints_distortions/checkpoint_epoch_9.pth")],
 "wam":        [_dt("DT_PEERS_ROOT", "/WAM/checkpoints/wam_mit.pth")],
 "vine":       [_dt("DT_PEERS_ROOT", "/VINE/ckpt/VINE-R-Enc/model.safetensors"),
                _dt("DT_PEERS_ROOT", "/VINE/ckpt/VINE-R-Dec/CustomConvNeXt.pth")],
}
def ck_bytes(m):
    return sum(os.path.getsize(p) for p in CKPT.get(m, []) if os.path.exists(p))

embed, score = peer_robust.load()
for i in range(3): score(embed(imgs[i]))
torch.cuda.synchronize(); t0 = time.time()
wms = [embed(imgs[i % len(imgs)]) for i in range(20)]
torch.cuda.synchronize(); t1 = time.time()
for w in wms: score(w)
torch.cuda.synchronize(); t2 = time.time()
mem = torch.cuda.max_memory_allocated() / 2**20
print(json.dumps({"method": METHOD, "embed_ms": (t1-t0)/20*1000, "verify_ms": (t2-t1)/20*1000,
                  "ckpt_MiB": ck_bytes(METHOD)/2**20, "peak_gpu_MiB": mem,
                  "per_asset_bytes": 0}))
