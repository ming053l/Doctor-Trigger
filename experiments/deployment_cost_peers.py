"""Deployment cost of each peer method (Table 16 peer rows): GPU embed/verify time, weights, memory.

Each peer embeds and scores 20 held-out FF++ crops (<peers_root>/heldout_ffpp, resized to
256 x 256) after three warm-up iterations, through the same loaders as the result tables
(peer_methods.load_peer). Prints one JSON line per method: embed_ms, verify_ms, ckpt_MiB (size of
the checkpoint files on disk), peak_gpu_MiB (torch peak allocation), per_asset_bytes (0: peers keep
no per-asset record).

Usage: deployment_cost_peers.py [METHOD ...]
With no METHOD, every Table 16 peer is measured, each in its own process so that peak GPU memory
is per method.
"""
import glob
import json
import os
import subprocess
import sys
import time

TABLE16_PEERS = ["stegastamp", "facesigns", "sepmark", "wam", "kadnet", "vine", "lidmark"]
TIMED_IMAGES = 20
WARM_UP = 3


def checkpoint_files(peers_root, ckpts_root):
    return {
        "facesigns":  [peers_root + "/FaceSigns/encoder_model.pth",
                       peers_root + "/FaceSigns/decoder_model.pth"],
        "sepmark":    [peers_root + "/SepMark/pretrained/results/Dual_watermark_256_128_0.1_0.0002_0.5_se_se_1_10_10_10_0.1_2023_04_18_16_29_54/models/EC_90.pth"],
        "stegastamp": [ckpts_root + "/stegastamp_pd/ffpp/encoder.pth",
                       ckpts_root + "/stegastamp_pd/ffpp/decoder.pth"],
        "kadnet":     [peers_root + "/KAD-Net/results/FD/256/models/EC_100.pth"],
        "lidmark":    [ckpts_root + "/lidmark/ffpp/weights/checkpoints_distortions/checkpoint_epoch_9.pth"],
        "wam":        [peers_root + "/WAM/checkpoints/wam_mit.pth"],
        "vine":       [peers_root + "/VINE/ckpt/VINE-R-Enc/model.safetensors",
                       peers_root + "/VINE/ckpt/VINE-R-Dec/CustomConvNeXt.pth"],
    }


def measure(method):
    import torch
    from PIL import Image

    from doctor_trigger import config
    from experiments.peer_methods import load_peer

    peers_root, ckpts_root = config.path("peers_root"), config.path("ckpts_root")
    files = sorted(glob.glob(peers_root + "/heldout_ffpp/*"))[:TIMED_IMAGES]
    images = [Image.open(f).convert("RGB").resize((256, 256)) for f in files]
    checkpoint_bytes = sum(os.path.getsize(p) for p in checkpoint_files(peers_root, ckpts_root).get(method, [])
                           if os.path.exists(p))

    embed, score = load_peer(method, "ffpp")
    for i in range(WARM_UP):
        score(embed(images[i]))
    torch.cuda.synchronize(); t0 = time.time()
    marked = [embed(images[i % len(images)]) for i in range(TIMED_IMAGES)]
    torch.cuda.synchronize(); t1 = time.time()
    for image in marked:
        score(image)
    torch.cuda.synchronize(); t2 = time.time()
    print(json.dumps({"method": method, "embed_ms": (t1 - t0) / TIMED_IMAGES * 1000,
                      "verify_ms": (t2 - t1) / TIMED_IMAGES * 1000,
                      "ckpt_MiB": checkpoint_bytes / 2**20,
                      "peak_gpu_MiB": torch.cuda.max_memory_allocated() / 2**20,
                      "per_asset_bytes": 0}))


if __name__ == "__main__":
    if len(sys.argv) == 2:
        measure(sys.argv[1])
    else:
        failed = [m for m in (sys.argv[1:] or TABLE16_PEERS)
                  if subprocess.run([sys.executable, os.path.abspath(__file__), m]).returncode != 0]
        if failed:
            sys.exit(f"failed: {' '.join(failed)}")
