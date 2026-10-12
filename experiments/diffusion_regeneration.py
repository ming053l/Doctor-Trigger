#!/usr/bin/env python
"""Diffusion regeneration against the deployed readings (Table: stress tests).

The attacker does not touch the record; it rebuilds the pixels. Both the released image and its
SimSwap re-rendering pass through the same image-to-image diffusion pass (sd-turbo, no template and
no key), at three strengths, in two scopes:

  global     the whole aligned crop is regenerated
  face-only  only the central face region is regenerated, the border is left untouched

Reported per setting: AUC of the record path (1 - H over the declared sub-band), the image-only
detector, and the standardized alarm, on FF++ test pairs. Chance is 0.500.

Paths from config.yaml (doctor_trigger.config): queries_csv, outputs_root, checkpoints.image_detector_aug;
optional env DT_DIFF_N (default 200). Writes <outputs_root>/adaptive/{diffusion_per_query.csv,diffusion_summary.csv}.
"""
import os, sys
from pathlib import Path

import numpy as np, pandas as pd, torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root
from doctor_trigger import config  # noqa: E402
from doctor_trigger.manifests import read_queries  # noqa: E402
from doctor_trigger.metrics import robust_z, roc_auc  # noqa: E402
from doctor_trigger.readings import SumForm, spectral_agreement  # noqa: E402
from doctor_trigger.record import load_record  # noqa: E402
from doctor_trigger.spectrum import luminance_phase, subband_b  # noqa: E402
from experiments.shared.image_only_detector import (checkpoint_path, image_only_score,  # noqa: E402
                                                    load_image_only_detector)

OUT = Path(config.path("outputs_root")) / "adaptive"
N = int(os.environ.get("DT_DIFF_N", "200")); CUT = 0.15; dev = "cuda"
STRENGTHS = [0.2, 0.3, 0.4]

image_detector = load_image_only_detector(checkpoint_path("ffpp"), dev)

from diffusers import AutoPipelineForImage2Image  # noqa: E402
pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16,
                                                  safety_checker=None).to(dev)
pipe.set_progress_bar_config(disable=True)


def prob(img):
    return image_only_score(image_detector, img, dev)


def rec(tp):
    """Template T_I, weights w and the bins K_eff and kappa <= CUT of a record, uncentered."""
    record = load_record(tp)
    tm = np.fft.ifftshift(record.template_phase)
    bins = np.fft.ifftshift(record.active_bins) & subband_b(tm.shape, centered=False, kappa_max=CUT)
    return tm, np.fft.ifftshift(record.amplitude_weights), bins


def rel_err(img, tm, w, bins):
    """1 - C_B (inline-product summation, as measured)."""
    ph = luminance_phase(np.asarray(img.convert("RGB")), centered=False)
    return 1.0 - spectral_agreement(ph, tm, w, bins, stabilizer=1e-8, form=SumForm.WEIGHT_MAP_VALUE_LAYOUT)


@torch.no_grad()
def purify(img, strength, seed):
    size = img.size
    steps = 8                      # sd-turbo denoises int(steps * strength) times; keep that >= 1
    assert int(steps * strength) >= 1, (steps, strength)
    out = pipe(prompt="a photograph of a face", image=img.resize((512, 512)), strength=strength,
               guidance_scale=0.0, num_inference_steps=steps,
               generator=torch.Generator(dev).manual_seed(seed)).images[0]
    return out.resize(size)


def face_only(img, strength, seed):
    w, h = img.size; m = 0.15                      # keep a border, regenerate the middle
    box = (int(w * m), int(h * m), int(w * (1 - m)), int(h * (1 - m)))
    out = img.copy(); out.paste(purify(img.crop(box), strength, seed).resize((box[2] - box[0], box[3] - box[1])), box)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = read_queries()
    ff = man[(man.dataset == "FF++") & (man.split == "test")]
    pos = ff[ff.protocol == "swap"].head(N); neg = ff[ff.protocol == "benign"].set_index("enrollment_id")
    train = man[(man.dataset == "FF++") & (man.split == "train") & (man.protocol == "benign")].head(200)
    cal_h, cal_d = [], []
    for q in train.itertuples():
        tm, w, bins = rec(q.template_artifact); img = Image.open(q.released_path).convert("RGB")
        cal_h.append(rel_err(img, tm, w, bins)); cal_d.append(prob(img))

    rows = []
    for k, q in enumerate(pos.itertuples(), 1):
        if q.enrollment_id not in neg.index: continue
        tm, w, bins = rec(q.template_artifact)
        for label, path, fake in (("benign", neg.loc[q.enrollment_id].released_path, 0), ("re-rendered", q.query_path, 1)):
            img = Image.open(path).convert("RGB")
            rows.append(dict(query_id=q.query_id, label=label, y=fake, setting="none", strength=0.0,
                             rel_err=rel_err(img, tm, w, bins), S_df=prob(img)))
            for s in STRENGTHS:
                for scope, fn in (("global", purify), ("face-only", face_only)):
                    out = fn(img, s, seed=(k * 31 + fake))
                    rows.append(dict(query_id=q.query_id, label=label, y=fake, setting=scope, strength=s,
                                     rel_err=rel_err(out, tm, w, bins), S_df=prob(out)))
        if k % 20 == 0: print(f"{k}/{len(pos)} pairs", flush=True)

    df = pd.DataFrame(rows)
    df["alarm"] = np.maximum(robust_z(df.rel_err, cal_h), np.minimum(robust_z(df.S_df, cal_d), 4))
    df.to_csv(OUT / "diffusion_per_query.csv", index=False)
    out = []
    for (setting, strength), g in df.groupby(["setting", "strength"]):
        out.append(dict(setting=setting, strength=strength, n_pairs=len(g) // 2,
                        record_only=roc_auc(g.y, g.rel_err), image_only=roc_auc(g.y, g.S_df),
                        alarm=roc_auc(g.y, g.alarm)))
    s = pd.DataFrame(out).sort_values(["setting", "strength"])
    s.to_csv(OUT / "diffusion_summary.csv", index=False); print(s.round(3).to_string(index=False))
    print("=== DIFFUSION DONE ===")


if __name__ == "__main__":
    main()
