#!/usr/bin/env python
"""Diffusion regeneration against the deployed readings (Table: stress tests).

The attacker does not touch the record; it rebuilds the pixels. Both the released image and its
SimSwap re-rendering pass through the same image-to-image diffusion pass (sd-turbo, no template and
no key), at three strengths, in two scopes:

  global     the whole aligned crop is regenerated
  face-only  only the central face region is regenerated, the border is left untouched

Reported per setting: AUC of the record path (1 - H over the declared sub-band), the image-only
detector, and the standardized alarm, on FF++ test pairs. Chance is 0.500.

Env: DT_CANON_ROOT, DT_OUTPUTS, DT_MODEL_SNAPSHOT, DT_IMAGE_DETECTOR_AUG; optional DT_DIFF_N (default 200).
Writes $DT_OUTPUTS/adaptive/{diffusion_per_query.csv,diffusion_summary.csv}.
"""
import os, sys
from pathlib import Path

import numpy as np, pandas as pd, torch, yaml
from PIL import Image
from sklearn.metrics import roc_auc_score
import torchvision.transforms as T


def _dt(env, rest=""):
    v = os.environ.get(env); assert v, f"{env} unset"; return v + rest


R = _dt("DT_CANON_ROOT"); OUT = Path(_dt("DT_OUTPUTS")) / "adaptive"
N = int(os.environ.get("DT_DIFF_N", "200")); CUT = 0.15; dev = "cuda"
STRENGTHS = [0.2, 0.3, 0.4]

sys.path.insert(0, _dt("DT_MODEL_SNAPSHOT"))
from src import models  # noqa: E402
cfg = yaml.safe_load(open(_dt("DT_MODEL_SNAPSHOT", "/config_ffpp_std_dfd_resnet50.yaml")))
image_detector = models.build_model(cfg).to(dev)
image_detector.load_state_dict(torch.load(_dt("DT_IMAGE_DETECTOR_AUG", "/ffpp/best_val_auc.pt"), map_location=dev, weights_only=False))
image_detector.eval()
tf = T.Compose([T.Resize((224, 224)), T.ToTensor()])

from diffusers import AutoPipelineForImage2Image  # noqa: E402
pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=torch.float16,
                                                  safety_checker=None).to(dev)
pipe.set_progress_bar_config(disable=True)


@torch.no_grad()
def prob(img):
    o = image_detector(tf(img).unsqueeze(0).to(dev))
    lg = o["logits"] if isinstance(o, dict) else o
    return float(torch.softmax(lg, 1)[0, 1]) if lg.ndim > 1 and lg.shape[1] == 2 else float(torch.sigmoid(lg.reshape(-1))[0])


def rec(tp):
    with np.load(tp) as z:
        tm = np.fft.ifftshift(z["template_phase"].astype(np.float64))
        sup = np.fft.ifftshift(z["support"].astype(bool))
        amp = np.fft.ifftshift(z["host_amplitude"].astype(np.float64))
    fy = np.fft.fftfreq(tm.shape[0])[:, None]; fx = np.fft.fftfreq(tm.shape[1])[None, :]
    return tm, np.where(sup & (np.sqrt(fy * fy + fx * fx) / np.sqrt(0.5) <= CUT), amp, 0.0)


def rel_err(img, tm, w):
    a = np.asarray(img.convert("RGB"), dtype=np.float64)
    lum = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
    ph = np.angle(np.fft.fft2(lum))
    return 1.0 - float((np.cos((ph - tm + np.pi) % (2 * np.pi) - np.pi) * w).sum() / (w.sum() + 1e-8))


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


def rz(v, cal):
    cal = np.asarray(cal); med = np.median(cal); mad = np.median(np.abs(cal - med)) + 1e-6
    return (np.asarray(v) - med) / mad


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(f"{R}/manifests/queries.csv"); man = man[man.status == "pass"]
    ff = man[(man.dataset == "FF++") & (man.split == "test")]
    pos = ff[ff.protocol == "swap"].head(N); neg = ff[ff.protocol == "benign"].set_index("enrollment_id")
    train = man[(man.dataset == "FF++") & (man.split == "train") & (man.protocol == "benign")].head(200)
    cal_h, cal_d = [], []
    for q in train.itertuples():
        tm, w = rec(q.template_artifact); img = Image.open(q.released_path).convert("RGB")
        cal_h.append(rel_err(img, tm, w)); cal_d.append(prob(img))

    rows = []
    for k, q in enumerate(pos.itertuples(), 1):
        if q.enrollment_id not in neg.index: continue
        tm, w = rec(q.template_artifact)
        for label, path, fake in (("benign", neg.loc[q.enrollment_id].released_path, 0), ("re-rendered", q.query_path, 1)):
            img = Image.open(path).convert("RGB")
            rows.append(dict(query_id=q.query_id, label=label, y=fake, setting="none", strength=0.0,
                             rel_err=rel_err(img, tm, w), S_df=prob(img)))
            for s in STRENGTHS:
                for scope, fn in (("global", purify), ("face-only", face_only)):
                    out = fn(img, s, seed=(k * 31 + fake))
                    rows.append(dict(query_id=q.query_id, label=label, y=fake, setting=scope, strength=s,
                                     rel_err=rel_err(out, tm, w), S_df=prob(out)))
        if k % 20 == 0: print(f"{k}/{len(pos)} pairs", flush=True)

    df = pd.DataFrame(rows)
    df["alarm"] = np.maximum(rz(df.rel_err, cal_h), np.minimum(rz(df.S_df, cal_d), 4))
    df.to_csv(OUT / "diffusion_per_query.csv", index=False)
    out = []
    for (setting, strength), g in df.groupby(["setting", "strength"]):
        out.append(dict(setting=setting, strength=strength, n_pairs=len(g) // 2,
                        record_only=roc_auc_score(g.y, g.rel_err), image_only=roc_auc_score(g.y, g.S_df),
                        alarm=roc_auc_score(g.y, g.alarm)))
    s = pd.DataFrame(out).sort_values(["setting", "strength"])
    s.to_csv(OUT / "diffusion_summary.csv", index=False); print(s.round(3).to_string(index=False))
    print("=== DIFFUSION DONE ===")


if __name__ == "__main__":
    main()
