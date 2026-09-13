"""Deployment cost of the enrolled path: closed-form relation, plus the two heads."""
import os as _dtos
def _dt(env, rest=""):
    v = _dtos.environ.get(env)
    assert v, f"{env} unset: run through run.py, or export it (see config.example.yaml)"
    return v + rest
import sys, os, glob, json, time
import numpy as np, torch, yaml
sys.path.insert(0, _dt("DT_MODEL_SNAPSHOT"))
from PIL import Image
from src.trigger_conditioned import (RingSpec, compute_delta_s, recover_host_phase,
                                     wrap_phase, compute_host_score)
IMG = 256
recs = [json.load(open(j)) for j in sorted(glob.glob(
    _dt("DT_LEGACY_TRIGGER_GLOB"), recursive=True))[:25]]
recs = [d for d in recs if os.path.exists(d['trigger_path'])]
imgs = [Image.open(d['trigger_path']).convert("RGB").resize((IMG, IMG)) for d in recs[:20]]
d0 = recs[0]
ann = RingSpec(shape=(IMG, IMG), bands=tuple(d0['bands']), half_plane=True)
dl = compute_delta_s(int(d0['phase_seed']), float(d0['alpha']), float(d0['beta']), ann, "zero_pi")
sup = dl.support

def phase_of(p):
    return np.angle(np.fft.fft2(np.asarray(p, np.float32).mean(2)))

# enrollment: one FFT, a keyed phase nudge, one inverse FFT
def enroll(p):
    a = np.asarray(p, np.float32).mean(2)
    F = np.fft.fft2(a); ph = np.angle(F)
    ph2 = ph.copy(); ph2[sup] = wrap_phase(ph[sup] + dl.delta[sup])
    return np.real(np.fft.ifft2(np.abs(F) * np.exp(1j * ph2))), ph2

# verification: undo the nudge, one cosine mean. No network at all.
def verify(p, tmpl):
    return compute_host_score(phase_of(p), recover_host_phase(tmpl, dl), sup)

for i in range(3): verify(imgs[i], enroll(imgs[i])[1])
t0 = time.time(); tm = [enroll(p)[1] for p in imgs]; t1 = time.time()
for p, t in zip(imgs, tm): verify(p, t)
t2 = time.time()

from src import models
cfg = yaml.safe_load(open(_dt("DT_MODEL_SNAPSHOT", "/config_ffpp_std_dfd_resnet50.yaml")))
image_detector = models.build_model(cfg)
image_detector_params = sum(q.numel() for q in image_detector.parameters())
record_model_params = 75*64 + 64 + 64*64 + 64 + 3*(64 + 1) + 2*(64*8 + 8)  # hidden 64 plus two 8-d embeddings
image_detector_ckpt = _dt("DT_IMAGE_DETECTOR_LEGACY")
print(json.dumps({
 "enroll_ms_cpu": (t1-t0)/len(imgs)*1000,
 "relation_verify_ms_cpu": (t2-t1)/len(imgs)*1000,
 "relation_params": 0,
 "record_model_params": record_model_params,
 "image_detector_params": image_detector_params,
 "image_detector_ckpt_MiB": os.path.getsize(image_detector_ckpt)/2**20 if os.path.exists(image_detector_ckpt) else None,
 "per_asset_template_bins": int(sup.sum()),
 "per_asset_KiB_fp32": int(sup.sum())*4/1024,
 "per_asset_KiB_fp16": int(sup.sum())*2/1024,
}, indent=1))
