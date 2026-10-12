"""Deployment cost of the enrolled path: closed-form relation, plus the two heads.

Timing is measured on the earlier trigger renders (config key legacy_trigger_glob) with the legacy
key path of workflow/record_model/src/trigger_conditioned.py (i.i.d. +-1 code on a half-plane
ring), as for the reported numbers; the record-model and image-only detector sizes are counted
from their definitions."""
import importlib.util
import sys, os, glob, json, time
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # repository root
from doctor_trigger import config  # noqa: E402
from experiments.shared.image_only_detector import build_image_only_detector, checkpoint_path  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "trigger_conditioned", REPO_ROOT / "workflow" / "record_model" / "src" / "trigger_conditioned.py")
trigger_conditioned = sys.modules["trigger_conditioned"] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trigger_conditioned)
RingSpec, compute_delta_s = trigger_conditioned.RingSpec, trigger_conditioned.compute_delta_s
recover_host_phase, wrap_phase = trigger_conditioned.recover_host_phase, trigger_conditioned.wrap_phase
compute_host_score = trigger_conditioned.compute_host_score

IMG = 256
recs = [json.load(open(j)) for j in sorted(glob.glob(config.path("legacy_trigger_glob"), recursive=True))[:25]]
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

image_detector = build_image_only_detector()
image_detector_params = sum(q.numel() for q in image_detector.parameters())
record_model_params = 75*64 + 64 + 64*64 + 64 + 3*(64 + 1) + 2*(64*8 + 8)  # hidden 64 plus two 8-d embeddings
image_detector_ckpt = checkpoint_path("ffpp", augmented=True)   # deployed detector; every head has the same size
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
