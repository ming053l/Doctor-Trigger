"""StarGAN-v2 / UniFace / inswapper / SD-inpainting attacks on the canonical releases, scored with
the original-reference agreement H against the records and with the image-only detector. Extends
the recorded-attack set; SimSwap is already canonical. Usage: engines_canonical.py ENGINE
(swaps are cached in <engine_cache>/ENGINE/<enrollment_id>.png)."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root
from doctor_trigger import config  # noqa: E402
from doctor_trigger.manifests import read_cases  # noqa: E402
from doctor_trigger.metrics import roc_auc  # noqa: E402
from doctor_trigger.readings import SizeMismatch, SumForm, query_phase, spectral_agreement  # noqa: E402
from doctor_trigger.record import load_record  # noqa: E402
from experiments.shared.image_only_detector import (deployed_checkpoint_path, image_only_score,  # noqa: E402
                                                    load_image_only_detector)
from experiments.shared.swap_engines import ENGINES  # noqa: E402

ENGINE = sys.argv[1]
EPS = 1e-12
cases = read_cases(split="test", dataset="FF++")


def H_of(pil, tp):
    record = load_record(tp)
    phase = query_phase(pil, record.shape, SizeMismatch.RESIZE, centered=True)
    return spectral_agreement(phase, record.original_phase, record.amplitude_weights, record.active_bins,
                              stabilizer=EPS, form=SumForm.WEIGHT_MAP)


image_detector = load_image_only_detector(deployed_checkpoint_path("FF++"), "cuda")


def dprob(pil):
    return image_only_score(image_detector, pil, "cuda")


eng = ENGINES[ENGINE]()
ys, ss, dd, drop = [], [], [], 0
for eid, g in cases.groupby("enrollment_id"):
    rel = g.doctor_trigger_released_path.iloc[0]; tp = g.doctor_trigger_template_path.iloc[0]
    src = g[g.protocol=="swap"].attack_source_path.iloc[0]
    _cache = Path(config.path("engine_cache") + f"/{ENGINE}"); _cache.mkdir(parents=True, exist_ok=True)
    _cf = _cache / f"{eid}.png"
    try:
        pil = Image.open(rel).convert("RGB")
        hb = H_of(pil, tp)
        if _cf.exists():
            sw = Image.open(_cf).convert("RGB")
        else:
            sw = eng(pil, Image.open(src).convert("RGB"))
            sw.save(_cf)
        hs = H_of(sw, tp)
    except Exception: drop += 1; continue
    ys += [0,1]; ss += [1-hb, 1-hs]; dd += [dprob(pil), dprob(sw)]
fus = [max(a,b) for a,b in zip(dd,ss)]
print(f"{ENGINE}: AUC(1-H)={roc_auc(ys,ss):.4f} AUC(image-only detector)={roc_auc(ys,dd):.4f} AUC(Fus)={roc_auc(ys,fus):.4f} n={len(ys)//2} dropped={drop} "
      f"benignH={1-np.mean([s for s,y in zip(ss,ys) if y==0]):.4f} "
      f"attackH={1-np.mean([s for s,y in zip(ss,ys) if y==1]):.4f}")
