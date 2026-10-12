#!/usr/bin/env python
"""Direct relation test at fixed false-alarm rates (appendix operating-points table).

The deployed reading (sub-band, against the stored template) scored on benign queries as the
negatives and re-rendered queries as the positives, per re-rendering engine and per corpus:
AUC, and the recall left at a false-alarm rate of 1e-2 and 1e-3 on the benign side.

Engines come from the cached swaps of the engines experiment; SimSwap rows are the canonical swap queries
queries of each corpus, scored against each asset's own record.

Paths: config.yaml (queries_csv, outputs_root, engine_cache). Writes $DT_OUTPUTS/operating_points.csv.
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np, pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root: doctor_trigger
from doctor_trigger import config  # noqa: E402
from doctor_trigger.manifests import read_queries, select  # noqa: E402
from doctor_trigger.metrics import roc_auc, tpr_at_fpr  # noqa: E402
from doctor_trigger.readings import SumForm, spectral_agreement  # noqa: E402
from doctor_trigger.record import load_record  # noqa: E402
from doctor_trigger.spectrum import luminance_phase, subband_b  # noqa: E402

OUT = Path(config.path("outputs_root"))
CACHE = Path(config.path("engine_cache"))
SUBBAND_STABILIZER = 1e-8


def rel_err(args):
    """1 - C_B of an image against a record, summed in the unshifted layout; NaN on a size mismatch."""
    path, tp = args
    record = load_record(tp)
    tm = np.fft.ifftshift(record.template_phase)
    ph = luminance_phase(np.asarray(Image.open(path).convert("RGB"), dtype=np.float64), centered=False)
    if ph.shape != tm.shape: return np.nan
    bins = np.fft.ifftshift(record.active_bins) & subband_b(record.shape, centered=False)
    agreement = spectral_agreement(ph, tm, np.fft.ifftshift(record.amplitude_weights), bins,
                                   stabilizer=SUBBAND_STABILIZER, form=SumForm.WEIGHT_MAP_VALUE_LAYOUT)
    return 1.0 - agreement


def main():
    man = read_queries()
    rows = []
    with Pool(24) as p:
        neg = {}
        for ds in ("FF++", "CelebDF", "DFDC"):
            b = select(man, dataset=ds, split="test", protocol="benign")
            neg[ds] = (b, np.array(p.map(rel_err, [(q.released_path, q.template_artifact) for q in b.itertuples()])))
            t = select(man, dataset=ds, split="test", protocol="swap")
            pos = np.array(p.map(rel_err, [(q.query_path, q.template_artifact) for q in t.itertuples()]))
            rows.append(("SimSwap", ds, neg[ds][1], pos))
            print(f"SimSwap {ds}: n={len(pos)}", flush=True)
        b, bs = neg["FF++"]
        for eng in sorted(d.name for d in CACHE.iterdir() if d.is_dir()):
            pairs = [(CACHE / eng / f"{q.enrollment_id}.png", q.template_artifact)
                     for q in b.itertuples() if (CACHE / eng / f"{q.enrollment_id}.png").exists()]
            pos = np.array(p.map(rel_err, pairs))
            rows.append((eng, "FF++", bs, pos))
            print(f"{eng} FF++: n={len(pos)}", flush=True)

    out = []
    for name, ds, negs, poss in rows:
        negs, poss = negs[np.isfinite(negs)], poss[np.isfinite(poss)]
        y = np.r_[np.zeros(len(negs)), np.ones(len(poss))]; s = np.r_[negs, poss]
        out.append(dict(re_rendering=name, dataset=ds, n_pos=len(poss), n_neg=len(negs),
                        auc=roc_auc(y, s), tpr_1e2=tpr_at_fpr(y, s, 1e-2), tpr_1e3=tpr_at_fpr(y, s, 1e-3)))
    df = pd.DataFrame(out)
    df.to_csv(OUT / "operating_points.csv", index=False)
    print(df.round(4).to_string(index=False)); print("=== OPERATING POINTS DONE ===")


if __name__ == "__main__":
    main()
