#!/usr/bin/env python
"""Direct relation test at fixed false-alarm rates (appendix operating-points table).

The deployed reading (sub-band, against the stored template) scored on benign queries as the
negatives and re-rendered queries as the positives, per re-rendering engine and per corpus:
AUC, and the recall left at a false-alarm rate of 1e-2 and 1e-3 on the benign side.

Engines come from the cached swaps of the engines experiment; SimSwap rows are the canonical swap queries
queries of each corpus, scored against each asset's own record.

Env: DT_CANON_ROOT, DT_OUTPUTS, DT_ENGINE_CACHE. Writes $DT_OUTPUTS/operating_points.csv.
"""
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np, pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve

R = os.environ["DT_CANON_ROOT"]; OUT = Path(os.environ["DT_OUTPUTS"])
CACHE = Path(os.environ["DT_ENGINE_CACHE"])
CUT = 0.15


def rel_err(args):
    path, tp = args
    with np.load(tp) as z:
        tm = np.fft.ifftshift(z["template_phase"].astype(np.float64))
        sup = np.fft.ifftshift(z["support"].astype(bool))
        amp = np.fft.ifftshift(z["host_amplitude"].astype(np.float64))
    fy = np.fft.fftfreq(tm.shape[0])[:, None]; fx = np.fft.fftfreq(tm.shape[1])[None, :]
    w = np.where(sup & (np.sqrt(fy * fy + fx * fx) / np.sqrt(0.5) <= CUT), amp, 0.0)
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    lum = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
    ph = np.angle(np.fft.fft2(lum))
    if ph.shape != tm.shape: return np.nan
    return 1.0 - float((np.cos((ph - tm + np.pi) % (2 * np.pi) - np.pi) * w).sum() / (w.sum() + 1e-8))


def tpr_at(y, s, target):
    fpr, tpr, _ = roc_curve(y, s); ok = fpr <= target
    return float(tpr[ok].max()) if ok.any() else 0.0


def main():
    man = pd.read_csv(f"{R}/manifests/queries.csv"); man = man[man.status == "pass"]
    rows = []
    with Pool(24) as p:
        neg = {}
        for ds in ("FF++", "CelebDF", "DFDC"):
            b = man[(man.dataset == ds) & (man.split == "test") & (man.protocol == "benign")]
            neg[ds] = (b, np.array(p.map(rel_err, [(q.released_path, q.template_artifact) for q in b.itertuples()])))
            t = man[(man.dataset == ds) & (man.split == "test") & (man.protocol == "swap")]
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
                        auc=roc_auc_score(y, s), tpr_1e2=tpr_at(y, s, 1e-2), tpr_1e3=tpr_at(y, s, 1e-3)))
    df = pd.DataFrame(out)
    df.to_csv(OUT / "operating_points.csv", index=False)
    print(df.round(4).to_string(index=False)); print("=== OPERATING POINTS DONE ===")


if __name__ == "__main__":
    main()
