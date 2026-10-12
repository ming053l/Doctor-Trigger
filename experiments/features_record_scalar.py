#!/usr/bin/env python
"""Record-model features with the release projection stored in the record as scalars.

The key-projection features (G_s, G_wrong_*, G_margin, G_b*, G_margin_b*) are divided by the
released copy's own projection onto the enrolled displacement, once over the full support and once
per band (9 numbers). Those numbers are fixed when the release is made, so enrollment can store them
in the NPZ record; verification then needs no released pixels. The host/template cosines C and C_b*
are the only features that need a full phase reference, and they read the stored template phase.

Assembly, not re-extraction: with the released-PNG reference (the pipeline's feature CSV) the G family equals
raw projection / release scalar, and with the template reference (features_record_ref.py) it equals
raw projection / template projection, which is 1 up to rounding. So this file takes the G family from
the pipeline's feature CSV and C, C_b* (and every other column) from the template-reference CSV, after checking
that no other column differs and that the two G families differ by one scale per row and band.

Paths: config.yaml (canonical_root, outputs_root). Output: $DT_OUTPUTS/features_record_scalar/recordscalar_features.csv
plus per dataset/split files.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root: doctor_trigger
from doctor_trigger import config  # noqa: E402

R = config.path("canonical_root")
OUTPUTS = Path(config.path("outputs_root"))
OUT = OUTPUTS / "features_record_scalar"
N_BANDS = 8

old = pd.read_csv(f"{R}/features/calibrated_features.csv")
new = pd.read_csv(OUTPUTS / "features_record_ref/recordref_features.csv")
assert len(old) == len(new) and (old.sample_id.values == new.sample_id.values).all(), "row order differs"

C_FAMILY = ["C"] + [f"C_b{i}" for i in range(N_BANDS)]
G_FULL = ["G_s", "G_wrong_mean", "G_wrong_std", "G_wrong_max", "G_margin"]
G_BANDS = [f"G_b{i}" for i in range(N_BANDS)] + [f"G_margin_b{i}" for i in range(N_BANDS)]
EXPECTED = set(C_FAMILY + G_FULL + G_BANDS + ["G_z"])

num = [c for c in old.columns if pd.api.types.is_numeric_dtype(old[c])]
differ = [c for c in num if not np.allclose(old[c], new[c], rtol=1e-9, atol=1e-9, equal_nan=True)]
print(f"columns that differ between the two references: {differ}")
assert set(differ) <= EXPECTED, f"unexpected columns changed: {set(differ) - EXPECTED}"

# one scale per row for the full-support family, one per row and band for the band family
scale = old.G_s / new.G_s
for c in G_FULL[1:4]:
    ok = np.isfinite(old[c]) & (np.abs(new[c]) > 1e-9)
    assert np.allclose(old[c][ok], new[c][ok] * scale[ok], rtol=1e-6, atol=1e-9), c
for i in range(N_BANDS):
    s = old[f"G_b{i}"] / new[f"G_b{i}"]
    ok = np.isfinite(s) & (np.abs(new[f"G_b{i}"]) > 1e-9)
    assert np.allclose(s[ok].groupby(old.template_id[ok]).std().fillna(0), 0, atol=1e-6), f"band {i} scale not per record"
print("G family: one release scalar per record over the support and per band (checked)")
b = old.protocol == "benign"
print(f"release scalar / template projection on benign rows: median {np.median((new.G_s / old.G_s)[b]):.3f}")

out = new.copy()
for c in G_FULL + G_BANDS + ["G_z"]:
    out[c] = old[c]
OUT.mkdir(parents=True, exist_ok=True)
main_csv = OUT / "recordscalar_features.csv"
out.to_csv(main_csv, index=False)
for (ds, split), part in out.groupby(["dataset", "split"]):
    part.to_csv(OUT / f"{main_csv.stem}_{str(ds).lower().replace('+', 'p')}_{split}.csv", index=False)
print(f"wrote {main_csv} rows={len(out)}")
print("=== FEATURES RECORD-SCALAR DONE ===")
