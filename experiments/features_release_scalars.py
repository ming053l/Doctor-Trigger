#!/usr/bin/env python
"""Record-model features with the nine release-normalization scalars stored in the record.

The key-projection features (G_s, G_wrong_*, G_margin, G_b*, G_margin_b*) are divided by the
encoded release's own projection onto the enrolled displacement, once over the full support and
once per band (9 numbers). Those numbers are fixed when the release is made, so enrollment can store
them in the record; verification then needs no released pixels. The agreements C and C_b* are the
only features that need a full phase reference, and they read the stored template phase T_I.

Assembly, not re-extraction: with the released-PNG reference (the pipeline's feature CSV) the G
family equals raw projection / release scalar, and with the template reference
(features_template_reference.py) it equals raw projection / template projection, which is 1 up to
rounding. So this file takes the G family from the pipeline's feature CSV and C, C_b* (and every
other column) from the template-reference CSV, after checking that no other column differs and that
the two G families differ by one scale per row and band.

Paths: config.yaml (canonical_root, outputs_root). Output:
$DT_OUTPUTS/features_record_scalar/recordscalar_features.csv plus per dataset/split files.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from doctor_trigger import config

PIPELINE_ROOT = config.path("canonical_root")
OUTPUTS = Path(config.path("outputs_root"))
OUT = OUTPUTS / "features_record_scalar"
BAND_COUNT = 8

released_reference = pd.read_csv(f"{PIPELINE_ROOT}/features/calibrated_features.csv")
template_reference = pd.read_csv(OUTPUTS / "features_record_ref/recordref_features.csv")
assert len(released_reference) == len(template_reference) and \
    (released_reference.sample_id.values == template_reference.sample_id.values).all(), "row order differs"

C_FAMILY = ["C"] + [f"C_b{i}" for i in range(BAND_COUNT)]
G_FULL = ["G_s", "G_wrong_mean", "G_wrong_std", "G_wrong_max", "G_margin"]
G_BANDS = [f"G_b{i}" for i in range(BAND_COUNT)] + [f"G_margin_b{i}" for i in range(BAND_COUNT)]
EXPECTED = set(C_FAMILY + G_FULL + G_BANDS + ["G_z"])

numeric = [c for c in released_reference.columns if pd.api.types.is_numeric_dtype(released_reference[c])]
differ = [c for c in numeric if not np.allclose(released_reference[c], template_reference[c],
                                                 rtol=1e-9, atol=1e-9, equal_nan=True)]
print(f"columns that differ between the two references: {differ}")
assert set(differ) <= EXPECTED, f"unexpected columns changed: {set(differ) - EXPECTED}"

# one scale per row for the full-support family, one per row and band for the band family
scale = released_reference.G_s / template_reference.G_s
for c in G_FULL[1:4]:
    ok = np.isfinite(released_reference[c]) & (np.abs(template_reference[c]) > 1e-9)
    assert np.allclose(released_reference[c][ok], template_reference[c][ok] * scale[ok], rtol=1e-6, atol=1e-9), c
for i in range(BAND_COUNT):
    band_scale = released_reference[f"G_b{i}"] / template_reference[f"G_b{i}"]
    ok = np.isfinite(band_scale) & (np.abs(template_reference[f"G_b{i}"]) > 1e-9)
    per_record_spread = band_scale[ok].groupby(released_reference.template_id[ok]).std().fillna(0)
    assert np.allclose(per_record_spread, 0, atol=1e-6), f"band {i} scale not per record"
print("G family: one release scalar per record over the support and per band (checked)")
benign = released_reference.protocol == "benign"
print(f"release scalar / template projection on benign rows: "
      f"median {np.median((template_reference.G_s / released_reference.G_s)[benign]):.3f}")

features = template_reference.copy()
for c in G_FULL + G_BANDS + ["G_z"]:
    features[c] = released_reference[c]
OUT.mkdir(parents=True, exist_ok=True)
main_csv = OUT / "recordscalar_features.csv"
features.to_csv(main_csv, index=False)
for (dataset, split), part in features.groupby(["dataset", "split"]):
    part.to_csv(OUT / f"{main_csv.stem}_{str(dataset).lower().replace('+', 'p')}_{split}.csv", index=False)
print(f"wrote {main_csv} rows={len(features)}")
print("=== FEATURES RECORD-SCALAR DONE ===")
