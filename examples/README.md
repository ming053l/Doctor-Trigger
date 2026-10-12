# Quick start without datasets

`quickstart.py` enrolls one face and verifies a few copies of it, on the CPU, in a few seconds,
with the `doctor_trigger` package:

- **enroll**: `doctor_trigger.enrollment.enroll()` derives the key s, applies the sign pattern h_s
  (±1 on the active ring bins) to the face's phase and returns the release and the private record R,
  with the pipeline's default settings and fidelity gate (PSNR ≥ 35 dB, SSIM ≥ 0.98).
- **verify**: `spectral_agreement()` against the template T_I on the sub-band κ ≤ 0.15 (face
  consistency `C`, the paper's C_B) and `key_projection()` of the phase drift onto δ_s (release
  signal `G_s`), from `doctor_trigger.readings`. The quickstart reads `G_s` with unit weights and
  without release normalization, so a release reads about 1.5 rather than about 1.
- **decide**: `doctor_trigger.decision.unit_status()`: Confirmed if `C ≥ τ_C` and `G_s ≥ τ_G`;
  Unconfirmed otherwise; Unavailable if the check cannot run (here: the copy's geometry does not
  match the record).

## Requirements

Python ≥ 3.10 with numpy, Pillow and scikit-image. From the repository root:

```bash
pip install -e .                 # or: export PYTHONPATH=.
python examples/quickstart.py
```

Tested with Python 3.10, numpy 1.26.4, Pillow 10.4.0, scikit-image 0.25.2. No GPU, no model
weights, no dataset.

## Output (real run)

```
[enroll]    office.png: 384x384, alpha=0.050, PSNR 51.15 dB, SSIM 0.9978 (released vs original)
[calibrate] TOY: 1% quantiles over 45 benign copies (JPEG q70-90, resize 0.6-0.9x) of 5 enrollments of market.png
            -> tau_C=0.9989, tau_G=0.868  (in practice: calibrate on your own releases)

copy                                              C     G_s  status
posted (released PNG)                        0.9999   1.529  Confirmed
JPEG re-save, q85                            0.9995   1.018  Confirmed
resized 0.75x and back                       0.9998   1.539  Confirmed
unreleased original                          0.9997   0.000  Unconfirmed (release signal missing)
face region replaced                         0.7468   0.716  Unconfirmed (face changed, release signal missing)
center crop 90% (no record of geometry)           -       -  Unavailable
(thresholds)                                 0.9989   0.868
```

How to read it:

- The **posted** image and its benign copies keep both the face (`C`) and the keyed signal (`G_s`).
- The **unreleased original** has the same face (`C` passes) but no keyed signal (`G_s` = 0): it was
  never published, so it is not confirmed.
- **Face region replaced** is a crude splice of another face into the release, a stand-in so this
  example needs no face-swap model; the paper's queries use SimSwap. `C` drops sharply.
- The **center crop** no longer matches the record's geometry, so no verdict is given. (The paper
  evaluates crops with a known crop mask; see its Table 4.)

## Thresholds: this is a toy calibration

`τ_C` and `τ_G` decide everything, and they must be calibrated on benign copies of **your own
releases**, over the channels you intend to support (the paper uses 1% benign rejection on training
releases). The script stands in for that with 45 benign copies of five enrollments made from a
second face (`market.png`, mirrored and rotated, each with its own key). Do not reuse these numbers.

## Options

```bash
python examples/quickstart.py --image my_face.png        # a square, face-centred crop
python examples/quickstart.py --out quickstart_out       # also save the images and the private record
```

`--out` writes `record_PRIVATE.npz` in the pipeline's record format, so the package CLI can read it:

```bash
python -m doctor_trigger verify quickstart_out/copy1.png quickstart_out/record_PRIVATE.npz
```

The record is image-derived (its phase reveals coarse facial structure), so keep it private.

## Sample images

`data/office.png` and `data/market.png` are **AI-generated people, not real persons**, made for the
Doctor Trigger project page and resized to 384 px. They are not from any benchmark dataset.
