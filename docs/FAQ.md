# FAQ

## Is this a watermark?
Not in the usual sense. A watermark carries a message that a decoder recovers from the image alone.
Doctor Trigger hides no message: it writes a weak keyed signal into the Fourier phase and keeps a
private record of the face (its spectral reference and key). Verification compares a circulating
copy with *that release's* record. The weak signal is only measurable because the record is
available, which is what lets the release stay near-identical to the original (51.2 dB PSNR on
average in the paper).

## Does it detect AI-generated or deepfake images in general?
No. It answers a narrower question: *is this circulating face still the version the publisher
released?* It can only check faces that were enrolled before release, and only against their own
records. A face that was never enrolled gets no verdict from it. The paper adds an optional,
separate image-only detector as a review flag; that detector is not the method.

## What do the two scores mean?
- *Spectral agreement* `C_B` (`C` in the quick start): amplitude-weighted agreement of the copy's
  luminance Fourier phase with the stored template T_I, on the active bins within the low sub-band
  B (κ ≤ 0.15). A face swap or generative edit lowers it.
- *Key projection* `G_s`: projection of the copy's phase drift from the original phase onto the
  enrolled displacement δ_s, whose direction is the key's binary ±1 sign pattern h_s. Normalized by
  the release's own projection (G_s^(release)) it is about 1 for the release, 0 for the
  never-released original, and around 0 on average for a wrong key.

Both are computed in [`doctor_trigger/readings.py`](../doctor_trigger/readings.py);
`python -m doctor_trigger verify IMAGE RECORD` prints them for one copy.

A copy is **Confirmed** only when both pass; otherwise it is **Unconfirmed** and goes to a person
for review. **Unavailable** means the check could not run (record missing, face or geometry
unresolved, or an unsupported channel), not that the copy failed.

## Why do I need the key projection if `C` already catches swaps?
The unreleased original looks like the release, so any similarity test (C_B, SSIM, ArcFace) that
tolerates JPEG also accepts the original (about 99% in the paper). Only the release carries the
keyed signal; `G_s` rejects every original while accepting 99.5% of JPEG releases.

## What must I keep private?
The per-face records R (the NPZ files: template T_I, original phase and amplitude weights, keyed
displacement δ_s, sign pattern h_s, key s) and the master seed the keys are derived from. The record is image-derived:
phase-only reconstruction of the template reveals coarse facial structure (paper App. K.4), so treat
it as sensitive data. Only the released image is public.

## How are thresholds set?
On benign copies of your own releases, over the channels you intend to support (JPEG, resizing,
...), at about 1% benign rejection (`python -m doctor_trigger calibrate SCORES.csv --column C_B`). Thresholds do not transfer automatically between corpora or
channels; the paper's Table 14 measures how far they do. The thresholds in
[`examples/quickstart.py`](../examples/quickstart.py) are a toy calibration on one AI-generated face.

## Does it survive cropping, video codecs, or many reposts?
Partly. See the Limitations section of the [README](../README.md#limitations-and-responsible-use)
and Table 4 of the paper.

## Does it work on video?
Not yet; video is future work and the paper evaluates single images (plus one-frame H.264/H.265
channels). The package keeps the door open: an enrolled unit is `UnitId(asset_id, frame_index)`
with `frame_index=None` for an image, `RecordStore` maps each unit to its own record file, readings
are per unit (`UnitReading`) and `decision.aggregate_status` combines unit statuses (today only the
single-unit rule), channels are composable per-image transforms (`channels.compose`), and
`enrollment.enroll_many` streams units. A video module would add per-frame enrollment, clip-level
aggregation and calibration, and clip codecs on top of these.

## Can I try it without the datasets?
Yes: after `pip install -e .`, `python examples/quickstart.py` enrolls a bundled AI-generated face and verifies a few copies
on the CPU in a few seconds. See [`examples/README.md`](../examples/README.md).
