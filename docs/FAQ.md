# FAQ

**Is this a watermark?**
Not in the usual sense. A watermark carries a message that a decoder recovers from the image alone.
Doctor Trigger hides no message: it writes a weak keyed signal into the Fourier phase and keeps a
private record of the face (its spectral reference and key). Verification compares a circulating
copy with *that release's* record. The weak signal is only measurable because the record is
available, which is what lets the release stay near-identical to the original (51.2 dB PSNR on
average in the paper).

**Does it detect AI-generated or deepfake images in general?**
No. It answers a narrower question: *is this circulating face still the version the publisher
released?* It can only check faces that were enrolled before release, and only against their own
records. A face that was never enrolled gets no verdict from it. The paper adds an optional,
separate image-only detector as a review flag; that detector is not the method.

**What do the two scores mean?**
- *Face consistency* `C` (the paper's C_B on the sub-band): agreement of the copy's Fourier phase
  with the stored template. A face swap or generative edit lowers it.
- *Release signal* `G_s`: projection of the copy's phase change onto the enrolled key. About 1 for
  the release, 0 for the never-released original, and around 0 on average for a wrong key.

A copy is **Confirmed** only when both pass; otherwise it is **Unconfirmed** and goes to a person
for review. **Unavailable** means the check could not run (record missing, face or geometry
unresolved, or an unsupported channel), not that the copy failed.

**Why do I need the key projection if `C` already catches swaps?**
The unreleased original looks like the release, so any similarity test (C, SSIM, ArcFace) that
tolerates JPEG also accepts the original (about 99% in the paper). Only the release carries the
keyed signal; `G_s` rejects every original while accepting 99.5% of JPEG releases.

**What must I keep private?**
The per-face records (the NPZ files: template phase, original phase and amplitude, keyed
displacement, key) and the master seed the keys are derived from. The record is image-derived:
phase-only reconstruction of the template reveals coarse facial structure (paper App. K.4), so treat
it as sensitive data. Only the released image is public.

**How are thresholds set?**
On benign copies of your own releases, over the channels you intend to support (JPEG, resizing,
...), at about 1% benign rejection. Thresholds do not transfer automatically between corpora or
channels; the paper's Table 14 measures how far they do. The thresholds in
[`examples/quickstart.py`](../examples/quickstart.py) are a toy calibration on one AI-generated face.

**Does it survive cropping, video codecs, or many reposts?**
Partly. See the Limitations section of the [README](../README.md#limitations-and-responsible-use)
and Table 4 of the paper.

**Can I try it without the datasets?**
Yes: `python examples/quickstart.py` enrolls a bundled AI-generated face and verifies a few copies
on the CPU in a few seconds. See [`examples/README.md`](../examples/README.md).
