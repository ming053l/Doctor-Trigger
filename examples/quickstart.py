#!/usr/bin/env python
"""Doctor Trigger in one image, without any dataset.

  python examples/quickstart.py                       # uses the bundled AI-generated faces
  python examples/quickstart.py --image my_face.png   # your own square face crop
  python examples/quickstart.py --out quickstart_out  # also write the images and the record

What it does, using the doctor_trigger package:

  1. enroll   doctor_trigger.enrollment.enroll(): derives the face's key s from a master seed and
              the sample id, writes the weak keyed phase signal into the released crop and returns
              the private record R, with the pipeline's fidelity gate (PSNR >= 35 dB, SSIM >= 0.98,
              trying alpha 0.050 -> 0.030 -> 0.015).
  2. calibrate  TOY calibration (doctor_trigger.calibration.lower_tail_threshold): thresholds are the
              1% quantiles of C_B and G_s over benign copies (JPEG q70-90, resizing) of five
              enrollments made from a *different* face. A real deployment calibrates on benign copies
              of its own releases, over the channels it intends to support (the paper: 1% benign
              rejection on training releases).
  3. verify   doctor_trigger.readings:
                C_B = spectral_agreement() against the stored template T_I on K_eff and the sub-band
                      kappa <= 0.15 ("face consistency").
                G_s = key_projection() of the phase drift onto the enrolled keyed displacement
                      ("release signal"). Here it is unweighted and not release-normalized, so a
                      release reads somewhat above 1 and the original about 0.
  4. decide   doctor_trigger.decision.unit_status(): Confirmed if C_B >= tau_C and G_s >= tau_G;
              Unconfirmed otherwise; Unavailable when the check cannot run (no record, or the
              copy's geometry does not match the record).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

REPO = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO))  # the doctor_trigger package

from doctor_trigger.calibration import lower_tail_threshold  # noqa: E402
from doctor_trigger.channels import jpeg, rescale_round_trip  # noqa: E402
from doctor_trigger.decision import DirectThresholds, Status, unit_status  # noqa: E402
from doctor_trigger.enrollment import DEFAULT_ALPHA_LADDER, EnrollmentSettings, enroll  # noqa: E402
from doctor_trigger.readings import SumForm, UnitReading, key_projection, phase_drift, spectral_agreement  # noqa: E402
from doctor_trigger.record import Record, UnitId, stored_arrays  # noqa: E402
from doctor_trigger.spectrum import SUBBAND_B_KAPPA_MAX, luminance_phase, subband_b  # noqa: E402

SETTINGS = EnrollmentSettings()  # the defaults of workflow/scripts/enroll.py
ALPHAS = DEFAULT_ALPHA_LADDER
RECORD_FIELDS = ("template_phase", "host_phase", "delta", "host_amplitude", "support", "code", "key", "alpha")
"""Record arrays written to record_PRIVATE.npz (beta, ring and amplitude floor are the defaults)."""
DATA = Path(__file__).resolve().parent / "data"


# ----------------------------------------------------------------------------- verify
def read_copy(query: np.ndarray | None, record: Record | None) -> UnitReading:
    """Face consistency C_B and release signal G_s of one copy against its record.

    Read on the uncentered Fourier grid with weight maps (``SumForm.WEIGHT_MAP``); G_s uses unit
    weights and is not release-normalized.
    """
    if record is None or query is None or query.shape[:2] != record.shape:
        return UnitReading(None, None, None, None, SUBBAND_B_KAPPA_MAX, False, "record or geometry unresolved")
    phase = luminance_phase(query, centered=False)
    active = np.fft.ifftshift(record.active_bins)
    agreement = spectral_agreement(
        phase,
        np.fft.ifftshift(record.template_phase),
        np.fft.ifftshift(record.amplitude_weights),
        active & subband_b(record.shape, centered=False),
        form=SumForm.WEIGHT_MAP,
    )
    projection = key_projection(
        phase_drift(phase, np.fft.ifftshift(record.original_phase)),
        np.fft.ifftshift(record.displacement),
        np.ones(record.shape),
        active,
        form=SumForm.WEIGHT_MAP,
    )
    return UnitReading(None, None, agreement, projection, SUBBAND_B_KAPPA_MAX, False)


def decide(query: np.ndarray | None, record: Record | None, thresholds: DirectThresholds) -> tuple[str, UnitReading]:
    reading = read_copy(query, record)
    status = unit_status(reading, thresholds)
    if status is not Status.UNCONFIRMED:
        return status.value, reading
    why = []
    if reading.spectral_agreement < thresholds.tau_C:
        why.append("face changed")
    if reading.key_projection < thresholds.tau_G:
        why.append("release signal missing")
    return status.value + " (" + ", ".join(why) + ")", reading


# ----------------------------------------------------------------------------- copies
def splice(target: np.ndarray, donor: np.ndarray) -> np.ndarray:
    """Crude face replacement: paste the donor's central region with a soft edge.

    Only a stand-in so this example needs no face-swap model; the paper's queries use SimSwap.
    """
    h, w = target.shape[:2]
    mask = Image.new("L", (w, h), 0)
    mask.paste(255, (int(0.28 * w), int(0.22 * h), int(0.72 * w), int(0.75 * h)))
    m = np.asarray(mask.filter(ImageFilter.GaussianBlur(radius=w / 40)), dtype=np.float64)[:, :, None] / 255.0
    return np.clip(m * donor + (1 - m) * target, 0, 255).round().astype(np.uint8)


def load(path: Path, size: int | None = None) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if size is not None and im.size != (size, size):
        im = im.resize((size, size), Image.LANCZOS)
    return np.asarray(im, dtype=np.uint8)


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--image", type=Path, default=DATA / "office.png", help="face to enroll (square crop)")
    ap.add_argument("--calibration-image", type=Path, default=DATA / "market.png",
                    help="a different face, used only for the toy calibration")
    ap.add_argument("--out", type=Path, default=None, help="optional folder for images and the record")
    args = ap.parse_args()

    original = load(args.image)
    size = original.shape[0]
    if original.shape[0] != original.shape[1]:
        raise SystemExit("please pass a square face crop")

    # 1. enroll
    enrolled = enroll(original, UnitId(args.image.stem), ALPHAS, SETTINGS)
    released, record = enrolled.released, enrolled.record
    print(f"[enroll]    {args.image.name}: {size}x{size}, alpha={enrolled.alpha:.3f}, "
          f"PSNR {enrolled.psnr:.2f} dB, SSIM {enrolled.ssim:.4f} (released vs original)")

    # 2. TOY calibration: five enrollments made from a different face (as given, mirrored, rotated;
    #    each gets its own key), each sent through the benign channels we want to tolerate.
    cal_img = load(args.calibration_image, size)
    cal_faces = [cal_img, cal_img[:, ::-1], np.rot90(cal_img), np.rot90(cal_img, 3), np.rot90(cal_img[:, ::-1])]
    cal = []
    for i, face in enumerate(cal_faces):
        face_enrolled = enroll(np.ascontiguousarray(face), UnitId(f"{args.calibration_image.stem}-cal{i}"), ALPHAS, SETTINGS)
        rel, rec = face_enrolled.released, face_enrolled.record
        benign = [jpeg(rel, q) for q in (70, 75, 80, 85, 90)] + [rescale_round_trip(rel, s) for s in (0.6, 0.75, 0.9)]
        benign.append(jpeg(rescale_round_trip(rel, 0.75), 80))
        cal += [read_copy(b, rec) for b in benign]
    tau_c = float(lower_tail_threshold([c.spectral_agreement for c in cal]))
    tau_g = float(lower_tail_threshold([c.key_projection for c in cal]))
    thresholds = DirectThresholds(tau_C=tau_c, tau_G=tau_g)
    print(f"[calibrate] TOY: 1% quantiles over {len(cal)} benign copies (JPEG q70-90, resize 0.6-0.9x) of "
          f"{len(cal_faces)} enrollments of {args.calibration_image.name}\n"
          f"            -> tau_C={tau_c:.4f}, tau_G={tau_g:.3f}  (in practice: calibrate on your own releases)")

    # 3-4. verify a few copies of the enrolled face
    donor = load(args.calibration_image, size)
    copies = {
        "posted (released PNG)": released,
        "JPEG re-save, q85": jpeg(released, 85),
        "resized 0.75x and back": rescale_round_trip(released, 0.75),
        "unreleased original": original,
        "face region replaced": splice(released, donor),
        "center crop 90% (no record of geometry)": released[size // 20: size - size // 20, size // 20: size - size // 20],
    }
    print(f"\n{'copy':42s} {'C':>8s} {'G_s':>7s}  status")
    for name, img in copies.items():
        status, reading = decide(img, record, thresholds)
        c = f"{reading.spectral_agreement:.4f}" if reading.available else "-"
        g = f"{reading.key_projection:.3f}" if reading.available else "-"
        print(f"{name:42s} {c:>8s} {g:>7s}  {status}")
    print(f"{'(thresholds)':42s} {tau_c:8.4f} {tau_g:7.3f}")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for name, img in [("original", original), ("released", released)] + [
                (f"copy{i}", im) for i, im in enumerate(copies.values())]:
            Image.fromarray(img).save(args.out / f"{name}.png")
        arrays = stored_arrays(record)
        np.savez_compressed(args.out / "record_PRIVATE.npz", **{name: arrays[name] for name in RECORD_FIELDS})
        print(f"\nwrote images and the private record to {args.out}/ (keep record_PRIVATE.npz private)")


if __name__ == "__main__":
    main()
