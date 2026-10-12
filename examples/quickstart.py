#!/usr/bin/env python
"""Doctor Trigger in one image, without any dataset.

  python examples/quickstart.py                       # uses the bundled AI-generated faces
  python examples/quickstart.py --image my_face.png   # your own square face crop
  python examples/quickstart.py --out quickstart_out  # also write the images and the record

(from the repository root, with the package importable: ``pip install -e .`` or ``PYTHONPATH=.``)

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
                G_s = key_projection() of the phase drift onto the enrolled displacement delta_s
                      ("release signal"). Here it is unweighted and not release-normalized (the
                      pipeline uses amplitude weights and divides by the release's own G_s), so a
                      release reads somewhat above 1 and the original about 0.
  4. decide   doctor_trigger.decision.unit_status(): Confirmed if C_B >= tau_C and G_s >= tau_G;
              Unconfirmed otherwise; Unavailable when the check cannot run (no record, or the
              copy's geometry does not match the record).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from doctor_trigger.calibration import lower_tail_threshold
from doctor_trigger.channels import jpeg, rescale_round_trip
from doctor_trigger.decision import DirectThresholds, Status, unit_status
from doctor_trigger.enrollment import DEFAULT_ALPHA_LADDER, EnrollmentSettings, enroll
from doctor_trigger.readings import UnitReading, key_projection, phase_drift, spectral_agreement, subband_bins
from doctor_trigger.record import Record, UnitId, save_record
from doctor_trigger.spectrum import SUBBAND_B_KAPPA_MAX, luminance_phase

SETTINGS = EnrollmentSettings()  # the pipeline's defaults (workflow/scripts/enroll.py)
ALPHAS = DEFAULT_ALPHA_LADDER
SAMPLE_IMAGES = Path(__file__).resolve().parent / "data"


# ----------------------------------------------------------------------------- verify
def read_copy(query: np.ndarray | None, record: Record | None) -> UnitReading:
    """Face consistency C_B and release signal G_s of one copy against its record.

    Read in record (centered) layout; G_s uses unit weights and is not release-normalized.
    """
    if record is None or query is None or query.shape[:2] != record.shape:
        return UnitReading(None, None, None, None, SUBBAND_B_KAPPA_MAX, False, "record or geometry unresolved")
    phase = luminance_phase(query, centered=True)
    agreement = spectral_agreement(phase, record.template_phase, record.amplitude_weights, subband_bins(record))
    projection = key_projection(phase_drift(phase, record.original_phase), record.displacement,
                                np.ones(record.shape), record.active_bins)
    return UnitReading(None, None, agreement, projection, SUBBAND_B_KAPPA_MAX, False)


def decide(query: np.ndarray | None, record: Record | None, thresholds: DirectThresholds) -> tuple[str, UnitReading]:
    """Status of one copy (Eq. 11), with the failed check(s) named when it is Unconfirmed."""
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


def load_rgb(path: Path, size: int | None = None) -> np.ndarray:
    """An image as uint8 RGB, resized to size x size (Lanczos) when a size is given."""
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != (size, size):
        image = image.resize((size, size), Image.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


# ----------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--image", type=Path, default=SAMPLE_IMAGES / "office.png", help="face to enroll (square crop)")
    parser.add_argument("--calibration-image", type=Path, default=SAMPLE_IMAGES / "market.png",
                        help="a different face, used only for the toy calibration")
    parser.add_argument("--out", type=Path, default=None, help="optional folder for images and the record")
    args = parser.parse_args()

    original = load_rgb(args.image)
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
    calibration_face = load_rgb(args.calibration_image, size)
    calibration_faces = [calibration_face, calibration_face[:, ::-1], np.rot90(calibration_face),
                         np.rot90(calibration_face, 3), np.rot90(calibration_face[:, ::-1])]
    calibration_readings = []
    for i, face in enumerate(calibration_faces):
        face_enrolled = enroll(np.ascontiguousarray(face), UnitId(f"{args.calibration_image.stem}-cal{i}"), ALPHAS, SETTINGS)
        face_release, face_record = face_enrolled.released, face_enrolled.record
        benign = ([jpeg(face_release, q) for q in (70, 75, 80, 85, 90)]
                  + [rescale_round_trip(face_release, s) for s in (0.6, 0.75, 0.9)])
        benign.append(jpeg(rescale_round_trip(face_release, 0.75), 80))
        calibration_readings += [read_copy(copy, face_record) for copy in benign]
    tau_c = float(lower_tail_threshold([r.spectral_agreement for r in calibration_readings]))
    tau_g = float(lower_tail_threshold([r.key_projection for r in calibration_readings]))
    thresholds = DirectThresholds(tau_C=tau_c, tau_G=tau_g)
    print(f"[calibrate] TOY: 1% quantiles over {len(calibration_readings)} benign copies (JPEG q70-90, resize 0.6-0.9x) of "
          f"{len(calibration_faces)} enrollments of {args.calibration_image.name}\n"
          f"            -> tau_C={tau_c:.4f}, tau_G={tau_g:.3f}  (in practice: calibrate on your own releases)")

    # 3-4. verify a few copies of the enrolled face
    donor = load_rgb(args.calibration_image, size)
    copies = {
        "posted (released PNG)": released,
        "JPEG re-save, q85": jpeg(released, 85),
        "resized 0.75x and back": rescale_round_trip(released, 0.75),
        "unreleased original": original,
        "face region replaced": splice(released, donor),
        "center crop 90% (no record of geometry)": released[size // 20: size - size // 20, size // 20: size - size // 20],
    }
    print(f"\n{'copy':42s} {'C':>8s} {'G_s':>7s}  status")
    for name, image in copies.items():
        status, reading = decide(image, record, thresholds)
        c = f"{reading.spectral_agreement:.4f}" if reading.available else "-"
        g = f"{reading.key_projection:.3f}" if reading.available else "-"
        print(f"{name:42s} {c:>8s} {g:>7s}  {status}")
    print(f"{'(thresholds)':42s} {tau_c:8.4f} {tau_g:7.3f}")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for name, image in [("original", original), ("released", released)] + [
                (f"copy{i}", copy) for i, copy in enumerate(copies.values())]:
            Image.fromarray(image).save(args.out / f"{name}.png")
        save_record(args.out / "record_PRIVATE.npz", record)  # the pipeline's record format
        print(f"\nwrote images and the private record to {args.out}/ (keep record_PRIVATE.npz private)")


if __name__ == "__main__":
    main()
