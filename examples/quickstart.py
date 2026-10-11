#!/usr/bin/env python
"""Doctor Trigger in one image, without any dataset.

  python examples/quickstart.py                       # uses the bundled AI-generated faces
  python examples/quickstart.py --image my_face.png   # your own square face crop
  python examples/quickstart.py --out quickstart_out  # also write the images and the record

What it does, using the repository's own functions (nothing here re-implements the method):

  1. enroll   workflow/scripts/enroll.py: stable_key() derives a per-face key, embed() writes the
              weak keyed phase signal and returns the private record (the same fidelity gate as the
              pipeline: PSNR >= 35 dB, SSIM >= 0.98, trying alpha 0.050 -> 0.030 -> 0.015).
  2. calibrate  TOY calibration: thresholds are the 1% quantiles of C and G_s over benign copies
              (JPEG q70-90, resizing) of five enrollments made from a *different* face. A real
              deployment calibrates on benign copies of its own releases, over the channels it
              intends to support (the paper: 1% benign rejection on training releases).
  3. verify   workflow/record_model/src/trigger_conditioned.py:
                C   = compute_template_score(): amplitude-weighted phase agreement with the stored
                      template on the sub-band kappa <= 0.15 (the paper's C_B, "face consistency").
                G_s = compute_trigger_projection(): projection of the phase change onto the enrolled
                      keyed displacement ("release signal"); about 1 for a release, 0 for the original.
  4. decide   Confirmed if C >= tau_C and G_s >= tau_G; Unconfirmed otherwise; Unavailable when the
              check cannot run (no record, or the copy's geometry does not match the record).
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from skimage.metrics import structural_similarity

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "workflow" / "scripts"))
from enroll import embed, frequency_radius, psnr, rgb_to_y, stable_key  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "trigger_conditioned", REPO / "workflow" / "record_model" / "src" / "trigger_conditioned.py")
tc = sys.modules["trigger_conditioned"] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tc)

# Enrollment settings: the defaults of workflow/scripts/enroll.py.
MASTER_SEED, BETA, BANDS, FLOOR = 17, 0.15, (0.10, 0.45), 1e-6
ALPHAS, MIN_PSNR, MIN_SSIM = (0.050, 0.030, 0.015), 35.0, 0.98
SUB_BAND = 0.15  # C is read on kappa <= 0.15 (the paper's C_B)
DATA = Path(__file__).resolve().parent / "data"


# ----------------------------------------------------------------------------- enroll
def enroll(image: np.ndarray, sample_id: str):
    """Release image + private record, with the pipeline's fidelity gate."""
    key = stable_key(MASTER_SEED, sample_id)
    for alpha in ALPHAS:
        released, record = embed(image, key, alpha, BETA, BANDS, FLOOR)
        p = psnr(image, released)
        s = float(structural_similarity(image, released, channel_axis=2, data_range=255))
        if p >= MIN_PSNR and s >= MIN_SSIM:
            record = dict(record, key=np.asarray([key], dtype=np.uint64), alpha=np.asarray([alpha], dtype=np.float32))
            return released, record, {"alpha": alpha, "psnr": p, "ssim": s}
    raise RuntimeError("fidelity gate failed for every alpha")


# ----------------------------------------------------------------------------- verify
def _unshift(record, name):
    return np.fft.ifftshift(record[name].astype(np.float64))


def read_copy(query: np.ndarray, record) -> dict[str, float]:
    """Face consistency C and release signal G_s of one copy against its record."""
    phase = np.angle(np.fft.fft2(rgb_to_y(query.astype(np.float64))))
    template, host = _unshift(record, "template_phase"), _unshift(record, "host_phase")
    delta, amp = _unshift(record, "delta"), _unshift(record, "host_amplitude")
    support = _unshift(record, "support").astype(bool)
    weights = np.where(support & (frequency_radius(template.shape) <= SUB_BAND), amp, 0.0)
    return {
        "C": tc.compute_template_score(phase, template, weights=weights),
        "G": tc.compute_trigger_projection(phase, host, delta, support=support),
    }


def decide(query: np.ndarray | None, record, tau_c: float, tau_g: float) -> tuple[str, dict]:
    if record is None or query is None or query.shape[:2] != record["template_phase"].shape:
        return "Unavailable", {}
    s = read_copy(query, record)
    if s["C"] >= tau_c and s["G"] >= tau_g:
        return "Confirmed", s
    why = []
    if s["C"] < tau_c:
        why.append("face changed")
    if s["G"] < tau_g:
        why.append("release signal missing")
    return "Unconfirmed (" + ", ".join(why) + ")", s


# ----------------------------------------------------------------------------- copies
def jpeg(a: np.ndarray, q: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "JPEG", quality=q)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def resized(a: np.ndarray, s: float) -> np.ndarray:
    """Shrink, then restore the enrollment geometry (as the paper's resize channels do)."""
    im = Image.fromarray(a)
    w, h = im.size
    return np.asarray(im.resize((int(w * s), int(h * s)), Image.BICUBIC).resize((w, h), Image.BICUBIC))


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
    released, record, fid = enroll(original, sample_id=args.image.stem)
    print(f"[enroll]    {args.image.name}: {size}x{size}, alpha={fid['alpha']:.3f}, "
          f"PSNR {fid['psnr']:.2f} dB, SSIM {fid['ssim']:.4f} (released vs original)")

    # 2. TOY calibration: five enrollments made from a different face (as given, mirrored, rotated;
    #    each gets its own key), each sent through the benign channels we want to tolerate.
    cal_img = load(args.calibration_image, size)
    cal_faces = [cal_img, cal_img[:, ::-1], np.rot90(cal_img), np.rot90(cal_img, 3), np.rot90(cal_img[:, ::-1])]
    cal = []
    for i, face in enumerate(cal_faces):
        rel, rec, _ = enroll(np.ascontiguousarray(face), sample_id=f"{args.calibration_image.stem}-cal{i}")
        benign = [jpeg(rel, q) for q in (70, 75, 80, 85, 90)] + [resized(rel, s) for s in (0.6, 0.75, 0.9)]
        benign.append(jpeg(resized(rel, 0.75), 80))
        cal += [read_copy(b, rec) for b in benign]
    tau_c = float(np.quantile([c["C"] for c in cal], 0.01))
    tau_g = float(np.quantile([c["G"] for c in cal], 0.01))
    print(f"[calibrate] TOY: 1% quantiles over {len(cal)} benign copies (JPEG q70-90, resize 0.6-0.9x) of "
          f"{len(cal_faces)} enrollments of {args.calibration_image.name}\n"
          f"            -> tau_C={tau_c:.4f}, tau_G={tau_g:.3f}  (in practice: calibrate on your own releases)")

    # 3-4. verify a few copies of the enrolled face
    donor = load(args.calibration_image, size)
    copies = {
        "posted (released PNG)": released,
        "JPEG re-save, q85": jpeg(released, 85),
        "resized 0.75x and back": resized(released, 0.75),
        "unreleased original": original,
        "face region replaced": splice(released, donor),
        "center crop 90% (no record of geometry)": released[size // 20: size - size // 20, size // 20: size - size // 20],
    }
    print(f"\n{'copy':42s} {'C':>8s} {'G_s':>7s}  status")
    for name, img in copies.items():
        status, s = decide(img, record, tau_c, tau_g)
        c = f"{s['C']:.4f}" if s else "-"
        g = f"{s['G']:.3f}" if s else "-"
        print(f"{name:42s} {c:>8s} {g:>7s}  {status}")
    print(f"{'(thresholds)':42s} {tau_c:8.4f} {tau_g:7.3f}")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for name, img in [("original", original), ("released", released)] + [
                (f"copy{i}", im) for i, im in enumerate(copies.values())]:
            Image.fromarray(img).save(args.out / f"{name}.png")
        np.savez_compressed(args.out / "record_PRIVATE.npz", **record)
        print(f"\nwrote images and the private record to {args.out}/ (keep record_PRIVATE.npz private)")


if __name__ == "__main__":
    main()
