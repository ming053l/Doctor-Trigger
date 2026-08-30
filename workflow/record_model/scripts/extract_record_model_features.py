#!/usr/bin/env python3
"""Extract record_model_release trigger-conditioned features from existing assets only."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.trigger_conditioned import feature_vector

ImageFile.LOAD_TRUNCATED_IMAGES = True


DEFAULT_SYNTHESIS_LOG = Path("data/manifests/ffpp_synthesis_log.csv")
DEFAULT_DATASET_CSV = Path("data/manifests/ffpp_dataset_test.csv")
DEFAULT_COPY_SCORES = ROOT / "analysis/tch_phase_ablation_20260427/scores.csv"


def load_config(path: Path) -> Dict[str, object]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def rgb_to_y(arr: np.ndarray) -> np.ndarray:
    arr01 = np.clip(arr / 255.0, 0.0, 1.0).astype(np.float64)
    return 0.299 * arr01[:, :, 0] + 0.587 * arr01[:, :, 1] + 0.114 * arr01[:, :, 2]


def clip_bbox(bbox: Iterable[float], shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    h, w = shape[:2]
    vals = list(bbox)
    if len(vals) != 4:
        return 0, 0, w, h
    x1, y1, x2, y2 = [int(round(float(v))) for v in vals]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))
    return x1, y1, x2, y2


@lru_cache(maxsize=64)
def phase_roi_cached(path_text: str, x1_raw: int, y1_raw: int, x2_raw: int, y2_raw: int) -> np.ndarray:
    path = Path(path_text)
    arr = load_rgb(path)
    x1, y1, x2, y2 = clip_bbox((x1_raw, y1_raw, x2_raw, y2_raw), arr.shape)
    roi = arr[y1:y2, x1:x2]
    if roi.size == 0:
        roi = arr
    return np.angle(np.fft.fft2(rgb_to_y(roi)))


def phase_roi(path: Path, bbox: Iterable[float]) -> np.ndarray:
    vals = list(bbox)
    if len(vals) != 4:
        vals = [0, 0, 10_000, 10_000]
    x1, y1, x2, y2 = [int(round(float(v))) for v in vals]
    return phase_roi_cached(str(path), x1, y1, x2, y2).copy()


def resize_phase_like(phi: np.ndarray, ref_shape: Tuple[int, int]) -> np.ndarray:
    if phi.shape == ref_shape:
        return phi
    # Map phase to [0,1] in float32 (PIL mode 'F') so BILINEAR resample is supported;
    # PIL refuses BILINEAR on 16-bit integer images ('I;16'), which fires whenever the
    # query and template resolutions differ (e.g. variable-size Celeb-DF frames).
    img = Image.fromarray(((phi + np.pi) / (2.0 * np.pi)).astype(np.float32))
    img = img.resize((ref_shape[1], ref_shape[0]), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float64) * (2.0 * np.pi) - np.pi
    return arr


def load_pairs(synthesis_log: Path, dataset_csv: Path | None, limit: int) -> pd.DataFrame:
    synth = pd.read_csv(synthesis_log)
    if dataset_csv and dataset_csv.exists():
        test = pd.read_csv(dataset_csv)
        if "order" in test.columns and "img_path" in test.columns:
            pos_paths = set(test.loc[test["order"].astype(str) == "swapped_render", "img_path"].astype(str))
            if pos_paths:
                synth = synth[synth["swapped_render"].astype(str).isin(pos_paths)].copy()
    for col in ["real_frame", "triggered_frame", "swapped_render"]:
        if col not in synth.columns:
            raise RuntimeError(f"missing required column in synthesis log: {col}")
        synth[col] = synth[col].astype(str)
        synth = synth[synth[col].map(lambda p: Path(p).exists())].copy()
    if "fake_then_trigger" in synth.columns:
        synth["fake_then_trigger"] = synth["fake_then_trigger"].astype(str)
    synth = synth[synth["triggered_frame"].map(lambda p: Path(p).with_suffix(".json").exists())].copy()
    synth = synth.sort_values([col for col in ["fake_type", "swapped_render"] if col in synth.columns]).reset_index(drop=True)
    if limit > 0:
        synth = synth.head(limit).copy()
    if synth.empty:
        raise RuntimeError("No valid FF++ rows found for record_model_release feature extraction.")
    return synth


def load_pair_manifest(pair_manifest: Path, limit: int) -> pd.DataFrame:
    df = pd.read_csv(pair_manifest)
    if "triggered_frame" not in df.columns or "swapped_render" not in df.columns:
        raise RuntimeError(f"pair_manifest must contain triggered_frame and swapped_render: {pair_manifest}")
    for col in ["triggered_frame", "swapped_render"]:
        df[col] = df[col].astype(str)
        df = df[df[col].map(lambda p: Path(p).exists())].copy()
    df = df[df["triggered_frame"].map(lambda p: Path(p).with_suffix(".json").exists())].copy()
    if "fake_then_trigger" in df.columns:
        df["fake_then_trigger"] = df["fake_then_trigger"].astype(str)
    else:
        df["fake_then_trigger"] = ""
    if "real_frame" not in df.columns:
        df["real_frame"] = ""
    if "fake_frame" not in df.columns:
        df["fake_frame"] = ""
    if "fake_type" not in df.columns:
        df["fake_type"] = "unknown"
    if limit > 0:
        df = df.head(limit).copy()
    if df.empty:
        raise RuntimeError(f"No valid rows found in pair_manifest: {pair_manifest}")
    return df.reset_index(drop=True)


def copy_ring_lookup(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out = {}
    for _, row in df.iterrows():
        if str(row.get("condition", "")) != "copy_ring_attack":
            continue
        attack = str(row.get("attack_path", ""))
        if not attack or not Path(attack).exists():
            continue
        out[str(row.get("swapped_render", ""))] = attack
    return out


def existing_file(value: object) -> Path | None:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    path = Path(text)
    if path.is_file():
        return path
    return None


def row_features(
    *,
    sample_id: str,
    dataset: str,
    protocol: str,
    query_path: Path,
    template_path: Path,
    meta: Dict[str, object],
    y_fake: int,
    y_rel: int,
    y_keycorr: int,
    key_override: int | None = None,
    host_key_override: int | None = None,
    notes: str = "",
    code_mode: str = "zero_pi",
    n_wrong: int = 4,
    n_bands: int = 4,
) -> Dict[str, object]:
    bbox = meta.get("bbox", [0, 0, 10_000, 10_000])
    bands_raw = meta.get("bands", [0.1, 0.45])
    bands = (float(bands_raw[0]), float(bands_raw[1]))
    alpha = float(meta.get("alpha", 0.05))
    beta = float(meta.get("beta", 0.15))
    key = int(meta["phase_seed"] if key_override is None else key_override)
    host_key = int(meta["phase_seed"] if host_key_override is None else host_key_override)
    template_phase = phase_roi(template_path, bbox)
    query_phase = resize_phase_like(phase_roi(query_path, bbox), template_phase.shape)
    feats = feature_vector(
        query_phase,
        template_phase,
        key=key,
        alpha=alpha,
        beta=beta,
        bands=bands,
        code_mode=code_mode,
        n_wrong=n_wrong,
        n_bands=n_bands,
        host_key=host_key,
    )
    return {
        "sample_id": sample_id,
        "dataset": dataset,
        "protocol": protocol,
        "source_path": str(query_path),
        "template_id": str(template_path),
        "key_id": int(key),
        "original_key_id": int(meta["phase_seed"]),
        "host_recovery_key_id": int(host_key),
        "alpha": alpha,
        "beta": beta,
        "kappa_min": bands[0],
        "kappa_max": bands[1],
        "code_mode": code_mode,
        "y_fake": int(y_fake),
        "y_rel": int(y_rel),
        "y_keycorr": int(y_keycorr),
        "notes": notes,
        **feats,
    }


_WORKER_CTX: Dict[str, object] = {}


def init_worker(ctx: Dict[str, object]) -> None:
    global _WORKER_CTX
    _WORKER_CTX = ctx


def build_pair_rows(task: Tuple[int, Dict[str, object]]) -> List[Dict[str, object]]:
    idx, row = task
    dataset_name = str(_WORKER_CTX["dataset_name"])
    sample_prefix = str(_WORKER_CTX["sample_prefix"])
    code_mode = str(_WORKER_CTX["code_mode"])
    n_wrong = int(_WORKER_CTX["n_wrong"])
    n_bands = int(_WORKER_CTX["n_bands"])
    copy_map = _WORKER_CTX["copy_map"]
    trig_path = Path(str(row["triggered_frame"]))
    swap_path = Path(str(row["swapped_render"]))
    meta = json.loads(trig_path.with_suffix(".json").read_text(encoding="utf-8"))
    sid = f"{sample_prefix}_{idx:06d}"
    out: List[Dict[str, object]] = []
    out.append(row_features(sample_id=f"{sid}_benign", dataset=dataset_name, protocol="benign", query_path=trig_path, template_path=trig_path, meta=meta, y_fake=0, y_rel=1, y_keycorr=1, code_mode=code_mode, n_wrong=n_wrong, n_bands=n_bands))
    out.append(row_features(sample_id=f"{sid}_ttf", dataset=dataset_name, protocol="swap", query_path=swap_path, template_path=trig_path, meta=meta, y_fake=1, y_rel=0, y_keycorr=0, code_mode=code_mode, n_wrong=n_wrong, n_bands=n_bands))
    ftt_path = existing_file(row.get("fake_then_trigger", ""))
    if ftt_path is not None:
        out.append(row_features(sample_id=f"{sid}_ftt", dataset=dataset_name, protocol="FTT", query_path=ftt_path, template_path=trig_path, meta=meta, y_fake=1, y_rel=0, y_keycorr=0, code_mode=code_mode, n_wrong=n_wrong, n_bands=n_bands))
    wrong_key = int(meta["phase_seed"]) ^ 0x13579BDF
    out.append(row_features(sample_id=f"{sid}_wrong_key", dataset=dataset_name, protocol="wrong_key", query_path=trig_path, template_path=trig_path, meta=meta, y_fake=0, y_rel=1, y_keycorr=0, key_override=wrong_key, host_key_override=int(meta["phase_seed"]), notes="benign content scored against wrong projection key while host recovery remains anchored to original enrolled key", code_mode=code_mode, n_wrong=n_wrong, n_bands=n_bands))
    attack_path = copy_map.get(str(swap_path))
    if attack_path:
        out.append(row_features(sample_id=f"{sid}_copy_ring", dataset=dataset_name, protocol="copy_ring", query_path=Path(attack_path), template_path=trig_path, meta=meta, y_fake=1, y_rel=0, y_keycorr=1, notes="annular phase transplanted from released I_prime; trigger presence is not relation preservation", code_mode=code_mode, n_wrong=n_wrong, n_bands=n_bands))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = ROOT / str(cfg["output_dir"])
    run_name = str(cfg.get("run_name", "smoke"))
    dataset_name = str(cfg.get("dataset_name", "FF++"))
    sample_prefix = str(cfg.get("sample_prefix", dataset_name.lower().replace("+", "p").replace("-", "_").replace(" ", "_")))
    features_dir = out_dir / "features"
    labels_dir = out_dir / "labels"
    logs_dir = out_dir / "logs"
    features_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    pair_manifest_cfg = cfg.get("pair_manifest")
    synthesis_log = Path(str(cfg.get("synthesis_log", DEFAULT_SYNTHESIS_LOG)))
    dataset_csv = Path(str(cfg.get("dataset_csv", DEFAULT_DATASET_CSV)))
    copy_scores = Path(str(cfg.get("copy_ring_scores", DEFAULT_COPY_SCORES)))
    rows_limit = int(cfg.get("smoke", {}).get("pair_limit", cfg.get("pair_limit", 0)))
    progress_every = int(cfg.get("progress_every", 500))
    n_wrong = int(cfg.get("feature", {}).get("n_wrong", 4))
    n_bands = int(cfg.get("feature", {}).get("n_bands", 4))
    code_mode = str(cfg.get("feature", {}).get("code_mode", "zero_pi"))
    num_workers = int(cfg.get("num_workers", 1))

    if pair_manifest_cfg:
        pairs = load_pair_manifest(Path(str(pair_manifest_cfg)), rows_limit)
    else:
        pairs = load_pairs(synthesis_log, dataset_csv, rows_limit)
    copy_map = copy_ring_lookup(copy_scores)
    rows: List[Dict[str, object]] = []
    tasks = [(int(idx), row.to_dict()) for idx, row in pairs.iterrows()]
    ctx = {
        "dataset_name": dataset_name,
        "sample_prefix": sample_prefix,
        "code_mode": code_mode,
        "n_wrong": n_wrong,
        "n_bands": n_bands,
        "copy_map": copy_map,
    }
    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, initargs=(ctx,)) as pool:
            for done, part in enumerate(pool.map(build_pair_rows, tasks, chunksize=16)):
                if progress_every > 0 and done % progress_every == 0:
                    print(f"[record_model_release_extract_features] {run_name}: {done}/{len(pairs)} pairs", flush=True)
                rows.extend(part)
    else:
        init_worker(ctx)
        for idx, task in enumerate(tasks):
            if progress_every > 0 and idx % progress_every == 0:
                print(f"[record_model_release_extract_features] {run_name}: {idx}/{len(pairs)} pairs", flush=True)
            rows.extend(build_pair_rows(task))

    feature_csv = features_dir / f"record_model_release_{run_name}_features.csv"
    pd.DataFrame(rows).to_csv(feature_csv, index=False)
    label_cols = ["sample_id", "dataset", "protocol", "y_fake", "y_rel", "y_keycorr", "source_path", "template_id", "key_id", "notes"]
    label_csv = labels_dir / f"label_audit_{run_name}.csv"
    pd.DataFrame(rows)[label_cols].to_csv(label_csv, index=False)
    canonical_label_csv = labels_dir / "label_audit.csv"
    pd.DataFrame(rows)[label_cols].to_csv(canonical_label_csv, index=False)
    (labels_dir / "label_schema.md").write_text(
        "\n".join(
            [
                "# Rev9 Label Schema",
                "",
                "- `y_fake`: 1 manipulated content, 0 genuine/benign content.",
                "- `y_rel`: 1 original enrolled face-image relation preserved under allowed benign channel; 0 disrupted/not applicable.",
                "- `y_keycorr`: 1 correct original release-bound trigger direction expected; 0 wrong key/no release-bound lock.",
                "- `copy_ring`: `y_keycorr=1` but `y_rel=0`, because trigger-like annular phase can be present while relation is not preserved.",
                "",
                f"source feature csv: `{feature_csv.resolve()}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {feature_csv}")
    print(f"Wrote {label_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
