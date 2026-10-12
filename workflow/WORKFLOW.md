# The pipeline

`run_workflow.sh` turns extracted dataset frames (FF++, DFDC, Celeb-DF) into releases, private
records, queries, record features and a trained record model. Source frames and SimSwap weights
are read-only inputs, named in `<repo>/config.yaml` (`source_manifests`, `simswap_root`). Every
stage writes only inside this directory (`pipeline_paths.py` refuses any other output path).

## Stages

| # | Script | Writes |
|---|---|---|
| 0 | `scripts/make_source_manifest.py FRAMES_ROOT OUT.csv` (run by hand, once per dataset) | source manifest `split,class,video_id,frame_path` |
| 1 | `scripts/build_base_manifest.py` | `manifests/base_samples.csv` |
| 2 | `scripts/align_faces.py` | `artifacts/aligned/`, `manifests/alignment.csv`, `manifests/alignment_quarantine.csv` |
| 3 | `scripts/enroll.py` | `artifacts/released/`, `private/templates/`, `manifests/enrollments.csv`, `manifests/enrollment_quarantine.csv` |
| 4 | `scripts/build_queries.py` | `artifacts/queries/`, `manifests/queries.csv`, `manifests/query_quarantine.csv` |
| 5 | `scripts/extract_features.py` | `features/calibrated_features.csv` (+ one file per dataset and split) |
| 6 | `scripts/integrity_gate.py` | `reports/calibrated_integrity.{json,md}` |
| 7 | `scripts/export_common_interface.py` | `common/cases.csv`, `common/peer_score_contract.json`, `common/workflow_contract.json` |
| 8 | `record_model/train_record_model.py --config configs/record_model.yaml` | `results/record_model/{checkpoints,scores,metrics,matrix}/` |
| 9 | `scripts/finalize_receipt.py`, then `scripts/verify_workflow.py` | `FINAL_RECEIPT.{json,md}` |

1. **Base manifest.** Merges the three source manifests and gives every frame a sample id and a
   video-group id. A dataset whose source splits put one video in two splits (Celeb-DF) is
   re-split by video group (70/15/15, by hash); FF++ and DFDC keep their source splits.
2. **Alignment.** SCRFD finds the face; YuNet and a Haar cascade validate it, and at least two of
   the three must agree (IoU ≥ 0.2). The crop keeps the face at width/height ratios 0.224/0.334,
   with no padding and no resize, saved as PNG. A frame that fails is quarantined; there is no
   full-frame fallback.
3. **Enrollment.** For each aligned real face: derive the key s from the master seed and the
   sample id, build the sign pattern h_s (±1 on each active bin of the luminance ring
   κ ∈ [0.10, 0.45], 0 on self-conjugate bins; the same pattern is applied to all three color
   channels), shift the phase by απh_s, blend with β = 0.15 and save the released image I′. α is
   the first value of the ladder (0.050, 0.030, 0.015; DFDC 0.015, 0.010, 0.005) whose release passes the
   fidelity gate (PSNR ≥ 35 dB, SSIM ≥ 0.98), checked again on the PNG read back from disk.
   The private record R is one NPZ per face; its format is below.
4. **Queries.** Each enrollment gets a benign query (the release itself) and a swap query: SimSwap
   applied to the release with an aligned fake face of the same dataset and split as identity
   source (FF++ manipulation types rotate across enrollments). Query ids are hashes, so paths never
   collide. A failed swap is retried with other source faces of the same class, then at a lower
   detector threshold (recorded in `simswap_det_thresh`); a swap that still fails stops the run.
   A finished query is reused only when its sidecar receipt matches the input and output hashes.
5. **Record features.** For each query J, the 75 record-model inputs x = Φ(J, R) (H, C, G_s, the
   wrong-key statistics over m = 8 wrong keys, residual statistics, and 8 radial bands). Key
   projections are divided by the release's own projection (read from `released_path`). Each benign
   query also gives a wrong-key row (labels y_rel = 1, y_key = 0; column `y_keycorr`).
6. **Integrity gate.** Unique ids, no video group in two splits, detector agreement, the fidelity
   gate, file hashes, complete records, no quarantined swap query, 75 finite features. Any failure
   stops the run before training.
7. **Comparison interface.** See below.
8. **Record model.** f_θ is trained on FF++ train rows only, for four input variants and three
   seeds, and scored on the FF++, DFDC and Celeb-DF test rows (`configs/record_model.yaml`).
9. **Receipt.** Row counts and SHA-256 of every manifest, contract and code file, the integrity
   report, metric means and package versions. `verify_workflow.py` re-checks a finished run
   without writing anything (`--full` also re-runs the integrity checks).

## Record format

`private/templates/<dataset>/<split>/<enrollment_id>.npz`, read and written by
`doctor_trigger.record` (`load_record`, `save_record`). Arrays are on the centered (fftshift) grid.

| NPZ key | Paper | dtype |
|---|---|---|
| `template_phase` | template T_I (Eq. 4) | float16 |
| `host_phase` | original phase φ̂_I | float16 |
| `delta` | displacement δ_s (Eq. 3) | float16 |
| `host_amplitude` | amplitude weights w_k = A_I / max A_I | float16 |
| `support` | active bins K_eff | uint8 |
| `code` | sign pattern h_s (−1, 0, +1) | int8 |
| `key`, `alpha`, `beta` | key s, strengths α, β | uint64, float32, float32 |
| `bands` | ring [κ_min, κ_max] | float32 |
| `amplitude_floor_ratio` | A_min / max A_I | float32 |

The record is image-derived (its phase reveals coarse facial structure): keep it private. The
JSON sidecar next to each NPZ repeats the enrollment row (paths, α, PSNR, SSIM, hashes).

## Comparison interface

`common/cases.csv` has one row per query with the exact bytes every method uses, each with its
SHA-256: the query (`input_path`), the aligned enrollment input, the swap identity source, and the
Doctor Trigger release and record. Detector-only methods score `input_path` unchanged. End-to-end
methods enroll `enrollment_input_path` themselves and apply the recorded swap with
`attack_source_path`. A peer method returns one row per query as described in
`peer_score_contract.json` (`score_fake` is required; relation and key scores are optional).
Failed cases are reported, never replaced by another crop.

## Run

From the repository root, after setting up the environment (`ENVIRONMENT.md`) and `config.yaml`:

```bash
workflow/run_workflow.sh --gpu 0          # run, or resume, the pipeline
workflow/run_workflow.sh --verify-only    # read-only check of a finished run (no GPU)
```

The runner writes `logs/runner.log`, `state/status.json`, a heartbeat every 30 s and one marker
per stage (`state/markers/`). A stage whose output file exists is reused; alignment and
enrollment also skip faces whose files and sidecars exist. The scripts can be run one at a time
with `PYTHONPATH=<repo>` (the runner sets it).
