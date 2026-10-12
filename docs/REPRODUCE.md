# Reproducing the paper

Every number in the paper comes from two steps:

1. **The pipeline** (`workflow/`) turns extracted dataset frames into enrolled releases, private
   records, benign and face-swap queries, and per-query features. See [DATA.md](DATA.md) for the
   inputs and [`workflow/WORKFLOW.md`](../workflow/WORKFLOW.md) for every stage.
2. **The evaluations** (`experiments/`) read that finished pipeline root and write one CSV (or
   figure) per paper artifact. Each is a named entry in
   [`experiments/registry.yaml`](../experiments/registry.yaml), run through
   `python -m doctor_trigger reproduce <id>`.

```bash
pip install -e ".[experiments]"         # the doctor_trigger package plus the evaluation dependencies
cp config.example.yaml config.yaml      # set canonical_root, weights and peer paths
python -m doctor_trigger status                       # which registered outputs exist
python -m doctor_trigger list                         # every entry
python -m doctor_trigger show threshold-transfer      # the resolved command, without running it
python -m doctor_trigger reproduce threshold-transfer # run one entry; results go under outputs_root
python -m doctor_trigger reproduce benign-channels dataset=FF++   # entries with {slots} take k=v arguments
```

`reproduce` runs the entry's command in its working directory with the `DT_*` path variables of
config.yaml exported. `python run.py run <id>` is the same command under its earlier name.

## The pipeline

```
raw videos (FF++ / Celeb-DF / DFDC)
  └─ extract frames with ffmpeg  ->  <frames_root>/<split>/<class>/<video_id>/*.png
      └─ scripts/make_source_manifest.py       stage 0: one manifest CSV per dataset
          └─ ./run_workflow.sh --gpu 0         then, in order:
              ├─ build_base_manifest.py        identifiers, leak-free video-group splits
              ├─ align_faces.py                SCRFD crops, second-detector agreement, lossless PNG
              ├─ enroll.py                     released image + private NPZ record per face
              ├─ build_queries.py              benign copies and SimSwap face-swap queries
              ├─ extract_features.py           spectral-agreement / key features for every query
              ├─ integrity_gate.py             ids, splits, fidelity gate, hashes; stops the run on failure
              ├─ export_common_interface.py    common/cases.csv and the peer-score contract
              ├─ record_model/train_record_model.py  trained record-model scorers (3 seeds)
              └─ finalize_receipt.py, verify_workflow.py   FINAL_RECEIPT: hashes and counts
```

The workflow is resumable and self-verifying: re-running it against a completed root re-checks
hashes instead of recomputing, and `./run_workflow.sh --verify-only` runs the acceptance checks
alone.

| Pipeline output | Meaning |
| --- | --- |
| `artifacts/released/.../*.png` | The published copy of each enrolled face (visually unchanged, keyed signal written). |
| `private/templates/.../*.npz` | The private per-face record R: template T_I, original phase and amplitude weights, keyed displacement δ_s, sign pattern h_s (±1), active bins, key. |
| `manifests/{alignment,enrollments,queries}.csv` | Stage manifests (plus `*_quarantine.csv` for rejected items). |
| `common/cases.csv` | The query inventory: every benign copy and swapped render, with paths and labels. |
| `features/calibrated_features.csv` | The verifier's reading of every query: spectral agreement H / C, key projection G_s, wrong-key statistics, band features (Φ). |
| `FINAL_RECEIPT.json` | Hashes and counts for every stage, so a finished root is checkable without recomputing. |

Everything downstream consumes only `cases.csv`, the features CSV and the records. Peer methods are
scored through the same files, per
[`workflow/contracts/peer_score_contract.json`](../workflow/contracts/peer_score_contract.json).

## Paper artifact → registry entry

Numbering follows the current paper (the PDF on the
[project page](https://ming053l.github.io/Doctor-Trigger/static/pdf/paper.pdf)); the `paper:` text
that `python -m doctor_trigger list` prints uses the same numbers.

**Marks:** ✓ = the entry produces this artifact; ~ = the entry produces part of it, or the
artifact needs inputs this repository does not ship (noted in the row); ✗ = no code in this
repository, with the reason. "Re-run" means the entry was run on the released pipeline outputs and
reproduced the printed numbers (figures: pixel-identical renders).

### Main paper

| Paper artifact | Registry entries (`python -m doctor_trigger reproduce <id>`) | |
| --- | --- | --- |
| Fig 1: two-stage verification interface | illustration (no code) | ✗ |
| Fig 2: release evidence with preserved fidelity | illustration drawn from one asset in the paper source (no code); the multi-asset version of its right half is Fig 9, `fig-channel-samples` | ✗ |
| Fig 3: enrollment for later verification | illustration (no code) | ✗ |
| Fig 4: from release evidence to verification and review | illustration (no code) | ✗ |
| Table 1: frequency-restricted spectral agreement | C_B row: `subband-table1 part=<benign\|datasets\|engines>` (engine columns read the swaps cached by `engines engine=<name>`); full-ring row: `benign-channels dataset=FF++`; exploratory cut sweep named in the caption: `subband-readout`. Clean SimSwap columns re-run. The WebP column has no script (the benign part measures an H.264 family instead) | ~ |
| Table 2: face-swap detection within and across datasets | peers: `peers-detect method=<peer>`; image-only S_df: `image-detector-train-aug dataset=<ds>` then `image-detector-eval`; direct 1 − C_B: `subband-table1 part=datasets` (re-run; a direct test has no training, so its transfer columns equal the target dataset's AUC); record model S_fake: `features-record-ref` → `features-record-scalar` → `record-model-recordscalar pin=<ffpp\|dfdc\|celebdf>` | ✓ |
| Table 3: release evidence from the key | swap AUC: `reference-baselines`; pass rates: `original-control calibration=1580` (re-run) | ✓ |
| Table 4: credential retention across delivery channels | `table4-artifact` (credential, SimSwap-reject and benign-pass columns; collect stage re-run on the frozen per-image scores). The reenactment and SD-inpainting columns are not produced | ~ |
| Fig 5: visual fidelity of released images | `fig-imperceptibility` (also reads release renders from an earlier enrollment run, `legacy_trigger_glob`, not shipped) | ✓ |
| Fig 6: complementary evidence for face consistency and release signal | `fig-signal` (re-run) | ✓ |
| Fig 7: direct spectral evidence across channels | `table4-artifact`, then `fig-roundtrip` (re-run from the frozen Table 4 scores) | ✓ |
| Sec. 4.2 verification cost and storage | `record-codec` (payload sizes), `deployment-cost` (timings) | ✓ |

### Appendix

| Paper artifact | Registry entries | |
| --- | --- | --- |
| Table 5: notation | text only | — |
| Table 6: release fidelity per dataset and resolution | `peer-fidelity` records the command, but the per-peer `*_fidelity.py` scripts run inside each peer's checkout and are not part of this repository | ✗ |
| Fig 8: release residuals | `fig-release-residual` (re-run) | ✓ |
| Fig 9: verification examples across datasets | `fig-channel-samples` | ✓ |
| Fig 10: keyed displacement and query residual | `fig-keymatch` (re-run) | ✓ |
| Table 7: private-reference comparisons | `reference-baselines` (swap columns), `attribution-matrix` (same-person release), `release-evidence` (release vs original) | ✓ |
| Table 8: original and release acceptance under channel calibration | `original-control calibration=100` (lossless panel), `original-control calibration=1580` (JPEG-q90 panel); re-run | ✓ |
| Fig 11: enrolled template and asset specificity | `fig-template` | ✓ |
| Table 9: key discrimination and learned record scores | `key-pure-canonical` (needs `features-record-scalar`, `record-model-recordscalar pin=ffpp`) | ✓ |
| Table 10: benign-channel robustness across datasets | `image-detector-train-aug dataset=<ds>`, then `benign-channels dataset=<FF++\|DFDC\|CelebDF>` | ✓ |
| Table 11: face-swap detection after image processing (additional sources) | not released: a separate cohort of 100 further FF++ videos with its own releases, swaps and scores, generated outside this pipeline | ✗ |
| Table 12: empirical operating points of the direct relation test | `operating-points` | ✓ |
| Table 13: joint verification with per-head and joint budgets | `joint-policy` (per-head and matched panels; re-run). The FF++ delivery panel is not released | ~ |
| Table 14: cross-dataset threshold transfer | `threshold-transfer` | ✓ |
| Fig 12: calibration and record-model sensitivity | `hyperparam_sensitivity`, then `fig-hyper` (drawing re-run from the stored sweep) | ✓ |
| Table 15, Figs 13 and 14: ring, strength, wrong-key count | `sensitivity-release dataset=<FF++\|DFDC\|CelebDF> setting=<alpha:0.02\|band:0.05-0.45\|...>` (grid; see `experiments/sensitivity_launch.sh`), then `sensitivity-fig` | ✓ |
| Table 16: runtime and records | `deployment-cost` (runs under `peers_root`), `record-codec`; the CPU timings of our paths come from a later benchmark that is not included | ~ |
| Table 17: regeneration and targeted attacks | `adaptive-attacks`, `diffusion-regeneration` (template and image-only columns; the record-model S_fake column is not produced) | ~ |
| Table 18: phase-ring transplantation | `copy-ring-canonical`, `copy-ring-detector` (FF++), `copy-ring-amplitude`; the Celeb-DF and DFDC image-only and either-flag rows are not produced | ~ |

`image-detector-train` trains the same detector without augmentation; it is the baseline behind the
augmentation remark of App. G.1 (read with `benign-channels plain_head=<ds>`).

## Expected variation

Record-model training fixes the random seed but not GPU nondeterminism, so a rerun can differ from
the paper in the third decimal. The threshold-transfer table (Table 14) is the one place where
printed values move, for example 3.9e-2 becoming 3.6e-2; its conclusion does not change.
ArcFace similarities also vary slightly between inference runs (up to about 3e-4). With only 100
calibration releases per dataset this can move a lossless-calibrated ArcFace cutoff by a few
queries: one rerun of `original-control calibration=100` gave 99.25% lossless acceptance where
Table 8 prints 98.75%; every other cell of Tables 3 and 8 reproduced exactly.

## What you need to provide

The repository contains the method, the pipeline, and the evaluations mapped above. It does not ship:

1. **The datasets** (FF++, Celeb-DF v2, DFDC), see [DATA.md](DATA.md).
2. **SimSwap weights** (`simswap_root`), used to build the face-swap queries.
3. **`ffmpeg` on `PATH`**, for frame extraction and the one-frame x264/x265 channels.
4. **Peer method repositories and weights** (`peers_root`), for the comparison tables only:
   StegaStamp, SepMark, LIDMark, FaceSigns, KAD-Net, WAM, VINE, each cloned from its authors with
   their released weights.
5. **Attack engine weights** (`engines_root`) for the engine columns: StarGAN-v2, UniFace,
   InSwapper, Stable Diffusion 1.5 inpainting.
6. **InsightFace `buffalo_l` weights** for the ArcFace references (`reference-baselines`,
   `original-control`); `insightface` downloads them on first use.

Items 4 to 6 are only needed for the comparison experiments. Two entries, `fig-imperceptibility`
and `deployment-cost`, also read release renders from an earlier enrollment run
(`legacy_trigger_glob`) that are not part of this release.

## How peers are evaluated

Every peer runs through the same interface: embed its mark into the enrollment input, write and
re-read a PNG (so everyone pays the same quantization cost), apply the same recorded face swap,
then score. No peer ever sees the private records. When our own retraining of a peer scored below
the authors' released checkpoint, the paper reports the stronger number; the retraining attempts
stay reproducible from this repository. The same rule applies to our image-only detector, which is
retrained on the same split the peers were.
