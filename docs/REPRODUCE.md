# Reproducing the paper

Every number in the paper comes from two steps:

1. **The pipeline** (`workflow/`) turns extracted dataset frames into enrolled releases, private
   records, benign and face-swap queries, and per-query features. See [DATA.md](DATA.md) for the
   inputs and [`workflow/WORKFLOW.md`](../workflow/WORKFLOW.md) for every stage.
2. **The evaluations** (`experiments/`) read that finished pipeline root and write one CSV (or
   figure) per paper artifact. Each is a named entry in
   [`experiments/registry.yaml`](../experiments/registry.yaml), run through `run.py`.

```bash
cp config.example.yaml config.yaml      # set canonical_root, weights and peer paths
python run.py status                    # which inputs/outputs resolve
python run.py list                      # every entry
python run.py show threshold-transfer   # the resolved command, without running it
python run.py run threshold-transfer    # run one entry; results go under outputs_root
python run.py run benign-channels dataset=FF++   # entries with {slots} take k=v arguments
```

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
              ├─ extract_features.py           relation / key features for every query
              ├─ train_eval_record_model_matrix  trained record-model scorers (3 seeds)
              └─ integrity gate + receipts     hashes, counts, quarantine accounting
```

The workflow is resumable and self-verifying: re-running it against a completed root re-checks
hashes instead of recomputing, and `./run_workflow.sh --verify-only` runs the acceptance checks
alone.

| Pipeline output | Meaning |
| --- | --- |
| `artifacts/released/.../*.png` | The published copy of each enrolled face (visually unchanged, keyed signal written). |
| `private/templates/.../*.npz` | The private per-face record: enrolled phase, original phase and amplitude, keyed displacement, key. |
| `common/cases.csv` | The query inventory: every benign copy and swapped render, with paths and labels. |
| `features/calibrated_features.csv` | The verifier's reading of every query: relation, key projection, wrong-key margins, band features. |
| `FINAL_RECEIPT.json` | Hashes and counts for every stage, so a finished root is checkable without recomputing. |

Everything downstream consumes only `cases.csv`, the features CSV and the records. Peer methods are
scored through the same files, per
[`workflow/contracts/peer_score_contract.json`](../workflow/contracts/peer_score_contract.json).

## Paper artifact → registry entry

Numbering follows the current paper PDF on the
[project page](https://ming053l.github.io/Doctor-Trigger/static/pdf/paper.pdf). The `paper:` text
inside `registry.yaml` (what `run.py list` prints) was written against an earlier draft, so its
table and figure numbers can differ from the ones below.

**Confidence:** ✓ = the script's docstring or registry entry describes exactly this artifact;
~ = likely, from matching content, but not confirmed against the script's output; ? = not found.

### Main paper

| Paper artifact | Registry entries (`python run.py run <id>`) | |
| --- | --- | --- |
| Table 1: sub-band agreement C_B (benign processing, datasets, other manipulations) | `subband-table1 part=<benign\|datasets\|engines>`; engine columns read the swaps cached by `engines engine=<name>` | ✓ (C_B row); full-ring C row ~ |
| Table 2: face-swap detection within and across datasets | peers: `peers-detect method=<peer>`; image-only S_df: `image-detector-train-aug dataset=<ffpp\|dfdc\|celebdf>` then `image-detector-eval`; record model S_fake: `features-record-ref` → `features-record-scalar` → `record-model-recordscalar pin=<ffpp\|dfdc\|celebdf>` | ✓ for these rows; source of the direct 1 − C_B row ? |
| Table 3: release evidence from the key | `reference-baselines` (swap AUC), `release-evidence` (original / release pass) | ✓ |
| Fig 5: visual fidelity | `fig-imperceptibility` (also reads earlier trigger renders, `legacy_trigger_glob`, not shipped) | ✓ |
| Sec. 4.2 record storage (2.3 KiB sub-band payload) | `record-codec` | ✓ |
| Table 4 and Fig 7: credential retention and channels | `table4-artifact` | ✓ |
| Fig 2 (right) | drawn in the paper source; appendix version: `fig-channel-samples` | ~ |
| Figs 1, 3, 4, 6 | drawn in the paper source | — |

### Appendix

| Paper artifact | Registry entries | |
| --- | --- | --- |
| Table 6: release fidelity per dataset and resolution | `peer-fidelity` (runs per-peer `*_fidelity.py` under `peers_root`; those scripts are **not** in this repository) | ~ |
| Fig 9: verification examples across datasets | `fig-channel-samples` | ~ |
| Table 7: private-reference comparisons | `reference-baselines` (swap), `attribution-matrix` (same-person release), `release-evidence` (release vs original) | ✓ |
| Table 8: original and release acceptance under channel calibration | candidates: `release-evidence`, `posthoc-attack` | ? |
| Fig 11: enrolled template and asset specificity | `fig-template` | ✓ |
| Table 9: key discrimination and learned record scores | `key-pure-canonical` (needs `features-record-scalar`, `record-model-recordscalar pin=ffpp`) | ~ |
| Table 10: benign-channel robustness across datasets | `image-detector-train-aug dataset=<ds>`, then `benign-channels dataset=<FF++\|DFDC\|CelebDF>` | ✓ |
| Table 11: face-swap detection after image processing (C vs C_B) | closest: `subband-readout` | ? |
| Table 12: operating points of the direct relation test | `operating-points` | ✓ |
| Table 13: joint verification (direct vs learned) | — | ? |
| Table 14: cross-dataset threshold transfer | `threshold-transfer` | ✓ |
| Fig 12: calibration and record-model sensitivity | `hyperparam_sensitivity` | ✓ |
| Table 15, Figs 13 and 14: ring, strength, wrong-key count | `sensitivity-release dataset=<FF++\|DFDC\|CelebDF> setting=<alpha:0.02\|band:0.05-0.45\|...>` (grid; see `experiments/sensitivity_launch.sh`), then `sensitivity-fig` | ✓ |
| Table 16: runtime and records | `deployment-cost` (runs under `peers_root`), `record-codec` | ✓ |
| Table 17: regeneration and adaptive attacks | `adaptive-attacks`, `diffusion-regeneration` | ✓ |
| Table 18 and App. K.2: phase-ring transplantation | `copy-ring-canonical`, `copy-ring-detector`, `copy-ring-amplitude` | ✓ |
| Figs 8 and 10 | — | ? |

Registry entries not placed above (supporting controls whose current table placement is not
confirmed): `subband-fusion`, `subband-freelunch`, `subband-side-effects`, `reference-channels`,
`defect-audit`, `posthoc-attack`, `image-detector-train` (detector without augmentation; read with
`benign-channels plain_head=<ds>`).

## Expected variation

Record-model training fixes the random seed but not GPU nondeterminism, so a rerun can differ from
the paper in the third decimal. The threshold-transfer table (Table 14) is the one place where
printed values move, for example 3.9e-2 becoming 3.6e-2; its conclusion does not change.

## What you need to provide

The repository contains the method, the pipeline, and every evaluation. It does not ship:

1. **The datasets** (FF++, Celeb-DF v2, DFDC), see [DATA.md](DATA.md).
2. **SimSwap weights** (`simswap_root`), used to build the face-swap queries.
3. **`ffmpeg` on `PATH`**, for frame extraction and the one-frame x264/x265 channels.
4. **Peer method repositories and weights** (`peers_root`), for the comparison tables only:
   StegaStamp, SepMark, LIDMark, FaceSigns, KAD-Net, WAM, VINE, each cloned from its authors with
   their released weights.
5. **Attack engine weights** (`engines_root`) for the engine columns: StarGAN-v2, UniFace,
   InSwapper, Stable Diffusion 1.5 inpainting.

Items 4 and 5 are only needed for the comparison experiments. Two entries, `fig-imperceptibility`
and `deployment-cost`, also read trigger renders from an earlier enrollment run
(`legacy_trigger_glob`) that are not part of this release.

## How peers are evaluated

Every peer runs through the same interface: embed its mark into the enrollment input, write and
re-read a PNG (so everyone pays the same quantization cost), apply the same recorded face swap,
then score. No peer ever sees the private records. When our own retraining of a peer scored below
the authors' released checkpoint, the paper reports the stronger number; the retraining attempts
stay reproducible from this repository. The same rule applies to our image-only detector, which is
retrained on the same split the peers were.
