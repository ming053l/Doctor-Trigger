# Doctor Trigger

Code for **DOCTOR TRIGGER: Private Face Records for Cooperative Content Provenance**.

Anonymous authors (under double-blind review).

**The idea in three sentences.** When a face image is released, the pipeline files a private
record of it: the face's own mid-band phase, displaced by a secret key, together with the
amplitude weights needed to read it back. When a copy of that image returns from the wild, the
verifier compares the copy against its own record instead of asking a classifier "does this look
manipulated?". A face swap breaks the recorded relation even when the pixels look clean, while
ordinary distribution (JPEG, WebP, resizing) largely preserves it.

## Where to look

| Path | What it is |
| --- | --- |
| [`workflow/WORKFLOW.md`](workflow/WORKFLOW.md) | Start here: every stage, its inputs and outputs, and why it exists. |
| [`workflow/ENVIRONMENT.md`](workflow/ENVIRONMENT.md) | Environment setup (conda env, model weights, dataset layout). |
| [`workflow/run_workflow.sh`](workflow/run_workflow.sh) | The one command that runs the whole pipeline, resumable. |
| [`workflow/scripts/`](workflow/scripts/) | The pipeline stages, one script per stage (listed below in order). |
| [`workflow/scripts/enroll.py`](workflow/scripts/enroll.py) | The heart of the method: writes the keyed phase displacement and files the private NPZ record. |
| [`workflow/scripts/extract_features.py`](workflow/scripts/extract_features.py) | The verifier's reading: relation, key projection, and wrong-key features per query. |
| [`workflow/record_model/`](workflow/record_model/) | The record model (a small MLP) that calibrates those features into scores. |
| [`workflow/contracts/`](workflow/contracts/) | Machine-readable specs of the output files and the peer-scoring interface. |
| [`workflow/assets/detectors/`](workflow/assets/detectors/) | Face-detector weights (SCRFD, YuNet) used for alignment. |
| [`run.py`](run.py) and [`experiments/registry.yaml`](experiments/registry.yaml) | The evaluations behind the paper: each table and figure mapped to one command. |

## The pipeline

One command drives everything, from extracted frames to verified records:

```
raw videos (FF++ / Celeb-DF / DFDC)
  └─ extract frames with ffmpeg  ->  <frames_root>/<split>/<class>/<video_id>/*.png
      └─ scripts/make_source_manifest.py       stage 0: one manifest CSV per dataset
          └─ ./run_workflow.sh --gpu 0         then, in order:
              ├─ build_base_manifest.py        identifiers, leak-free video-group splits
              ├─ align_faces.py                SCRFD crops, second-detector agreement, lossless PNG
              ├─ enroll.py                     released image + private NPZ phase record per asset
              ├─ build_queries.py              benign copies and SimSwap face-swap queries
              ├─ extract_features.py           relation / key features for every query
              ├─ train_eval_record_model_matrix  trained record-model scorers (3 seeds)
              └─ integrity gate + receipts     hashes, counts, quarantine accounting
```

The workflow is resumable and self-verifying: re-running it against a completed root re-checks
hashes instead of recomputing, and `./run_workflow.sh --verify-only` runs the acceptance
procedure alone.

## What comes out

| Artifact | Meaning |
| --- | --- |
| `artifacts/released/.../*.png` | The published copy of each enrolled face (visually unchanged, trigger written). |
| `private/templates/.../*.npz` | The private per-face record: enrolled phase, host phase and amplitudes, keyed displacement, key. |
| `common/cases.csv` | The query inventory: every benign copy and swapped render, with paths and labels. |
| `features/calibrated_features.csv` | The verifier's reading of every query: relation, key projection, wrong-key margins, band features. |
| `FINAL_RECEIPT.json` | Hashes and counts for every stage, so a finished root is checkable without recomputing. |

Everything downstream (scoring, tables, plots) consumes only `cases.csv` and the features CSV.
Peer methods are scored through the same two files, per
[`workflow/contracts/peer_score_contract.json`](workflow/contracts/peer_score_contract.json).

## Quick start

Build the records:

```bash
cd workflow
pip install -r requirements-workflow.txt          # see ENVIRONMENT.md for the full setup
python scripts/make_source_manifest.py --help     # stage 0: point it at your extracted frames
./run_workflow.sh --gpu 0                         # runs every stage, resumable
./run_workflow.sh --verify-only                   # acceptance checks on a finished root
```

Then run the evaluations against that root:

```bash
cp config.example.yaml config.yaml      # set canonical_root to the finished workflow root, then the weights
python run.py status                    # which inputs resolve
python run.py list                      # every evaluation, with the paper artifact it produces
python run.py show threshold-transfer   # the resolved command, without running it
python run.py run threshold-transfer
```

## Reproducing the paper

Each row lists the registry entries to run (`python run.py run <id>`). Entries print where they
write; results go under `outputs_root`.

| Paper artifact | Registry entries |
| --- | --- |
| Table 1: verification paths across channels, datasets, and re-rendering | `subband-table1`, `subband-fusion`, `engines engine=<name>`, `image-detector-train-aug dataset=<ds>` then `image-detector-eval` |
| Table 2: peer comparison and transfer | `peers-detect method=<peer>` (once per peer), `image-detector-eval`; record-model rows: `features-record-ref`, `features-record-scalar`, then `record-model-recordscalar pin=<ffpp\|dfdc\|celebdf>` |
| Table 3: reference controls | `reference-baselines`, `reference-channels`, `attribution-matrix`, `release-evidence` |
| Table 4 and Fig 7: credential retention and channel limits | `table4-artifact` |
| Fig 5: imperceptibility | `fig-imperceptibility` |
| Table 8: benign channels per dataset | `benign-channels dataset=<FF++\|DFDC\|CelebDF>` |
| Figs 9 and 10, Table 12: release-mechanism sensitivity | `sensitivity-release` (grid), then `sensitivity-fig` |
| Fig 11: hyperparameters | `hyperparam_sensitivity` |
| Table 11: post-hoc enrollment attack | `posthoc-attack` |
| Table 13: release fidelity | `peer-fidelity` |
| Table 14: operating points | `operating-points` |
| Tables 15 and 16: wrong keys and key-only operating points | `key-pure-canonical` |
| Table 17: threshold transfer | `threshold-transfer` |
| Table 18: stress tests | `adaptive-attacks`, `diffusion-regeneration` |
| Table 19: deployment cost | `deployment-cost` |
| Table 20: ring transplantation | `copy-ring-canonical`, `copy-ring-detector`, `copy-ring-amplitude` |
| Appendix E.1: template source and channel cancellation | `defect-audit` |
| Appendix I.1: sub-band readout | `subband-readout`, `subband-freelunch`, `subband-side-effects` |
| Figs 12 and 16 | `fig-template`, `fig-channel-samples` |

The remaining figures are drawn in the paper source; the entries above produce the data behind
them.

Record-model training fixes the random seed but not GPU nondeterminism, so a rerun can differ from
the paper in the third decimal. The threshold-transfer table (Table 17) is the one place where printed
values move, for example 3.9e-2 becoming 3.6e-2; its conclusion does not change.

## What you need to provide

The repository contains the method, the pipeline, and every evaluation. It does not ship:

1. **The datasets.** Obtain FaceForensics++, Celeb-DF v2, and DFDC from their maintainers under
   their terms, extract frames, and run stage 0.
2. **SimSwap weights** (`simswap_root` in config), used to build the face-swap queries.
3. **`ffmpeg` on `PATH`**, for frame extraction and the one-frame x264/x265 channels.
4. **Peer method repositories and weights** (`peers_root`), for the comparison tables only:
   StegaStamp, SepMark, LIDMark, FaceSigns, KAD-Net, WAM, VINE, each cloned from its authors with
   their released weights.
5. **Attack engine weights** (`engines_root`) for the engine columns: StarGAN-v2, UniFace,
   InSwapper, Stable Diffusion 1.5 inpainting.

Items 4 and 5 are only needed for the comparison experiments; the pipeline and the method itself
need items 1 to 3. Two entries, `fig-imperceptibility` and `deployment-cost`, also read trigger
renders from an earlier enrollment run (`legacy_trigger_glob`) that are not part of this release.

## How peers are evaluated

Every peer runs through the same interface: embed its mark into the enrollment input, write and
re-read a PNG (so everyone pays the same quantization cost), apply the same recorded face swap,
then score. No peer ever sees the private records. When our own retraining of a peer scored below
the authors' released checkpoint, the paper reports the stronger number; the retraining attempts
stay reproducible from this repository. The same rule applies to our image-only detector, which is
retrained on the same split the peers were.

## Repository layout

```
workflow/               the pipeline: raw frames -> records -> features -> record model
  WORKFLOW.md           stage-by-stage documentation
  run_workflow.sh       pipeline driver (resumable, self-verifying)
  scripts/              pipeline stages
  record_model/         record model: code and configs
  contracts/            the peer scoring interface and stage contracts
  assets/detectors/     face detector ONNX models used for alignment
run.py                  evaluation entry point: list / show / run / status
config.example.yaml     template for machine-local paths (copy to config.yaml)
configs/record_model/   record-model training configs used by the evaluations (templates)
experiments/            the evaluations; registry.yaml maps paper artifacts to commands
  _snapshot/            trimmed model code the evaluations import: image-only detector,
                        benign channels, SimSwap wrapper, FaceSigns reimplementation
dt/                     path resolution shared by run.py and the evaluations
```

## License

MIT. See [LICENSE](LICENSE). Datasets, peer checkpoints, and engine weights remain under their
original licenses and terms.
