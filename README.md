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

```bash
cd workflow
pip install -r requirements-workflow.txt          # see ENVIRONMENT.md for the full setup
python scripts/make_source_manifest.py --help     # stage 0: point it at your extracted frames
./run_workflow.sh --gpu 0                         # runs every stage, resumable
./run_workflow.sh --verify-only                   # acceptance checks on a finished root
```

## License

MIT. See [LICENSE](LICENSE).
