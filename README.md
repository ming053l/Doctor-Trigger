<h1 align="center">Doctor Trigger</h1>

<p align="center"><b>A Framework for Release-Bound Face Verification</b><br>
Check whether a face circulating online is still the version that was released.</p>

<p align="center">
<a href="https://ming053l.github.io/">Chia-Ming Lee</a><sup>1,2</sup>,
Chia-Yu Lin<sup>3</sup>,
Hung-Kai Huang<sup>2</sup>,
Yi-Ting Ku<sup>3</sup>,
Yu-Chen Liang<sup>3</sup>,
<a href="https://cchsu.info/">Chih-Chung Hsu</a><sup>2</sup>
</p>

<p align="center">
<sup>1</sup>University at Albany, SUNY<br>
<sup>2</sup>National Yang Ming Chiao Tung University<br>
<sup>3</sup>National Cheng Kung University
</p>

<p align="center">
<a href="https://ming053l.github.io/Doctor-Trigger/"><img alt="Project page" src="docs/assets/badges/project-page.svg" height="20"></a>
<a href="https://ming053l.github.io/Doctor-Trigger/static/pdf/paper.pdf"><img alt="Paper" src="docs/assets/badges/paper.svg" height="20"></a>
<a href="https://ming053l.github.io/Doctor-Trigger/#video"><img alt="Video" src="docs/assets/badges/video.svg" height="20"></a>
<a href="LICENSE"><img alt="License: MIT" src="docs/assets/badges/license.svg" height="20"></a>
</p>

<p align="center"><img src="docs/assets/workflow.png" width="900" alt="Doctor Trigger workflow: the publisher enrolls each face and keeps a private record; a requester submits a circulating copy and receives Confirmed, Unconfirmed or Unavailable."></p>

## What it does

Posted faces get reposted, compressed, swapped and edited as they spread. Doctor Trigger lets a
publisher check any circulating copy against the specific release it published. Before release, it
adds an invisible keyed signal to the face (51.2 dB PSNR on average) and files a small private
record. Later, a copy is compared with that record: ordinary sharing (JPEG, resizing) passes, a face
swap or generative edit fails, and the never-released original is told apart from the release.

<p align="center"><img src="docs/assets/demo_ai_face.jpg" width="820" alt="Four copies of one AI-generated face checked against its record: the posted image and a JPEG repost are Confirmed; a face swap and the never-posted original are Unconfirmed."><br>
<sub>Four copies checked against one private record, run with this code (from the project page). All people shown are AI-generated.</sub></p>

## How it works

| Step | What happens | Code |
| --- | --- | --- |
| **1. Enroll** | A per-face secret key `s` expands into a binary ±1 sign pattern `h_s` on a mid-frequency ring of the luminance Fourier phase; the release carries a weak phase displacement `δ_s` along it. The publisher keeps a private record `R`: template, original phase and amplitude weights, key. | [`doctor_trigger/enrollment.py`](doctor_trigger/enrollment.py), [`key.py`](doctor_trigger/key.py) |
| **2. Verify** | Two readings against the record. *Spectral agreement* `C_B`: is this still the enrolled face? *Key projection* `G_s`: is the keyed displacement still there? | [`doctor_trigger/readings.py`](doctor_trigger/readings.py) |
| **3. Decide** | Thresholds come from benign copies (about 1% benign rejection). An optional learned record model can replace the direct thresholds. | [`calibration.py`](doctor_trigger/calibration.py), [`decision.py`](doctor_trigger/decision.py), [`record_model.py`](doctor_trigger/record_model.py) |

| Status | Meaning |
| --- | --- |
| **Confirmed** | Both checks pass: consistent with the enrolled release. |
| **Unconfirmed** | Spectral agreement or key projection fails (e.g. a swapped face, or an original that was never released). Flagged for human review. |
| **Unavailable** | The record, the face, or a supported channel is missing, so no verdict is given. |

See [`docs/assets/verification.png`](docs/assets/verification.png) for the full verification diagram
and [docs/FAQ.md](docs/FAQ.md) for common questions (is it a watermark? what must stay private?).

## Quick start (no dataset, CPU, a few seconds)

```bash
git clone https://github.com/ming053l/Doctor-Trigger.git && cd Doctor-Trigger
pip install -e .                 # numpy, Pillow, scikit-image, PyYAML, pandas
python examples/quickstart.py
```

```
copy                                              C     G_s  status
posted (released PNG)                        0.9999   1.529  Confirmed
JPEG re-save, q85                            0.9995   1.018  Confirmed
resized 0.75x and back                       0.9998   1.539  Confirmed
unreleased original                          0.9997   0.000  Unconfirmed (release signal missing)
face region replaced                         0.7468   0.716  Unconfirmed (face changed, release signal missing)
center crop 90% (no record of geometry)           -       -  Unavailable
(thresholds)                                 0.9989   0.868
```

The thresholds there come from a toy calibration; see [examples/README.md](examples/README.md).
The same steps on your own image:

```bash
python -m doctor_trigger enroll face.png --sample-id face --out out/   # out/face_released.png + out/face_record_PRIVATE.npz
python -m doctor_trigger verify copy.png out/face_record_PRIVATE.npz             # prints H_B, C_B, G_s
python -m doctor_trigger calibrate scores.csv --column C_B --frr 0.01           # τ from your benign copies' scores
python -m doctor_trigger verify copy.png out/face_record_PRIVATE.npz --tau-c <τ_C> --tau-g <τ_G>   # adds the status
```

For the full pipeline (GPU, datasets, SimSwap), see [`workflow/ENVIRONMENT.md`](workflow/ENVIRONMENT.md)
and [`workflow/WORKFLOW.md`](workflow/WORKFLOW.md).

## Data

The paper enrolls 7,366 real faces from FaceForensics++, DFDC and Celeb-DF v2. Request each dataset
from its maintainers ([FF++](https://github.com/ondyari/FaceForensics),
[DFDC](https://ai.meta.com/datasets/dfdc/),
[Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics)); nothing is hosted here. Extract
frames to `<frames_root>/<split>/<class>/<video_id>/*.png`, then:

```bash
python workflow/scripts/make_source_manifest.py <frames_root> <out>/ffpp_manifest.csv   # once per dataset
cp config.example.yaml config.yaml    # set source_manifests.{ffpp,dfdc,celebdf} and simswap_root
workflow/run_workflow.sh --gpu 0      # align faces, enroll, build queries, extract features, train record model
```

Details: [docs/DATA.md](docs/DATA.md).

## Reproducing the paper

Each table and figure is a named entry in [`experiments/registry.yaml`](experiments/registry.yaml),
run against a finished pipeline root:

```bash
pip install -e ".[experiments]" && cp config.example.yaml config.yaml   # then fill in the paths
python -m doctor_trigger list                          # every entry and the paper artifact it feeds
python -m doctor_trigger reproduce threshold-transfer  # `python run.py run <id>` does the same
```

[docs/REPRODUCE.md](docs/REPRODUCE.md) maps every paper table and figure to its entry, lists the
external weights each one needs, and explains how peer methods are scored.

## Repository map

```
doctor_trigger/  the method (python -m doctor_trigger list|show|status|reproduce|enroll|verify|calibrate)
  spectrum.py      luminance Y, phase φ, wrap(·), normalized radius κ, sub-band B (κ ≤ 0.15)
  key.py           key s, sign pattern h_s (±1), active bins K_eff, displacement δ_s, wrong keys
  record.py        the private record R (NPZ), enrolled units and where their records live
  enrollment.py    encoded release I′ and record R for one face, α ladder, fidelity gate
  readings.py      spectral agreement H / C / C_B, key projection G_s, wrong-key statistics, features Φ
  calibration.py   thresholds τ at a target benign false-rejection rate (FRR)
  decision.py      Confirmed / Unconfirmed / Unavailable per unit
  channels.py      benign delivery channels (JPEG, resizing, one-frame codecs, ...)
  metrics.py       AUC, TPR at a fixed FPR, EER
  record_model.py  the optional learned record model f_θ (S_rel, S_key, S_fake; needs torch)
  config.py, manifests.py, cli.py   config.yaml paths, query manifests, the command line
examples/        dataset-free quick start (two AI-generated sample faces)
workflow/        the pipeline: frames -> aligned faces -> releases + private records -> queries -> features -> record model
experiments/     the paper's evaluations; registry.yaml maps them to commands
configs/         record-model training configs used by the evaluations
run.py           same as python -m doctor_trigger (run = reproduce)
docs/            DATA, REPRODUCE, FAQ and README figures
```

## Limitations and responsible use

- **Only enrolled faces.** It verifies copies of faces that were enrolled before release, against
  their own records. It is not a general deepfake detector.
- **Known weak spots** (paper Table 4, 200 FF++ releases): face-swap rejection drops to 63% after a
  repost of a repost, 11% under per-image H.264 intra coding, and 6% under a 90% center crop, while
  benign copies still pass. Strong diffusion regeneration and a phase-ring transplant attack also
  weaken it (App. K). Validate on the channels you intend to support.
- **Images only.** Video (per-frame enrollment, clip-level decisions) is future work; the package
  is laid out for it, see [docs/FAQ.md](docs/FAQ.md#faq) ("Does it work on video?").
- **Records are sensitive.** Templates reveal coarse facial structure; keep records, keys and the
  master seed private.
- **Responsible use.** The code is meant for publishers verifying copies of their own releases. Use
  the benchmark datasets under their own terms. All faces in `examples/` and the README figures are
  AI-generated.

## Citation

```bibtex
@misc{lee2026doctortrigger,
  title  = {Doctor Trigger: A Framework for Release-Bound Face Verification},
  author = {Lee, Chia-Ming and Lin, Chia-Yu and Huang, Hung-Kai and Ku, Yi-Ting and Liang, Yu-Chen and Hsu, Chih-Chung},
  year   = {2026},
  url    = {https://ming053l.github.io/Doctor-Trigger/},
  note   = {Under review}
}
```

## License

MIT, see [LICENSE](LICENSE). Datasets, peer checkpoints, and engine weights remain under their
original licenses and terms.
