# Record model training

`train_record_model.py` trains the record model f_θ (`doctor_trigger.record_model`) on one
record-feature table and scores it on others. The model maps the 75 features x = Φ(J, R) to three
scores: S_rel (relation to the enrolled face), S_key (enrolled key present) and S_fake.

```bash
PYTHONPATH=. python workflow/record_model/train_record_model.py --config workflow/configs/record_model.yaml
```

The YAML config names the training table (`train_features_csv`), the tables to score
(`eval_feature_csvs`), the input variants, the seeds and the training settings (`train.steps`,
`batch_size`, `hidden_dim`, `lr`). Relative paths resolve against the config file.

| Variant | Inputs |
|---|---|
| `host_only` | agreement readings: H, C, residual mean/std, mean cos r |
| `trigger_only` | key readings: G_s, wrong-key statistics, mean δ sin r, and all per-band columns |
| `host_plus_trigger` | both |
| `full_record_model` | both, plus context columns (`detector_score`, `alignment_quality`, `D_face_context`) when the table has them; the pipeline's tables do not, so this is the 75 features |

Outputs under `output_dir`: `checkpoints/<variant>_seed<k>_<run>.pt`,
`scores/<run>_<variant>_seed<k>_<eval>_scores.csv` (S_fake, S_rel, S_key and the targets per query),
`metrics/record_model_release_<run>_metrics.csv` (AUC, EER, TPR at FPR 1e-2/1e-3/1e-4 per score)
and `matrix/` (run manifest and seed-mean summary).
