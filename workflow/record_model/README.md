# Phase-Relation RecordModel Code Pack

This is the RecordModel component of the release. It contains the trigger-conditioned
feature extractor, RecordModel model/training/evaluation scripts, variant configs, and
unit tests needed to reproduce RecordModel experiments from existing templates, keys,
frames, and feature CSVs.

Excluded by design:

- no old TCH / V0 checkpoint package
- no image-detector or passive-backbone weights
- no paper data or private server paths
- no generated score dumps
- no author/user names or machine-specific paths

Typical usage:

```bash
python -m pytest tests/test_trigger_conditioned_features.py
python scripts/extract_record_model_features.py --config configs/record_model_feature_extraction.yaml
python scripts/train_eval_record_model_matrix.py --config configs/record_model_train_eval.yaml
```

Input manifests should be supplied under `data/manifests/` or configured via
the YAML files. This pack does not rebuild datasets and does not embed new
triggers.
