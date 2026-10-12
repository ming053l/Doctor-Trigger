"""doctor_trigger.record_model against the code it replaces, on the shipped checkpoints.

References (frozen outputs, see golden_values.py):
* the former workflow/record_model/src/record_model.py: variant columns, in
  tests/golden/record_model_library.json
* the record-model trainer's former standardize, fit and predict (now calls into this module):
  tests/golden/record_model_trainer.json
* experiments/joint_policy.py load_models / predict (NaN -> column mean) and
  experiments/copy_ring_canonical.py predict (NaN -> 0), now calls into this module: frozen outputs
  in tests/golden/experiment_record_model.json
Every comparison is exact (``np.array_equal`` / ``torch.equal``, or SHA-256 digests of the bytes).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from doctor_trigger import record_model as rm
from golden_values import digest, frame_digest, load

ROOT = Path("/ssd2/ming/ttf_run/public_view")
OUTPUTS = Path("/ssd2/ming/ttf_run/public_view_outputs")
CALIBRATED_CKPTS = ROOT / "results/record_model/checkpoints"
RECORDSCALAR_RUN = OUTPUTS / "record_model_runs/recordscalar_ffpp"
RECORDSCALAR_FEATURES = OUTPUTS / "features_record_scalar"

pytestmark = pytest.mark.skipif(not CALIBRATED_CKPTS.exists(), reason="public_view checkpoints not available")


@pytest.fixture(scope="module")
def old_lib():
    return load("record_model_library")


@pytest.fixture(scope="module")
def old_trainer():
    return load("record_model_trainer")


def test_columns(old_lib):
    for csv in (ROOT / "features/calibrated_features_ffpp_test.csv", RECORDSCALAR_FEATURES / "recordscalar_features_ffpp_test.csv"):
        cols = list(pd.read_csv(csv, nrows=1).columns)
        for variant in rm.VARIANTS:
            assert rm.feature_columns(variant, cols) == old_lib["feature_columns_for_variant"][csv.name][variant]
    assert rm.AGREEMENT_COLUMNS == old_lib["HOST_COLUMNS"] and rm.KEY_COLUMNS == old_lib["TRIGGER_COLUMNS"]


def test_checkpoint_columns_match_variant():
    for ckpt in sorted(CALIBRATED_CKPTS.glob("*.pt")):
        fitted = rm.load_checkpoint(ckpt)
        cols = list(pd.read_csv(ROOT / "features/calibrated_features_ffpp_train.csv", nrows=1).columns)
        assert fitted.feature_columns == rm.feature_columns(fitted.variant, cols)


@pytest.mark.parametrize("eval_name", ["ffpp_test", "dfdc_test", "celebdf_test"])
def test_trainer_predict_bit_identical(old_trainer, eval_name):
    frame = pd.read_csv(ROOT / f"features/calibrated_features_{eval_name}.csv")
    for ckpt_file in sorted(CALIBRATED_CKPTS.glob("*.pt")):
        fitted = rm.load_checkpoint(ckpt_file)
        new_scores = rm.score_table(fitted, frame)
        assert frame_digest(new_scores) == old_trainer["predict"][f"{eval_name}|{ckpt_file.name}"], ckpt_file.name
        # and the score CSV the trainer wrote when the checkpoint was made
        stem = ckpt_file.stem.replace("_calibrated_ffpp_to_all", "")
        stored = pd.read_csv(ROOT / f"results/record_model/scores/calibrated_ffpp_to_all_{stem}_{eval_name}_scores.csv")
        for name in rm.SCORE_NAMES:
            assert np.array_equal(stored[name].to_numpy(np.float32), new_scores[name].to_numpy()), (ckpt_file.name, name)


@pytest.fixture(scope="module")
def old_experiments():
    return load("experiment_record_model")


@pytest.mark.skipif(not RECORDSCALAR_RUN.exists(), reason="record-scalar run not available")
def test_joint_policy_and_copy_ring_predict(old_experiments):
    frame = pd.read_csv(RECORDSCALAR_FEATURES / "recordscalar_features_ffpp_train.csv")
    frame = frame[frame.protocol == "benign"].reset_index(drop=True)
    with_nan = frame.head(300).copy()
    with_nan.loc[::7, "G_z"] = np.nan  # exercise the two NaN policies
    with_nan["C_b3"] = with_nan["C_b3"].astype(object)
    with_nan.loc[::11, "C_b3"] = "not-a-number"
    frames = {"frame": frame, "with_nan": with_nan}
    checkpoint = RECORDSCALAR_RUN / "checkpoints/full_record_model_seed{}_recordscalar_ffpp_to_all.pt"

    # joint_policy.py: (S_rel, S_key) per seed, NaN -> column mean
    joint = old_experiments["joint_policy_predict"]
    assert len(joint) == 6
    for seed in range(3):
        fitted = rm.load_checkpoint(str(checkpoint).format(seed))
        for frame_name, df in frames.items():
            scores = rm.record_model_scores(fitted, df)
            assert digest(np.stack([scores["S_rel"], scores["S_key"]], axis=1)) == joint[f"{seed}|{frame_name}"]

    # copy_ring_canonical.py: mean over the three seeds, NaN -> 0
    for frame_name, df in frames.items():
        per_seed = [rm.record_model_scores(rm.load_checkpoint(str(checkpoint).format(s)), df, missing="zero")
                    for s in range(3)]
        for name in ("S_rel", "S_key", "S_fake"):
            mean = np.mean([p[name] for p in per_seed], axis=0)
            assert digest(mean) == old_experiments["copy_ring_predict"][f"{frame_name}|{name}"], name


def test_standardize(old_trainer):
    frame = pd.read_csv(ROOT / "features/calibrated_features_ffpp_train.csv", nrows=2000)
    cols = rm.feature_columns("full_record_model", list(frame.columns))
    x = rm.feature_matrix(frame, cols)
    x[::13, 2] = np.nan
    x[:, 4] = 1.0  # constant column -> sigma floor
    assert [digest(b) for b in rm.standardize(x.copy())] == old_trainer["standardize"]


@pytest.mark.parametrize("variant", ["host_only", "full_record_model"])
def test_fit_bit_identical(old_trainer, variant):
    frame = pd.read_csv(ROOT / "features/calibrated_features_ffpp_train.csv", nrows=3000)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        new = rm.fit_record_model(frame, variant, seed=1, steps=25, hidden_dim=64, lr=1e-3, batch_size=512)
    finally:
        torch.set_num_threads(threads)
    old = old_trainer["fit"][variant]
    assert old["columns"] == new.feature_columns
    assert digest(new.mean) == old["mean"] and digest(new.std) == old["std"]
    assert old["final_loss"] == new.final_loss
    new_state = new.model.state_dict()
    assert list(old["state"]) == list(new_state)
    for key, value in new_state.items():
        assert digest(value) == old["state"][key], key


def test_checkpoint_round_trip(tmp_path):
    fitted = rm.load_checkpoint(next(iter(sorted(CALIBRATED_CKPTS.glob("full_record_model_seed0*.pt")))))
    out = tmp_path / "ck.pt"
    rm.save_checkpoint(fitted, out, "train.csv")
    ck = torch.load(out, map_location="cpu", weights_only=False)
    assert set(ck) == {"state_dict", "variant", "seed", "feature_columns", "mean", "std", "train_features_csv"}
    again = rm.load_checkpoint(out)
    frame = pd.read_csv(ROOT / "features/calibrated_features_ffpp_test.csv", nrows=500)
    a, b = rm.record_model_scores(fitted, frame), rm.record_model_scores(again, frame)
    assert all(np.array_equal(a[k], b[k]) for k in rm.SCORE_NAMES)
