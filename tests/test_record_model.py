"""doctor_trigger.record_model against the code it replaces, on the shipped checkpoints.

References (imported by file path):
* workflow/record_model/src/record_model.py           columns, model, loss
* workflow/record_model/scripts/train_eval_record_model_matrix.py   standardize, fit, predict
* experiments/joint_policy.py  load_models / predict  (NaN -> column mean)
* experiments/copy_ring_canonical.py  predict          (NaN -> 0)
Every comparison is exact (``np.array_equal`` / ``torch.equal``).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from doctor_trigger import record_model as rm

REPO = Path(__file__).resolve().parents[1]
ROOT = Path("/ssd2/ming/ttf_run/public_view")
OUTPUTS = Path("/ssd2/ming/ttf_run/public_view_outputs")
CALIBRATED_CKPTS = ROOT / "results/record_model/checkpoints"
RECORDSCALAR_RUN = OUTPUTS / "record_model_runs/recordscalar_ffpp"
RECORDSCALAR_FEATURES = OUTPUTS / "features_record_scalar"

pytestmark = pytest.mark.skipif(not CALIBRATED_CKPTS.exists(), reason="public_view checkpoints not available")


def import_by_path(name: str, file: Path, extra_path: Path | None = None):
    for mod in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
        del sys.modules[mod]
    if extra_path is not None:
        sys.path.insert(0, str(extra_path))
    try:
        spec = importlib.util.spec_from_file_location(name, file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    finally:
        if extra_path is not None:
            sys.path.remove(str(extra_path))
    return module


@pytest.fixture(scope="module")
def old_lib():
    return import_by_path("old_record_model", REPO / "workflow/record_model/src/record_model.py")


@pytest.fixture(scope="module")
def old_trainer():
    return import_by_path("old_record_model_trainer",
                          REPO / "workflow/record_model/scripts/train_eval_record_model_matrix.py")


def test_columns(old_lib):
    for csv in (ROOT / "features/calibrated_features_ffpp_test.csv", RECORDSCALAR_FEATURES / "recordscalar_features_ffpp_test.csv"):
        cols = list(pd.read_csv(csv, nrows=1).columns)
        for variant in rm.VARIANTS:
            assert rm.feature_columns(variant, cols) == old_lib.feature_columns_for_variant(variant, cols)
    assert rm.AGREEMENT_COLUMNS == old_lib.HOST_COLUMNS and rm.KEY_COLUMNS == old_lib.TRIGGER_COLUMNS


def test_checkpoint_columns_match_variant():
    for ckpt in sorted(CALIBRATED_CKPTS.glob("*.pt")):
        fitted = rm.load_checkpoint(ckpt)
        cols = list(pd.read_csv(ROOT / "features/calibrated_features_ffpp_train.csv", nrows=1).columns)
        assert fitted.feature_columns == rm.feature_columns(fitted.variant, cols)


@pytest.mark.parametrize("eval_name", ["ffpp_test", "dfdc_test", "celebdf_test"])
def test_trainer_predict_bit_identical(old_lib, old_trainer, eval_name):
    frame = pd.read_csv(ROOT / f"features/calibrated_features_{eval_name}.csv")
    for ckpt_file in sorted(CALIBRATED_CKPTS.glob("*.pt")):
        ck = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        old_model = old_lib.RecordModelMLP(old_lib.RecordModelConfig(
            variant=ck["variant"], input_dim=len(ck["feature_columns"]),
            hidden_dim=ck["state_dict"]["encoder.0.weight"].shape[0]))
        old_model.load_state_dict(ck["state_dict"])
        old_model.eval()
        old_scores = old_trainer.predict(old_model, frame, ck["feature_columns"], ck["mean"], ck["std"])
        fitted = rm.load_checkpoint(ckpt_file)
        new_scores = rm.score_table(fitted, frame)
        pd.testing.assert_frame_equal(old_scores, new_scores, check_exact=True)
        # and the score CSV the trainer wrote when the checkpoint was made
        stem = ckpt_file.stem.replace("_calibrated_ffpp_to_all", "")
        stored = pd.read_csv(ROOT / f"results/record_model/scores/calibrated_ffpp_to_all_{stem}_{eval_name}_scores.csv")
        for name in rm.SCORE_NAMES:
            assert np.array_equal(stored[name].to_numpy(np.float32), new_scores[name].to_numpy()), (ckpt_file.name, name)


@pytest.fixture(scope="module")
def experiment_env():
    env = {"DT_CANON_ROOT": str(ROOT), "DT_OUTPUTS": str(OUTPUTS),
           "DT_RECORD_MODEL_RUNS": str(OUTPUTS / "record_model_runs"),
           "DT_CKPT_DIR": str(RECORDSCALAR_RUN / "checkpoints")}
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.mark.skipif(not RECORDSCALAR_RUN.exists(), reason="record-scalar run not available")
def test_joint_policy_and_copy_ring_predict(experiment_env):
    frame = pd.read_csv(RECORDSCALAR_FEATURES / "recordscalar_features_ffpp_train.csv")
    frame = frame[frame.protocol == "benign"].reset_index(drop=True)
    with_nan = frame.head(300).copy()
    with_nan.loc[::7, "G_z"] = np.nan  # exercise the two NaN policies
    with_nan["C_b3"] = with_nan["C_b3"].astype(object)
    with_nan.loc[::11, "C_b3"] = "not-a-number"

    joint = import_by_path("old_joint_policy", REPO / "experiments/joint_policy.py")
    for seed, model, ck in joint.load_models():
        fitted = rm.load_checkpoint(RECORDSCALAR_RUN / f"checkpoints/full_record_model_seed{seed}_recordscalar_ffpp_to_all.pt")
        for df in (frame, with_nan):
            scores = rm.record_model_scores(fitted, df)
            assert np.array_equal(joint.predict(model, ck, df), np.stack([scores["S_rel"], scores["S_key"]], axis=1))

    copy_ring = import_by_path("old_copy_ring_canonical", REPO / "experiments/copy_ring_canonical.py")
    for df in (frame, with_nan):
        old = copy_ring.predict(df)
        per_seed = [rm.record_model_scores(rm.load_checkpoint(
            RECORDSCALAR_RUN / f"checkpoints/full_record_model_seed{s}_recordscalar_ffpp_to_all.pt"), df, missing="zero")
            for s in range(3)]
        for name in ("S_rel", "S_key", "S_fake"):
            assert np.array_equal(old[name], np.mean([p[name] for p in per_seed], axis=0)), name


def test_standardize(old_trainer):
    frame = pd.read_csv(ROOT / "features/calibrated_features_ffpp_train.csv", nrows=2000)
    cols = rm.feature_columns("full_record_model", list(frame.columns))
    x = rm.feature_matrix(frame, cols)
    x[::13, 2] = np.nan
    x[:, 4] = 1.0  # constant column -> sigma floor
    for a, b in zip(old_trainer.standardize(x.copy()), rm.standardize(x.copy())):
        assert a.dtype == b.dtype and np.array_equal(a, b)


@pytest.mark.parametrize("variant", ["host_only", "full_record_model"])
def test_fit_bit_identical(old_trainer, variant):
    frame = pd.read_csv(ROOT / "features/calibrated_features_ffpp_train.csv", nrows=3000)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        old = old_trainer.fit_model(frame, variant, seed=1, steps=25, hidden_dim=64, lr=1e-3, batch_size=512)
        new = rm.fit_record_model(frame, variant, seed=1, steps=25, hidden_dim=64, lr=1e-3, batch_size=512)
    finally:
        torch.set_num_threads(threads)
    assert old["columns"] == new.feature_columns
    assert np.array_equal(old["mean"], new.mean) and np.array_equal(old["std"], new.std)
    assert old["final_loss"] == new.final_loss
    old_state, new_state = old["model"].state_dict(), new.model.state_dict()
    assert list(old_state) == list(new_state)
    for key in old_state:
        assert torch.equal(old_state[key], new_state[key]), key


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
