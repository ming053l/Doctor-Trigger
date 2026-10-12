"""doctor_trigger.manifests against the pasted pandas filters it replaces, on a pipeline root.

Each reference below is a filter copied from an experiment (file noted); the loader must return
an identical DataFrame (same rows, order, index, dtypes). ``sha256_file`` must equal the
whole-file SHA-256 and the hashes recorded in cases.csv.
"""

from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from data_roots import PIPELINE_ROOT, requires, under
from doctor_trigger import config, manifests

ROOT = PIPELINE_ROOT
CASES = under(ROOT, "common/cases.csv")
QUERIES = under(ROOT, "manifests/queries.csv")

pytestmark = requires(CASES, QUERIES)


@pytest.fixture(scope="module")
def cases() -> pd.DataFrame:
    return pd.read_csv(CASES)


@pytest.fixture(scope="module")
def queries() -> pd.DataFrame:
    return pd.read_csv(QUERIES)


def same(a: pd.DataFrame, b: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert list(a.index) == list(b.index)


def test_case_filters(cases):
    new = lambda **kw: manifests.read_cases(CASES, **kw)  # noqa: E731
    same(new(), cases)
    same(new(split="test", dataset="FF++"), cases[(cases.split == "test") & (cases.dataset == "FF++")])  # benign_channels
    same(new(dataset="FF++", split="test", protocol="benign"),
         cases[(cases.dataset == "FF++") & (cases.split == "test") & (cases.protocol == "benign")])  # build_fig_keymatch
    same(new(split="test", protocol="benign"), cases[(cases.split == "test") & (cases.protocol == "benign")])  # release_evidence
    same(new(dataset="DFDC", protocol=["benign", "swap"]),
         cases[(cases.dataset == "DFDC") & (cases.protocol.isin(["benign", "swap"]))])  # train_image_detector
    same(new(split=("train", "val")), cases[cases.split.isin(["train", "val"])])
    sub = cases[cases.protocol == "benign"]
    same(manifests.select(sub, split="test"), sub[sub.split == "test"])  # original_control


def test_query_filters(queries):
    man = queries[queries.status == "pass"]
    new = lambda **kw: manifests.read_queries(QUERIES, **kw)  # noqa: E731
    same(new(), man)
    same(new(dataset="FF++", split="test", protocol="swap"),
         man[(man.dataset == "FF++") & (man.split == "test") & (man.protocol == "swap")])  # ring_transplant
    same(new(dataset="FF++", split="train", protocol="benign").head(200),
         man[(man.dataset == "FF++") & (man.split == "train") & (man.protocol == "benign")].head(200))
    same(new(dataset="FF++", split="test"), man[(man.dataset == "FF++") & (man.split == "test")])  # adaptive_attacks
    same(new(passed_only=False), queries)


def test_queries_path_comes_from_config(tmp_path, monkeypatch):
    file = tmp_path / "config.yaml"
    file.write_text(f"canonical_root: {ROOT}\ncases_csv: '{{canonical_root}}/common/cases.csv'\n")
    monkeypatch.setenv(config.CONFIG_ENV, str(file))
    assert config.path("queries_csv") == str(QUERIES)
    same(manifests.read_queries(dataset="CelebDF"), manifests.read_queries(QUERIES, dataset="CelebDF"))
    same(manifests.read_cases(split="val"), manifests.read_cases(CASES, split="val"))


def test_sha256_file(cases):
    assert manifests.sha256_file(CASES) == hashlib.sha256(CASES.read_bytes()).hexdigest()
    for row in cases.head(5).itertuples():
        assert manifests.sha256_file(row.input_path) == row.input_sha256
        assert manifests.sha256_file(row.doctor_trigger_template_path) == row.doctor_trigger_template_sha256


def test_vocabularies(cases):
    assert set(cases.dataset) == set(manifests.DATASETS)
    assert set(cases.split) == set(manifests.SPLITS)
    assert set(cases.protocol) == set(manifests.PROTOCOLS)
    assert tuple(cases.columns) == manifests.CASE_COLUMNS
    assert tuple(pd.read_csv(QUERIES, nrows=1).columns) == manifests.QUERY_COLUMNS
