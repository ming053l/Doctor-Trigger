"""Loading the two case manifests of a finished pipeline root, with the usual filters.

* ``cases.csv`` (``config`` key ``cases_csv``; schema ``doctor-trigger-common-case-v2``): one row
  per query J with its released image I', record R (``doctor_trigger_template_path``), the
  original crop, the protocol and the targets ``y_fake``, ``y_rel``, ``y_keycorr`` (y_key in the
  paper).
* ``queries.csv`` (``config`` key ``queries_csv``): the query manifest written by the query stage,
  with the same queries under the pipeline's own column names (``template_artifact``,
  ``released_path``, ``query_path``) and a ``status`` column.

Both loaders return ``pandas.DataFrame`` objects exactly as ``pd.read_csv`` reads the file, then
keep the rows matching the filters. A filter is ``None`` (keep all), one value, or a collection of
values. Row order and index are those of the file, so ``read_cases(dataset="FF++", split="test")``
equals ``cases[(cases.dataset == "FF++") & (cases.split == "test")]``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Collection, Union

import pandas as pd

from doctor_trigger import config

DATASETS = ("FF++", "DFDC", "CelebDF")
"""Dataset labels as stored in the manifests."""

DATASET_TAGS = {"FF++": "ffpp", "DFDC": "dfdc", "CelebDF": "celebdf"}
"""Dataset label -> lower-case tag used in file and checkpoint directory names."""

SPLITS = ("train", "val", "test")
PROTOCOLS = ("benign", "swap")
"""Query protocols in the manifests: benign releases and SimSwap face swaps. Wrong-key rows exist
only in the feature tables (``protocol == "wrong_key"``), not in these manifests."""

CASE_COLUMNS = (
    "case_schema_version", "query_id", "enrollment_id", "dataset", "split", "source_group_id",
    "input_path", "input_sha256", "enrollment_input_path", "enrollment_input_sha256",
    "attack_source_path", "attack_source_sha256", "doctor_trigger_released_path",
    "doctor_trigger_released_sha256", "doctor_trigger_template_path", "doctor_trigger_template_sha256",
    "protocol", "backend", "simswap_det_thresh", "fallback_attempt", "y_fake", "y_rel", "y_keycorr",
)
"""Columns of cases.csv (schema v2), in file order."""

QUERY_COLUMNS = (
    "query_id", "sample_id", "enrollment_id", "dataset", "split", "source_group_id", "fake_sample_id",
    "fake_source_group_id", "fake_class", "template_artifact", "released_path", "query_path", "protocol",
    "backend", "y_fake", "y_rel", "y_keycorr", "method_version", "query_sha256", "status",
    "fake_aligned_path", "simswap_det_thresh", "fallback_attempt",
)
"""Columns of queries.csv, in file order."""

Filter = Union[None, str, Collection[str]]


def _matches(column: pd.Series, wanted: Filter) -> pd.Series:
    if wanted is None:
        return pd.Series(True, index=column.index)
    if isinstance(wanted, str):
        return column == wanted
    return column.isin(list(wanted))


def select(frame: pd.DataFrame, *, dataset: Filter = None, split: Filter = None,
           protocol: Filter = None) -> pd.DataFrame:
    """Rows of ``frame`` whose ``dataset``, ``split`` and ``protocol`` match the filters."""
    mask = pd.Series(True, index=frame.index)
    for column, wanted in (("dataset", dataset), ("split", split), ("protocol", protocol)):
        if wanted is not None:
            mask &= _matches(frame[column], wanted)
    return frame[mask]


_REQUIRED = ("query_id", "enrollment_id", "dataset", "split", "protocol")


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], file: Path) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{file}: missing columns {missing}")


def read_cases(file: str | Path | None = None, *, dataset: Filter = None, split: Filter = None,
               protocol: Filter = None) -> pd.DataFrame:
    """cases.csv (default: config key ``cases_csv``), filtered by dataset / split / protocol."""
    file = Path(file) if file is not None else Path(config.path("cases_csv"))
    frame = pd.read_csv(file)
    _require_columns(frame, _REQUIRED, file)
    return select(frame, dataset=dataset, split=split, protocol=protocol)


def read_queries(file: str | Path | None = None, *, dataset: Filter = None, split: Filter = None,
                 protocol: Filter = None, passed_only: bool = True) -> pd.DataFrame:
    """queries.csv (default: config key ``queries_csv``), filtered by dataset / split / protocol.

    ``passed_only`` keeps rows with ``status == "pass"`` (every reader of this file does that).
    """
    file = Path(file) if file is not None else Path(config.path("queries_csv"))
    frame = pd.read_csv(file)
    _require_columns(frame, _REQUIRED + ("status",), file)
    if passed_only:
        frame = frame[frame["status"] == "pass"]
    return select(frame, dataset=dataset, split=split, protocol=protocol)


def sha256_file(file: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hex SHA-256 of a file's bytes, read in chunks."""
    digest = hashlib.sha256()
    with open(file, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
