"""Where the data-backed tests find their inputs.

Most tests compare ``doctor_trigger`` with frozen outputs of the code it replaced, on real records
and images of a finished pipeline run. Point these environment variables at that run:

  DOCTOR_TRIGGER_TEST_ROOT         the finished pipeline root in the public layout (common/cases.csv,
                                   manifests/queries.csv, features/, private/templates/, results/)
  DOCTOR_TRIGGER_TEST_OUTPUTS      the experiment outputs made from that root (features_record_scalar/,
                                   record_model_runs/, ...)
  DOCTOR_TRIGGER_TEST_ENROLLMENTS  the run's enrollment manifest (one row per enrollment: aligned_path,
                                   released_path, template_artifact, alpha, ...)

A test whose inputs are not set or not present is skipped. The regression subsets (64 enrollments,
one per test video, and 512 queries) ship in tests/data/.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent / "data"
SUBSET_ENROLLMENTS = DATA / "subset_enrollments.csv"
SUBSET_QUERIES = DATA / "subset_queries.csv"


def _path(variable: str) -> Path | None:
    value = os.environ.get(variable)
    return Path(value) if value else None


PIPELINE_ROOT = _path("DOCTOR_TRIGGER_TEST_ROOT")
OUTPUTS = _path("DOCTOR_TRIGGER_TEST_OUTPUTS")
ENROLLMENTS = _path("DOCTOR_TRIGGER_TEST_ENROLLMENTS")


def under(root: Path | None, relative: str) -> Path | None:
    """``root / relative``, or ``None`` when the root is not set."""
    return None if root is None else root / relative


def available(*paths: Path | None) -> bool:
    return all(path is not None and path.exists() for path in paths)


def requires(*paths: Path | None) -> pytest.MarkDecorator:
    """Skip marker for a test module or test that reads ``paths``."""
    return pytest.mark.skipif(not available(*paths), reason="test data not available (see tests/data_roots.py)")


def skip_module_unless(*paths: Path | None) -> None:
    """Skip the calling test module (call at import time, before it reads ``paths``)."""
    if not available(*paths):
        pytest.skip("test data not available (see tests/data_roots.py)", allow_module_level=True)


def enrollment_subset(per_dataset: dict[str, int] | None = None):
    """Enrollment-manifest rows of the 64-enrollment subset (all, or the first ``count`` per dataset).

    Empty when the manifest is not available, so tests parametrized over it are skipped.
    """
    import pandas as pd

    if not available(ENROLLMENTS):
        return pd.DataFrame()
    rows = pd.read_csv(ENROLLMENTS).merge(pd.read_csv(SUBSET_ENROLLMENTS)[["enrollment_id"]], on="enrollment_id")
    if per_dataset is None:
        return rows
    return pd.concat([rows[rows.dataset == name].head(count) for name, count in per_dataset.items()])
