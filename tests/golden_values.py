"""Frozen outputs of the workflow scripts' old helper functions, for the bit-identity tests.

The old helpers (workflow/scripts/enroll.py, extract_features.py, the record-model trainer's
standardize/fit/predict, the quickstart's jpeg/resized) were replaced by calls into
``doctor_trigger``. Their outputs on the test inputs were saved before the removal, as SHA-256
digests of the exact bytes (arrays), exact JSON values (scalars) or digests of feature rows,
in ``tests/golden/<name>.json``. ``digest`` here is the function the goldens were made with.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"


def digest(array: Any) -> str:
    """SHA-256 over dtype, shape and the C-order bytes of an array (a torch tensor is converted)."""
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    array = np.asarray(array)
    if array.dtype == object:
        raise TypeError("object arrays have no stable bytes; use frame_digest")
    return hashlib.sha256(f"{array.dtype.str}|{array.shape}|".encode() + array.tobytes()).hexdigest()


def features_digest(values: Mapping[str, Any]) -> str:
    """SHA-256 over the ordered (column, Python type, value) triples of a feature row (NaN-safe)."""
    triples = [[column, type(value).__name__, value] for column, value in values.items()]
    return hashlib.sha256(json.dumps(triples).encode()).hexdigest()


def frame_digest(frame) -> str:
    """SHA-256 over a DataFrame's column names, dtypes and values (object columns as JSON)."""
    parts = []
    for column in frame.columns:
        values = frame[column].to_numpy()
        body = json.dumps(values.tolist(), default=str) if values.dtype == object else digest(values)
        parts.append(f"{column}|{values.dtype.str}|{body}")
    parts.append(f"index|{digest(frame.index.to_numpy())}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


@lru_cache(maxsize=None)
def load(name: str) -> dict:
    """The golden file ``tests/golden/<name>.json``."""
    return json.loads((GOLDEN_DIR / f"{name}.json").read_text())
