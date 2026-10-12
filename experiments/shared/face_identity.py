"""ArcFace identity similarity, the face-recognition reference control (paper Tables 3, 7, 8).

InsightFace ``buffalo_l`` (SCRFD detector + ArcFace recognizer) on an RGB image; the identity
similarity of two images is the dot product of the first detected face's normalized embeddings
(cosine similarity), NaN when either image has no detected face. Settings used by
attribution_matrix.py, reference_baselines.py and original_control.py: detection size 320 × 320,
detection threshold 0.3, CUDA with CPU fallback.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image

DEFAULT_PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")


def face_analysis(allowed_modules: Sequence[str] | None = None, det_size: tuple[int, int] = (320, 320),
                  det_thresh: float = 0.3, ctx_id: int = 0,
                  providers: Sequence[str] = DEFAULT_PROVIDERS):
    """A prepared ``insightface.app.FaceAnalysis("buffalo_l")``.

    ``allowed_modules=("detection", "recognition")`` skips the landmark and attribute models
    (original_control.py); the embedding is the same either way.
    """
    from insightface.app import FaceAnalysis

    kwargs = {"allowed_modules": list(allowed_modules)} if allowed_modules is not None else {}
    app = FaceAnalysis(name="buffalo_l", providers=list(providers), **kwargs)
    app.prepare(ctx_id=ctx_id, det_size=det_size, det_thresh=det_thresh)
    return app


def identity_embedding(app, image: Image.Image | np.ndarray) -> np.ndarray | None:
    """Normalized ArcFace embedding of the first detected face in an RGB image, or None."""
    rgb = np.asarray(image.convert("RGB")) if isinstance(image, Image.Image) else np.asarray(image)
    faces = app.get(rgb[:, :, ::-1])
    return faces[0].normed_embedding if faces else None


def identity_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Cosine similarity of two embeddings; NaN when either is missing."""
    return float(np.dot(a, b)) if a is not None and b is not None else np.nan
