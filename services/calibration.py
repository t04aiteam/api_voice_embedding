"""Score calibration: center raw embeddings against a background mean vector.

calibration/mean_embedding.npy holds the average raw embedding over a
background set of speakers. Subtracting it before comparing/normalizing keeps
cosine similarity scores centered around 0 for different speakers regardless
of recording conditions, which is what makes a single fixed threshold usable.
Re-run scripts/calibrate.py on real data before relying on this in production.
"""
import os
import threading

import numpy as np

from services.model import EMBEDDING_DIM

CALIBRATION_PATH = os.getenv("CALIBRATION_PATH", "calibration/mean_embedding.npy")

_mean_embedding: np.ndarray | None = None
_lock = threading.Lock()


def get_mean_embedding() -> np.ndarray:
    global _mean_embedding
    if _mean_embedding is not None:
        return _mean_embedding
    with _lock:
        if _mean_embedding is not None:
            return _mean_embedding
        if os.path.exists(CALIBRATION_PATH):
            _mean_embedding = np.load(CALIBRATION_PATH).astype(np.float32)
        else:
            # Identity calibration until scripts/calibrate.py produces a real vector.
            _mean_embedding = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        return _mean_embedding


def reload_mean_embedding() -> np.ndarray:
    """Force a re-read from disk (used by tooling after recalibrating)."""
    global _mean_embedding
    _mean_embedding = None
    return get_mean_embedding()


def calibrate_embedding(raw: np.ndarray) -> np.ndarray:
    """Center a raw embedding against the background mean and L2-normalize it."""
    centered = raw - get_mean_embedding()
    norm = np.linalg.norm(centered)
    if norm > 1e-8:
        centered = centered / norm
    return centered.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)
