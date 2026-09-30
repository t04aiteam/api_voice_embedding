"""Persistence for enrolled speakers.

Each speaker's calibrated embeddings are stacked into
data/speakers/{speaker_id}.npy as a (samples, 2048) float32 array. Enrolling
again with a new recording appends a row; verification compares against the
centroid of all stored rows.
"""
import os
import re
import threading

import numpy as np

DATA_DIR = os.getenv("SPEAKER_DATA_DIR", "data/speakers")
SPEAKER_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

_lock = threading.Lock()


def is_valid_speaker_id(speaker_id: str) -> bool:
    return bool(SPEAKER_ID_RE.match(speaker_id))


def _ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)


def _path(speaker_id: str) -> str:
    return os.path.join(DATA_DIR, f"{speaker_id}.npy")


def enroll(speaker_id: str, embedding: np.ndarray) -> int:
    """Append a calibrated embedding for speaker_id and return the new sample count."""
    _ensure_data_dir()
    path = _path(speaker_id)
    with _lock:
        row = embedding.astype(np.float32)[None, :]
        if os.path.exists(path):
            existing = np.load(path)
            updated = np.vstack([existing, row])
        else:
            updated = row
        np.save(path, updated)
        return int(updated.shape[0])


def get_embeddings(speaker_id: str) -> np.ndarray | None:
    path = _path(speaker_id)
    if not os.path.exists(path):
        return None
    return np.load(path)


def exists(speaker_id: str) -> bool:
    return os.path.exists(_path(speaker_id))


def list_speakers() -> list[dict]:
    _ensure_data_dir()
    result = []
    for name in sorted(os.listdir(DATA_DIR)):
        if not name.endswith(".npy"):
            continue
        speaker_id = name[: -len(".npy")]
        arr = np.load(os.path.join(DATA_DIR, name))
        result.append({"speaker_id": speaker_id, "samples": int(arr.shape[0])})
    return result


def centroids() -> dict[str, np.ndarray]:
    """Map every enrolled speaker to the mean of its stored embeddings.

    The same centroid /speakers/{id}/verify scores against, so identify and
    verify give one score for one file and one person.

    Returns:
        speaker_id to centroid, in speaker_id order.
    """
    # ponytail: reads every .npy on each call, fine to a few thousand speakers;
    # keep centroids in memory (invalidated by enroll/delete) if identify gets slow.
    _ensure_data_dir()
    return {
        name[: -len(".npy")]: np.load(os.path.join(DATA_DIR, name)).mean(axis=0)
        for name in sorted(os.listdir(DATA_DIR))
        if name.endswith(".npy")
    }


def delete(speaker_id: str) -> bool:
    path = _path(speaker_id)
    with _lock:
        if os.path.exists(path):
            os.remove(path)
            return True
        return False
