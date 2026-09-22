"""Upload validation and audio decoding.

wav/flac/ogg decode natively via libsndfile; mp3/m4a fall back to librosa's
audioread backend, which requires ffmpeg to be installed on the host.
"""
import asyncio
import io
import os

import librosa
import numpy as np
from fastapi import HTTPException, UploadFile

MIN_AUDIO_SECONDS = float(os.getenv("MIN_AUDIO_SECONDS", "1"))
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "25"))


def _decode(raw: bytes) -> tuple[np.ndarray, int, float]:
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file / cannot decode audio.")
    try:
        waveform, sample_rate = librosa.load(io.BytesIO(raw), sr=None, mono=True)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot decode audio file: {exc}",
        ) from exc
    if waveform is None or waveform.size == 0:
        raise HTTPException(status_code=400, detail="Empty file / cannot decode audio.")
    duration = float(len(waveform) / sample_rate)
    if duration < MIN_AUDIO_SECONDS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Audio too short: {duration:.2f}s "
                f"(minimum {MIN_AUDIO_SECONDS}s required)."
            ),
        )
    return waveform, sample_rate, duration


async def read_and_decode(upload: UploadFile) -> tuple[np.ndarray, int, float]:
    """Read an UploadFile, enforce the size limit, and decode it to a mono waveform."""
    raw = await upload.read()
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    if len(raw) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_MB} MB limit.",
        )
    return await asyncio.to_thread(_decode, raw)
