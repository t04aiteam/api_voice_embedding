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
# Activation memory grows linearly with duration (measured 7.6 MiB per second
# of audio, so a 6-minute song needs about 2.9 GB, see README "Memory"), so cap
# it. 0 disables the cap.
MAX_AUDIO_SECONDS = float(os.getenv("MAX_AUDIO_SECONDS", "600"))
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "25"))


def _decode(raw: bytes) -> tuple[np.ndarray, int, float]:
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file / cannot decode audio.")
    # Stop decoding just past the cap: a 25 MB mp3 can hold 30 minutes, and
    # decoding all of it to float32 only to reject it costs seconds of CPU.
    limit = MAX_AUDIO_SECONDS + 1 if MAX_AUDIO_SECONDS > 0 else None
    try:
        waveform, sample_rate = librosa.load(
            io.BytesIO(raw), sr=None, mono=True, duration=limit
        )
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
    if MAX_AUDIO_SECONDS > 0 and duration > MAX_AUDIO_SECONDS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Audio too long: more than {MAX_AUDIO_SECONDS:g}s "
                f"(maximum {MAX_AUDIO_SECONDS:g}s allowed, see MAX_AUDIO_SECONDS)."
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
