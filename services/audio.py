"""Upload validation and audio decoding.

wav/flac/ogg/mp3 decode in memory via libsndfile. Everything libsndfile cannot
open (mp4/mov/mkv/webm video, m4a/aac, wma, ...) is decoded by the host's ffmpeg,
so video evidence is accepted as uploaded.
"""

import asyncio
import io
import os
import subprocess
import tempfile

import librosa
import numpy as np
import soundfile as sf
from fastapi import HTTPException, UploadFile

from services.model import InferenceOutOfMemory

MIN_AUDIO_SECONDS = float(os.getenv("MIN_AUDIO_SECONDS", "1"))
# Optional cap, off by default: the user manual (HDSD) lists 25 MB as the only
# size limit. Activation memory grows linearly with duration (measured 7.6 MiB
# per second of audio, see README "Memory"), so a deployment can opt in.
MAX_AUDIO_SECONDS = float(os.getenv("MAX_AUDIO_SECONDS", "0"))
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "25"))
# The ECAPA feature extractor works at 24 kHz; decoding straight to that rate
# saves it a second resample.
FFMPEG_SAMPLE_RATE = 24000


def _ffmpeg_decode(raw: bytes, limit: float | None) -> tuple[np.ndarray, int]:
    """Decode any file ffmpeg can read to a mono float32 waveform.

    The upload goes through a temp file, not a pipe: an mp4 whose index (moov
    atom) sits at the end, the default for phone and camera recordings, cannot
    be read from a stream that does not seek.

    Args:
        raw: Uploaded file bytes.
        limit: Decode at most this many seconds, or everything if None.

    Returns:
        The waveform and its sample rate (FFMPEG_SAMPLE_RATE).

    Raises:
        ValueError: ffmpeg failed, e.g. the file has no audio track.
    """
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(raw)
        tmp.flush()
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", tmp.name, "-vn", "-ac", "1"]
        cmd += ["-ar", str(FFMPEG_SAMPLE_RATE)]
        if limit:
            cmd += ["-t", f"{limit:g}"]
        proc = subprocess.run(
            [*cmd, "-f", "f32le", "pipe:1"], capture_output=True, timeout=120
        )
        if proc.returncode != 0:
            err = proc.stderr.decode(errors="replace").replace(tmp.name, "upload")
            # The reason ("does not contain any stream") comes before the
            # generic last line ("Error opening output files").
            why = " | ".join(x.strip() for x in err.splitlines() if x.strip())
            raise ValueError(f"ffmpeg: {why[:300] or 'exit ' + str(proc.returncode)}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy(), FFMPEG_SAMPLE_RATE


def _decode(raw: bytes) -> tuple[np.ndarray, int, float]:
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file / cannot decode audio.")
    # Stop decoding just past the cap: a 25 MB mp3 can hold 30 minutes, and
    # decoding all of it to float32 only to reject it costs seconds of CPU.
    limit = MAX_AUDIO_SECONDS + 1 if MAX_AUDIO_SECONDS > 0 else None
    try:
        try:
            waveform, sample_rate = librosa.load(
                io.BytesIO(raw), sr=None, mono=True, duration=limit
            )
        except sf.SoundFileRuntimeError:
            # "Format not recognised": librosa's own fallback (audioread) only
            # takes a path, never the BytesIO an upload arrives as.
            waveform, sample_rate = _ffmpeg_decode(raw, limit)
    except MemoryError as exc:
        # Decoding allocates too (a 378 s mp3 decodes to 64 MiB); under memory
        # pressure numpy raises MemoryError and the generic branch below would
        # blame the file with a 400 (oom_forced round 2). It is the server's
        # memory, so it is the OOM 500.
        raise InferenceOutOfMemory("cpu", exc) from exc
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
