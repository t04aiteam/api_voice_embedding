"""/embed, /compare, /speakers... — see README.md for the full endpoint reference."""
import asyncio
import os
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from schemas.voice import (
    CompareResponse,
    DeleteResponse,
    EmbedResponse,
    EnrollResponse,
    IdentifyMatch,
    IdentifyResponse,
    SpeakerInfo,
    VerifyResponse,
)
from services import audio as audio_service
from services import calibration as calibration_service
from services import model as model_service
from services import store as store_service

router = APIRouter()

DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.96"))
DEFAULT_TOP_K = 5


def _parse_threshold(threshold: Optional[str]) -> float:
    if threshold is None or threshold.strip() == "":
        return DEFAULT_THRESHOLD
    try:
        value = float(threshold)
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail="threshold must be a number between 0 and 1."
        ) from exc
    if not (0.0 <= value <= 1.0):
        raise HTTPException(status_code=422, detail="threshold must be between 0 and 1.")
    return value


def _parse_top_k(top_k: Optional[str]) -> int:
    """Read the top_k form field; blank means the default.

    Args:
        top_k: Raw form value.

    Returns:
        A whole number >= 1.

    Raises:
        HTTPException: 422 when the value is not a whole number >= 1.
    """
    if top_k is None or top_k.strip() == "":
        return DEFAULT_TOP_K
    try:
        value = int(top_k)
    except ValueError:
        value = 0
    if value < 1:
        raise HTTPException(status_code=422, detail="top_k must be a whole number >= 1.")
    return value


def _parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "y")


def _require_valid_speaker_id(speaker_id: str) -> None:
    if not store_service.is_valid_speaker_id(speaker_id):
        raise HTTPException(
            status_code=422,
            detail=(
                "speaker_id must be 1-64 characters: letters, digits, "
                "'.', '_' or '-'."
            ),
        )


def _require_enrolled(speaker_id: str) -> None:
    if not store_service.exists(speaker_id):
        raise HTTPException(status_code=404, detail=f"Speaker '{speaker_id}' is not enrolled.")


async def _embed_calibrated(waveform, sample_rate):
    raw = await asyncio.to_thread(model_service.extract_embedding, waveform, sample_rate)
    return calibration_service.calibrate_embedding(raw)


@router.post("/compare", response_model=CompareResponse)
async def compare(
    file_a: UploadFile = File(..., description="First recording (>= 1s)."),
    file_b: UploadFile = File(..., description="Second recording (>= 1s)."),
    threshold: Optional[str] = Form(None, description="Decision threshold; default 0.96."),
):
    """Compare two recordings 1:1 and report whether they are the same speaker."""
    thr = _parse_threshold(threshold)
    wave_a, sr_a, _ = await audio_service.read_and_decode(file_a)
    wave_b, sr_b, _ = await audio_service.read_and_decode(file_b)
    emb_a = await _embed_calibrated(wave_a, sr_a)
    emb_b = await _embed_calibrated(wave_b, sr_b)
    similarity = calibration_service.cosine_similarity(emb_a, emb_b)
    return CompareResponse(
        similarity=round(similarity, 6),
        threshold=thr,
        same_speaker=similarity >= thr,
    )


@router.post("/speakers/enroll", response_model=EnrollResponse)
async def enroll_speaker(
    speaker_id: str = Form(..., description="1-64 chars: letters, digits, '.', '_', '-'."),
    file: UploadFile = File(..., description="Voice sample (>= 1s)."),
):
    """Enroll a voice sample for speaker_id. Call repeatedly with different
    recordings to build a more accurate profile."""
    _require_valid_speaker_id(speaker_id)
    waveform, sample_rate, _ = await audio_service.read_and_decode(file)
    embedding = await _embed_calibrated(waveform, sample_rate)
    samples = store_service.enroll(speaker_id, embedding)
    return EnrollResponse(speaker_id=speaker_id, samples=samples)


@router.post("/speakers/{speaker_id}/verify", response_model=VerifyResponse)
async def verify_speaker(
    speaker_id: str,
    file: UploadFile = File(..., description="Recording to verify (>= 1s)."),
    threshold: Optional[str] = Form(None, description="Decision threshold; default 0.96."),
):
    """Check whether a new recording matches the enrolled speaker_id."""
    _require_valid_speaker_id(speaker_id)
    _require_enrolled(speaker_id)
    thr = _parse_threshold(threshold)
    stored = store_service.get_embeddings(speaker_id)
    waveform, sample_rate, _ = await audio_service.read_and_decode(file)
    probe = await _embed_calibrated(waveform, sample_rate)
    centroid = stored.mean(axis=0)
    similarity = calibration_service.cosine_similarity(probe, centroid)
    return VerifyResponse(
        speaker_id=speaker_id,
        similarity=round(similarity, 6),
        threshold=thr,
        verified=similarity >= thr,
        samples=int(stored.shape[0]),
    )


@router.post("/speakers/identify", response_model=IdentifyResponse)
async def identify_speaker(
    file: UploadFile = File(..., description="Recording to identify (>= 1s)."),
    top_k: Optional[str] = Form(None, description="Most matches returned; default 5."),
    threshold: Optional[str] = Form(None, description="Decision threshold; default 0.96."),
):
    """Match a recording against every enrolled speaker (1:N), best first.

    Candidates below the threshold stay in the list with verified=false, so a
    UC 68/69 screen can show them. Nobody enrolled gives an empty list; the file
    is still decoded, so a bad upload is a 400 either way.

    Args:
        file: Recording to identify.
        top_k: Most matches returned, blank for 5.
        threshold: Decision threshold, blank for DEFAULT_THRESHOLD.

    Returns:
        The query duration, the threshold used and the ranked matches.
    """
    thr = _parse_threshold(threshold)
    k = _parse_top_k(top_k)
    waveform, sample_rate, duration = await audio_service.read_and_decode(file)
    gallery = store_service.centroids()
    scored = []
    if gallery:
        probe = await _embed_calibrated(waveform, sample_rate)
        scored = sorted(
            (
                (calibration_service.cosine_similarity(probe, centroid), speaker_id)
                for speaker_id, centroid in gallery.items()
            ),
            key=lambda pair: (-pair[0], pair[1]),
        )[:k]
    return IdentifyResponse(
        query_duration_seconds=round(duration, 3),
        threshold=thr,
        matches=[
            IdentifyMatch(
                speaker_id=speaker_id,
                similarity=round(similarity, 6),
                verified=similarity >= thr,
            )
            for similarity, speaker_id in scored
        ],
    )


@router.get("/speakers", response_model=list[SpeakerInfo])
async def list_speakers():
    """List every enrolled speaker and how many samples each has."""
    return store_service.list_speakers()


@router.delete("/speakers/{speaker_id}", response_model=DeleteResponse)
async def delete_speaker(speaker_id: str):
    """Delete a speaker and all of their stored embeddings."""
    _require_valid_speaker_id(speaker_id)
    _require_enrolled(speaker_id)
    store_service.delete(speaker_id)
    return DeleteResponse(speaker_id=speaker_id, deleted=True)


@router.post("/embed", response_model=EmbedResponse)
async def embed(
    file: UploadFile = File(..., description="Recording to embed (>= 1s)."),
    raw: Optional[str] = Form(
        None, description="false (default): calibrated embedding; true: raw model output."
    ),
):
    """Return the 2048-d speaker embedding for a recording, for integration
    with other systems."""
    waveform, sample_rate, duration = await audio_service.read_and_decode(file)
    raw_embedding = await asyncio.to_thread(
        model_service.extract_embedding, waveform, sample_rate
    )
    embedding = (
        raw_embedding
        if _parse_bool(raw, default=False)
        else calibration_service.calibrate_embedding(raw_embedding)
    )
    return EmbedResponse(
        embedding=[round(float(x), 6) for x in embedding.tolist()],
        dim=int(embedding.shape[0]),
        duration_seconds=round(duration, 3),
    )
