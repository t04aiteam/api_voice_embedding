"""Model loading and embedding extraction for the Qwen3 ECAPA-TDNN voice encoder.

The model is loaded exactly once (see load_model, invoked from main.py's lifespan
startup hook) and reused for every request.
"""
import os
import threading

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

MODEL_ID = os.getenv("MODEL_ID", "marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B")
# Pinned to a specific commit so the service behaves identically across restarts
# and can run fully offline once the revision is cached locally.
MODEL_REVISION = os.getenv("MODEL_REVISION", "7577f61c42737fc8064bba773e2a18602df92803")
EMBEDDING_DIM = 2048

_device: str | None = None
_processor = None
_model = None
_load_lock = threading.Lock()


def resolve_device() -> str:
    """Resolve DEVICE=auto|cpu|cuda to an actual torch device string, cached."""
    global _device
    if _device is not None:
        return _device
    requested = os.getenv("DEVICE", "auto").lower()
    if requested == "auto":
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        _device = requested
    return _device


def load_model():
    """Load the processor + model once. Safe to call repeatedly (idempotent)."""
    global _processor, _model
    if _model is not None:
        return _model, _processor
    with _load_lock:
        if _model is not None:
            return _model, _processor
        device = resolve_device()
        hf_token = os.getenv("HF_TOKEN") or None
        _processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            token=hf_token,
        )
        model = AutoModel.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            token=hf_token,
        )
        model.eval()
        model.to(device)
        _model = model
        return _model, _processor


def get_model():
    return load_model()[0]


def get_processor():
    return load_model()[1]


def extract_embedding(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """Run inference and return the raw (uncalibrated) 2048-d embedding as float32."""
    model, processor = load_model()
    device = resolve_device()
    inputs = processor(waveform, sampling_rate=sample_rate)
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.no_grad():
        output = model(**inputs).last_hidden_state
    return output.squeeze().detach().cpu().numpy().astype(np.float32)
