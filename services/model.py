"""Model loading and embedding extraction for the Qwen3 ECAPA-TDNN voice encoder.

The model is loaded exactly once (see load_model, invoked from main.py's lifespan
startup hook) and reused for every request.

Out-of-memory handling (the GPU is shared with other jobs, so free memory can
drop to almost nothing at any time):

1. Inference is serialized through a lock, so peak memory is one request's
   activations, never N concurrent requests' worth.
2. On CUDA OOM the cached allocator is flushed and the forward pass retried once.
3. If it still fails and CPU_FALLBACK_ON_OOM is on (default), the request runs
   on a CPU copy of the model (~12M parameters, ~50 MB RAM) loaded once from the
   local model cache. The copy never touches the GPU: deep-copying a CUDA model
   would allocate the weights on the GPU that just ran out of memory.
4. If the fallback is off, or the CPU itself runs out of memory,
   InferenceOutOfMemory is raised and main.py turns it into HTTP 503.
"""

import logging
import os
import threading

# Must be set before the CUDA allocator is initialised (first CUDA call), so
# a value from the environment or .env wins and this is only the default.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModel, AutoProcessor  # noqa: E402

logger = logging.getLogger(__name__)

MODEL_ID = os.getenv("MODEL_ID", "marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B")
# Pinned to a specific commit so the service behaves identically across restarts
# and can run fully offline once the revision is cached locally.
MODEL_REVISION = os.getenv("MODEL_REVISION", "7577f61c42737fc8064bba773e2a18602df92803")
EMBEDDING_DIM = 2048

CPU_FALLBACK_ON_OOM = os.getenv("CPU_FALLBACK_ON_OOM", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "y",
)

# torch.OutOfMemoryError exists since 2.5; torch.cuda.OutOfMemoryError is the
# older name and is kept as an alias in newer releases.
CudaOutOfMemoryError = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)

# torch's CPU allocator raises a plain RuntimeError and the wording changed
# across releases: "DefaultCPUAllocator: not enough memory" (older) and
# "DefaultCPUAllocator: can't allocate memory" (2.13 in emb-env, checked with
# strings on libc10.so).
_CPU_OOM_MARKERS = ("not enough memory", "can't allocate memory")
# Besides torch.OutOfMemoryError, a starved GPU surfaces as a plain RuntimeError
# when a library handle cannot be created (cuBLAS/cuDNN *_ALLOC_FAILED) or a raw
# driver call fails ("CUDA error: out of memory").
_CUDA_OOM_MARKERS = (
    "CUDA out of memory",
    "CUDA error: out of memory",
    "CUBLAS_STATUS_ALLOC_FAILED",
    "CUDNN_STATUS_ALLOC_FAILED",
)

_device: str | None = None
_processor = None
_model = None
_cpu_model = None
_load_lock = threading.Lock()
_cpu_model_lock = threading.Lock()
# ponytail: one global lock, so every request queues behind the running forward
# pass (2.6 s per 6-minute file on CPU). Bounded semaphore + 503 when the queue
# is deep if latency under bursts matters.
_infer_lock = threading.Lock()


class InferenceOutOfMemory(RuntimeError):
    """Raised when a forward pass cannot get the memory it needs on any device."""

    def __init__(self, device: str, cause: BaseException):
        """Record which device ran out of memory and the underlying error.

        Args:
            device: Device string the failing forward pass ran on.
            cause: The exception torch or the interpreter raised.
        """
        super().__init__(f"Out of memory on {device}: {type(cause).__name__}")
        self.device = device
        """Device that ran out of memory, "cuda" or "cpu"."""
        self.cause = cause
        """Original exception, kept for logging."""


def resolve_device() -> str:
    """Resolve DEVICE=auto|cpu|cuda to an actual torch device string, cached.

    Returns:
        The torch device string the model runs on.
    """
    global _device
    if _device is not None:
        return _device
    requested = os.getenv("DEVICE", "auto").lower()
    if requested == "auto":
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        _device = requested
    return _device


def _hf_token() -> str | None:
    """Read HF_TOKEN, ignoring blanks and a stray inline comment.

    python-dotenv turns ``HF_TOKEN=   # comment`` into the literal comment text,
    which would be sent to the Hub as a bearer token.

    Returns:
        The token, or None when unset.
    """
    token = (os.getenv("HF_TOKEN") or "").strip()
    if not token or token.startswith("#"):
        return None
    return token


def _load_pretrained_model():
    """Load the pinned model revision onto the CPU, from the local cache once downloaded.

    Returns:
        The model in eval mode, on CPU.
    """
    model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        token=_hf_token(),
    )
    return model.eval()


def load_model():
    """Load the processor + model once. Safe to call repeatedly (idempotent).

    Returns:
        The (model, processor) pair.
    """
    global _processor, _model
    if _model is not None:
        return _model, _processor
    with _load_lock:
        if _model is not None:
            return _model, _processor
        device = resolve_device()
        _processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            token=_hf_token(),
        )
        _model = _load_pretrained_model().to(device)
        return _model, _processor


def get_model():
    """Return the loaded model, loading it first if needed.

    Returns:
        The model.
    """
    return load_model()[0]


def get_processor():
    """Return the loaded processor, loading it first if needed.

    Returns:
        The processor.
    """
    return load_model()[1]


def _get_cpu_model():
    """Return a CPU copy of the loaded model, loading it on first use.

    The copy is re-read from the local model cache (same pinned revision, so the
    weights are identical) instead of deep-copied: ``copy.deepcopy`` of a CUDA
    module clones every parameter on the GPU first, and that is the device that
    just ran out of memory.

    Returns:
        The model on CPU. If the main model already lives on CPU it is returned as is.
    """
    global _cpu_model
    model = get_model()
    if resolve_device() == "cpu":
        return model
    if _cpu_model is not None:
        return _cpu_model
    with _cpu_model_lock:
        if _cpu_model is None:
            logger.warning("Loading CPU fallback copy of the model after CUDA OOM.")
            _cpu_model = _load_pretrained_model().to("cpu")
        return _cpu_model


def _forward(model, inputs: dict, device: str) -> np.ndarray:
    """Run one forward pass on device and return the embedding as float32.

    Args:
        model: Model whose parameters live on device.
        inputs: Processor output (tensors are moved to device here).
        device: Torch device string to run on.

    Returns:
        The raw embedding as a float32 numpy array.
    """
    moved = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.no_grad():
        output = model(**moved).last_hidden_state
    return output.squeeze().detach().cpu().numpy().astype(np.float32)


def _release_cuda_cache() -> None:
    """Return cached, unused blocks to the driver so a retry can succeed."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_cuda_oom(exc: BaseException) -> bool:
    """Tell whether an exception means the GPU ran out of memory.

    Args:
        exc: The exception to classify.

    Returns:
        True for torch.OutOfMemoryError and for a RuntimeError carrying a CUDA
        allocation failure message.
    """
    if isinstance(exc, CudaOutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and any(
        m in str(exc) for m in _CUDA_OOM_MARKERS
    )


def _is_cpu_oom(exc: BaseException) -> bool:
    """Tell whether an exception is a host-memory exhaustion.

    Args:
        exc: The exception to classify.

    Returns:
        True for MemoryError and for torch's CPU allocator RuntimeError.
    """
    if isinstance(exc, MemoryError):
        return True
    return isinstance(exc, RuntimeError) and any(
        m in str(exc) for m in _CPU_OOM_MARKERS
    )


def _forward_cpu_fallback(inputs: dict) -> np.ndarray:
    """Run inputs through the CPU copy of the model.

    Args:
        inputs: Processor output.

    Returns:
        The raw embedding as a float32 numpy array.

    Raises:
        InferenceOutOfMemory: If the host runs out of memory too.
    """
    try:
        return _forward(_get_cpu_model(), inputs, "cpu")
    except (MemoryError, RuntimeError) as err:
        if _is_cpu_oom(err):
            raise InferenceOutOfMemory("cpu", err) from err
        raise


def extract_embedding(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """Run inference and return the raw (uncalibrated) 2048-d embedding as float32.

    Args:
        waveform: Mono waveform.
        sample_rate: Sample rate of waveform in Hz.

    Returns:
        The raw embedding as a float32 numpy array.

    Raises:
        InferenceOutOfMemory: If the forward pass runs out of memory on every
            device it was allowed to try.
    """
    model, processor = load_model()
    device = resolve_device()
    inputs = processor(waveform, sampling_rate=sample_rate)
    with _infer_lock:
        cause = None
        for attempt in (1, 2):
            try:
                return _forward(model, inputs, device)
            except (MemoryError, RuntimeError) as err:
                if _is_cuda_oom(err):
                    # Drop the traceback: it pins the failed pass's frames and
                    # their CUDA tensors, which must be gone before the flush.
                    cause = err.with_traceback(None)
                elif _is_cpu_oom(err):
                    raise InferenceOutOfMemory(device, err) from err
                else:
                    raise
            # Outside the except block, so the flush sees the freed activations.
            _release_cuda_cache()
            if attempt == 1:
                logger.warning(
                    "CUDA OOM on %s, flushed the cache, retrying once.", device
                )
        if not CPU_FALLBACK_ON_OOM:
            raise InferenceOutOfMemory(device, cause) from cause
        logger.warning("CUDA OOM again, running this request on CPU.")
        return _forward_cpu_fallback(inputs)
