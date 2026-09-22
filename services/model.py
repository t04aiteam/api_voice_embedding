"""Model loading and embedding extraction for the Qwen3 ECAPA-TDNN voice encoder.

The model is loaded exactly once (see load_model, invoked from main.py's lifespan
startup hook) and reused for every request.

Out-of-memory handling (the GPU is shared with other jobs, so free memory can
drop to almost nothing at any time):

1. Inference is serialized through a lock, so peak memory is one request's
   activations, never N concurrent requests' worth.
2. On CUDA OOM the cached allocator is flushed and the forward pass retried once.
3. If it still fails and CPU_FALLBACK_ON_OOM is on (default), the request runs
   on a lazily created CPU copy of the model (~12M parameters, ~50 MB RAM).
4. If the fallback is off, or the CPU itself runs out of memory,
   InferenceOutOfMemory is raised and main.py turns it into HTTP 503.
"""

import copy
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

_device: str | None = None
_processor = None
_model = None
_cpu_model = None
_load_lock = threading.Lock()
_cpu_model_lock = threading.Lock()
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
        hf_token = _hf_token()
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
    """Return a CPU copy of the loaded model, creating it on first use.

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
            logger.warning("Creating CPU fallback copy of the model after CUDA OOM.")
            _cpu_model = copy.deepcopy(model).to("cpu").eval()
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


def _is_cpu_oom(exc: BaseException) -> bool:
    """Tell whether an exception is a host-memory exhaustion.

    Python raises MemoryError; torch's CPU allocator raises a RuntimeError whose
    message says "DefaultCPUAllocator: not enough memory".

    Args:
        exc: The exception to classify.

    Returns:
        True if exc means the host ran out of memory.
    """
    if isinstance(exc, MemoryError):
        return True
    return isinstance(exc, RuntimeError) and "not enough memory" in str(exc)


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
        try:
            return _forward(model, inputs, device)
        except CudaOutOfMemoryError:
            logger.warning("CUDA OOM on %s, flushing cache and retrying once.", device)
            _release_cuda_cache()
        except (MemoryError, RuntimeError) as err:
            if _is_cpu_oom(err):
                raise InferenceOutOfMemory(device, err) from err
            raise
        try:
            return _forward(model, inputs, device)
        except CudaOutOfMemoryError as second:
            _release_cuda_cache()
            if not CPU_FALLBACK_ON_OOM:
                raise InferenceOutOfMemory(device, second) from second
            logger.warning("CUDA OOM again, running this request on CPU.")
        return _forward_cpu_fallback(inputs)
