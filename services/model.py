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
4. Before any CPU pass (fallback or DEVICE=cpu) the activation memory is
   estimated from the processor output and checked against the smaller of
   MemAvailable and the process's cgroup v2 cap (docker mem_limit, systemd
   MemoryMax), so a recording that cannot fit fails as one request instead
   of feeding the kernel's OOM killer on a shared box.
5. If the fallback is off, or the CPU itself runs out of memory (in the
   forward pass or in feature extraction), InferenceOutOfMemory is raised and
   main.py turns it into HTTP 500 whose detail says "out of memory" (the HDSD
   manual documents 500 only). services/audio.py raises the same for a
   MemoryError while decoding.
"""

import logging
import os
import threading

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

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
# pass (2.6 s per 6-minute file on CPU). Bounded semaphore + early 500 when the
# queue is deep if latency under bursts matters.
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


# Activation memory of one CPU forward pass per byte of processor output.
# Measured on 222 (fp32, 80 threads, probe_pass2.log): 157, 159 and 160 for
# 60 s, 378 s and 756 s of audio, so it is linear in the recording length.
# ponytail: one constant from one box and one model revision; re-measure
# with probe_pass2.py if MODEL_ID changes.
_CPU_ACTIVATION_BYTES_PER_INPUT_BYTE = 160


_MEMINFO_PATH = "/proc/meminfo"
_CGROUP_ROOT = "/sys/fs/cgroup"
_PROC_SELF_CGROUP = "/proc/self/cgroup"


def _meminfo_available() -> int | None:
    """Read MemAvailable from /proc/meminfo.

    Returns:
        Bytes the kernel expects to be usable without swapping, or None when
        /proc/meminfo cannot be read (non-Linux).
    """
    try:
        with open(_MEMINFO_PATH) as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _cgroup_memory_available() -> int | None:
    """Read how much is left under this process's cgroup v2 memory cap.

    A container ``mem_limit`` or a systemd ``MemoryMax=`` caps the process
    below the host's MemAvailable, and the kernel enforces it with SIGKILL,
    so the guard has to see it. Measured on 222 (oom_cgroup_before.log): with
    a 3 GB cap and 80 GB MemAvailable the 378 s song got the process killed.

    Returns:
        ``memory.max - memory.current`` in bytes, or None when the cgroup has
        no cap ("max") or the files cannot be read (cgroup v1, non-Linux).
    """
    # ponytail: reads the process's own cgroup only; a cap on an ancestor is
    # not seen. Walk the parents if a deployment nests cgroups.
    try:
        with open(_PROC_SELF_CGROUP) as f:
            path = f.read().strip().split(":", 2)[2]
        base = _CGROUP_ROOT + path.rstrip("/")
        with open(base + "/memory.max") as f:
            limit = f.read().strip()
        if limit == "max":
            return None
        with open(base + "/memory.current") as f:
            current = int(f.read().strip())
        return max(int(limit) - current, 0)
    except (OSError, ValueError, IndexError):
        return None


def _host_memory_available() -> int | None:
    """Bytes a CPU pass may allocate: the smaller of host MemAvailable and the cgroup cap.

    Returns:
        The tighter of the two limits, or None when neither can be read.
    """
    values = [
        v for v in (_meminfo_available(), _cgroup_memory_available()) if v is not None
    ]
    return min(values) if values else None


def _check_host_memory(inputs: dict) -> None:
    """Refuse a CPU forward pass whose activations would not fit in host RAM.

    The pass allocates one activation tensor at a time and the kernel accepts
    each of them, so without this check an over-long recording drives the
    host into swap and the OOM killer ends this process or a neighbour's,
    instead of one request failing.

    Args:
        inputs: Processor output.

    Raises:
        MemoryError: If the estimated activation memory exceeds MemAvailable.
    """
    available = _host_memory_available()
    if available is None:
        return
    input_bytes = sum(int(getattr(v, "nbytes", 0)) for v in inputs.values())
    needed = input_bytes * _CPU_ACTIVATION_BYTES_PER_INPUT_BYTE
    if needed > available:
        raise MemoryError(
            f"CPU forward pass needs about {needed >> 20} MiB of activations, "
            f"{available >> 20} MiB available"
        )


def _forward(model, inputs: dict, device: str) -> np.ndarray:
    """Run one forward pass on device and return the embedding as float32.

    Args:
        model: Model whose parameters live on device.
        inputs: Processor output (tensors are moved to device here).
        device: Torch device string to run on.

    Returns:
        The raw embedding as a float32 numpy array.
    """
    if device == "cpu":
        _check_host_memory(inputs)
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
    # Feature extraction (mel spectrogram, always on CPU) allocates too; under
    # memory pressure torch.stft fails with the same allocator error as the
    # forward pass (oom_forced round 1: it surfaced as a generic 500).
    try:
        inputs = processor(waveform, sampling_rate=sample_rate)
    except (MemoryError, RuntimeError) as err:
        if _is_cpu_oom(err):
            raise InferenceOutOfMemory("cpu", err) from err
        raise
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
