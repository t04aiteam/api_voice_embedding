"""Gate tests for out-of-memory handling in services/model.py and main.py.

No real model is loaded: the module globals are swapped for fakes, and the CUDA
cache flush is stubbed so the tests never touch a GPU.

Run:
    emb-env/bin/python -m unittest discover -s tests -v
"""

import io
import os
import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import librosa
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
from services import audio as audio_service  # noqa: E402
from services import model as model_service  # noqa: E402

OOM = model_service.CudaOutOfMemoryError


class FakeProcessor:
    """Stand-in processor whose outputs carry no tensors, so nothing is moved to a device."""

    def __call__(self, waveform, sampling_rate=None):
        """Return a placeholder input dict.

        Args:
            waveform: Ignored.
            sampling_rate: Ignored.

        Returns:
            A dict with one non-tensor value.
        """
        return {"input_values": "fake"}


class FakeModel:
    """Stand-in model that fails a configurable number of times before succeeding.

    Records the device it was moved to and every device a forward pass ran on.
    """

    def __init__(self, failures=0, exc=None, delay=0.0):
        """Configure the failure schedule.

        Args:
            failures: How many leading forward calls raise exc.
            exc: Exception instance to raise, defaults to a CUDA OOM.
            delay: Seconds each forward pass sleeps, for the concurrency test.
        """
        self.failures = failures
        self.exc = (
            exc
            if exc is not None
            else OOM("CUDA out of memory. Tried to allocate 374.00 MiB")
        )
        self.delay = delay
        self.device = "cuda"
        self.calls = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def to(self, device):
        """Record the device move and return self.

        Args:
            device: Target device string.

        Returns:
            This model.
        """
        self.device = device
        return self

    def eval(self):
        """Return self, mirroring nn.Module.eval.

        Returns:
            This model.
        """
        return self

    def __deepcopy__(self, memo):
        """Fail: deep-copying the CUDA model would allocate its weights on the GPU.

        Args:
            memo: deepcopy memo dict, unused.

        Raises:
            AssertionError: Always, the fallback must load from the cache instead.
        """
        raise AssertionError(
            "copy.deepcopy(model) allocates on the GPU that just ran out of memory"
        )

    def __call__(self, **kwargs):
        """Run one fake forward pass.

        Args:
            **kwargs: Ignored model inputs.

        Returns:
            An object with a (1, 2048) last_hidden_state.

        Raises:
            Exception: The configured exception while failures remain.
        """
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            self.calls.append(self.device)
            if len(self.calls) <= self.failures:
                raise self.exc
            return SimpleNamespace(
                last_hidden_state=torch.ones(1, model_service.EMBEDDING_DIM)
            )
        finally:
            with self.lock:
                self.active -= 1


def wav_bytes(seconds: float, sr: int = 16000) -> bytes:
    """Encode a sine tone as an in-memory WAV.

    Args:
        seconds: Duration of the tone.
        sr: Sample rate.

    Returns:
        The WAV file contents.
    """
    t = np.arange(int(seconds * sr)) / sr
    buf = io.BytesIO()
    sf.write(
        buf, (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr, format="WAV"
    )
    return buf.getvalue()


class ModelServiceState:
    """Install fakes into services.model and restore the originals afterwards."""

    def __init__(self, model, device="cuda", fallback=True):
        """Capture the fakes to install.

        Args:
            model: FakeModel to serve as the loaded model.
            device: Value resolve_device should report.
            fallback: Value for CPU_FALLBACK_ON_OOM.
        """
        self.model = model
        self.device = device
        self.fallback = fallback
        self.release_calls = 0
        self.release_inside_except = 0
        self.cache_loads = 0
        self._patches = []

    def _count_release(self):
        self.release_calls += 1
        # Inside an except block the failed pass's traceback (and its CUDA
        # tensors) is still alive, so a flush there frees nothing useful.
        if sys.exc_info()[0] is not None:
            self.release_inside_except += 1

    def _load_from_cache(self):
        self.cache_loads += 1
        return FakeModel(failures=0, exc=self.model.exc, delay=self.model.delay)

    def __enter__(self):
        """Apply the patches.

        Returns:
            This state object.
        """
        self._patches = [
            mock.patch.object(model_service, "_model", self.model),
            mock.patch.object(model_service, "_processor", FakeProcessor()),
            mock.patch.object(model_service, "_cpu_model", None),
            mock.patch.object(model_service, "_device", self.device),
            mock.patch.object(model_service, "CPU_FALLBACK_ON_OOM", self.fallback),
            mock.patch.object(
                model_service, "_release_cuda_cache", self._count_release
            ),
            mock.patch.object(
                model_service, "_load_pretrained_model", self._load_from_cache
            ),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        """Undo the patches.

        Args:
            *exc: Exception info, ignored.
        """
        for p in reversed(self._patches):
            p.stop()


WAVE = np.zeros(16000, dtype=np.float32)


class ExtractEmbeddingOomTests(unittest.TestCase):
    """extract_embedding: retry, CPU fallback and error mapping."""

    def test_no_oom_runs_once_on_gpu(self):
        model = FakeModel()
        with ModelServiceState(model) as st:
            out = model_service.extract_embedding(WAVE, 16000)
        self.assertEqual(out.shape, (model_service.EMBEDDING_DIM,))
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(model.calls, ["cuda"])
        self.assertEqual(st.release_calls, 0)

    def test_single_oom_flushes_cache_and_retries_on_gpu(self):
        model = FakeModel(failures=1)
        with ModelServiceState(model) as st:
            out = model_service.extract_embedding(WAVE, 16000)
        self.assertEqual(out.shape, (model_service.EMBEDDING_DIM,))
        self.assertEqual(model.calls, ["cuda", "cuda"])
        self.assertEqual(st.release_calls, 1)
        self.assertIsNone(model_service._cpu_model)

    def test_double_oom_falls_back_to_cpu_copy(self):
        model = FakeModel(failures=2)
        with ModelServiceState(model) as st:
            out = model_service.extract_embedding(WAVE, 16000)
            cpu_copy = model_service._cpu_model
            # A second request reuses the same CPU copy.
            model.failures = 4
            model_service.extract_embedding(WAVE, 16000)
            self.assertIs(model_service._cpu_model, cpu_copy)
        self.assertEqual(out.shape, (model_service.EMBEDDING_DIM,))
        self.assertEqual(model.calls, ["cuda"] * 4)
        self.assertEqual(cpu_copy.device, "cpu")
        self.assertEqual(cpu_copy.calls, ["cpu", "cpu"])
        self.assertEqual(st.release_calls, 4)
        self.assertEqual(st.cache_loads, 1)

    def test_cpu_copy_is_loaded_from_cache_not_deepcopied(self):
        # FakeModel.__deepcopy__ raises: deep-copying a CUDA module clones the
        # weights on the GPU that just ran out of memory (9540308 did that).
        model = FakeModel(failures=2)
        with ModelServiceState(model) as st:
            out = model_service.extract_embedding(WAVE, 16000)
        self.assertEqual(out.shape, (model_service.EMBEDDING_DIM,))
        self.assertEqual(st.cache_loads, 1)

    def test_cache_flush_runs_outside_the_except_block(self):
        # Inside the except block the traceback keeps the failed pass's CUDA
        # tensors alive, so empty_cache() there cannot return them.
        model = FakeModel(failures=2)
        with ModelServiceState(model, fallback=False) as st:
            with self.assertRaises(model_service.InferenceOutOfMemory) as ctx:
                model_service.extract_embedding(WAVE, 16000)
        self.assertEqual(st.release_calls, 2)
        self.assertEqual(st.release_inside_except, 0)
        self.assertIsNone(ctx.exception.cause.__traceback__)

    def test_cublas_alloc_failure_counts_as_cuda_oom(self):
        # On a full GPU cuBLAS handle creation fails with a plain RuntimeError,
        # not torch.OutOfMemoryError; it must take the retry/fallback path.
        exc = RuntimeError(
            "CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling `cublasCreate(handle)`"
        )
        model = FakeModel(failures=2, exc=exc)
        with ModelServiceState(model) as st:
            out = model_service.extract_embedding(WAVE, 16000)
            cpu_calls = list(model_service._cpu_model.calls)
        self.assertEqual(out.shape, (model_service.EMBEDDING_DIM,))
        self.assertEqual(model.calls, ["cuda", "cuda"])
        self.assertEqual(st.release_calls, 2)
        self.assertEqual(cpu_calls, ["cpu"])

    def test_double_oom_without_fallback_raises_503_error(self):
        model = FakeModel(failures=2)
        with ModelServiceState(model, fallback=False) as st:
            with self.assertRaises(model_service.InferenceOutOfMemory) as ctx:
                model_service.extract_embedding(WAVE, 16000)
        self.assertEqual(ctx.exception.device, "cuda")
        self.assertIsInstance(ctx.exception.cause, OOM)
        self.assertEqual(model.calls, ["cuda", "cuda"])
        self.assertEqual(st.release_calls, 2)

    def test_cpu_memory_error_maps_to_cpu_oom(self):
        model = FakeModel(failures=1, exc=MemoryError())
        with ModelServiceState(model, device="cpu"):
            with self.assertRaises(model_service.InferenceOutOfMemory) as ctx:
                model_service.extract_embedding(WAVE, 16000)
        self.assertEqual(ctx.exception.device, "cpu")

    def test_torch_cpu_allocator_error_maps_to_cpu_oom(self):
        # Both wordings torch has used; emb-env's torch 2.13 emits the second
        # (checked with strings on libc10.so), which 9540308 did not match.
        messages = (
            "[enforce fail at alloc_cpu.cpp:117] DefaultCPUAllocator: not enough memory",
            "[enforce fail at alloc_cpu.cpp:121] DefaultCPUAllocator: can't allocate "
            "memory: you tried to allocate 2999999488 bytes.",
        )
        for message in messages:
            model = FakeModel(failures=1, exc=RuntimeError(message))
            with ModelServiceState(model, device="cpu"):
                with self.assertRaises(model_service.InferenceOutOfMemory) as ctx:
                    model_service.extract_embedding(WAVE, 16000)
            self.assertEqual(ctx.exception.device, "cpu", message)

    def test_torch_cpu_allocator_error_during_fallback_is_503_not_500(self):
        cpu_exc = RuntimeError(
            "DefaultCPUAllocator: can't allocate memory: you tried to allocate 1 bytes."
        )
        model = FakeModel()
        with ModelServiceState(model):
            with mock.patch.object(model_service, "_forward") as fwd:
                fwd.side_effect = [OOM("a"), OOM("b"), cpu_exc]
                with self.assertRaises(model_service.InferenceOutOfMemory) as ctx:
                    model_service.extract_embedding(WAVE, 16000)
        self.assertEqual(ctx.exception.device, "cpu")

    def test_fallback_cpu_oom_reports_cpu(self):
        model = FakeModel()
        with ModelServiceState(model):
            with mock.patch.object(model_service, "_forward") as fwd:
                fwd.side_effect = [OOM("a"), OOM("b"), MemoryError()]
                with self.assertRaises(model_service.InferenceOutOfMemory) as ctx:
                    model_service.extract_embedding(WAVE, 16000)
        self.assertEqual(ctx.exception.device, "cpu")
        self.assertEqual(
            [c.args[2] for c in fwd.call_args_list], ["cuda", "cuda", "cpu"]
        )

    def test_other_runtime_errors_propagate_unchanged(self):
        model = FakeModel(failures=1, exc=RuntimeError("shape mismatch"))
        with ModelServiceState(model):
            with self.assertRaises(RuntimeError) as ctx:
                model_service.extract_embedding(WAVE, 16000)
        self.assertNotIsInstance(ctx.exception, model_service.InferenceOutOfMemory)

    def test_inference_is_serialized(self):
        model = FakeModel(delay=0.05)
        with ModelServiceState(model):
            threads = [
                threading.Thread(
                    target=model_service.extract_embedding, args=(WAVE, 16000)
                )
                for _ in range(4)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        self.assertEqual(len(model.calls), 4)
        self.assertEqual(model.max_active, 1)

    def test_alloc_conf_default_is_expandable_segments(self):
        self.assertIn(
            "expandable_segments", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        )

    def test_hf_token_ignores_blank_and_dotenv_comment(self):
        cases = {
            "": None,
            "   ": None,
            "# only needed for private/gated models": None,
            " hf_abc ": "hf_abc",
        }
        for raw, expected in cases.items():
            with mock.patch.dict(os.environ, {"HF_TOKEN": raw}):
                self.assertEqual(model_service._hf_token(), expected, raw)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(model_service._hf_token())


class AudioLengthCapTests(unittest.TestCase):
    """MAX_AUDIO_SECONDS rejects recordings that would need too much activation memory."""

    def test_too_long_is_422(self):
        with mock.patch.object(audio_service, "MAX_AUDIO_SECONDS", 1.5):
            with self.assertRaises(HTTPException) as ctx:
                audio_service._decode(wav_bytes(2.0))
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertIn("Audio too long", ctx.exception.detail)

    def test_within_cap_passes(self):
        with mock.patch.object(audio_service, "MAX_AUDIO_SECONDS", 3.0):
            wave, sr, duration = audio_service._decode(wav_bytes(2.0))
        self.assertAlmostEqual(duration, 2.0, places=3)

    def test_zero_disables_cap(self):
        with mock.patch.object(audio_service, "MAX_AUDIO_SECONDS", 0):
            with mock.patch.object(
                audio_service.librosa, "load", wraps=librosa.load
            ) as spy:
                wave, sr, duration = audio_service._decode(wav_bytes(2.0))
        self.assertAlmostEqual(duration, 2.0, places=3)
        self.assertIsNone(spy.call_args.kwargs["duration"])

    def test_decoder_stops_one_second_past_the_cap(self):
        # A 25 MB mp3 holds ~30 minutes; without the bound the whole file is
        # decoded to float32 (seconds of CPU) just to be rejected.
        with mock.patch.object(audio_service, "MAX_AUDIO_SECONDS", 1.5):
            with mock.patch.object(
                audio_service.librosa, "load", wraps=librosa.load
            ) as spy:
                with self.assertRaises(HTTPException) as ctx:
                    audio_service._decode(wav_bytes(10.0))
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertEqual(spy.call_args.kwargs["duration"], 2.5)
        self.assertIn("more than 1.5s", ctx.exception.detail)


class HttpErrorMappingTests(unittest.TestCase):
    """main.py handlers: OOM is a 503 with Retry-After, other failures a generic 500."""

    def setUp(self):
        # No context manager: lifespan (real model load) must not run.
        self.client = TestClient(main.app, raise_server_exceptions=False)
        self.files = {
            "file_a": ("a.wav", wav_bytes(1.5), "audio/wav"),
            "file_b": ("b.wav", wav_bytes(1.5), "audio/wav"),
        }

    def test_compare_oom_is_503_with_retry_after(self):
        err = model_service.InferenceOutOfMemory(
            "cuda", OOM("CUDA out of memory. Process 6763 has")
        )
        with mock.patch.object(model_service, "extract_embedding", side_effect=err):
            r = self.client.post("/compare", files=self.files)
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.headers["Retry-After"], str(main.OOM_RETRY_AFTER_SECONDS))
        body = r.json()
        self.assertIn("out of memory on cuda", body["detail"])
        self.assertEqual(body["retry_after_seconds"], main.OOM_RETRY_AFTER_SECONDS)
        self.assertNotIn("Process", body["detail"])

    def test_embed_oom_is_503(self):
        err = model_service.InferenceOutOfMemory("cpu", MemoryError())
        with mock.patch.object(model_service, "extract_embedding", side_effect=err):
            r = self.client.post("/embed", files={"file": self.files["file_a"]})
        self.assertEqual(r.status_code, 503)
        self.assertIn("on cpu", r.json()["detail"])

    def test_unexpected_error_is_generic_500(self):
        with mock.patch.object(
            model_service,
            "extract_embedding",
            side_effect=RuntimeError("secret pid 402265"),
        ):
            with self.assertLogs("voice_api", level="ERROR") as logs:
                r = self.client.post("/compare", files=self.files)
        self.assertEqual(r.status_code, 500)
        self.assertIn("RuntimeError", r.json()["detail"])
        self.assertNotIn("402265", r.json()["detail"])
        self.assertTrue(any("secret pid 402265" in line for line in logs.output))

    def test_compare_success_path(self):
        emb = np.ones(model_service.EMBEDDING_DIM, dtype=np.float32)
        with mock.patch.object(model_service, "extract_embedding", return_value=emb):
            r = self.client.post("/compare", files=self.files)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.json()["same_speaker"])


if __name__ == "__main__":
    unittest.main()
