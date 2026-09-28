"""Gate tests: uploads libsndfile cannot read (mp4 video, m4a) decode through ffmpeg.

UC 68 matches voices taken from audio and video evidence, so /embed, /compare
and /speakers/* must accept an mp4 as uploaded. Before the fix, librosa got the
upload as a BytesIO; its audioread fallback only works on paths, so every
container outside libsndfile (mp4, m4a, webm, ...) answered 400 "Format not
recognised". Fixtures are generated with the host's ffmpeg, the same binary
the service calls.

Run:
    emb-env/bin/python -m unittest discover -s tests -v
"""

import functools
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
from services import audio as audio_service  # noqa: E402
from services import model as model_service  # noqa: E402

SINE = ["-f", "lavfi", "-i", "sine=frequency=220:sample_rate=44100:duration={d}"]
VIDEO = ["-f", "lavfi", "-i", "color=c=black:s=64x64:r=10:d={d}"]


def make(inputs: list[str], codecs: list[str], suffix: str, seconds: float) -> bytes:
    """Render a lavfi fixture with ffmpeg and return the file bytes.

    Args:
        inputs: ffmpeg input arguments, with ``{d}`` standing for the duration.
        codecs: ffmpeg output codec arguments.
        suffix: Output file suffix, which picks the container.
        seconds: Fixture duration.

    Returns:
        The encoded file contents.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "fixture" + suffix)
        args = [a.format(d=seconds) for a in inputs]
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", *args, *codecs, "-shortest", out],
            check=True,
        )
        with open(out, "rb") as fh:
            return fh.read()


@functools.cache
def fixtures() -> dict[str, bytes]:
    """Render every fixture once per test run; each ffmpeg call costs ~0.1 s.

    Returns:
        Fixture name to file bytes.
    """
    av = ["-c:v", "mpeg4", "-c:a", "aac"]
    return {
        "mp4": make(SINE + VIDEO, av, ".mp4", 3.0),
        "m4a": make(SINE, ["-c:a", "aac"], ".m4a", 3.0),
        "silent_mp4": make(VIDEO, ["-c:v", "mpeg4"], ".mp4", 3.0),
        "short_mp4": make(SINE + VIDEO, av, ".mp4", 0.5),
        "wav": make(SINE, ["-c:a", "pcm_s16le", "-ar", "16000"], ".wav", 3.0),
    }


class VideoDecodeTests(unittest.TestCase):
    """_decode: containers outside libsndfile go through ffmpeg, the rest stay native."""

    @classmethod
    def setUpClass(cls):
        """Attach the shared fixtures."""
        f = fixtures()
        cls.mp4, cls.m4a, cls.wav = f["mp4"], f["m4a"], f["wav"]
        cls.silent_mp4, cls.short_mp4 = f["silent_mp4"], f["short_mp4"]

    def assert_sine(self, wave, sr, duration, seconds):
        """Check a decode of the 220 Hz fixture.

        Args:
            wave: Decoded waveform.
            sr: Decoded sample rate.
            duration: Reported duration.
            seconds: Expected duration.
        """
        self.assertEqual(wave.ndim, 1)
        self.assertEqual(wave.dtype, np.float32)
        self.assertEqual(sr, audio_service.FFMPEG_SAMPLE_RATE)
        self.assertAlmostEqual(duration, seconds, delta=0.1)
        # lavfi's sine has amplitude 1/8, RMS ~0.088; silence would be ~0.
        self.assertGreater(float(np.sqrt(np.mean(wave**2))), 0.05)

    def test_mp4_video_decodes_its_audio_track(self):
        self.assert_sine(*audio_service._decode(self.mp4), 3.0)

    def test_m4a_decodes(self):
        self.assert_sine(*audio_service._decode(self.m4a), 3.0)

    def test_video_without_audio_track_is_400(self):
        with self.assertRaises(HTTPException) as ctx:
            audio_service._decode(self.silent_mp4)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Cannot decode audio file", ctx.exception.detail)
        # ffmpeg says why on an earlier line than its last one; keep it.
        self.assertIn("stream", ctx.exception.detail)

    def test_garbage_is_400(self):
        with self.assertRaises(HTTPException) as ctx:
            audio_service._decode(b"not audio at all" * 100)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_native_formats_do_not_spawn_ffmpeg(self):
        # wav/flac/ogg/mp3 keep the in-memory libsndfile path and native rate.
        with mock.patch.object(audio_service.subprocess, "run") as run:
            wave, sr, duration = audio_service._decode(self.wav)
        run.assert_not_called()
        self.assertEqual(sr, 16000)
        self.assertAlmostEqual(duration, 3.0, places=2)

    def test_length_cap_bounds_the_ffmpeg_decode(self):
        with mock.patch.object(audio_service, "MAX_AUDIO_SECONDS", 1.5):
            with mock.patch.object(
                audio_service.subprocess, "run", wraps=subprocess.run
            ) as spy:
                with self.assertRaises(HTTPException) as ctx:
                    audio_service._decode(self.mp4)
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertIn("Audio too long", ctx.exception.detail)
        cmd = spy.call_args.args[0]
        self.assertEqual(cmd[cmd.index("-t") + 1], "2.5")

    def test_short_video_is_422(self):
        with self.assertRaises(HTTPException) as ctx:
            audio_service._decode(self.short_mp4)
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertIn("Audio too short", ctx.exception.detail)

    def test_temp_file_is_removed_on_success_and_failure(self):
        seen = []
        real = tempfile.NamedTemporaryFile

        def spy(*args, **kwargs):
            """Record the path of each temp file the decoder creates.

            Args:
                *args: Passed through.
                **kwargs: Passed through.

            Returns:
                The real temp file.
            """
            fh = real(*args, **kwargs)
            seen.append(fh.name)
            return fh

        with mock.patch.object(audio_service.tempfile, "NamedTemporaryFile", spy):
            audio_service._decode(self.mp4)
            with self.assertRaises(HTTPException):
                audio_service._decode(self.silent_mp4)
        self.assertEqual(len(seen), 2)
        self.assertFalse(any(os.path.exists(p) for p in seen))

    def test_missing_ffmpeg_is_named_in_the_error(self):
        err = FileNotFoundError(2, "No such file or directory", "ffmpeg")
        with mock.patch.object(audio_service.subprocess, "run", side_effect=err):
            with self.assertRaises(HTTPException) as ctx:
                audio_service._decode(self.mp4)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("ffmpeg", ctx.exception.detail)

    def test_memory_error_on_ffmpeg_path_is_oom(self):
        with mock.patch.object(
            audio_service.np, "frombuffer", side_effect=MemoryError("32 MiB")
        ):
            with self.assertRaises(model_service.InferenceOutOfMemory) as ctx:
                audio_service._decode(self.mp4)
        self.assertEqual(ctx.exception.device, "cpu")


class VideoRouteTests(unittest.TestCase):
    """The HTTP surface: every voice route decodes through read_and_decode."""

    @classmethod
    def setUpClass(cls):
        """Attach the shared mp4 and wav fixtures."""
        cls.mp4, cls.wav = fixtures()["mp4"], fixtures()["wav"]

    def setUp(self):
        # No context manager: lifespan (real model load) must not run.
        self.client = TestClient(main.app, raise_server_exceptions=False)
        emb = np.ones(model_service.EMBEDDING_DIM, dtype=np.float32)
        patcher = mock.patch.object(
            model_service, "extract_embedding", return_value=emb
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_embed_accepts_mp4(self):
        r = self.client.post(
            "/embed", files={"file": ("clip.mp4", self.mp4, "video/mp4")}
        )
        self.assertEqual(r.status_code, 200, r.text)
        self.assertAlmostEqual(r.json()["duration_seconds"], 3.0, delta=0.1)

    def test_compare_accepts_mp4_against_wav(self):
        files = {
            "file_a": ("clip.mp4", self.mp4, "video/mp4"),
            "file_b": ("clip.wav", self.wav, "audio/wav"),
        }
        r = self.client.post("/compare", files=files)
        self.assertEqual(r.status_code, 200, r.text)


if __name__ == "__main__":
    unittest.main()
