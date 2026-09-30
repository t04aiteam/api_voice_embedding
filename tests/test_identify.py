"""Gate tests: POST /speakers/identify, 1:N against every enrolled speaker (UC 68).

Contract (HopDong_API_AmThanh.md section 2.1, sent to the app team as a stub
shape): {query_duration_seconds, threshold, matches: [{speaker_id, similarity,
verified}]}, best first, at most top_k (default 5), candidates below the
threshold kept with verified=false, [] with status 200 when nobody is enrolled.

Failure modes pinned here:
- wrong or extra response fields (the app built a stub against the exact shape);
- ascending or unstable order, top_k ignored, default not 5;
- below-threshold candidates dropped instead of flagged;
- a score that differs from /speakers/{id}/verify for the same file and person
  (both must use the centroid of the stored samples);
- a bad top_k or threshold accepted, a bad file answered 200 because the
  gallery was empty;
- identify reading or writing outside SPEAKER_DATA_DIR.

The model is faked: each test clip has its own length, and the fake embedding
is looked up by that length, so every similarity below is exact by design.

Run:
    emb-env/bin/python -m unittest discover -s tests -v
"""

import io
import math
import os
import subprocess
import sys
import tempfile
import unittest
import wave
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
from services import calibration as calibration_service  # noqa: E402
from services import model as model_service  # noqa: E402
from services import store as store_service  # noqa: E402

SR = 16000
DIM = model_service.EMBEDDING_DIM
FIELDS = {"query_duration_seconds", "threshold", "matches"}
MATCH_FIELDS = {"speaker_id", "similarity", "verified"}


def axis(i: int) -> np.ndarray:
    """Unit vector along one axis.

    Args:
        i: Axis index.

    Returns:
        The unit vector.
    """
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def at_cos(c: float, side: int) -> np.ndarray:
    """Unit vector whose cosine with axis 0 is exactly ``c``.

    Args:
        c: Target cosine with the probe (axis 0).
        side: Orthogonal axis that takes the remainder.

    Returns:
        The unit vector.
    """
    return c * axis(0) + math.sqrt(1 - c * c) * axis(side)


# Clip length (seconds) -> the embedding the fake model returns for it.
EMB = {
    1.5: axis(0),  # probe
    2.0: at_cos(0.99, 1),  # speaker a
    2.5: at_cos(0.95, 2),  # speaker b
    3.0: at_cos(0.50, 3),  # speaker c
    3.5: at_cos(0.97, 4),  # a second sample for speaker a
}


def clip(seconds: float) -> bytes:
    """A 16 kHz mono 220 Hz sine WAV of the given length.

    Args:
        seconds: Clip length.

    Returns:
        The WAV bytes.
    """
    n = int(seconds * SR)
    pcm = (0.1 * np.sin(2 * np.pi * 220 * np.arange(n) / SR) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def fake_extract(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """Stand-in for the model: the embedding registered for this clip length.

    Args:
        waveform: Decoded audio.
        sample_rate: Its sample rate.

    Returns:
        The registered embedding.
    """
    return EMB[round(len(waveform) / sample_rate, 2)].copy()


class IdentifyTests(unittest.TestCase):
    """The HTTP surface of /speakers/identify against a temp speaker store."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.data_dir = os.path.join(tmp.name, "speakers")
        for p in (
            mock.patch.object(store_service, "DATA_DIR", self.data_dir),
            mock.patch.object(
                calibration_service, "_mean_embedding", np.zeros(DIM, np.float32)
            ),
            mock.patch.object(model_service, "extract_embedding", fake_extract),
        ):
            p.start()
            self.addCleanup(p.stop)
        # No context manager: lifespan (real model load) must not run.
        self.client = TestClient(main.app, raise_server_exceptions=False)

    def enroll(self, speaker_id: str, seconds: float) -> None:
        """Enroll one clip for a speaker and require a 200.

        Args:
            speaker_id: Speaker to enroll.
            seconds: Clip length, which picks the embedding.
        """
        r = self.client.post(
            "/speakers/enroll",
            data={"speaker_id": speaker_id},
            files={"file": ("s.wav", clip(seconds), "audio/wav")},
        )
        self.assertEqual(r.status_code, 200, r.text)

    def identify(self, seconds: float = 1.5, body: bytes | None = None, **form):
        """POST /speakers/identify with one clip.

        Args:
            seconds: Clip length, which picks the embedding.
            body: Raw upload bytes instead of a generated clip.
            **form: Extra form fields (top_k, threshold).

        Returns:
            The response.
        """
        data = clip(seconds) if body is None else body
        return self.client.post(
            "/speakers/identify",
            data=form,
            files={"file": ("q.wav", data, "audio/wav")},
        )

    def enroll_abc(self) -> None:
        """Enroll speakers a (0.99), b (0.95) and c (0.50) against the probe."""
        self.enroll("c", 3.0)
        self.enroll("a", 2.0)
        self.enroll("b", 2.5)

    def test_nobody_enrolled_is_200_with_no_matches(self):
        r = self.identify()
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(
            r.json(),
            {"query_duration_seconds": 1.5, "threshold": 0.96, "matches": []},
        )

    def test_exact_fields_best_first_and_below_threshold_kept(self):
        self.enroll_abc()
        r = self.identify()
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(set(body), FIELDS)
        self.assertTrue(all(set(m) == MATCH_FIELDS for m in body["matches"]))
        self.assertEqual(
            body["matches"],
            [
                {"speaker_id": "a", "similarity": 0.99, "verified": True},
                {"speaker_id": "b", "similarity": 0.95, "verified": False},
                {"speaker_id": "c", "similarity": 0.5, "verified": False},
            ],
        )
        self.assertEqual(body["threshold"], 0.96)
        self.assertEqual(body["query_duration_seconds"], 1.5)

    def test_top_k_cuts_the_list(self):
        self.enroll_abc()
        r = self.identify(top_k="2")
        self.assertEqual([m["speaker_id"] for m in r.json()["matches"]], ["a", "b"])

    def test_top_k_defaults_to_5_and_ties_sort_by_id(self):
        for sid in ("s6", "s2", "s4", "s0", "s5", "s1", "s3"):
            self.enroll(sid, 2.0)
        r = self.identify()
        ids = [m["speaker_id"] for m in r.json()["matches"]]
        self.assertEqual(ids, ["s0", "s1", "s2", "s3", "s4"])

    def test_empty_top_k_and_threshold_fields_use_the_defaults(self):
        # Postman sends a disabled or blank form field as "".
        self.enroll_abc()
        r = self.identify(top_k="", threshold="")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(len(r.json()["matches"]), 3)
        self.assertEqual(r.json()["threshold"], 0.96)

    def test_threshold_is_echoed_and_decides_verified(self):
        self.enroll_abc()
        r = self.identify(threshold="0.9")
        body = r.json()
        self.assertEqual(body["threshold"], 0.9)
        self.assertEqual([m["verified"] for m in body["matches"]], [True, True, False])

    def test_bad_top_k_is_422(self):
        for bad in ("0", "-1", "abc", "1.5"):
            with self.subTest(top_k=bad):
                r = self.identify(top_k=bad)
                self.assertEqual(r.status_code, 422, r.text)
                self.assertIn("top_k", r.json()["detail"])

    def test_bad_threshold_is_422(self):
        for bad in ("2", "-0.1", "x"):
            with self.subTest(threshold=bad):
                self.assertEqual(self.identify(threshold=bad).status_code, 422)

    def test_score_equals_verify_for_the_same_file_and_person(self):
        # Two samples: both endpoints must score against their centroid.
        self.enroll("a", 2.0)
        self.enroll("a", 3.5)
        ident = self.identify().json()["matches"][0]
        ver = self.client.post(
            "/speakers/a/verify", files={"file": ("q.wav", clip(1.5), "audio/wav")}
        ).json()
        self.assertEqual(ident["speaker_id"], "a")
        self.assertEqual(ident["similarity"], ver["similarity"])
        self.assertEqual(ident["verified"], ver["verified"])
        centroid = (EMB[2.0] + EMB[3.5]) / 2
        expected = float(centroid[0] / np.linalg.norm(centroid))
        self.assertAlmostEqual(ident["similarity"], expected, places=6)

    def test_bad_file_is_400_even_with_nobody_enrolled(self):
        r = self.identify(body=b"not audio at all" * 100)
        self.assertEqual(r.status_code, 400, r.text)

    def test_too_short_file_is_422(self):
        self.assertEqual(self.identify(seconds=0.5).status_code, 422)

    def test_deleted_speaker_is_no_longer_matched(self):
        self.enroll_abc()
        self.assertEqual(self.client.delete("/speakers/a").status_code, 200)
        ids = [m["speaker_id"] for m in self.identify().json()["matches"]]
        self.assertEqual(ids, ["b", "c"])

    def test_identify_leaves_the_store_untouched(self):
        self.enroll_abc()
        before = {
            n: os.path.getmtime(os.path.join(self.data_dir, n))
            for n in os.listdir(self.data_dir)
        }
        self.identify()
        after = {
            n: os.path.getmtime(os.path.join(self.data_dir, n))
            for n in os.listdir(self.data_dir)
        }
        self.assertEqual(before, after)

    def test_mp4_video_is_accepted(self):
        self.enroll_abc()
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "q.mp4")
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=220:sample_rate=44100:duration=3",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=black:s=64x64:r=10:d=3",
                    "-c:v",
                    "mpeg4",
                    "-c:a",
                    "aac",
                    "-shortest",
                    out,
                ],
                check=True,
            )
            with open(out, "rb") as fh:
                data = fh.read()
        # AAC priming changes the decoded length, so the fake keys on 1.5 s.
        with mock.patch.object(
            model_service, "extract_embedding", return_value=EMB[1.5].copy()
        ):
            r = self.client.post(
                "/speakers/identify", files={"file": ("q.mp4", data, "video/mp4")}
            )
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["matches"][0]["speaker_id"], "a")
        self.assertAlmostEqual(r.json()["query_duration_seconds"], 3.0, delta=0.1)


if __name__ == "__main__":
    unittest.main()
