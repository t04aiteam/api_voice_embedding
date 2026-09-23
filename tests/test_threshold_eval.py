"""Gate tests for evals/threshold_eval.py: pure scoring helpers, no model, no audio."""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evals import threshold_eval  # noqa: E402


class RatesTests(unittest.TestCase):
    def test_counts_false_rejects_and_false_accepts(self):
        same = np.array([0.9, 0.8, 0.6])
        diff = np.array([0.5, 0.75, 0.2, 0.71])
        out = threshold_eval.rates(same, diff, 0.7)
        self.assertEqual(out["false_reject"], 1)
        self.assertEqual(out["false_accept"], 2)
        self.assertAlmostEqual(out["false_reject_rate"], 1 / 3, places=4)
        self.assertAlmostEqual(out["false_accept_rate"], 0.5, places=4)

    def test_score_equal_to_threshold_is_accepted(self):
        out = threshold_eval.rates(np.array([0.7]), np.array([0.7]), 0.7)
        self.assertEqual(out["false_reject"], 0)
        self.assertEqual(out["false_accept"], 1)


class EerTests(unittest.TestCase):
    def test_separable_scores_give_zero_eer_between_the_groups(self):
        same = np.array([0.8, 0.9, 0.95])
        diff = np.array([0.1, 0.2, 0.3])
        thr, rate = threshold_eval.eer(same, diff)
        self.assertEqual(rate, 0.0)
        self.assertGreater(thr, 0.3)
        self.assertLessEqual(thr, 0.8)

    def test_identical_distributions_give_half_error(self):
        scores = np.array([0.4, 0.5, 0.6, 0.7])
        _, rate = threshold_eval.eer(scores, scores)
        self.assertAlmostEqual(rate, 0.5, places=1)


class CenterTests(unittest.TestCase):
    def test_zero_mean_is_plain_l2_normalisation(self):
        raw = np.array([3.0, 4.0], dtype=np.float32)
        out = threshold_eval.center(raw, np.zeros(2))
        np.testing.assert_allclose(out, [0.6, 0.8], rtol=1e-6)

    def test_mean_is_subtracted_before_normalising(self):
        raw = np.array([1.0, 1.0], dtype=np.float32)
        out = threshold_eval.center(raw, np.array([1.0, 0.0]))
        np.testing.assert_allclose(out, [0.0, 1.0], rtol=1e-6)


class CollectTests(unittest.TestCase):
    def test_one_folder_per_speaker_sorted_and_capped(self):
        with tempfile.TemporaryDirectory() as tmp:
            for spk, files in (
                ("b", ["2.wav", "1.wav", "3.wav"]),
                ("a", ["x.mp3", "notes.txt"]),
            ):
                os.makedirs(os.path.join(tmp, spk))
                for f in files:
                    open(os.path.join(tmp, spk, f), "wb").close()
            open(os.path.join(tmp, "stray.wav"), "wb").close()
            got = threshold_eval.collect(tmp, per_speaker=2, max_speakers=0)
        self.assertEqual(list(got), ["a", "b"])
        self.assertEqual([os.path.basename(p) for p in got["a"]], ["x.mp3"])
        self.assertEqual([os.path.basename(p) for p in got["b"]], ["1.wav", "2.wav"])

    def test_max_speakers_stops_early(self):
        with tempfile.TemporaryDirectory() as tmp:
            for spk in ("a", "b", "c"):
                os.makedirs(os.path.join(tmp, spk))
                open(os.path.join(tmp, spk, "1.wav"), "wb").close()
            got = threshold_eval.collect(tmp, per_speaker=5, max_speakers=2)
        self.assertEqual(list(got), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
