"""Gate tests for evals/identify_eval.py: scoring and folder walk, no server, no model."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evals import identify_eval  # noqa: E402


def q(speaker: str, *matches: tuple[str, float]) -> dict:
    """One query record as the eval stores it.

    Args:
        speaker: True speaker of the query clip.
        *matches: (speaker_id, similarity) pairs, best first.

    Returns:
        The query record.
    """
    return {
        "speaker": speaker,
        "matches": [{"speaker_id": s, "similarity": v} for s, v in matches],
    }


GENUINE = [
    q("a", ("a", 0.99), ("b", 0.90)),  # right, high
    q("a", ("a", 0.95), ("b", 0.94)),  # right, low
    q("b", ("a", 0.97), ("b", 0.96)),  # wrong person on top, b second
    q("b", ("c", 0.50)),  # b not in the list at all
]
IMPOSTOR = [q("x", ("a", 0.98)), q("y", ("b", 0.80)), q("z")]


class SummarizeTests(unittest.TestCase):
    def test_rank1_and_topk_hit(self):
        out = identify_eval.summarize(GENUINE, IMPOSTOR, [])
        self.assertEqual(out["rank1_accuracy"], 0.5)
        self.assertEqual(out["topk_hit_rate"], 0.75)
        self.assertEqual((out["genuine_queries"], out["impostor_queries"]), (4, 3))

    def test_threshold_splits_right_wrong_and_false_match(self):
        out = identify_eval.summarize(GENUINE, IMPOSTOR, [0.96])["thresholds"][0]
        self.assertEqual(out["genuine_accept_rate"], 0.25)  # only the 0.99
        self.assertEqual(out["wrong_accept_rate"], 0.25)  # a at 0.97 for b
        self.assertAlmostEqual(out["impostor_false_match_rate"], 0.3333, places=4)

    def test_score_equal_to_threshold_counts_as_verified(self):
        out = identify_eval.at_threshold([q("a", ("a", 0.95))], [], 0.95)
        self.assertEqual(out["genuine_accept_rate"], 1.0)

    def test_empty_match_lists_and_empty_sets_do_not_crash(self):
        out = identify_eval.summarize([q("a")], [], [0.5])
        self.assertEqual(out["rank1_accuracy"], 0.0)
        self.assertEqual(out["thresholds"][0]["impostor_false_match_rate"], 0.0)
        self.assertIsNone(out["equal_error"])

    def test_equal_error_sits_between_the_groups_when_separable(self):
        genuine = [q("a", ("a", 0.99)), q("b", ("b", 0.98))]
        impostor = [q("x", ("a", 0.90)), q("y", ("b", 0.85))]
        eer = identify_eval.summarize(genuine, impostor, [])["equal_error"]
        self.assertEqual(eer["genuine_miss_rate"], 0.0)
        self.assertEqual(eer["impostor_false_match_rate"], 0.0)
        self.assertGreater(eer["threshold"], 0.90)
        self.assertLessEqual(eer["threshold"], 0.98)


class CollectTests(unittest.TestCase):
    def test_first_n_audio_files_per_speaker_folder(self):
        with tempfile.TemporaryDirectory() as root:
            for spk, names in {
                "S2": ["b.wav", "a.wav", "c.wav", "notes.txt"],
                "S1": ["x.mp4"],
            }.items():
                os.makedirs(os.path.join(root, spk))
                for n in names:
                    open(os.path.join(root, spk, n), "w").close()
            open(os.path.join(root, "stray.wav"), "w").close()
            out = identify_eval.collect(root, 2)
        self.assertEqual(list(out), ["S1", "S2"])
        self.assertEqual([os.path.basename(p) for p in out["S2"]], ["a.wav", "b.wav"])
        self.assertEqual([os.path.basename(p) for p in out["S1"]], ["x.mp4"])


if __name__ == "__main__":
    unittest.main()
