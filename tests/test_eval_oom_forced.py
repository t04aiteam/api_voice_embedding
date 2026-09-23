"""Gate tests for evals/eval_oom_forced.py: outcome labelling and verdict, no server."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evals"))

import eval_oom_forced as forced  # noqa: E402


def _res(status, detail):
    return {"status": status, "latency": 0.1, "detail": detail}


class OutcomeTests(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(forced.outcome(_res(200, {"similarity": 1.0})), "200")
        self.assertEqual(
            forced.outcome(_res(500, {"detail": "Server out of memory on cpu"})),
            "oom",
        )
        self.assertEqual(
            forced.outcome(_res(500, {"detail": "Internal server error: X"})), "500"
        )
        self.assertEqual(forced.outcome(_res("error", "ConnectError")), "error")


class VerdictTests(unittest.TestCase):
    def test_clean_run_passes(self):
        steps = [
            {"name": "a", "expect": "200", "outcomes": ["200"], "alive": True},
            {"name": "b", "expect": "oom", "outcomes": ["oom", "oom"], "alive": True},
        ]
        self.assertEqual(forced.verdict(steps), [])

    def test_wrong_outcome_and_dead_server_are_both_reported(self):
        steps = [
            {"name": "b", "expect": "oom", "outcomes": ["oom", "500"], "alive": False},
        ]
        failures = forced.verdict(steps)
        self.assertEqual(len(failures), 2)
        self.assertIn("expected oom, got ['500']", failures[0])
        self.assertIn("server died", failures[1])


if __name__ == "__main__":
    unittest.main()
