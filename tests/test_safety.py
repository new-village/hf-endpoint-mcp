import unittest
from unittest.mock import patch

from hf_endpoint_mcp import ops
from test_ops import CFG, ep


class SafetyTest(unittest.TestCase):
    def test_failed_endpoint_never_arms_timer_or_resumes(self):
        with patch.object(ops, "locked"), patch.object(ops, "token", return_value="test"), \
             patch.object(ops, "get", return_value=ep("failed")), \
             patch.object(ops, "timer") as timer, patch.object(ops, "request") as request:
            with self.assertRaisesRegex(RuntimeError, "failed"):
                ops.start(CFG)
            timer.assert_not_called()
            request.assert_not_called()

    def test_timer_disable_checks_inactive(self):
        def run(argv, **kwargs):
            class Result:
                stdout = "disabled" if "is-enabled" in argv else "active"
            return Result()

        with patch.object(ops, "run", side_effect=run):
            with self.assertRaisesRegex(RuntimeError, "still active"):
                ops.timer(CFG, "disable")


if __name__ == "__main__":
    unittest.main()
