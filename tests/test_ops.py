import tempfile
import unittest
from unittest.mock import patch

from hf_endpoint_mcp import ops

CFG = {"namespace": "example", "endpoint": "example", "timer": "hf-endpoint-watch.timer",
       "lock_file": "/unused", "switch_command": ["/bin/true"], "price_per_hour_usd": 1}


def ep(state="running", ready=1):
    return {"status": {"state": state, "readyReplica": ready, "url": "https://example.invalid"},
            "compute": {"scaling": {"maxReplica": 1, "minReplica": 0, "scaleToZeroTimeout": 15}}}


class OperationsTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.cfg = {**CFG, "lock_file": self.dir.name + "/lock"}

    def test_start_enables_monitor_before_resume_and_switches_only_after_smoke(self):
        events = []
        def req(tok, method, url, payload=None, timeout=30):
            events.append(method + " " + url.split("/")[-1])
            return {}
        with patch.object(ops, "token", return_value="test"), patch.object(ops, "get", side_effect=[ep("paused"), ep()]), \
             patch.object(ops, "timer", side_effect=lambda c, a: events.append("timer " + a)), \
             patch.object(ops, "request", side_effect=req), patch.object(ops, "smoke", return_value={"inference_verified": True}), \
             patch.object(ops, "switch", side_effect=lambda c, m: events.append("switch " + m)):
            self.assertEqual(ops.start(self.cfg)["state"], "running")
        self.assertEqual(events, ["timer enable", "POST resume", "switch hf"])

    def test_start_refuses_unbounded_replica_before_timer_or_resume(self):
        bad = ep()
        bad["compute"]["scaling"]["maxReplica"] = 3
        with patch.object(ops, "token", return_value="test"), patch.object(ops, "get", return_value=bad), \
             patch.object(ops, "timer") as timer:
            with self.assertRaisesRegex(RuntimeError, "maxReplica"):
                ops.start(self.cfg)
            timer.assert_not_called()

    def test_shutdown_fallback_precedes_pause_and_timer_disable(self):
        events = []
        with patch.object(ops, "token", return_value="test"), patch.object(ops, "get", side_effect=[ep("scaledToZero"), ep("paused")]), \
             patch.object(ops, "switch", side_effect=lambda c, m: events.append("switch " + m)), \
             patch.object(ops, "request", side_effect=lambda *a, **k: events.append("pause")), \
             patch.object(ops, "timer", side_effect=lambda c, a: events.append("timer " + a)):
            self.assertEqual(ops.shutdown(self.cfg)["timer"], "disabled")
        self.assertEqual(events, ["switch sol", "pause", "timer disable"])

    def test_pause_failure_leaves_monitor_enabled(self):
        with patch.object(ops, "token", return_value="test"), patch.object(ops, "get", return_value=ep("scaledToZero")), \
             patch.object(ops, "switch"), patch.object(ops, "request"), patch.object(ops, "timer") as timer:
            with self.assertRaisesRegex(RuntimeError, "pause unconfirmed"):
                ops.shutdown(self.cfg)
            timer.assert_not_called()

    def test_reconcile_never_resumes(self):
        with patch.object(ops, "token", return_value="test"), patch.object(ops, "get", return_value=ep()), \
             patch.object(ops, "request") as req:
            self.assertEqual(ops.reconcile(self.cfg)["state"], "running")
            req.assert_not_called()


if __name__ == "__main__":
    unittest.main()
