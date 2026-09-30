"""--vram-settle-s: wait until a previous process has released its GPU memory."""
import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_csv_recording import benchmark as b


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def settle(readings, timeout_s=30.0):
    clock = FakeClock()
    values = iter(readings)
    last = [None]

    def read():
        last[0] = next(values, last[0])
        return last[0]
    return b.wait_for_vram_settle(read, timeout_s, clock=clock, sleep=clock.sleep)


class WaitTests(unittest.TestCase):
    def test_stable_memory_settles_after_the_window(self):
        result = settle([15000.0] * 10)
        self.assertEqual((result["settled"], result["waited_s"], result["released_mib"]), (True, 3.0, 0.0))

    def test_waits_while_a_previous_process_releases_memory(self):
        # 15.6 GiB used, then released in steps over 2 s, then stable.
        result = settle([15600, 15600, 12000, 8000, 3000, 400] + [400] * 20)
        self.assertTrue(result["settled"])
        self.assertEqual(result["released_mib"], 15200.0)
        self.assertGreaterEqual(result["waited_s"], 2.5 + 3.0)

    def test_small_fluctuations_do_not_block(self):
        result = settle([5000, 4980, 5010, 4990, 5000, 4995, 5005, 5000])
        self.assertTrue(result["settled"])
        self.assertEqual(result["waited_s"], 3.0)

    def test_timeout_when_memory_keeps_dropping(self):
        result = settle([20000 - 200 * i for i in range(100)], timeout_s=5.0)
        self.assertFalse(result["settled"])
        self.assertEqual(result["waited_s"], 5.0)

    def test_unreadable_memory_skips_the_wait(self):
        self.assertIsNone(settle([None]))


class ServerStartTests(unittest.TestCase):
    def run_start(self, settle_s, result):
        args = SimpleNamespace(vram_settle_s=settle_s, vram_gpu="all")
        read, close = Mock(), Mock()
        events = []
        with patch.object(b, "nvml_used_mib_reader", return_value=(read, close)) as reader, \
             patch.object(b, "wait_for_vram_settle", side_effect=lambda *a, **k: events.append("wait") or result), \
             contextlib.redirect_stderr(io.StringIO()) as err:
            vram_settle = b.settle_vram_before_server_start(args)
        return vram_settle, reader, close, events, err.getvalue()

    def test_waits_before_start_and_reports_released_memory(self):
        result = {"initial_mib": 15600.0, "final_mib": 400.0, "released_mib": 15200.0,
                  "waited_s": 5.5, "settled": True}
        vram_settle, reader, close, events, err = self.run_start(30.0, result)
        self.assertEqual(vram_settle, result)
        reader.assert_called_once_with("all")
        close.assert_called_once()
        self.assertEqual(events, ["wait"])
        self.assertIn("waited 5.5 s while 15200 MiB were released", err)

    def test_reports_usage_by_others_and_warns_on_timeout(self):
        quiet = {"initial_mib": 400.0, "final_mib": 400.0, "released_mib": 0.0, "waited_s": 3.0, "settled": True}
        err = self.run_start(30.0, quiet)[4]
        self.assertEqual(err, "GPU memory before server start: 400 MiB used by other processes\n")
        busy = {"initial_mib": 9000.0, "final_mib": 5000.0, "released_mib": 4000.0, "waited_s": 30.0,
                "settled": False}
        self.assertIn("WARNING: GPU memory was still being released", self.run_start(30.0, busy)[4])

    def test_disabled_or_without_nvml(self):
        vram_settle, reader, _, events, _ = self.run_start(0.0, None)
        self.assertIsNone(vram_settle)
        reader.assert_not_called()
        with patch.object(b, "nvml_used_mib_reader", return_value=(None, None)):
            self.assertIsNone(b.settle_vram_before_server_start(SimpleNamespace(vram_settle_s=30.0, vram_gpu="all")))


class AbortMessageTests(unittest.TestCase):
    def test_sysmem_abort_names_vram_used_by_other_processes(self):
        args = SimpleNamespace(sysmem_guard="abort", sysmem_guard_mb_s=1000.0)
        runner = b.BenchmarkRunner(args, "http://x", object(), None, Mock(), None, None, None)
        spilled = {"prompt_n": 100, "prefill_tps": 526.2, "pcie_prefill_rx_median_mb_s": 1019.0}
        self.assertNotIn("other processes", runner.check_sysmem_fallback(spilled, 10000, 0))
        runner.vram_before_start_mib = 1079.1
        self.assertIn("other processes already used 1079 MiB", runner.check_sysmem_fallback(spilled, 10000, 0))


if __name__ == "__main__":
    unittest.main()
