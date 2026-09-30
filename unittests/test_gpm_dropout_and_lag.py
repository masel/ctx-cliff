"""Complete GPM dropouts (cross-checked with a second sensor), GPM lag correction,
legacy PCIe calibration scale and cached link queries (review of 2026-09-26)."""
import contextlib
import io
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_csv_recording import SCRIPT, benchmark as b
from test_gpm_boundaries import FakeNvml as FakeGpmNvml


def gpm_row(start, end, gpu="0", graphics=90.0, sm=60.0, occupancy=30.0, tensor=10.0, dram=40.0,
            rx=300.0, tx=20.0):
    return {"gpu_index": gpu, "interval_start_mono": start, "interval_end_mono": end,
            "graphics_util_pct": graphics, "sm_util_pct": sm, "sm_occupancy_pct": occupancy,
            "tensor_util_pct": tensor, "dram_bw_util_pct": dram, "pcie_rx_mib_s": rx,
            "pcie_tx_mib_s": tx, "pcie_theoretical_mib_s": 7512.0}


def zero_row(start, end, gpu="0"):
    return gpm_row(start, end, gpu, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def store_of(rows, key="interval_end_mono"):
    store = b.SampleStore(key)
    store.archive_enabled = False
    store.extend(rows)
    return store


class GpmDropoutTests(unittest.TestCase):
    def test_all_zero_detection(self):
        self.assertTrue(b.gpm_all_zero(zero_row(0, 0.25)))
        self.assertFalse(b.gpm_all_zero(gpm_row(0, 0.25)))
        self.assertFalse(b.gpm_all_zero(dict(zero_row(0, 0.25), pcie_tx_mib_s=0.3)))
        missing = {k: None for k in b.GPM_VALUE_KEYS}
        self.assertFalse(b.gpm_all_zero(dict(missing, pcie_rx_mib_s=0.0)))  # too little evidence

    def test_zero_intervals_with_activity_are_excluded_from_all_gpm_statistics(self):
        rows = [gpm_row(0.0, 0.25), zero_row(0.25, 0.5), zero_row(0.5, 0.75), gpm_row(0.75, 1.0)]
        store = store_of(rows)
        self.addCleanup(store.close)
        busy = lambda gpu, start, end: "legacy_pcie"  # noqa: E731
        summary = b.summarize_gpm(store, 0.0, 1.0, activity=busy)
        self.assertEqual((summary["gpm_dropout_samples"], summary["gpm_dropout_ms"]), (2, 500.0))
        self.assertEqual(summary["gpm_samples"], 4)
        self.assertAlmostEqual(summary["gpm_graphics_avg_pct"], 90.0)
        self.assertAlmostEqual(summary["gpm_dram_avg_pct"], 40.0)
        self.assertAlmostEqual(summary["gpm_pcie_rx_median_mib_s"], 300.0)
        self.assertEqual(summary["gpm_pcie_samples"], 2)
        self.assertAlmostEqual(summary["gpm_coverage_pct"], 50.0)

    def test_real_idle_zeros_stay_without_activity_evidence(self):
        rows = [gpm_row(0.0, 0.25), zero_row(0.25, 0.5)]
        store = store_of(rows)
        self.addCleanup(store.close)
        for activity in (None, lambda gpu, start, end: None):
            summary = b.summarize_gpm(store, 0.0, 0.5, activity=activity)
            self.assertEqual(summary["gpm_dropout_samples"], 0)
            self.assertAlmostEqual(summary["gpm_graphics_avg_pct"], 45.0)

    def test_activity_evidence_from_legacy_pcie_or_nvidia_smi(self):
        monitor = b.NvidiaVramMonitor()
        for store in (monitor.samples, monitor.pcie_samples):
            store.archive_enabled = False
            self.addCleanup(store.close)
        monitor.pcie_samples.extend([
            {"t_mono": 1.1, "gpu_index": "0", "pcie_rx_mb_s": 400.0, "pcie_tx_mb_s": 40.0},
            {"t_mono": 1.2, "gpu_index": "1", "pcie_rx_mb_s": 0.5, "pcie_tx_mb_s": 0.5},
            {"t_mono": 2.1, "gpu_index": "0", "pcie_rx_mb_s": 0.6, "pcie_tx_mb_s": 0.4},
        ])
        monitor.samples.extend([
            {"t_mono": 2.2, "gpu_index": "0", "gpu_util_pct": 97.0},
            {"t_mono": 3.2, "gpu_index": "0", "gpu_util_pct": 0.0},
        ])
        self.assertEqual(monitor.activity_evidence("0", 1.0, 1.25), "legacy_pcie")
        self.assertIsNone(monitor.activity_evidence("1", 1.0, 1.25))
        self.assertEqual(monitor.activity_evidence(0, 2.0, 2.25), "nvidia_smi_util")
        self.assertIsNone(monitor.activity_evidence("0", 3.0, 3.25))

    def test_dropouts_are_summed_over_repeats_and_marked_in_console(self):
        sample = dict(cache_n=0, prompt_n=10, prompt_ms=10, prefill_tps=100, decode_tps=5, predicted_n=8,
                      predicted_ms=100, draft_n=0, draft_acc=0, wall_s=1, truncated=False, status="OK",
                      gpm_decode_dropout_samples=2, gpm_decode_dropout_ms=500.0)
        row = b.aggregate_point([sample, dict(sample, gpm_decode_dropout_samples=1, gpm_decode_dropout_ms=250.0)],
                                1, 1, 1, SimpleNamespace(cache_mode="cold"))
        self.assertEqual((row["gpm_decode_dropout_samples"], row["gpm_decode_dropout_ms"]), (3, 750.0))
        self.assertIn("gpm_prefill_dropout_samples", b.RESULT_CSV_FIELDS)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            b.print_live_row(row, drafting_on=False)
        self.assertEqual(out.getvalue().count("!"), 1)


class GpmLagTests(unittest.TestCase):
    def run_reader(self, lag_s, duration=0.45):
        nv = FakeGpmNvml()
        monitor = b.NvidiaGpmMonitor(interval_ms=120)
        monitor.samples.archive_enabled = False
        self.addCleanup(monitor.samples.close)
        monitor._pynvml = nv
        monitor.metric_ids = [(key, index) for index, (key, _) in enumerate(b.NvidiaGpmMonitor.METRIC_DEFS)]
        monitor._handles = [("0", "handle-0")]
        monitor._sample_pairs = {"0": (object(), object())}
        monitor._prev_times = {"0": time.perf_counter()}
        monitor.lag_s = lag_s
        thread = threading.Thread(target=monitor._reader)
        thread.start()
        time.sleep(duration)
        monitor.stop_event.set()
        thread.join(2)
        rows, _ = monitor.samples.window(-1e9, 1e9)
        return monitor, rows

    def test_intervals_are_shifted_back_and_raw_times_kept(self):
        _, rows = self.run_reader(0.07)
        self.assertGreaterEqual(len(rows), 2)
        for row in rows:
            self.assertAlmostEqual(row["sample_start_mono"] - row["interval_start_mono"], 0.07)
            self.assertAlmostEqual(row["sample_end_mono"] - row["interval_end_mono"], 0.07)
            self.assertAlmostEqual(row["interval_ms"], (row["sample_end_mono"] - row["sample_start_mono"]) * 1000)
        self.assertIn("sample_start_mono", b.GPM_CSV_FIELDS)

    def test_zero_lag_keeps_times(self):
        _, rows = self.run_reader(0.0, duration=0.3)
        self.assertTrue(rows)
        self.assertEqual(rows[0]["sample_end_mono"], rows[0]["interval_end_mono"])

    def test_wait_for_lagged_only_waits_for_missing_intervals(self):
        monitor = b.NvidiaGpmMonitor()
        self.addCleanup(monitor.samples.close)
        monitor.running = True
        monitor.lag_s = 0.05
        end = time.perf_counter()
        monitor.wait_for_lagged(end)
        self.assertGreaterEqual(time.perf_counter(), end + 0.07 - 0.005)
        started = time.perf_counter()
        monitor.wait_for_lagged(started - 1.0)  # long ago: nothing to wait for
        monitor.lag_s = 0.0
        monitor.wait_for_lagged(time.perf_counter())
        self.assertLess(time.perf_counter() - started, 0.02)


class LegacyScaleAndLinkCacheTests(unittest.TestCase):
    def test_scale_divides_legacy_rates_and_saturation(self):
        vram = store_of([])
        pcie = store_of([{"t_mono": 1.0 + i * 0.1, "gpu_index": "0", "pcie_rx_mb_s": 4000.0,
                          "pcie_tx_mb_s": 100.0, "pcie_link_gen": 3, "pcie_link_width": 8}
                         for i in range(5)], key="t_mono")
        for store in (vram, pcie):
            self.addCleanup(store.close)
        raw = b.summarize_vram(vram, pcie, 0.9, 1.6, 250)
        scaled = b.summarize_vram(vram, pcie, 0.9, 1.6, 250, pcie_scale=2.0)
        self.assertAlmostEqual(scaled["pcie_rx_median_mb_s"], raw["pcie_rx_median_mb_s"] / 2)
        self.assertAlmostEqual(scaled["pcie_peak_pct_theoretical"], raw["pcie_peak_pct_theoretical"] / 2)
        self.assertEqual(raw["pcie_over90_pct"], 0.0)
        # Raw rows in the store are not modified.
        self.assertEqual(pcie.window(0, 2)[0][0]["pcie_rx_mb_s"], 4000.0)

    def test_link_state_is_queried_at_most_once_per_refresh_interval(self):
        calls = {"cur": 0, "max": 0}

        class Nvml:
            NVML_PCIE_UTIL_RX_BYTES, NVML_PCIE_UTIL_TX_BYTES = 1, 0

            def nvmlDeviceGetPcieThroughput(self, handle, counter):
                time.sleep(0.002)
                return 1000

            def nvmlDeviceGetCurrPcieLinkGeneration(self, handle):
                calls["cur"] += 1
                return 3

            def nvmlDeviceGetCurrPcieLinkWidth(self, handle):
                return 8

            def nvmlDeviceGetMaxPcieLinkGeneration(self, handle):
                calls["max"] += 1
                return 3

            def nvmlDeviceGetMaxPcieLinkWidth(self, handle):
                return 16

        monitor = b.NvidiaVramMonitor(pcie_interval_ms=20)
        monitor.pcie_samples.archive_enabled = False
        self.addCleanup(monitor.pcie_samples.close)
        monitor._pynvml = Nvml()
        monitor._nvml_handles = [("0", object())]
        with patch.object(b.NvidiaVramMonitor, "LINK_REFRESH_S", 0.2):
            thread = threading.Thread(target=monitor._pcie_reader)
            thread.start()
            time.sleep(0.5)
            monitor.stop_event.set()
            thread.join(2)
        rows, _ = monitor.pcie_samples.window(-1e9, 1e9)
        self.assertGreater(len(rows), 10)
        self.assertLessEqual(calls["cur"], 4)
        self.assertEqual(calls["max"], 1)
        self.assertTrue(all(r["pcie_link_gen"] == 3 and r["pcie_link_width_max"] == 16 for r in rows))

    def test_timing_summary_hints_at_uncorrected_values(self):
        summary = {"requested_interval_ms": 250, "achieved_interval_median_ms": 250.0,
                   "achieved_interval_p90_ms": 260.0, "rx_tx_query_median_ms": 63.0,
                   "time_coverage_pct_per_direction": 8.0, "scale": 1.0}
        with contextlib.redirect_stdout(io.StringIO()) as out:
            b.print_pcie_timing(summary)
        self.assertIn("--pcie-legacy-scale", out.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as out:
            b.print_pcie_timing(dict(summary, scale=1.55))
        self.assertNotIn("uncorrected", out.getvalue())


class CliTests(unittest.TestCase):
    def test_defaults_and_validation(self):
        seen = []
        argv = [str(SCRIPT), "--file", str(SCRIPT), "--vram-log", "off", "--gpm-log", "off",
                "--win-gpu-mem", "off"]
        with patch("sys.argv", argv), patch.object(b, "run_benchmark", side_effect=lambda a, *r: seen.append(a)):
            b.main()
        self.assertEqual((seen[0].gpm_lag_ms, seen[0].pcie_legacy_scale), (70.0, 1.0))
        for extra in (["--gpm-lag-ms", "-1"], ["--gpm-lag-ms", "2000"], ["--pcie-legacy-scale", "0"]):
            with self.subTest(extra=extra), patch("sys.argv", argv + extra), self.assertRaises(SystemExit), \
                    contextlib.redirect_stderr(io.StringIO()):
                b.main()

    def test_result_row_records_corrections(self):
        sample = dict(cache_n=0, prompt_n=10, prompt_ms=10, prefill_tps=100, decode_tps=5, predicted_n=8,
                      predicted_ms=100, draft_n=0, draft_acc=0, wall_s=1, truncated=False, status="OK")
        row = b.aggregate_point([sample], 1, 1, 1, SimpleNamespace(cache_mode="cold", gpm_lag_ms=70.0,
                                                                  pcie_legacy_scale=1.55))
        self.assertEqual((row["gpm_lag_ms"], row["pcie_legacy_scale"]), (70.0, 1.55))


if __name__ == "__main__":
    unittest.main()


class GpmRestartTests(unittest.TestCase):
    def reader_monitor(self):
        nv = FakeGpmNvml()
        monitor = b.NvidiaGpmMonitor(interval_ms=120)
        monitor.samples.archive_enabled = False
        self.addCleanup(monitor.samples.close)
        monitor._pynvml = nv
        monitor.metric_ids = [(key, index) for index, (key, _) in enumerate(b.NvidiaGpmMonitor.METRIC_DEFS)]
        monitor._handles = [("0", "handle-0")]
        monitor._sample_pairs = {"0": (object(), object())}
        monitor._prev_times = {"0": time.perf_counter()}
        monitor.running = True
        monitor.thread = threading.Thread(target=monitor._reader)
        monitor.thread.start()
        self.addCleanup(lambda: (monitor.stop_event.set(), monitor.thread.join(2)))
        return monitor, nv

    def test_realloc_and_reinit_run_in_reader_thread(self):
        monitor, nv = self.reader_monitor()
        old_pair = monitor._sample_pairs["0"]
        event = monitor.request_restart("realloc")
        self.assertTrue(event["ok"])
        self.assertIsNot(monitor._sample_pairs["0"], old_pair)
        self.assertEqual(nv.nvmlGpmSampleFree.call_count, 2)
        nv.nvmlShutdown.assert_not_called()
        event = monitor.request_restart("reinit")
        self.assertTrue(event["ok"])
        nv.nvmlShutdown.assert_called_once_with()
        nv.nvmlInit.assert_called_once_with()
        self.assertEqual(monitor._handles, [("0", "handle-0")])
        time.sleep(0.3)
        self.assertTrue(monitor.thread.is_alive())
        self.assertIsNone(monitor.error)
        self.assertEqual([e["kind"] for e in monitor.restart_events], ["realloc", "reinit"])
        with self.assertRaises(ValueError):
            monitor.request_restart("bogus")

    def test_failed_restart_is_reported(self):
        monitor, nv = self.reader_monitor()
        nv.nvmlGpmSampleAlloc.side_effect = RuntimeError("no memory")
        event = monitor.request_restart("realloc")
        self.assertFalse(event["ok"])
        self.assertIn("realloc restart failed", monitor.error)

    def test_not_running_monitor_ignores_restart(self):
        monitor = b.NvidiaGpmMonitor()
        self.addCleanup(monitor.samples.close)
        self.assertIsNone(monitor.request_restart("realloc"))

    def runner_with(self, mode, dropout_gpus, phase_values):
        calls = []

        class Monitor:
            restart_events = []

            def request_restart(self, kind):
                calls.append(kind)
                event = {"kind": kind, "ok": True}
                self.restart_events.append(event)
                return event

        args = SimpleNamespace(gpm_restart=mode)
        runner = b.BenchmarkRunner(args, "http://x", None, None, None, None, Monitor(), None)
        with contextlib.redirect_stderr(io.StringIO()) as err:
            result = runner._maybe_restart_gpm(dropout_gpus, phase_values, 100, 0, "measure")
        return result, calls, err.getvalue()

    def test_runner_triggers_only_on_dropouts_and_when_enabled(self):
        self.assertEqual(self.runner_with("off", {"0"}, {})[:2], ("", []))
        self.assertEqual(self.runner_with("realloc", set(), {"gpm_decode_dropout_samples": 0})[:2], ("", []))
        result, calls, err = self.runner_with("realloc", {"0"}, {})
        self.assertEqual((result, calls), ("realloc", ["realloc"]))
        self.assertIn("GPM dropout (sm_zero); GPM realloc restart done", err)
        result, calls, err = self.runner_with("reinit", set(), {"gpm_prefill_dropout_samples": 2})
        self.assertEqual((result, calls), ("reinit", ["reinit"]))
        self.assertIn("(all_zero)", err)

    def test_result_counts_restarts_and_cli_default(self):
        sample = dict(cache_n=0, prompt_n=10, prompt_ms=10, prefill_tps=100, decode_tps=5, predicted_n=8,
                      predicted_ms=100, draft_n=0, draft_acc=0, wall_s=1, truncated=False, status="OK")
        row = b.aggregate_point([dict(sample, gpm_restart="realloc"), dict(sample, gpm_restart=""), sample],
                                1, 1, 1, SimpleNamespace(cache_mode="cold"))
        self.assertEqual(row["gpm_restarts"], 1)
        self.assertIn("gpm_restart", b.SAMPLE_CSV_FIELDS)
        seen = []
        argv = [str(SCRIPT), "--file", str(SCRIPT), "--vram-log", "off", "--gpm-log", "off", "--win-gpu-mem", "off"]
        with patch("sys.argv", argv), patch.object(b, "run_benchmark", side_effect=lambda a, *r: seen.append(a)):
            b.main()
        self.assertEqual(seen[0].gpm_restart, "off")
