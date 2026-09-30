"""Suspect GPM SM/occupancy/tensor dropouts are excluded from phase statistics;
the legacy NVML PCIe sampler records its achieved timing (review of 2026-09-26)."""
import contextlib
import io
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_csv_recording import SCRIPT, benchmark as b


def gpm_row(start, end, *, gpu=0, graphics=90.0, sm=0.0, occupancy=0.0, tensor=0.0,
            dram=50.0, rx=100.0, tx=20.0):
    return {"gpu_index": gpu, "interval_start_mono": start, "interval_end_mono": end,
            "interval_ms": (end - start) * 1000.0,
            "graphics_util_pct": graphics, "sm_util_pct": sm, "sm_occupancy_pct": occupancy,
            "tensor_util_pct": tensor, "dram_bw_util_pct": dram,
            "pcie_rx_mib_s": rx, "pcie_tx_mib_s": tx, "pcie_theoretical_mib_s": 7512.0}


def series(start, count, **values):
    return [gpm_row(start + i * 0.25, start + (i + 1) * 0.25, **values) for i in range(count)]


class GpmSuspectExclusionTests(unittest.TestCase):
    def store(self, rows):
        store = b.SampleStore("interval_end_mono")
        store.archive_enabled = False
        self.addCleanup(store.close)
        store.extend(rows)
        return store

    def test_suspect_run_is_excluded_only_from_sm_occupancy_tensor(self):
        rows = (series(0, 4) +  # dropout: graphics 90 %, SM/O/T = 0
                series(1, 4, sm=90.0, occupancy=40.0, tensor=20.0, dram=70.0))
        store = self.store(rows)
        excluded = b.summarize_gpm(store, 0, 2, exclude_suspect=True)
        kept = b.summarize_gpm(store, 0, 2, exclude_suspect=False)

        self.assertAlmostEqual(excluded["gpm_sm_avg_pct"], 90.0)
        self.assertAlmostEqual(excluded["gpm_occupancy_avg_pct"], 40.0)
        self.assertAlmostEqual(excluded["gpm_tensor_avg_pct"], 20.0)
        self.assertAlmostEqual(excluded["gpm_sm_valid_ms"], 1000.0)
        self.assertEqual(excluded["gpm_sm_valid_samples"], 4)
        # Graphics, DRAM and PCIe keep every sample.
        self.assertAlmostEqual(excluded["gpm_dram_avg_pct"], 60.0)
        self.assertAlmostEqual(excluded["gpm_graphics_avg_pct"], 90.0)
        self.assertEqual(excluded["gpm_dram_valid_samples"], 8)
        self.assertEqual(excluded["gpm_pcie_samples"], 8)
        self.assertEqual(excluded["gpm_suspect_samples"], 4)
        self.assertEqual(excluded["gpm_suspect_phases"], 1)

        self.assertAlmostEqual(kept["gpm_sm_avg_pct"], 45.0)
        self.assertEqual(kept["gpm_sm_valid_samples"], 8)
        self.assertEqual(kept["gpm_suspect_samples"], 4)

    def test_fully_suspect_phase_has_no_sm_value_but_keeps_dram(self):
        summary = b.summarize_gpm(self.store(series(0, 8)), 0, 2)
        for engine in ("sm", "occupancy", "tensor"):
            self.assertIsNone(summary[f"gpm_{engine}_avg_pct"])
            self.assertEqual(summary[f"gpm_{engine}_valid_ms"], 0.0)
        self.assertAlmostEqual(summary["gpm_dram_avg_pct"], 50.0)

    def test_short_runs_are_not_suspect_and_stay_in_the_average(self):
        rows = series(0, 3) + series(0.75, 5, sm=80.0)
        summary = b.summarize_gpm(self.store(rows), 0, 2)
        self.assertEqual(summary["gpm_suspect_samples"], 0)
        self.assertAlmostEqual(summary["gpm_sm_avg_pct"], 50.0)

    def test_point_average_uses_clean_repeats_only(self):
        # Both repeats within the store's 5 s idle history.
        store = self.store(series(0, 4) + series(2, 4, sm=80.0))
        repeats = []
        for start in (0, 2):
            summary = b.summarize_gpm(store, start, start + 1)
            repeats.append(b.phase_fields(summary, "gpm_", "decode"))
        samples = [dict(cache_n=0, prompt_n=10, prompt_ms=10, prefill_tps=100, decode_tps=5,
                        predicted_n=8, predicted_ms=100, draft_n=0, draft_acc=0, wall_s=1,
                        truncated=False, status="OK", **fields) for fields in repeats]
        row = b.aggregate_point(samples, 100, 100, 400, SimpleNamespace(cache_mode="cold"))
        self.assertAlmostEqual(row["gpm_decode_sm_avg_pct"], 80.0)
        self.assertEqual(row["gpm_decode_suspect_phases"], 1)
        self.assertEqual(row["gpm_suspect_handling"], "excluded_from_sm_occupancy_tensor")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            b.print_live_row(row, drafting_on=False)
        self.assertIn("80/0/0/50*", out.getvalue())

    def test_request_level_dropout_also_excludes_low_graphics_zeros(self):
        # Prefill window: mostly idle-looking samples (graphics 3 %) with SM 0 plus
        # one clearly real SM value; decode: a qualifying dropout run.
        prefill = series(0, 4, graphics=3.0) + [gpm_row(1, 1.25, graphics=40.0, sm=35.0, occupancy=10.0)]
        decode = series(1.25, 6, graphics=97.0)
        store = self.store(prefill + decode)
        flagged = b.gpm_dropout_gpus(store, 0, 2.75)
        self.assertEqual(flagged, {0})

        alone = b.summarize_gpm(store, 0, 1.25)  # phase alone: nothing qualifies
        self.assertEqual(alone["gpm_suspect_samples"], 0)
        self.assertAlmostEqual(alone["gpm_sm_avg_pct"], 7.0)

        judged = b.summarize_gpm(store, 0, 1.25, dropout_gpus=flagged)
        self.assertEqual(judged["gpm_suspect_samples"], 4)
        self.assertEqual(judged["gpm_suspect_phases"], 1)
        self.assertEqual(judged["gpm_suspect_max_run_ms"], 0.0)  # no run inside the prefill itself
        self.assertAlmostEqual(judged["gpm_sm_avg_pct"], 35.0)  # the real nonzero value stays
        self.assertEqual(judged["gpm_sm_valid_samples"], 1)
        self.assertEqual(judged["gpm_graphics_valid_samples"], 5)

        kept = b.summarize_gpm(store, 0, 1.25, exclude_suspect=False, dropout_gpus=flagged)
        self.assertEqual(kept["gpm_suspect_samples"], 4)
        self.assertAlmostEqual(kept["gpm_sm_avg_pct"], 7.0)

    def test_dropout_is_judged_per_gpu(self):
        rows = sorted(series(0, 6, gpu="0") + series(0, 6, gpu="1", sm=50.0) +
                      series(1.5, 2, gpu="1", graphics=5.0), key=lambda r: r["interval_end_mono"])
        store = self.store(rows)
        flagged = b.gpm_dropout_gpus(store, 0, 2)
        self.assertEqual(flagged, {"0"})
        summary = b.summarize_gpm(store, 0, 2, dropout_gpus=flagged)
        # GPU 1's idle zeros are real (GPU 1 is not in dropout) and stay in the mean.
        self.assertEqual(summary["gpm_suspect_samples"], 6)
        self.assertAlmostEqual(summary["gpm_sm_avg_pct"], 37.5)

    def test_runner_judges_the_whole_request(self):
        calls = []

        class Monitor:
            def dropout_gpus(self, start, end):
                calls.append(("request", round(end - start, 1) >= 0))
                return {"0"}

            def summarize(self, start, end, dropout_gpus=None):
                calls.append(("phase", dropout_gpus))
                return b.empty_gpm()

            def register_phase_window(self, *args):
                pass

        http = SimpleNamespace(post=lambda url, json=None, timeout=None: SimpleNamespace(
            ok=True, status_code=200, json=lambda: {"timings": {"prompt_n": 5, "prompt_ms": 5,
                                                                "predicted_n": 2, "predicted_ms": 5}}))
        args = SimpleNamespace(cache_mode="incremental", repeat=1, slot_id=0, settle=0, n_predict=2,
                               deterministic=True, ignore_eos=True)
        runner = b.BenchmarkRunner(args, "http://fake", http, None, SimpleNamespace(check=lambda: None),
                                   None, Monitor(), None)
        runner.take_sample([1, 2, 3], 10, 0)
        self.assertEqual(calls[0][0], "request")
        self.assertGreaterEqual(len(calls), 2)
        self.assertEqual([c for c in calls[1:]], [("phase", {"0"})] * (len(calls) - 1))

    def test_monitor_flag_controls_exclusion(self):
        monitor = b.NvidiaGpmMonitor()
        monitor.samples.archive_enabled = False
        self.addCleanup(monitor.samples.close)
        monitor.samples.extend(series(0, 4) + series(1, 4, sm=60.0))
        self.assertAlmostEqual(monitor.summarize(0, 2)["gpm_sm_avg_pct"], 60.0)
        monitor.exclude_suspect = False
        self.assertAlmostEqual(monitor.summarize(0, 2)["gpm_sm_avg_pct"], 30.0)

    def test_handling_column_reflects_keep_option(self):
        sample = dict(cache_n=0, prompt_n=10, prompt_ms=10, prefill_tps=100, decode_tps=5,
                      predicted_n=8, predicted_ms=100, draft_n=0, draft_acc=0, wall_s=1,
                      truncated=False, status="OK")
        row = b.aggregate_point([sample], 1, 1, 1, SimpleNamespace(cache_mode="cold", gpm_suspect="keep"))
        self.assertEqual(row["gpm_suspect_handling"], "kept")
        self.assertIn("gpm_suspect_handling", b.RESULT_CSV_FIELDS)


class FakeNvml:
    NVML_PCIE_UTIL_RX_BYTES = 1
    NVML_PCIE_UTIL_TX_BYTES = 0

    def __init__(self, delay):
        self.delay = delay

    def nvmlDeviceGetPcieThroughput(self, handle, counter):
        time.sleep(self.delay)
        return 50000 if counter == self.NVML_PCIE_UTIL_RX_BYTES else 10000  # KB/s

    def nvmlDeviceGetCurrPcieLinkGeneration(self, handle):
        return 3

    def nvmlDeviceGetCurrPcieLinkWidth(self, handle):
        return 8


class LegacyPcieTimingTests(unittest.TestCase):
    def run_sampler(self, interval_ms, delay_s, duration_s=0.45):
        monitor = b.NvidiaVramMonitor(pcie_interval_ms=interval_ms)
        monitor.pcie_samples.archive_enabled = False
        self.addCleanup(monitor.pcie_samples.close)
        monitor._pynvml = FakeNvml(delay_s)
        monitor._nvml_handles = [("0", object())]
        thread = threading.Thread(target=monitor._pcie_reader)
        started = time.perf_counter()
        thread.start()
        time.sleep(duration_s)
        monitor.stop_event.set()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        rows, _ = monitor.pcie_samples.window(started, time.perf_counter() + 1)
        return monitor, rows

    def test_slow_queries_are_recorded_and_summarized(self):
        monitor, rows = self.run_sampler(interval_ms=20, delay_s=0.03)
        self.assertGreaterEqual(len(rows), 3)
        self.assertIsNone(rows[0]["pcie_poll_interval_ms"])
        for row in rows:
            self.assertGreaterEqual(row["pcie_query_ms"], 55)
            self.assertEqual((row["pcie_rx_mb_s"], row["pcie_tx_mb_s"]), (50.0, 10.0))
        for row in rows[1:]:
            self.assertGreaterEqual(row["pcie_poll_interval_ms"], row["pcie_query_ms"] - 1)
        summary = monitor.pcie_timing_summary()
        self.assertEqual(summary["requested_interval_ms"], 20)
        self.assertGreater(summary["achieved_interval_median_ms"], 50)
        self.assertEqual(summary["polls"], len(rows))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            b.print_pcie_timing(summary)
        self.assertIn("requested 20 ms, achieved median", out.getvalue())
        self.assertIn("the requested interval was not reached; use --pcie-interval-ms", out.getvalue())

    def test_sample_time_is_centre_of_query(self):
        before = time.perf_counter()
        _, rows = self.run_sampler(interval_ms=100, delay_s=0.02, duration_s=0.15)
        first = rows[0]
        self.assertGreaterEqual(first["t_mono"], before + 0.015)

    def test_reachable_interval_gives_no_hint(self):
        monitor, rows = self.run_sampler(interval_ms=100, delay_s=0.001, duration_s=0.55)
        summary = monitor.pcie_timing_summary()
        self.assertLess(summary["achieved_interval_median_ms"], 130)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            b.print_pcie_timing(summary)
        self.assertNotIn("not reached", out.getvalue())

    def test_no_samples_no_summary(self):
        monitor = b.NvidiaVramMonitor()
        self.addCleanup(monitor.pcie_samples.close)
        self.assertIsNone(monitor.pcie_timing_summary())
        with contextlib.redirect_stdout(io.StringIO()) as out:
            b.print_pcie_timing(None)
        self.assertEqual(out.getvalue(), "")

    def test_pcie_csv_has_timing_columns(self):
        self.assertIn("pcie_query_ms", b.PCIE_CSV_FIELDS)
        self.assertIn("pcie_poll_interval_ms", b.PCIE_CSV_FIELDS)


class CliDefaultsTests(unittest.TestCase):
    def test_defaults_and_choices(self):
        seen = []
        argv = [str(SCRIPT), "--file", str(SCRIPT), "--vram-log", "off", "--gpm-log", "off",
                "--win-gpu-mem", "off"]
        with patch("sys.argv", argv), patch.object(b, "run_benchmark",
                                                   side_effect=lambda args, *a: seen.append(args)):
            b.main()
        self.assertEqual((seen[0].pcie_interval_ms, seen[0].gpm_suspect), (250, "exclude"))
        with patch("sys.argv", argv + ["--gpm-suspect", "bogus"]), self.assertRaises(SystemExit), \
                contextlib.redirect_stderr(io.StringIO()):
            b.main()


if __name__ == "__main__":
    unittest.main()
