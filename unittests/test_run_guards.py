"""--sysmem-guard and --abort-below-pct: stop a sweep whose further points are meaningless."""
import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_csv_recording import benchmark as b
from test_measurement import sample


def runner(**options):
    args = SimpleNamespace(cache_mode="cold", slot_id=0, settle=0, n_predict=8, deterministic=False,
                           ignore_eos=False, repeat=2, **options)
    builder = b.PromptBuilder("x" * 5000, "", lambda text: [0] * len(text), 1, 0)
    return b.BenchmarkRunner(args, "http://test", object(), builder, Mock(), None, None, None)


def spilled(**extra):
    return sample(pcie_prefill_rx_median_mb_s=11000.0, prefill_tps=60, **extra)


class SignalTests(unittest.TestCase):
    def test_highest_available_prefill_rx_is_used(self):
        self.assertEqual(b.sysmem_fallback_signal({"pcie_prefill_rx_median_mb_s": 20.0,
                                                   "gpm_prefill_pcie_rx_median_mib_s": 9000.0}), 9000.0)
        self.assertEqual(b.sysmem_fallback_signal({"pcie_prefill_rx_median_mb_s": 18.5}), 18.5)
        self.assertIsNone(b.sysmem_fallback_signal({"pcie_prefill_rx_median_mb_s": None}))


class SysmemGuardTests(unittest.TestCase):
    def test_abort_warn_off_and_threshold(self):
        abort = runner(sysmem_guard="abort", sysmem_guard_mb_s=1000.0)
        self.assertIn("shared system memory", abort.check_sysmem_fallback(spilled(), 20000, 0))
        self.assertEqual(abort.check_sysmem_fallback(sample(pcie_prefill_rx_median_mb_s=21.0), 20000, 0), "")
        self.assertEqual(runner(sysmem_guard="abort", sysmem_guard_mb_s=20000.0)
                         .check_sysmem_fallback(spilled(), 20000, 0), "")
        self.assertEqual(runner(sysmem_guard="off").check_sysmem_fallback(spilled(), 20000, 0), "")
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(runner(sysmem_guard="warn", sysmem_guard_mb_s=1000.0)
                             .check_sysmem_fallback(spilled(), 20000, 0), "")
        self.assertIn("NOTE", err.getvalue())

    def test_missing_telemetry_is_reported_once(self):
        guard = runner(sysmem_guard="abort", sysmem_guard_mb_s=1000.0)
        with contextlib.redirect_stderr(io.StringIO()) as err:
            for _ in range(3):
                self.assertEqual(guard.check_sysmem_fallback(sample(), 10000, 0), "")
        self.assertEqual(err.getvalue().count("inactive"), 1)

    def test_measure_point_records_the_sample_and_stops_after_first_spill(self):
        guard = runner(sysmem_guard="abort", sysmem_guard_mb_s=1000.0)
        with patch.object(b, "tokenize", return_value=list(range(100))), \
             patch.object(guard, "take_sample", side_effect=[spilled(), sample()]) as take:
            with self.assertRaises(b.SysmemFallbackError):
                guard.measure_point(100)
        self.assertEqual(take.call_count, 1)
        written = guard.recording.write_sample.call_args.args[0]
        self.assertIn("MB/s", written["validation_error"])
        guard.recording.write_result.assert_not_called()


class PrefillFloorTests(unittest.TestCase):
    def test_floor_is_relative_to_first_recorded_point(self):
        floor = runner(abort_below_pct=25.0)
        with patch.object(b, "tokenize", return_value=list(range(100))), \
             patch.object(floor, "take_sample", side_effect=[sample(prefill_tps=800)] * 2
                          + [sample(prefill_tps=400)] * 2 + [sample(prefill_tps=150)]):
            floor.measure_point(100)
            self.assertEqual(floor.reference_prefill_tps, 800)
            with patch.object(b, "tokenize", return_value=list(range(200))):
                floor.measure_point(200)  # 50 %: normal decline
            with patch.object(b, "tokenize", return_value=list(range(300))), \
                 self.assertRaises(b.PrefillFloorError) as raised:
                floor.measure_point(300)  # 19 %
        self.assertIn("19%", str(raised.exception))
        self.assertEqual(raised.exception.stop_reason, "prefill_floor")

    def test_zero_or_missing_setting_disables_the_floor(self):
        for floor in (runner(abort_below_pct=0.0), runner()):
            floor.reference_prefill_tps = 800
            self.assertEqual(floor.check_prefill_floor(sample(prefill_tps=10), 90000, 0), "")


class SweepStopTests(unittest.TestCase):
    def test_guard_stops_sweep_keeps_completed_points_and_skips_drift_check(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "input.txt"
            source.write_text("x" * 5000, encoding="utf-8")
            calls = []

            def complete(base, prompt, n_predict, *args, **kwargs):
                calls.append(len(prompt))
                return {"timings": {"cache_n": 0, "prompt_n": len(prompt), "prompt_ms": 10,
                                    "predicted_n": n_predict, "predicted_ms": 10}}

            def guard(self, sample, ctx, repeat_idx):
                return "spill" if ctx >= 300 else ""

            argv = ["ctx-cliff.py", "--file", str(source), "--start", "100", "--end", "400", "--step", "100",
                    "--n-predict", "8", "--repeat", "1", "--warmup", "0", "--cache-mode", "cold",
                    "--settle", "0", "--vram-log", "off", "--gpm-log", "off", "--win-gpu-mem", "off",
                    "--cliff-min-repeats", "1"]
            meta = {}
            with patch.object(b.sys, "argv", argv), \
                 patch.object(b, "tokenize", side_effect=lambda base, text, add_bos=True, **kw: [1] * (len(text) + 1)), \
                 patch.object(b, "completion", side_effect=complete), \
                 patch.object(b, "server_is_ready", return_value=True), \
                 patch.object(b, "detect_slot_n_ctx", return_value=4096), \
                 patch.object(b, "reset_slot", return_value=True), \
                 patch.object(b, "probe_prefill_repeats"), \
                 patch.object(b.BenchmarkRunner, "check_sysmem_fallback", guard), \
                 patch.object(b.RunMetadata, "update", lambda self, **fields: meta.update(fields)), \
                 contextlib.redirect_stdout(io.StringIO()) as out, \
                 contextlib.redirect_stderr(io.StringIO()) as err, \
                 self.assertRaises(SystemExit) as raised:
                b.main()
        self.assertEqual(raised.exception.code, 1)
        self.assertEqual((meta["stop_reason"], meta["status"], meta["completed_points"]),
                         ("sysmem_fallback", "stopped_sysmem_fallback", 2))
        self.assertIsNone(meta["drift_check"])
        self.assertEqual(len(calls), 3)  # 100, 200 and the one spilled sample at 300
        self.assertIn("SysmemFallbackError: spill", err.getvalue())
        self.assertIn("RUN ENDED EARLY at target=300", out.getvalue())


class SweepNotesTests(unittest.TestCase):
    def test_notes_from_the_sweep_appear_below_the_table(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "input.txt"
            source.write_text("x" * 5000, encoding="utf-8")

            def complete(base, prompt, n_predict, *args, **kwargs):
                predicted = 3 if len(prompt) == 200 else n_predict  # early EOS at the second point
                return {"timings": {"cache_n": 0, "prompt_n": len(prompt), "prompt_ms": 10,
                                    "predicted_n": predicted, "predicted_ms": 10},
                        "stop_type": "eos" if predicted < n_predict else "limit"}

            argv = ["ctx-cliff.py", "--file", str(source), "--start", "100", "--end", "300", "--step", "100",
                    "--n-predict", "8", "--repeat", "1", "--warmup", "0", "--cache-mode", "cold",
                    "--settle", "0", "--vram-log", "off", "--gpm-log", "off", "--win-gpu-mem", "off",
                    "--no-drift-check", "--cliff-min-repeats", "1"]
            meta = {}
            with patch.object(b.sys, "argv", argv), \
                 patch.object(b, "tokenize", side_effect=lambda base, text, add_bos=True, **kw: [1] * (len(text) + 1)), \
                 patch.object(b, "completion", side_effect=complete), \
                 patch.object(b, "server_is_ready", return_value=True), \
                 patch.object(b, "detect_slot_n_ctx", return_value=4096), \
                 patch.object(b, "reset_slot", return_value=True), \
                 patch.object(b, "probe_prefill_repeats"), \
                 patch.object(b.RunMetadata, "update", lambda self, **fields: meta.update(fields)), \
                 contextlib.redirect_stdout(io.StringIO()) as out, \
                 contextlib.redirect_stderr(io.StringIO()) as err:
                b.main()
        text = out.getvalue()
        self.assertNotIn("STOP@3", err.getvalue())
        table_end = text.index("=" * 20, text.index("ctx"))
        notes = text.index("NOTES during the sweep (1):")
        self.assertGreater(notes, table_end)
        self.assertIn("  - target=200 repeat=1: STOP@3; decode sample excluded", text)
        self.assertEqual(len(meta["sweep_notes"]), 1)


class GuardCliTests(unittest.TestCase):
    def parse(self, *arguments):
        captured = {}
        argv = ["ctx-cliff.py", "--file", "input.txt", *arguments]
        with patch.object(b.sys, "argv", argv), \
             patch.object(b, "run_benchmark", side_effect=lambda args, *_: captured.update(args=args)), \
             patch.object(b, "termination_handler", return_value=contextlib.nullcontext()), \
             patch.object(b, "CsvRecording", return_value=contextlib.nullcontext()), \
             contextlib.redirect_stderr(io.StringIO()):
            b.main()
        return captured["args"]

    def test_defaults_and_validation(self):
        args = self.parse()
        self.assertEqual((args.sysmem_guard, args.sysmem_guard_mb_s, args.abort_below_pct),
                         ("abort", 1000.0, 20.0))
        self.assertEqual(self.parse("--abort-below-pct", "0").abort_below_pct, 0.0)
        for bad in (("--abort-below-pct", "-1"), ("--abort-below-pct", "100"), ("--sysmem-guard-mb-s", "0")):
            with self.subTest(bad=bad), self.assertRaises(SystemExit):
                self.parse(*bad)


if __name__ == "__main__":
    unittest.main()
