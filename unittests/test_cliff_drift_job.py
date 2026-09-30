"""Robust cliff analysis, drift check and Windows job binding (review of 2026-09-26)."""
import contextlib
import ctypes
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_csv_recording import SCRIPT, benchmark as b


def point(ctx, decode, low=None, high=None, prefill=1000.0, prompt_n=1000, pmin=None, pmax=None, repeats=3):
    return {"total_ctx": ctx, "target_ctx": ctx, "decode_tps_median": decode,
            "decode_tps_min": decode if low is None else low, "decode_tps_max": decode if high is None else high,
            "prefill_tps": prefill, "prefill_tps_min": prefill if pmin is None else pmin,
            "prefill_tps_max": prefill if pmax is None else pmax, "prompt_n": prompt_n,
            "valid_repeats": repeats, "prefill_valid_repeats": repeats}


def cliff_args(**overrides):
    values = dict(cliff_min_repeats=2, cliff_pct=15.0, repeat=3, cache_mode="incremental")
    values.update(overrides)
    return SimpleNamespace(**values)


class CliffAnalysisTests(unittest.TestCase):
    def test_all_confirmed_candidates_are_listed(self):
        rows = [point(1000, 100), point(2000, 80), point(3000, 78), point(4000, 50)]
        analysis = b.analyze_cliffs(rows, "decode_tps_median", 2, 15.0)
        self.assertEqual([(d["prev"]["total_ctx"], d["cur"]["total_ctx"]) for d in analysis["candidates"]],
                         [(3000, 4000), (1000, 2000)])
        with contextlib.redirect_stdout(io.StringIO()) as out:
            b.print_cliff_report("DECODE", analysis, cliff_args())
        text = out.getvalue()
        self.assertEqual(text.count("DECODE CLIFF CANDIDATE"), 2)
        self.assertEqual(text.count("-> verify with"), 2)
        self.assertIn("overall decode change ctx 1000 -> 4000: -50.0%", text)

    def test_overlapping_repeat_ranges_are_not_confirmed(self):
        # Median drops 20 %, but one noisy repeat makes the ranges overlap.
        rows = [point(1000, 100, low=70, high=105), point(2000, 80, low=75, high=90)]
        analysis = b.analyze_cliffs(rows, "decode_tps_median", 2, 15.0)
        self.assertEqual(analysis["candidates"], [])
        self.assertEqual(analysis["unconfirmed"][0]["reasons"], ["repeat ranges overlap"])
        with contextlib.redirect_stdout(io.StringIO()) as out:
            b.print_cliff_report("DECODE", analysis, cliff_args())
        self.assertIn("not counted as a cliff (repeat ranges overlap): 20.0%", out.getvalue())
        self.assertNotIn("CLIFF CANDIDATE", out.getvalue())

    def test_prefill_needs_comparable_prompt_n(self):
        rows = [point(10000, 50, prefill=2000, prompt_n=10000),
                point(12000, 50, prefill=1200, prompt_n=2000),   # small batch: slower by nature
                point(14000, 50, prefill=900, prompt_n=2200)]
        analysis = b.analyze_cliffs(rows, "prefill_tps", 2, 15.0)
        self.assertEqual([(d["prev"]["total_ctx"], d["reasons"]) for d in analysis["unconfirmed"]],
                         [(10000, ["prompt_n differs"])])
        self.assertEqual([(d["prev"]["total_ctx"], d["cur"]["total_ctx"]) for d in analysis["candidates"]],
                         [(12000, 14000)])

    def test_gradual_decline_is_reported(self):
        rows = [point(1000 * i, 100 - 4 * i) for i in range(1, 8)]
        analysis = b.analyze_cliffs(rows, "decode_tps_median", 2, 15.0)
        self.assertEqual(analysis["candidates"], [])
        with contextlib.redirect_stdout(io.StringIO()) as out:
            b.print_cliff_report("DECODE", analysis, cliff_args())
        self.assertIn("-> overall decode change ctx 1000 -> 7000: -25.0% (gradual, no single step >= 15%)",
                      out.getvalue())
        self.assertIn("below --cliff-pct", out.getvalue())
        self.assertNotIn("gradual decline", out.getvalue())

    def test_small_overall_change_has_no_gradual_hint(self):
        rows = [point(1000 * i, 100 - i) for i in range(1, 8)]
        with contextlib.redirect_stdout(io.StringIO()) as out:
            b.print_cliff_report("DECODE", b.analyze_cliffs(rows, "decode_tps_median", 2, 15.0), cliff_args())
        self.assertIn("-> overall decode change ctx 1000 -> 7000: -6.1%\n", out.getvalue())

    def test_low_quality_points_are_not_used(self):
        rows = [point(1000, 100), point(2000, 50, repeats=1), point(3000, 40)]
        analysis = b.analyze_cliffs(rows, "decode_tps_median", 2, 15.0)
        self.assertEqual(analysis["drops"], [])  # no adjacent usable pair
        self.assertEqual(analysis["overall"]["change_pct"], -60.0)

    def test_summary_digest(self):
        rows = [point(1000, 100), point(2000, 50)]
        digest = b.cliff_summary(b.analyze_cliffs(rows, "decode_tps_median", 2, 15.0))
        self.assertEqual(digest["candidates"], [{"from_ctx": 1000, "to_ctx": 2000, "drop_pct": 50.0,
                                                 "reasons": []}])
        json.dumps(digest)

    def test_prefill_range_is_aggregated(self):
        base = dict(cache_n=0, prompt_n=100, prompt_ms=100, decode_tps=5, predicted_n=8, predicted_ms=100,
                    draft_n=0, draft_acc=0, wall_s=1, truncated=False, status="OK")
        row = b.aggregate_point([dict(base, prefill_tps=v) for v in (900, 1000, 1100)], 1, 1, 1,
                                SimpleNamespace(cache_mode="cold"))
        self.assertEqual((row["prefill_tps_min"], row["prefill_tps"], row["prefill_tps_max"]), (900, 1000, 1100))
        self.assertIn("prefill_tps_min", b.RESULT_CSV_FIELDS)


class DriftTests(unittest.TestCase):
    def test_compare_drift(self):
        original = point(1000, 50.0, low=49.0, high=51.0, prefill=1000.0, pmin=990.0, pmax=1010.0)
        stable = b.compare_drift(original, point(1000, 50.5, prefill=1005.0))
        self.assertFalse(stable["drift"])
        drifted = b.compare_drift(original, point(1000, 45.0, prefill=1000.0))
        self.assertTrue(drifted["drift"])
        self.assertAlmostEqual(drifted["decode"]["change_pct"], -10.0)
        self.assertTrue(drifted["decode"]["outside_original_range"])
        # Large change but still within the original (noisy) range: not flagged.
        noisy = b.compare_drift(point(1000, 50.0, low=40.0, high=60.0), point(1000, 44.0))
        self.assertFalse(noisy["decode"]["drift"])
        with contextlib.redirect_stdout(io.StringIO()) as out:
            b.print_drift(drifted)
            b.print_drift(stable)
            b.print_drift({"error": "boom"})
            b.print_drift(None)
        text = out.getvalue()
        self.assertIn("decode 50.00 -> 45.00 tok/s (-10.0%)", text)
        self.assertIn("WARNING", text)
        self.assertIn("stable: no relevant drift", text)
        self.assertIn("drift check failed: boom", text)

    def run_main(self, completion, *extra):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "in.txt").write_text("text " * 10000, encoding="utf-8")
            argv = [str(SCRIPT), "--file", str(root / "in.txt"), "--csv", str(root / "run.csv"),
                    "--start", "100", "--end", "300", "--step", "100", "--n-predict", "8", "--repeat", "1",
                    "--warmup", "0", "--vram-log", "off", "--gpm-log", "off", "--win-gpu-mem", "off",
                    "--settle", "0", *extra]
            with patch("sys.argv", argv), patch.object(b, "server_is_ready", return_value=True), \
                    patch.object(b, "detect_slot_n_ctx", return_value=10000), \
                    patch.object(b, "tokenize", side_effect=lambda base, text, **kw: [0] * len(text)), \
                    patch.object(b, "reset_slot", return_value=True), \
                    patch.object(b, "fetch_server_props", return_value={}), \
                    patch.object(b, "completion", side_effect=completion), \
                    contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
                b.main()
            meta = json.loads((root / "run.meta.json").read_text(encoding="utf-8"))
            rows = (root / "run.csv").read_text(encoding="utf-8").splitlines()
            samples = (root / "run.samples.csv").read_text(encoding="utf-8").splitlines()
        return out.getvalue(), meta, rows, samples

    def test_first_point_is_remeasured_without_csv_rows(self):
        calls = []

        def complete(*a, **k):
            calls.append(k.get("cache_prompt"))
            predicted_ms = 160 if len(calls) == 4 else 80  # 4th request = drift check, slower
            return {"timings": {"predicted_n": 8, "predicted_ms": predicted_ms, "prompt_n": 100,
                                "prompt_ms": 50}, "content": "x"}

        out, meta, rows, samples = self.run_main(complete)
        self.assertEqual(len(calls), 4)
        self.assertFalse(calls[3])  # like the first point: no prompt reuse
        self.assertEqual(len(rows), 1 + 3)
        self.assertEqual(len(samples), 1 + 3)
        # Result directly below the announcement, before the cliff report.
        announce = out.index("drift check: re-measuring the first point (target=100) ...")
        result = out.index("decode ", announce)
        self.assertLess(result, out.index("PREFILL"))
        self.assertIn("WARNING", out[result:out.index("PREFILL")])
        self.assertTrue(meta["drift_check"]["drift"])
        self.assertIn("cliffs", meta)

    def test_managed_server_stops_after_drift_check_before_the_reports(self):
        stdout = io.StringIO()
        stopped_at = []

        class FakeServer:
            def __init__(self, command, **kwargs):
                self.command = command

            def start(self):
                pass

            def stop(self):
                if not stopped_at:
                    stopped_at.append(stdout.getvalue())

        def complete(*a, **k):
            return {"timings": {"predicted_n": 8, "predicted_ms": 80, "prompt_n": 100, "prompt_ms": 50}}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "in.txt").write_text("text " * 10000, encoding="utf-8")
            argv = [str(SCRIPT), "--file", str(root / "in.txt"), "--server-command", "llama-server --port 8080",
                    "--server-log", str(root / "server.log"),
                    "--start", "100", "--end", "300", "--step", "100", "--n-predict", "8", "--repeat", "1",
                    "--warmup", "0", "--vram-log", "off", "--gpm-log", "off", "--win-gpu-mem", "off",
                    "--settle", "0"]
            with patch("sys.argv", argv), patch.object(b, "server_is_ready", return_value=False), \
                    patch.object(b, "ManagedLlamaServer", FakeServer), \
                    patch.object(b, "settle_vram_before_server_start", return_value=None), \
                    patch.object(b, "detect_slot_n_ctx", return_value=10000), \
                    patch.object(b, "tokenize", side_effect=lambda base, text, **kw: [0] * len(text)), \
                    patch.object(b, "reset_slot", return_value=True), \
                    patch.object(b, "fetch_server_props", return_value={}), \
                    patch.object(b, "completion", side_effect=complete), \
                    contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                b.main()
        self.assertEqual(len(stopped_at), 1)
        self.assertIn("-> stable: no relevant drift", stopped_at[0])
        self.assertNotIn("PREFILL", stopped_at[0])  # reports come after the server was stopped
        self.assertIn("PREFILL", stdout.getvalue())

    def test_drift_check_can_be_disabled(self):
        calls = []

        def complete(*a, **k):
            calls.append(1)
            return {"timings": {"predicted_n": 8, "predicted_ms": 80, "prompt_n": 100, "prompt_ms": 50}}

        out, meta, _, _ = self.run_main(complete, "--no-drift-check")
        self.assertEqual(len(calls), 3)
        self.assertNotIn("drift check", out)
        self.assertIsNone(meta["drift_check"])


class FakeKernel32:
    def __init__(self, create=1234, set_ok=1, assign_ok=1):
        self.create, self.set_ok, self.assign_ok = create, set_ok, assign_ok
        self.calls = []

    def CreateJobObjectW(self, attributes, name):
        self.calls.append("create")
        return self.create

    def SetInformationJobObject(self, job, info_class, info, size):
        info_struct = ctypes.cast(info, ctypes.POINTER(ctypes.c_longlong * 3)).contents
        self.limit_flags = info_struct[2] & 0xFFFFFFFF
        self.calls.append(("set", info_class, size))
        return self.set_ok

    def AssignProcessToJobObject(self, job, process):
        self.calls.append(("assign", process))
        return self.assign_ok

    def CloseHandle(self, handle):
        self.calls.append(("close", handle))
        return 1


class JobObjectTests(unittest.TestCase):
    def test_job_with_kill_on_close_is_created_and_assigned(self):
        kernel32 = FakeKernel32()
        self.assertEqual(b.bind_to_kill_job(99, kernel32), 1234)
        self.assertEqual(kernel32.limit_flags, b.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)
        self.assertEqual(kernel32.calls[1][1], b.JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS)
        self.assertIn(("assign", 99), kernel32.calls)
        self.assertNotIn(("close", 1234), kernel32.calls)

    def test_failures_close_the_job_and_return_none(self):
        self.assertIsNone(b.bind_to_kill_job(99, FakeKernel32(create=0)))
        for kernel32 in (FakeKernel32(set_ok=0), FakeKernel32(assign_ok=0)):
            self.assertIsNone(b.bind_to_kill_job(99, kernel32))
            self.assertIn(("close", 1234), kernel32.calls)

    def start_server(self, kill_with_script, bind_result):
        proc = Mock()
        proc.poll.return_value = None
        proc._handle = 42
        server = b.ManagedLlamaServer("llama-server --port 8080", "http://127.0.0.1:8080", 5, None,
                                      http=SimpleNamespace(get=lambda *a, **k: SimpleNamespace(ok=True)),
                                      kill_with_script=kill_with_script)
        bind = Mock(return_value=bind_result)
        with patch.object(b.subprocess, "Popen", return_value=proc), patch.object(b, "bind_to_kill_job", bind), \
                patch.object(b.os, "name", "nt"), contextlib.redirect_stderr(io.StringIO()) as err:
            server.start()
        return server, bind, err.getvalue()

    def test_managed_server_is_bound_unless_kept(self):
        server, bind, err = self.start_server(True, 777)
        bind.assert_called_once_with(42)
        self.assertEqual(server.job, 777)
        server, bind, err = self.start_server(False, 777)
        bind.assert_not_called()
        self.assertIsNone(server.job)

    def test_binding_failure_only_warns(self):
        server, bind, err = self.start_server(True, None)
        self.assertIn("could not bind llama-server", err)
        self.assertIsNone(server.job)


if __name__ == "__main__":
    unittest.main()
