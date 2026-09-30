"""Decode steps with MTP/drafting: tokens per step (content) vs ms per step (context)."""
import contextlib
import io
from types import SimpleNamespace
import unittest

from test_csv_recording import benchmark as b
from test_measurement import sample


def point(ctx, tps, tokens_per_step, ms, spread=0.01):
    return dict(total_ctx=ctx, valid_repeats=2, decode_tps_median=tps,
                decode_tps_min=tps * (1 - spread), decode_tps_max=tps * (1 + spread),
                tokens_per_step_median=tokens_per_step, ms_per_step_median=ms,
                ms_per_step_min=ms * (1 - spread), ms_per_step_max=ms * (1 + spread))


class StepStatsTests(unittest.TestCase):
    def test_without_drafting_a_step_is_a_token(self):
        self.assertEqual(b.decode_step_stats(512, 20000.0, 0),
                         {"decode_steps": 512, "tokens_per_step": 1.0, "ms_per_step": 20000.0 / 512})

    def test_accepted_drafts_add_tokens_to_a_step(self):
        stats = b.decode_step_stats(512, 14000.0, 256)
        self.assertEqual((stats["decode_steps"], stats["tokens_per_step"]), (256, 2.0))
        self.assertAlmostEqual(stats["ms_per_step"], 14000.0 / 256)

    def test_invalid_timings_give_no_values(self):
        for args in ((0, 100.0, 0), (10, 0.0, 0), (10, 100.0, 10)):
            self.assertIsNone(b.decode_step_stats(*args)["ms_per_step"])

    def test_point_aggregates_valid_decode_repeats_only(self):
        samples = [sample(tokens_per_step=2.0, ms_per_step=60.0),
                   sample(tokens_per_step=2.4, ms_per_step=62.0),
                   sample(status="STOP@3", tokens_per_step=9.0, ms_per_step=999.0)]
        row = b.aggregate_point(samples, 100, 100, 400, SimpleNamespace(cache_mode="cold"))
        self.assertEqual((row["tokens_per_step_median"], row["ms_per_step_median"],
                          row["ms_per_step_min"], row["ms_per_step_max"]), (2.2, 61.0, 60.0, 62.0))

    def test_csv_fields_contain_step_values(self):
        for field in ("decode_steps", "tokens_per_step", "ms_per_step"):
            self.assertIn(field, b.SAMPLE_CSV_FIELDS)
        for field in ("tokens_per_step_median", "ms_per_step_median", "ms_per_step_min", "ms_per_step_max"):
            self.assertIn(field, b.RESULT_CSV_FIELDS)


class StepCliffTests(unittest.TestCase):
    def test_decode_drop_from_fewer_tokens_per_step_is_a_content_effect(self):
        # Real run 173539: 70000 -> 80000, decode -21 %, step cost only +5.7 %.
        rows = [point(70000, 30.3, 2.47, 81.1), point(80000, 24.0, 2.06, 85.7)]
        analysis = b.analyze_cliffs(rows, "decode_tps_median", 2, 15.0)
        self.assertEqual(analysis["candidates"], [])
        self.assertIn("content effect: tokens/step 2.47->2.06", analysis["unconfirmed"][0]["reasons"][0])

    def test_decode_drop_with_rising_step_cost_stays_a_candidate(self):
        rows = [point(70000, 30.3, 2.3, 80.0), point(80000, 22.0, 2.3, 110.0)]
        analysis = b.analyze_cliffs(rows, "decode_tps_median", 2, 15.0)
        self.assertEqual(len(analysis["candidates"]), 1)

    def test_without_drafting_the_content_check_does_not_apply(self):
        rows = [point(70000, 20.0, 1.0, 50.0), point(80000, 15.0, 1.0, 51.0)]
        self.assertEqual(len(b.analyze_cliffs(rows, "decode_tps_median", 2, 15.0)["candidates"]), 1)

    def test_step_rate_rows_and_their_cliff_analysis(self):
        rows = b.step_rate_rows([point(10000, 36.0, 2.1, 50.0), point(20000, 35.0, 2.6, 80.0)])
        self.assertAlmostEqual(rows[0]["decode_step_rate"], 20.0)
        self.assertAlmostEqual(rows[0]["decode_step_rate_min"], 1000.0 / 50.5)
        self.assertAlmostEqual(rows[0]["decode_step_rate_max"], 1000.0 / 49.5)
        analysis = b.analyze_cliffs(rows, "decode_step_rate", 2, 15.0)
        self.assertEqual(len(analysis["candidates"]), 1)  # hidden in decode tok/s by more tokens per step
        with contextlib.redirect_stdout(io.StringIO()) as out:
            b.print_cliff_report("DECODE STEP RATE", analysis, SimpleNamespace(
                cliff_min_repeats=2, cliff_pct=15.0, repeat=2))
        self.assertIn("steps/s", out.getvalue())

    def test_live_row_shows_step_cost_only_with_mtp(self):
        row = {"total_ctx": 10000, "prompt_n": 10000, "prefill_tps": 850.0, "decode_tps_median": 36.3,
               "draft_acc_pct": 55.0, "ms_per_step_median": 57.45, "status": "OK"}
        with contextlib.redirect_stdout(io.StringIO()) as out:
            b.print_live_row(row, True)
            b.print_live_row(row, False)
        with_mtp, without = out.getvalue().splitlines()
        self.assertIn("57.5", with_mtp)
        self.assertNotIn("57.5", without)


class MtpRunTests(unittest.TestCase):
    def test_mtp_run_shows_step_column_and_step_rate_analysis(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "input.txt"
            source.write_text("x" * 5000, encoding="utf-8")

            def complete(base, prompt, n_predict, *args, **kwargs):
                return {"timings": {"cache_n": 0, "prompt_n": len(prompt), "prompt_ms": 10,
                                    "predicted_n": n_predict, "predicted_ms": 400 + len(prompt),
                                    "draft_n": 8, "draft_n_accepted": 4}}

            argv = ["ctx-cliff.py", "--file", str(source), "--start", "100", "--end", "300", "--step", "100",
                    "--n-predict", "8", "--repeat", "2", "--warmup", "1", "--cache-mode", "cold",
                    "--settle", "0", "--vram-log", "off", "--gpm-log", "off", "--win-gpu-mem", "off",
                    "--no-drift-check"]
            meta = {}
            with patch.object(b.sys, "argv", argv), \
                 patch.object(b, "tokenize", side_effect=lambda base, text, add_bos=True, **kw: [1] * (len(text) + 1)), \
                 patch.object(b, "completion", side_effect=complete), \
                 patch.object(b, "server_is_ready", return_value=True), \
                 patch.object(b, "detect_slot_n_ctx", return_value=4096), \
                 patch.object(b, "reset_slot", return_value=True), \
                 patch.object(b, "probe_prefill_repeats"), \
                 patch.object(b.RunMetadata, "update", lambda self, **fields: meta.update(fields)), \
                 contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
                b.main()
        text = out.getvalue()
        self.assertIn("step ms = decode cost per verification step", text)
        self.assertIn("DECODE STEP RATE", text)
        self.assertEqual(set(meta["cliffs"]), {"prefill", "decode", "decode_step_rate"})
        # 8 tokens, 4 accepted -> 4 steps; predicted_ms = 400 + 100 at the first point
        self.assertIn("125.0", text.split("---")[-1])


if __name__ == "__main__":
    unittest.main()
