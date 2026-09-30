"""Regression coverage for the review fixes around outputs and failures.

These tests deliberately exercise ``main`` through the same small offline
harness used by ``test_csv_recording``.  ``RecordingTests`` is composed into
the test class instead of being subclassed, so its tests are not discovered a
second time.
"""

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from test_csv_recording import benchmark, read_rows


def _recording_test_class():
    """Load the helper lazily so unittest does not rediscover its tests here."""
    from test_csv_recording import RecordingTests
    return RecordingTests


class ReviewFixTests(unittest.TestCase):
    def setUp(self):
        # RecordingTests.setUp only prepares a temporary directory, arguments,
        # and the standard output patches; composition avoids inherited tests.
        _recording_test_class().setUp(self)

    def _run_main(self, completion, repeat=1):
        argv = [
            str(Path(benchmark.__file__)), "--file", str(self.input),
            "--csv", str(self.csv), "--start", "100", "--end", "200",
            "--step", "100", "--n-predict", "8", "--repeat", str(repeat),
            "--warmup", "0", "--vram-log", "off", "--gpm-log", "off",
            "--win-gpu-mem", "off", "--settle", "0",
        ]

        def ready(*args, **kwargs):
            self.assertEqual(
                read_rows(self.csv),
                (list(benchmark.RESULT_CSV_FIELDS), []),
            )
            return True

        with patch("sys.argv", argv), \
             patch.object(benchmark, "server_is_ready", side_effect=ready), \
             patch.object(benchmark, "detect_slot_n_ctx", return_value=10000), \
             patch.object(benchmark, "tokenize", side_effect=lambda base, text, **kw: [0] * len(text)), \
             patch.object(benchmark, "reset_slot", return_value=True), \
             patch.object(benchmark, "probe_prefill_repeats",
                          side_effect=lambda args, *a, **kw: setattr(args, "prefill_repeat_enabled", False)), \
             patch.object(benchmark, "completion", side_effect=completion):
            benchmark.main()

    @staticmethod
    def _ok_response():
        return {
            "timings": {
                "cache_n": 0,
                "prompt_n": 100,
                "prompt_ms": 50,
                "predicted_n": 8,
                "predicted_ms": 80,
            }
        }

    def test_server_log_collision_is_rejected_without_results_csv(self):
        args = type(self.args)(**vars(self.args))
        args.csv = None

        hardlink = self.root / "input-hardlink.txt"
        server_logs = [self.input]
        try:
            os.link(self.input, hardlink)
        except OSError:
            pass
        else:
            server_logs.append(hardlink)

        for server_log in server_logs:
            with self.subTest(server_log=server_log):
                args.server_log = str(server_log)
                with self.assertRaises(ValueError):
                    benchmark.CsvRecording(args)

    def test_http_500_on_first_measurement_exits_one_without_retry(self):
        calls = []

        def fail(*args, **kwargs):
            calls.append(True)
            raise benchmark.CompletionRequestError(500, "server failure")

        with self.assertRaises(SystemExit) as raised:
            self._run_main(fail)
        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(read_rows(self.csv)[1], [])

    def test_http_500_after_a_completed_measurement_exits_one_without_retry(self):
        calls = []

        def complete(*args, **kwargs):
            calls.append(True)
            if len(calls) == 2:
                raise benchmark.CompletionRequestError(500, "server failure")
            return self._ok_response()

        with self.assertRaises(SystemExit) as raised:
            self._run_main(complete)
        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(read_rows(self.csv)[1]), 1)

    def test_context_overflow_on_first_measurement_exits_one(self):
        calls = []

        def overflow(*args, **kwargs):
            calls.append(True)
            raise benchmark.CompletionRequestError(
                400, "context size exceeded", "exceed_context_size"
            )

        with self.assertRaises(SystemExit) as raised:
            self._run_main(overflow)
        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(read_rows(self.csv)[1], [])

    def test_context_overflow_after_a_completed_measurement_keeps_results(self):
        calls = []

        def complete(*args, **kwargs):
            calls.append(True)
            if len(calls) == 2:
                raise benchmark.CompletionRequestError(
                    400, "context size exceeded", "exceed_context_size"
                )
            return self._ok_response()

        try:
            self._run_main(complete)
        except SystemExit as raised:
            self.assertEqual(raised.exception.code, 0)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(read_rows(self.csv)[1]), 1)

    def test_write_sample_creates_and_flushes_individual_measurement_csv(self):
        row = {
            "target_ctx": 100,
            "total_ctx": 100,
            "target_chars": 400,
            "cache_mode": "incremental",
            "repeat": 1,
            "status": "OK",
            "prompt_n": 100,
            "prompt_ms": 50,
            "predicted_n": 8,
            "predicted_ms": 80,
        }
        with benchmark.CsvRecording(self.args) as recording:
            recording.write_sample(row)
            sample_path = self.root / "run.samples.csv"
            fields, rows = read_rows(sample_path)
            self.assertEqual(len(rows), 1)
            self.assertTrue({
                "target_ctx", "total_ctx", "target_chars", "cache_mode",
                "repeat", "status", "prompt_n", "prompt_ms",
                "predicted_n", "predicted_ms",
            } <= set(fields))
            self.assertEqual(rows[0]["repeat"], "1")
            self.assertEqual(rows[0]["status"], "OK")
            self.assertEqual(rows[0]["predicted_n"], "8")

    def test_samples_are_written_per_repeat_and_survive_later_failure(self):
        calls = []

        def complete(*args, **kwargs):
            calls.append(True)
            if len(calls) == 2:
                raise benchmark.CompletionRequestError(500, "server failure")
            return self._ok_response()

        with self.assertRaises(SystemExit) as raised:
            self._run_main(complete, repeat=2)
        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(len(calls), 2)

        sample_path = self.root / "run.samples.csv"
        fields, rows = read_rows(sample_path)
        self.assertTrue({
            "target_ctx", "total_ctx", "target_chars", "cache_mode",
            "repeat", "status", "prompt_n", "prompt_ms",
            "predicted_n", "predicted_ms",
        } <= set(fields))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["repeat"], "1")
        self.assertEqual(rows[0]["status"], "OK")
        self.assertEqual(rows[0]["predicted_n"], "8")
        self.assertEqual(rows[0]["predicted_ms"], "80.0")

    def test_inferred_samples_csv_is_checked_for_input_collisions(self):
        collision_input = self.root / "run.samples.csv"
        collision_input.write_text("input that must survive", encoding="utf-8")
        args = type(self.args)(**vars(self.args))
        args.file = str(collision_input)
        with self.assertRaises(ValueError):
            benchmark.CsvRecording(args)
        self.assertEqual(collision_input.read_text(encoding="utf-8"), "input that must survive")
        self.assertFalse(self.csv.exists())


if __name__ == "__main__":
    unittest.main()
