"""Offline tests for CLI parsing and validation boundaries."""

import contextlib
import io
import sys
import unittest
from unittest.mock import patch

from test_csv_recording import benchmark as b


class CliBoundaryTests(unittest.TestCase):
    def invoke(self, *arguments):
        """Run main() through argument parsing without starting a benchmark."""
        captured = {}

        def capture(args, _parser, _resources, _recording):
            captured["args"] = args

        argv = ["ctx-cliff.py", "--file", "input.txt", *arguments]
        with patch.object(sys, "argv", argv), \
             patch.object(b, "run_benchmark", side_effect=capture), \
             patch.object(b, "termination_handler", return_value=contextlib.nullcontext()), \
             patch.object(b, "CsvRecording", return_value=contextlib.nullcontext()):
            b.main()
        return captured["args"]

    def assert_rejected(self, *arguments, message=None):
        argv = ["ctx-cliff.py", "--file", "input.txt", *arguments]
        with patch.object(sys, "argv", argv), \
             patch.object(b, "termination_handler", return_value=contextlib.nullcontext()), \
             patch.object(b, "CsvRecording", return_value=contextlib.nullcontext()), \
             contextlib.redirect_stderr(io.StringIO()), \
             self.assertRaises(SystemExit) as raised:
            b.main()
        self.assertEqual(raised.exception.code, 2)

    def test_defaults_and_csv_alias_are_normalized(self):
        args = self.invoke("--csv-export", "results.csv")
        self.assertEqual(args.cache_mode, "incremental")
        self.assertEqual(args.csv, "results.csv")
        self.assertEqual(args.start, 10000)
        self.assertEqual(args.end, 120000)
        self.assertEqual(args.step, 5000)
        self.assertEqual(args.vram_log, "auto")
        self.assertEqual(args.gpm_log, "auto")
        self.assertEqual(args.win_gpu_mem, "auto")
        self.assertEqual(args.cliff_min_repeats, 2)

    def test_csv_without_path_uses_empty_sentinel_before_allocation(self):
        with patch.object(b, "allocate_csv_path", return_value="allocated.csv") as allocate:
            args = self.invoke("--csv")
        self.assertEqual(args.csv, "allocated.csv")
        allocate.assert_called_once_with("outputs")

    def test_all_numeric_lower_boundaries_are_accepted(self):
        args = self.invoke(
            "--start", "1", "--end", "1", "--step", "1", "--n-predict", "1",
            "--repeat", "1", "--warmup", "0", "--cliff-pct", "0",
            "--vram-interval-ms", "100", "--pcie-interval-ms", "20",
            "--gpm-interval-ms", "101", "--win-gpu-mem-interval-ms", "1000",
            "--server-start-timeout", "0.001",
        )
        self.assertEqual((args.start, args.end, args.step), (1, 1, 1))
        self.assertEqual((args.n_predict, args.repeat, args.warmup), (1, 1, 0))
        self.assertEqual((args.vram_interval_ms, args.pcie_interval_ms), (100, 20))
        self.assertEqual((args.gpm_interval_ms, args.win_gpu_mem_interval_ms), (101, 1000))

    def test_invalid_context_range_is_rejected(self):
        for arguments in (("--start", "0"), ("--start", "2", "--end", "1"),
                          ("--step", "0"), ("--step", "-1")):
            with self.subTest(arguments=arguments):
                self.assert_rejected(*arguments)

    def test_invalid_measurement_values_are_rejected(self):
        for option, value in (("--n-predict", "0"), ("--repeat", "0"),
                              ("--warmup", "-1"), ("--cliff-pct", "-0.1"),
                              ("--cliff-min-repeats", "0")):
            with self.subTest(option=option):
                self.assert_rejected(option, value)

    def test_invalid_telemetry_intervals_are_rejected(self):
        for option, value in (("--vram-interval-ms", "99"),
                              ("--pcie-interval-ms", "19"),
                              ("--gpm-interval-ms", "100"),
                              ("--win-gpu-mem-interval-ms", "999"),
                              ("--server-start-timeout", "0"),
                              ("--server-start-timeout", "-1")):
            with self.subTest(option=option):
                self.assert_rejected(option, value)

    def test_empty_or_whitespace_server_command_is_rejected(self):
        for command in ("", "   ", "\t\n"):
            with self.subTest(command=repr(command)):
                self.assert_rejected("--server-command", command)

    def test_server_command_is_preserved_as_one_complete_string(self):
        command = 'llama-server --model "model with spaces.gguf" --port 8080'
        args = self.invoke("--server-command", command)
        self.assertEqual(args.server_command, command)

    def test_choice_aliases_and_linux_win_gpu_mode_parse_without_server(self):
        args = self.invoke("--vram-log", "off", "--gpm-log", "on",
                           "--win-gpu-mem", "on", "--cache-mode", "cold",
                           "--deterministic", "--ignore-eos")
        self.assertEqual((args.vram_log, args.gpm_log, args.win_gpu_mem),
                         ("off", "on", "on"))
        self.assertEqual(args.cache_mode, "cold")
        self.assertTrue(args.deterministic and args.ignore_eos)

    def test_removed_character_cut_options_are_rejected(self):
        for option in ("--safe-cut", "--cut-mode", "--no-cache-prompt"):
            with self.subTest(option=option):
                self.assert_rejected(option)


if __name__ == "__main__":
    unittest.main()
