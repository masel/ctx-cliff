"""Offline regression tests for GPM counter plausibility diagnostics."""

from contextlib import redirect_stdout
import csv
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest


from test_csv_recording import benchmark


def gpm_row(start, end, *, gpu=0, graphics=90.0, sm=0.0, occupancy=0.0,
            tensor=0.0, dram=50.0, **extra):
    row = {
        "gpu_index": gpu,
        "interval_start_mono": start,
        "interval_end_mono": end,
        "graphics_util_pct": graphics,
        "sm_util_pct": sm,
        "sm_occupancy_pct": occupancy,
        "tensor_util_pct": tensor,
        "dram_bw_util_pct": dram,
    }
    row.update(extra)
    return row


class GpmPlausibilityTests(unittest.TestCase):
    def test_sustained_counter_dropout_is_flagged(self):
        rows = [gpm_row(index * 0.25, (index + 1) * 0.25)
                for index in range(4)]

        result = benchmark.gpm_plausibility(rows)

        self.assertEqual(result["gpm_suspect_samples"], 4)
        self.assertAlmostEqual(result["gpm_suspect_ms"], 1000.0)
        self.assertEqual(result["gpm_suspect_phases"], 1)
        self.assertAlmostEqual(result["gpm_suspect_max_run_ms"], 1000.0)

    def test_short_idle_and_missing_runs_are_not_flagged(self):
        short = [gpm_row(index * 0.25, (index + 1) * 0.25)
                 for index in range(3)]
        idle = [gpm_row(2 + index * 0.25, 2 + (index + 1) * 0.25,
                        graphics=20.0)
                for index in range(4)]
        missing_break = [
            gpm_row(4.00, 4.25),
            gpm_row(4.25, 4.50),
            gpm_row(4.50, 4.75, sm=None),
            gpm_row(4.75, 5.00),
            gpm_row(5.00, 5.25),
        ]

        result = benchmark.gpm_plausibility(short + idle + missing_break)

        self.assertEqual(result["gpm_suspect_samples"], 0)
        self.assertEqual(result["gpm_suspect_ms"], 0.0)
        self.assertEqual(result["gpm_suspect_phases"], 0)
        self.assertEqual(result["gpm_suspect_max_run_ms"], 0.0)

    def test_gap_breaks_a_run(self):
        rows = [gpm_row(0.00, 0.25), gpm_row(0.25, 0.50),
                gpm_row(0.75, 1.00), gpm_row(1.00, 1.25)]

        result = benchmark.gpm_plausibility(rows)

        self.assertEqual(result["gpm_suspect_samples"], 0)
        self.assertEqual(result["gpm_suspect_phases"], 0)

    def test_gpu_groups_are_evaluated_independently(self):
        gpu0 = [gpm_row(index * 0.25, (index + 1) * 0.25, gpu=0)
                for index in range(4)]
        gpu1 = [gpm_row(index * 0.25, (index + 1) * 0.25, gpu=1)
                for index in range(2)]

        result = benchmark.gpm_plausibility(gpu0 + gpu1)

        self.assertEqual(result["gpm_suspect_samples"], 4)
        self.assertEqual(result["gpm_suspect_phases"], 1)
        self.assertAlmostEqual(result["gpm_suspect_ms"], 1000.0)

    def test_mixed_repeats_retain_averages_and_suspect_flag(self):
        first = dict(
            gpm_sm_avg_pct=0.0,
            gpm_occupancy_avg_pct=0.0,
            gpm_tensor_avg_pct=0.0,
            gpm_dram_avg_pct=50.0,
            gpm_sm_valid_ms=1000.0,
            gpm_occupancy_valid_ms=1000.0,
            gpm_tensor_valid_ms=1000.0,
            gpm_dram_valid_ms=1000.0,
            gpm_suspect_samples=4,
            gpm_suspect_ms=1000.0,
            gpm_suspect_phases=1,
            gpm_suspect_max_run_ms=1000.0,
        )
        second = dict(
            gpm_sm_avg_pct=94.0,
            gpm_occupancy_avg_pct=46.0,
            gpm_tensor_avg_pct=1.0,
            gpm_dram_avg_pct=58.0,
            gpm_sm_valid_ms=1000.0,
            gpm_occupancy_valid_ms=1000.0,
            gpm_tensor_valid_ms=1000.0,
            gpm_dram_valid_ms=1000.0,
            gpm_suspect_samples=0,
            gpm_suspect_ms=0.0,
            gpm_suspect_phases=0,
            gpm_suspect_max_run_ms=0.0,
        )

        result = benchmark.aggregate_telemetry(
            [first, second], benchmark.empty_gpm())

        self.assertAlmostEqual(result["gpm_sm_avg_pct"], 47.0)
        self.assertAlmostEqual(result["gpm_occupancy_avg_pct"], 23.0)
        self.assertAlmostEqual(result["gpm_tensor_avg_pct"], 0.5)
        self.assertAlmostEqual(result["gpm_dram_avg_pct"], 54.0)
        self.assertEqual(result["gpm_suspect_samples"], 4)
        self.assertAlmostEqual(result["gpm_suspect_ms"], 1000.0)
        self.assertEqual(result["gpm_suspect_phases"], 1)

    def test_live_table_marks_suspect_phase(self):
        row = {
            "total_ctx": 100,
            "prompt_n": 10,
            "prefill_tps": 20.0,
            "decode_tps_median": 5.0,
            "vram_free_min_mib": 1000,
            "power_avg_w": 100,
            "status": "OK",
            "pcie_prefill_bus_avg_pct": 10,
            "pcie_decode_bus_avg_pct": 10,
        }
        for phase in ("prefill", "decode"):
            prefix = f"gpm_{phase}_"
            row.update({
                prefix + "pcie_rx_p95_mib_s": 1,
                prefix + "pcie_tx_p95_mib_s": 2,
                prefix + "pcie_over90_pct": 0,
                prefix + "sm_avg_pct": 0,
                prefix + "occupancy_avg_pct": 0,
                prefix + "tensor_avg_pct": 0,
                prefix + "dram_avg_pct": 50,
                prefix + "suspect_phases": 1 if phase == "prefill" else 0,
            })

        output = io.StringIO()
        with redirect_stdout(output):
            benchmark.print_live_row(row, drafting_on=False)

        self.assertIn("0/0/0/50*", output.getvalue())
        self.assertEqual(output.getvalue().count("*"), 1)

    def test_result_csv_schema_contains_plausibility_metadata(self):
        expected = {
            "gpm_prefill_suspect_samples",
            "gpm_prefill_suspect_ms",
            "gpm_prefill_suspect_phases",
            "gpm_prefill_suspect_max_run_ms",
            "gpm_decode_suspect_samples",
            "gpm_decode_suspect_ms",
            "gpm_decode_suspect_phases",
            "gpm_decode_suspect_max_run_ms",
        }
        self.assertTrue(expected.issubset(set(benchmark.RESULT_CSV_FIELDS)))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "result.csv"
            input_path = root / "input.txt"
            input_path.write_text("input", encoding="utf-8")
            args = SimpleNamespace(
                csv=str(csv_path), file=str(input_path), server_log=None,
                vram_log="off", gpm_log="off", win_gpu_mem="off",
                vram_csv=None, pcie_csv=None, gpm_csv=None,
                win_gpu_mem_csv=None,
            )
            with benchmark.CsvRecording(args):
                with csv_path.open(newline="", encoding="utf-8") as handle:
                    header = next(csv.reader(handle))
            self.assertEqual(header, list(benchmark.RESULT_CSV_FIELDS))
            self.assertTrue(expected.issubset(set(header)))


if __name__ == "__main__":
    unittest.main()
