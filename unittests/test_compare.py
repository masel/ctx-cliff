"""--reference / --compare: compare runs of any configuration per target context."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_csv_recording import benchmark as b


def write_run(folder, name, rows, meta=None, samples=None):
    stem = Path(folder) / name
    fields = sorted({key for row in rows for key in row})
    with open(f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if meta is not None:
        Path(f"{stem}.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    if samples is not None:
        with open(f"{stem}.samples.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted({key for row in samples for key in row}))
            writer.writeheader()
            writer.writerows(samples)
    return f"{stem}.csv"


def meta(scenario="agent", server="llama-server.exe --model m.gguf --ctx-size 160000", **sampling):
    return {"arguments": {"cache_mode": "incremental", "n_predict": 512, "deterministic": False},
            "prompt": {"scenario": scenario, "sampling_first_repeat": {"temperature": 0.6, **sampling},
                       "agent": {"task": "t", "thinking": "auto"}},
            "input_file": {"sha256": "abc"}, "server": {"command": server}}


BASE = [{"target_ctx": 10000, "total_ctx": 10000, "prefill_tps": 878.0, "decode_tps_median": 25.0,
         "draft_n": 0},
        {"target_ctx": 20000, "total_ctx": 20000, "prefill_tps": 790.0, "decode_tps_median": 20.0,
         "draft_n": 0},
        {"target_ctx": 159487, "total_ctx": 159487, "prefill_tps": 350.0, "decode_tps_median": 14.0,
         "draft_n": 0}]
# Older MTP result CSV without step columns; the samples provide them.
MTP = [{"target_ctx": 10000, "total_ctx": 10000, "prefill_tps": 850.0, "decode_tps_median": 35.0,
        "draft_n": 900},
       {"target_ctx": 20000, "total_ctx": 20000, "prefill_tps": 770.0, "decode_tps_median": 30.0,
        "draft_n": 900},
       {"target_ctx": 89599, "total_ctx": 89599, "prefill_tps": 460.0, "decode_tps_median": 24.0,
        "draft_n": 900}]
MTP_SAMPLES = [{"target_ctx": 10000, "status": "OK", "predicted_n": 512, "predicted_ms": 14000, "draft_acc": 256},
               {"target_ctx": 10000, "status": "OK", "predicted_n": 512, "predicted_ms": 15000, "draft_acc": 256},
               {"target_ctx": 20000, "status": "OK", "predicted_n": 512, "predicted_ms": 17000, "draft_acc": 256},
               {"target_ctx": 20000, "status": "STOP@3", "predicted_n": 3, "predicted_ms": 1, "draft_acc": 0}]


class HelperTests(unittest.TestCase):
    def test_command_items_pair_options_with_values(self):
        self.assertEqual(b.command_items("srv.exe --model a b -ngl -1 --flash-attn on --jinja -np 1"),
                         ["srv.exe", "--model a", "b", "-ngl -1", "--flash-attn on", "--jinja", "-np 1"])
        self.assertEqual(b.command_items(None), [])

    def test_step_cost_without_drafting_is_ms_per_token(self):
        points = b.comparison_points(BASE)
        self.assertAlmostEqual(points[10000]["ms_per_step_median"], 40.0)
        self.assertEqual(points[10000]["tokens_per_step_median"], 1.0)

    def test_older_mtp_column_names_are_still_read(self):
        points = b.comparison_points([{"target_ctx": "10000", "decode_tps_median": "35.0",
                                       "mtp_draft_n": "900", "mtp_acc_pct": "55.0"}])
        self.assertEqual(points[10000]["draft_n"], 900)
        self.assertEqual(points[10000]["draft_acc_pct"], 55.0)
        # Drafting was active, so the step cost is not simply ms per token.
        self.assertIsNone(points[10000]["ms_per_step_median"])

    def test_setting_differences_are_described_per_key(self):
        self.assertEqual(b.describe_setting_difference({"temperature": 0.6}, {"temperature": 0.6, "min_p": 0.0}),
                         "min_p default -> 0.0")
        self.assertEqual(b.describe_setting_difference("file", "agent"), "file -> agent")


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def runs(self, run_meta=None):
        ref = write_run(self.folder.name, "ref", BASE, meta())
        run = write_run(self.folder.name, "mtp", MTP, run_meta or meta(
            server="llama-server.exe --model m.gguf --ctx-size 90000 --spec-type draft-mtp", min_p=0.0),
            MTP_SAMPLES)
        return ref, run

    def test_common_points_changes_and_differences(self):
        ref, run = self.runs()
        comparison = b.compare_runs(b.load_run(ref), b.load_run(run))
        self.assertEqual([p["target_ctx"] for p in comparison["points"]], [10000, 20000])
        first = comparison["points"][0]
        self.assertAlmostEqual(first["decode_tps_median"]["change_pct"], 40.0)
        self.assertAlmostEqual(first["ms_per_step_median"]["run"], 14500 / 256)  # from samples
        self.assertEqual(first["tokens_per_step"], {"reference": 1.0, "run": 2.0})
        self.assertEqual((comparison["unmatched_reference_ctx"], comparison["unmatched_run_ctx"]), ([159487], [89599]))
        self.assertEqual(comparison["server_reference_only"], ["--ctx-size 160000"])
        self.assertEqual(comparison["server_run_only"], ["--ctx-size 90000", "--spec-type draft-mtp"])
        self.assertEqual([d["setting"] for d in comparison["setting_differences"]], ["sampling"])
        self.assertAlmostEqual(comparison["summary"]["decode_tps_median"]["median_change_pct"], 45.0)

    def test_workload_differences_warn_and_text_differences_only_note(self):
        ref, run = self.runs(meta(scenario="file", min_p=0.0))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            b.print_comparison(b.compare_runs(b.load_run(ref), b.load_run(run)))
        text = out.getvalue()
        self.assertIn("WARNING: scenario differs (agent -> file)", text)
        self.assertIn("NOTE: sampling differs (min_p default -> 0.0)", text)
        self.assertIn("+40.0%", text)

    def test_runs_without_metadata_are_compared_without_setting_checks(self):
        ref = write_run(self.folder.name, "ref", BASE)
        run = write_run(self.folder.name, "run", BASE)
        comparison = b.compare_runs(b.load_run(ref), b.load_run(run))
        self.assertEqual(comparison["setting_differences"], [])
        self.assertEqual(comparison["summary"]["decode_tps_median"]["median_change_pct"], 0.0)

    def test_offline_compare_prints_matrix_for_several_runs(self):
        ref, run = self.runs()
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(b.run_compare([ref, run, run]), 0)
        self.assertIn("DECODE CHANGE vs ref", out.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as out:
            b.run_compare([ref, run])
        self.assertNotIn("DECODE CHANGE", out.getvalue())

    def test_offline_compare_reports_unreadable_files(self):
        not_a_result = Path(self.folder.name) / "other.csv"
        not_a_result.write_text("a,b\n1,2\n", encoding="utf-8")
        for paths in ([str(Path(self.folder.name) / "missing.csv"), str(not_a_result)],
                      [str(not_a_result), str(not_a_result)]):
            with contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(b.run_compare(paths), 1)
            self.assertIn("ERROR", err.getvalue())


LOG_F16_DRAFT = """\
0.00.55 I load_tensors:        CUDA0 model buffer size = 12005.90 MiB
0.00.56 I load_tensors:   CPU_Mapped model buffer size =   500.00 MiB
0.02.60 I llama_kv_cache: size =    0.00 MiB ( 91136 cells,   0 layers,  1/1 seqs), K (none):    0.00 MiB, V (none):    0.00 MiB
0.02.63 I llama_kv_cache_kvarn: type = kvarn_k4v4_g128, layers = 16, groups/stream = 712, streams = 1, KVarN = 1589.62 MiB, equivalent F16 = 5696.00 MiB
0.02.64 I llama_memory_recurrent:      CUDA0 RS buffer size =   448.88 MiB
0.02.67 I sched_reserve:      CUDA0 compute buffer size =   211.65 MiB
0.02.67 I sched_reserve:  CUDA_Host compute buffer size =   109.65 MiB
0.02.79 I llama_kv_cache:      CUDA0 KV buffer size =   356.00 MiB
0.02.80 I llama_kv_cache: size =  356.00 MiB ( 91136 cells,   1 layers,  1/1 seqs), K (f16):  178.00 MiB, V (f16):  178.00 MiB
0.02.81 I sched_reserve:      CUDA0 compute buffer size =   157.02 MiB
0.05.86 I sched_reserve:      CUDA0 compute buffer size =   999.00 MiB
"""
LOG_KVARN3_DRAFT = LOG_F16_DRAFT.replace(
    "0.02.79 I llama_kv_cache:      CUDA0 KV buffer size =   356.00 MiB\n"
    "0.02.80 I llama_kv_cache: size =  356.00 MiB ( 91136 cells,   1 layers,  1/1 seqs), K (f16):  178.00 MiB, V (f16):  178.00 MiB\n"
    "0.02.81 I sched_reserve:      CUDA0 compute buffer size =   157.02 MiB",
    "0.02.80 I llama_kv_cache_kvarn: type = kvarn_k3v3_g128, layers = 1, groups/stream = 712, streams = 1, KVarN = 77.10 MiB, equivalent F16 = 356.00 MiB\n"
    "0.02.81 I sched_reserve:      CUDA0 compute buffer size =   159.65 MiB")


class ServerMemoryTests(unittest.TestCase):
    def test_log_is_split_into_main_and_draft_contexts(self):
        memory = b.parse_server_memory(LOG_F16_DRAFT.splitlines())
        self.assertEqual(memory["model_mib"], {"CUDA0": 12005.9, "CPU_Mapped": 500.0})
        main, draft = memory["contexts"]
        self.assertEqual((main["name"], main["kv_type"], main["kv_layers"], main["kv_mib"],
                          main["recurrent_mib"], main["compute_mib"]),
                         ("main", "kvarn_k4v4_g128", 16, 1589.62, 448.88, 211.65))
        # The later re-reserve (999) does not replace the first device compute buffer.
        self.assertEqual((draft["name"], draft["kv_type"], draft["kv_mib"], draft["compute_mib"]),
                         ("draft", "f16", 356.0, 157.02))
        self.assertAlmostEqual(memory["device_total_mib"], 12005.9 + 1589.62 + 448.88 + 211.65 + 356.0 + 157.02)
        self.assertIn("draft KV 356.0 (f16)", b.describe_server_memory(memory))
        self.assertIsNone(b.parse_server_memory(["nothing here"]))

    def test_differences_ignore_rounding_and_show_type_changes(self):
        ref = b.parse_server_memory(LOG_F16_DRAFT.splitlines())
        run = b.parse_server_memory(LOG_KVARN3_DRAFT.splitlines())
        run["contexts"][0]["kv_mib"] += 0.1  # logged rounding noise
        buffers = {d["buffer"]: (d["reference"], d["run"]) for d in b.memory_differences(ref, run)}
        self.assertEqual(buffers["draft KV"], ((356.0, "f16"), (77.1, "kvarn_k3v3_g128")))
        self.assertEqual(buffers["draft compute"], ((157.02, None), (159.65, None)))
        self.assertNotIn("KV", buffers)
        self.assertEqual(b.memory_differences(None, run), [])

    def test_older_runs_read_memory_from_their_server_log(self):
        with tempfile.TemporaryDirectory() as folder:
            ref_log, run_log = Path(folder) / "ref.log", Path(folder) / "run.log"
            ref_log.write_text(LOG_F16_DRAFT, encoding="utf-8")
            run_log.write_text(LOG_KVARN3_DRAFT, encoding="utf-8")
            ref = write_run(folder, "ref", BASE, {**meta(), "server": {"command": "srv", "log": str(ref_log)}})
            run = write_run(folder, "run", BASE, {**meta(), "server": {"command": "srv", "log": str(run_log)}})
            with contextlib.redirect_stdout(io.StringIO()) as out:
                b.run_compare([ref, run])
        self.assertIn("server memory (MiB): draft KV 356.0 f16 -> 77.1 kvarn_k3v3_g128 (-278.9)", out.getvalue())


class CompareCliTests(unittest.TestCase):
    def main(self, *arguments):
        with patch.object(b.sys, "argv", ["ctx-cliff.py", *arguments]), \
             patch.object(b, "run_benchmark", side_effect=lambda *a: None), \
             patch.object(b, "termination_handler", return_value=contextlib.nullcontext()), \
             patch.object(b, "CsvRecording", return_value=contextlib.nullcontext()), \
             contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err, \
             self.assertRaises(SystemExit) as raised:
            b.main()
        return raised.exception.code, out.getvalue(), err.getvalue()

    def test_compare_runs_without_file_or_server(self):
        with tempfile.TemporaryDirectory() as folder:
            ref = write_run(folder, "ref", BASE)
            code, out, _ = self.main("--compare", ref, ref)
        self.assertEqual(code, 0)
        self.assertIn("COMPARISON ref vs reference ref", out)

    def test_invalid_usage(self):
        self.assertEqual(self.main("--compare", "only-one.csv")[0], 2)
        code, _, err = self.main()
        self.assertEqual(code, 2)
        self.assertIn("--file is required", err)
        code, _, err = self.main("--file", "input.txt", "--reference", "missing.csv")
        self.assertEqual(code, 2)
        self.assertIn("--reference", err)


class ReferenceRunTests(unittest.TestCase):
    def test_sweep_with_reference_prints_and_records_comparison(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "input.txt"
            source.write_text("x" * 5000, encoding="utf-8")
            ref = write_run(folder, "ref", [
                {"target_ctx": 100, "total_ctx": 100, "prefill_tps": 5.0, "decode_tps_median": 10.0},
                {"target_ctx": 200, "total_ctx": 200, "prefill_tps": 5.0, "decode_tps_median": 10.0}])

            def complete(base, prompt, n_predict, *args, **kwargs):
                return {"timings": {"cache_n": 0, "prompt_n": len(prompt), "prompt_ms": 10,
                                    "predicted_n": n_predict, "predicted_ms": 400}}

            argv = ["ctx-cliff.py", "--file", str(source), "--start", "100", "--end", "200", "--step", "100",
                    "--n-predict", "8", "--repeat", "1", "--warmup", "0", "--cache-mode", "cold",
                    "--settle", "0", "--vram-log", "off", "--gpm-log", "off", "--win-gpu-mem", "off",
                    "--no-drift-check", "--reference", ref]
            recorded = {}
            with patch.object(b.sys, "argv", argv), \
                 patch.object(b, "tokenize", side_effect=lambda base, text, add_bos=True, **kw: [1] * (len(text) + 1)), \
                 patch.object(b, "completion", side_effect=complete), \
                 patch.object(b, "server_is_ready", return_value=True), \
                 patch.object(b, "detect_slot_n_ctx", return_value=4096), \
                 patch.object(b, "reset_slot", return_value=True), \
                 patch.object(b, "probe_prefill_repeats"), \
                 patch.object(b.RunMetadata, "update", lambda self, **fields: recorded.update(fields)), \
                 contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()):
                b.main()
        self.assertIn("COMPARISON this run vs reference ref", out.getvalue())
        points = recorded["comparison"]["points"]
        self.assertEqual([p["target_ctx"] for p in points], [100, 200])
        self.assertAlmostEqual(points[0]["decode_tps_median"]["change_pct"], 100.0)  # 8 tok / 0.4 s = 20 tok/s


if __name__ == "__main__":
    unittest.main()
