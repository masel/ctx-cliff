"""Offline regression tests for repeated prefill measurements in ctx-cliff.py."""

import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from test_csv_recording import benchmark


class FakeResponse:
    def __init__(self, payload=None, status_code=200, text=""):
        self._payload = payload if payload is not None else {}
        self.status_code = status_code
        self.text = text

    @property
    def ok(self):
        return 200 <= self.status_code < 400

    def json(self):
        return self._payload

    def raise_for_status(self):
        if not self.ok:
            raise benchmark.requests.HTTPError(self.text)


class MockLlamaServer:
    """Small in-process HTTP double for completion, erase, save and restore."""

    def __init__(self, completions=None, save_count=None, restore_counts=None,
                 snapshot_failure=None):
        self.completions = list(completions or [])
        self.save_count = save_count
        self.restore_counts = list(restore_counts or [])
        self.snapshot_failure = snapshot_failure
        self.actions = []
        self.events = []
        self.completion_payloads = []
        self.tokens = []
        self.snapshot_tokens = []

    @staticmethod
    def _timings(**overrides):
        timings = {
            "cache_n": 4,
            "prompt_n": 20,
            "prompt_ms": 100,
            "predicted_n": 4,
            "predicted_ms": 200,
            "draft_n": 0,
            "draft_n_accepted": 0,
        }
        timings.update(overrides)
        return timings

    def post(self, url, json=None, timeout=None):
        parsed = urlparse(url)
        action = parse_qs(parsed.query).get("action", [None])[0]
        if parsed.path.endswith("/tokenize"):
            return FakeResponse({"tokens": list(range(int(json["content"].split()[-1])))})
        if parsed.path.endswith("/completion"):
            self.events.append("completion")
            self.completion_payloads.append(dict(json or {}))
            prompt = json["prompt"]
            if isinstance(prompt, list):
                # Recurrent state can extend its full prefix, but cannot roll
                # back a divergent decode tail without discarded checkpoints.
                cached = (len(self.tokens) if json["cache_prompt"]
                          and prompt[:len(self.tokens)] == self.tokens else 0)
                timings = self._timings(cache_n=cached, prompt_n=len(prompt) - cached,
                                        predicted_n=json["n_predict"])
                self.tokens = prompt + [-1] * (json["n_predict"] - 1)
                if json["n_predict"] != 1 and self.completions:
                    timings = self.completions.pop(0)
            else:
                timings = self.completions.pop(0) if self.completions else self._timings()
            return FakeResponse({"timings": timings})
        if action in ("save", "restore"):
            self.actions.append(action)
            self.events.append(action)
            if self.snapshot_failure is not None:
                return self.snapshot_failure
            if action == "save":
                self.snapshot_tokens = list(self.tokens)
                count = self.save_count if self.save_count is not None else len(self.tokens)
            else:
                self.tokens = list(self.snapshot_tokens)
                count = self.restore_counts.pop(0) if self.restore_counts else (
                    self.save_count if self.save_count is not None else len(self.tokens))
            return FakeResponse({"n_saved" if action == "save" else "n_restored": count})
        raise AssertionError(f"unexpected POST {url}")

    def request(self, method, url, timeout=None):
        parsed = urlparse(url)
        action = parse_qs(parsed.query).get("action", [None])[0]
        self.actions.append(action or method.lower())
        self.events.append(action or method.lower())
        if action == "erase":
            self.tokens = []
            return FakeResponse({})
        raise AssertionError(f"unexpected request {method} {url}")


class FixedBuilder:
    """Token IDs come from the (possibly patched) module tokenizer via the mock server."""
    def __init__(self, http=None):
        self.http, self.last_tokens = http, None

    def build(self, ctx):
        text = f"fixed prompt {ctx}"
        self.last_tokens = (text, benchmark.tokenize("http://mock-server", text, http=self.http))
        return text, ctx + 10, ctx * 2


class RecordingSpy:
    def __init__(self):
        self.rows = []
        self.samples = []

    def check(self):
        return None

    def write_result(self, row):
        self.rows.append(row)

    def write_sample(self, row):
        self.samples.append(row)


def args(cache_mode="incremental", repeat=3):
    return SimpleNamespace(
        cache_mode=cache_mode,
        repeat=repeat,
        slot_id=7,
        settle=0,
        n_predict=4,
        deterministic=True,
        ignore_eos=True,
    )


def runner(server, recording, **kwargs):
    config = args(**kwargs)
    return benchmark.BenchmarkRunner(
        config,
        "http://mock-server",
        server,
        FixedBuilder(server),
        recording,
        None,
        None,
        None,
    )


class PrefillRepeatTests(unittest.TestCase):
    def test_snapshot_directory_from_command_handles_quoted_separate_equals_relative_and_absent(self):
        with tempfile.TemporaryDirectory(prefix="ctx cliff ") as temp_dir:
            directory = Path(temp_dir) / "snapshots with spaces"
            commands = (
                f'server --slot-save-path "{directory}"',
                f'server --slot-save-path="{directory}"',
            )
            for command in commands:
                with self.subTest(command=command):
                    self.assertEqual(
                        benchmark.snapshot_directory_from_command(command),
                        str(directory.resolve()),
                    )

        relative = "relative snapshots with spaces"
        self.assertEqual(
            benchmark.snapshot_directory_from_command(
                f'server --slot-save-path "{relative}"'
            ),
            str(Path(relative).resolve()),
        )
        self.assertIsNone(benchmark.snapshot_directory_from_command("server --model model.gguf"))

    def test_cleanup_snapshot_removes_only_this_run_file_and_tolerates_missing_file(self):
        with tempfile.TemporaryDirectory(prefix="ctx cliff cleanup ") as temp_dir:
            directory = Path(temp_dir)
            generated = directory / "ctx-cliff-generated.bin"
            other_file = directory / "ctx-cliff-other.bin"
            other_folder = directory / "keep-this-folder"
            generated.write_bytes(b"generated")
            other_file.write_bytes(b"keep")
            other_folder.mkdir()
            (other_folder / "nested.bin").write_bytes(b"keep")

            benchmark.cleanup_snapshot(str(directory), generated.name)
            benchmark.cleanup_snapshot(str(directory), generated.name)

            self.assertFalse(generated.exists())
            self.assertTrue(other_file.exists())
            self.assertTrue(other_folder.is_dir())
            self.assertTrue((other_folder / "nested.bin").exists())

    def test_cleanup_snapshot_warns_when_unlink_fails(self):
        with tempfile.TemporaryDirectory(prefix="ctx cliff cleanup ") as temp_dir:
            with patch.object(benchmark.os, "unlink", side_effect=OSError("busy")), redirect_stderr(
                io.StringIO()
            ) as stderr:
                benchmark.cleanup_snapshot(temp_dir, "ctx-cliff-generated.bin")

        self.assertIn("WARNING: could not remove snapshot", stderr.getvalue())
        self.assertIn("busy", stderr.getvalue())

    def _run_snapshot_cleanup_lifecycle(self, *, keep_snapshot=False, managed=True,
                                        fail=False, external=False):
        """Run just far enough to register snapshot cleanup callbacks."""
        events = []
        with tempfile.TemporaryDirectory(prefix="ctx cliff lifecycle ") as temp_dir:
            input_file = Path(temp_dir) / "input.txt"
            input_file.write_text("test input", encoding="utf-8")
            snapshot_dir = Path(temp_dir) / "snapshots with spaces"
            snapshot_dir.mkdir()
            command = f'server --slot-save-path "{snapshot_dir}"'

            class FakeSession:
                def __enter__(self):
                    events.append("http-enter")
                    return self

                def __exit__(self, exc_type, exc_value, traceback):
                    events.append("http-close")

            class FakeManagedServer:
                def __init__(self, command, **kwargs):
                    self.command = command
                    self.running = False

                def start(self):
                    self.running = True
                    events.append("managed-start")

                def stop(self):
                    # Idempotent like ManagedLlamaServer.stop(): the sweep stops the
                    # server early, the exit stack calls stop() again.
                    if self.running:
                        self.running = False
                        events.append("managed-stop")

            class FakeRunner:
                snapshot_filename = "ctx-cliff-generated.bin"

            args = SimpleNamespace(
                server_command=None if external else command,
                server_log=None,
                server_log_dir=temp_dir,
                base_url="http://remote.example:8080" if external else "http://127.0.0.1:8080",
                reuse_running_server=not managed and not external,
                server_start_timeout=1,
                keep_server=False,
                cache_mode="incremental",
                repeat=2,
                keep_snapshot=keep_snapshot,
                file=str(input_file),
                slot_id=7,
                n_predict=4,
                deterministic=True,
                ignore_eos=True,
                start=1,
                end=0,
                step=1,
                warmup=0,
                settle=0,
                cliff_pct=10.0,
                vram_log="off",
                gpm_log="off",
                win_gpu_mem="off",
            )
            def fake_probe(*probe_args, **probe_kwargs):
                events.append("probe")
                if fail:
                    raise RuntimeError("probe failed")

            def fake_cleanup(*cleanup_args):
                events.append("snapshot-cleanup")

            resources = benchmark.ExitStack()
            try:
                patches = [
                    patch.object(benchmark.requests, "Session", return_value=FakeSession()),
                    patch.object(benchmark, "server_is_ready", return_value=not managed),
                    patch.object(benchmark, "tokenize", return_value=[1]),
                    patch.object(benchmark, "detect_slot_n_ctx", return_value=None),
                    patch.object(benchmark, "PromptBuilder", return_value=SimpleNamespace(reset=lambda: None)),
                    patch.object(benchmark, "BenchmarkRunner", return_value=FakeRunner()),
                    patch.object(benchmark, "reset_slot", return_value=True),
                    patch.object(benchmark, "probe_prefill_repeats", side_effect=fake_probe),
                    patch.object(benchmark, "cleanup_snapshot", side_effect=fake_cleanup),
                ]
                if managed:
                    patches.append(patch.object(benchmark, "ManagedLlamaServer", FakeManagedServer))
                with benchmark.ExitStack() as patch_stack:
                    for patcher in patches:
                        patch_stack.enter_context(patcher)
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "probe failed"):
                            benchmark.run_benchmark(args, object(), resources, object())
                    else:
                        benchmark.run_benchmark(args, object(), resources, object())
            finally:
                resources.close()
        return events

    def test_snapshot_cleanup_runs_after_managed_server_stop_on_success_and_exception(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                events = self._run_snapshot_cleanup_lifecycle(fail=fail)
                self.assertEqual(
                    events,
                    ["http-enter", "managed-start", "probe", "managed-stop",
                     "snapshot-cleanup", "http-close"],
                )

    def test_keep_snapshot_suppresses_cleanup_callback(self):
        events = self._run_snapshot_cleanup_lifecycle(keep_snapshot=True)
        self.assertEqual(
            events,
            ["http-enter", "managed-start", "probe", "managed-stop", "http-close"],
        )

    def test_reused_or_external_server_does_not_infer_snapshot_directory_from_command(self):
        for external in (False, True):
            with self.subTest(external=external), tempfile.TemporaryDirectory(prefix="ctx cliff server ") as temp_dir:
                events = []
                with patch.object(benchmark, "snapshot_directory_from_command") as infer:
                    # The lifecycle helper models a reused local server. An external
                    # server has no command at all, so its command cannot be inferred.
                    if external:
                        with patch.object(benchmark, "server_is_ready", return_value=True):
                            events = self._run_snapshot_cleanup_lifecycle(managed=False, external=True)
                    else:
                        events = self._run_snapshot_cleanup_lifecycle(managed=False)
                infer.assert_not_called()
                self.assertNotIn("snapshot-cleanup", events)

    def test_prepare_snapshot_command_adds_directory_beside_script_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix="ctx cliff ") as temp_dir:
            script = Path(temp_dir) / "ctx-cliff.py"
            snapshot_dir = Path(temp_dir) / "slot-snapshots"
            command = r'server --model "D:\models\model with spaces.gguf"'
            with patch.object(benchmark, "__file__", str(script)):
                prepared = benchmark.prepare_snapshot_command(command, args(repeat=3))

            self.assertNotEqual(prepared, command)
            self.assertIn("--slot-save-path", prepared)
            self.assertIn(str(snapshot_dir), prepared)
            self.assertTrue(snapshot_dir.is_dir())

    def test_prepare_snapshot_command_preserves_existing_slot_save_path_without_creation(self):
        commands = (
            r'server --slot-save-path "D:\existing snapshots"',
            r'server --slot-save-path="D:\existing snapshots"',
        )
        with tempfile.TemporaryDirectory(prefix="ctx cliff ") as temp_dir:
            script = Path(temp_dir) / "ctx-cliff.py"
            snapshot_dir = Path(temp_dir) / "slot-snapshots"
            for command in commands:
                with self.subTest(command=command), patch.object(benchmark, "__file__", str(script)):
                    prepared = benchmark.prepare_snapshot_command(command, args(repeat=3))
                self.assertEqual(prepared, command)
                self.assertFalse(snapshot_dir.exists())

    def test_prepare_snapshot_command_does_not_create_directory_for_cold_or_single_repeat(self):
        with tempfile.TemporaryDirectory(prefix="ctx cliff ") as temp_dir:
            script = Path(temp_dir) / "ctx-cliff.py"
            snapshot_dir = Path(temp_dir) / "slot-snapshots"
            for cache_mode, repeat in (("cold", 3), ("incremental", 1)):
                config = args(cache_mode=cache_mode, repeat=repeat)
                command = "server --model model.gguf"
                with self.subTest(cache_mode=cache_mode, repeat=repeat), patch.object(
                    benchmark, "__file__", str(script)
                ):
                    prepared = benchmark.prepare_snapshot_command(command, config)
                self.assertEqual(prepared, command)
                self.assertFalse(snapshot_dir.exists())

    def test_probe_prefill_repeats_success_enables_repeats(self):
        config = args(repeat=3)
        config.prefill_repeat_enabled = False
        http = object()
        with patch.object(benchmark, "slot_snapshot", return_value=20) as snapshot, redirect_stderr(
            io.StringIO()
        ) as stderr:
            benchmark.probe_prefill_repeats(config, "http://mock-server", "snapshot.bin", http=http)

        self.assertTrue(config.prefill_repeat_enabled)
        snapshot.assert_called_once_with(
            "http://mock-server", config.slot_id, "save", "snapshot.bin", http=http
        )
        self.assertIn("slot snapshots available", stderr.getvalue())

    def test_probe_prefill_repeats_rejection_disables_repeats_with_warning(self):
        config = args(repeat=3)
        with patch.object(benchmark, "slot_snapshot", side_effect=RuntimeError("save rejected")), redirect_stderr(
            io.StringIO()
        ) as stderr:
            benchmark.probe_prefill_repeats(config, "http://mock-server", "snapshot.bin", http=object())

        self.assertFalse(config.prefill_repeat_enabled)
        self.assertIn("WARNING: prefill repeats disabled", stderr.getvalue())
        self.assertIn("save rejected", stderr.getvalue())

    def test_probe_prefill_repeats_skips_probe_for_cold_and_single_repeat(self):
        for cache_mode, repeat in (("cold", 3), ("incremental", 1)):
            config = args(cache_mode=cache_mode, repeat=repeat)
            with self.subTest(cache_mode=cache_mode, repeat=repeat), patch.object(
                benchmark, "slot_snapshot"
            ) as snapshot:
                benchmark.probe_prefill_repeats(config, "http://mock-server", "snapshot.bin", http=object())
            self.assertTrue(config.prefill_repeat_enabled)
            snapshot.assert_not_called()

    def test_probe_prefill_repeats_propagates_network_exception(self):
        class FailingHttp:
            def post(self, *args, **kwargs):
                raise benchmark.requests.ConnectionError("server unavailable")

        config = args(repeat=3)
        with self.assertRaises(benchmark.requests.ConnectionError):
            benchmark.probe_prefill_repeats(
                config, "http://mock-server", "snapshot.bin", http=FailingHttp()
            )
        self.assertTrue(config.prefill_repeat_enabled)

    def test_rejected_snapshot_fallback_repeats_decode_without_snapshots_and_uses_first_prefill(self):
        def timing(prompt_ms, predicted_ms):
            return MockLlamaServer._timings(prompt_ms=prompt_ms, predicted_ms=predicted_ms)

        server = MockLlamaServer(completions=[
            timing(10, 400), timing(20, 200), timing(30, 100),
            timing(40, 100), timing(50, 80), timing(60, 50),
        ])
        recording = RecordingSpy()
        config = args(repeat=3)
        config.prefill_repeat_enabled = False
        run = benchmark.BenchmarkRunner(
            config,
            "http://mock-server",
            server,
            FixedBuilder(server),
            recording,
            None,
            None,
            None,
        )

        first = run.measure_point(100)
        second = run.measure_point(200)

        self.assertEqual(server.actions, [])
        self.assertEqual(
            [payload["cache_prompt"] for payload in server.completion_payloads],
            [False, True, True, True, True, True],
        )
        self.assertEqual(first["prompt_ms"], 10)
        self.assertEqual(first["prefill_tps"], 2000.0)
        self.assertEqual(first["decode_tps_median"], 20.0)
        self.assertEqual(second["prompt_ms"], 40)
        self.assertEqual(second["prefill_tps"], 500.0)
        self.assertEqual(second["decode_tps_median"], 50.0)
        self.assertEqual(len(recording.rows), 2)

    def test_managed_snapshot_command_is_prepared_only_for_a_new_server(self):
        class LaunchSentinel(Exception):
            pass

        with tempfile.TemporaryDirectory(prefix="ctx cliff ") as temp_dir:
            launch_args = SimpleNamespace(
                server_command=r"server --slot-save-path D:\existing-snapshots",
                server_log=None,
                server_log_dir=temp_dir,
                base_url="http://127.0.0.1:8080",
                reuse_running_server=False,
                server_start_timeout=1,
                keep_server=False,
                cache_mode="incremental",
                repeat=3,
            )
            launch_resources = benchmark.ExitStack()
            try:
                with patch.object(benchmark, "server_is_ready", return_value=False), patch.object(
                    benchmark, "prepare_snapshot_command", wraps=benchmark.prepare_snapshot_command
                ) as prepare, patch.object(
                    benchmark, "ManagedLlamaServer", side_effect=LaunchSentinel()
                ):
                    with self.assertRaises(LaunchSentinel):
                        benchmark.run_benchmark(launch_args, object(), launch_resources, object())
                prepare.assert_called_once_with(launch_args.server_command, launch_args)
            finally:
                launch_resources.close()

            reuse_args = SimpleNamespace(
                **{**vars(launch_args), "reuse_running_server": True, "file": "missing-input.txt"}
            )
            reuse_resources = benchmark.ExitStack()
            try:
                with patch.object(benchmark, "server_is_ready", return_value=True), patch.object(
                    benchmark, "prepare_snapshot_command", wraps=benchmark.prepare_snapshot_command
                ) as prepare, patch.object(benchmark, "ManagedLlamaServer") as managed:
                    with self.assertRaises(SystemExit):
                        benchmark.run_benchmark(reuse_args, object(), reuse_resources, object())
                prepare.assert_not_called()
                managed.assert_not_called()
            finally:
                reuse_resources.close()

    def test_first_point_disables_cache_for_every_repeat_and_later_point_restores_each_repeat(self):
        server = MockLlamaServer()
        recording = RecordingSpy()
        run = runner(server, recording, repeat=3)

        run.measure_point(100)
        self.assertEqual([p["cache_prompt"] for p in server.completion_payloads],
                         [False, False, False])
        self.assertEqual(server.actions, ["erase", "erase", "erase"])
        self.assertEqual(server.events,
                         ["erase", "completion", "erase", "completion", "erase", "completion"])
        self.assertEqual(len(recording.rows), 1)

        run.measure_point(200)
        self.assertEqual(server.actions,
                         ["erase", "erase", "erase", "save", "restore", "restore", "restore"])
        self.assertEqual(server.events,
                         ["erase", "completion", "erase", "completion", "erase", "completion",
                          "completion", "save", "restore", "completion", "restore", "completion", "restore",
                          "completion"])
        self.assertEqual([p["cache_prompt"] for p in server.completion_payloads],
                         [False, False, False, False, True, True, True])
        self.assertEqual(server.completion_payloads[3]["prompt"], list(range(100)))
        self.assertEqual(server.completion_payloads[3]["n_predict"], 1)
        self.assertEqual(recording.rows[-1]["cache_n"], 100)
        self.assertEqual(recording.rows[-1]["prompt_n"], 100)
        self.assertEqual(len(recording.rows), 2)

    def test_restore_count_mismatch_fails_before_writing_csv_row(self):
        server = MockLlamaServer(save_count=20, restore_counts=[19])
        recording = RecordingSpy()
        run = runner(server, recording, repeat=2)
        run.measurement_started = True
        run.previous_prompt_tokens = list(range(20))

        with self.assertRaisesRegex(RuntimeError, r"Slot restore token mismatch: 19 != 20"):
            run.measure_point(100)
        self.assertEqual(server.actions, ["save", "restore"])
        self.assertEqual(recording.rows, [])

    def test_prefix_uses_common_token_ids_when_boundary_token_changes(self):
        server = MockLlamaServer()
        recording = RecordingSpy()
        run = runner(server, recording, repeat=2)
        with patch.object(benchmark, "tokenize", side_effect=[
            [10, 20, 30], [10, 20, 31, 40, 50],
        ]), patch.object(run, "take_sample", wraps=run.take_sample) as measured:
            run.measure_point(100)
            result = run.measure_point(200)
        self.assertEqual(server.completion_payloads[2]["prompt"], [10, 20])
        self.assertEqual(server.completion_payloads[2]["n_predict"], 1)
        self.assertEqual(measured.call_count, 4)  # preparation is not a measured sample
        self.assertEqual((result["total_ctx"], result["cache_n"], result["prompt_n"]), (5, 2, 3))

    def test_identical_full_cache_rebuilds_are_rejected_before_aggregation(self):
        server = MockLlamaServer()
        recording = RecordingSpy()
        run = runner(server, recording, repeat=3)
        run.measure_point(100)
        server.completions = [server._timings(cache_n=0, prompt_n=200)] * 3
        with self.assertRaisesRegex(RuntimeError, "Incremental prefix reuse lost.*cache_n=0, expected 100"):
            run.measure_point(200)
        self.assertEqual(len(recording.rows), 1)

    def test_prepared_snapshot_must_not_include_a_decode_tail(self):
        server = MockLlamaServer(save_count=101)
        recording = RecordingSpy()
        run = runner(server, recording, repeat=2)
        run.measure_point(100)
        with self.assertRaisesRegex(RuntimeError, "Prefix snapshot token mismatch.*101 != 100"):
            run.measure_point(200)
        self.assertEqual(len(recording.rows), 1)
        self.assertNotIn("restore", server.actions)

    def test_cache_and_prompt_count_mismatch_fails_before_writing_csv_row(self):
        server = MockLlamaServer(completions=[
            MockLlamaServer._timings(cache_n=4, prompt_n=20),
            MockLlamaServer._timings(cache_n=5, prompt_n=20),
        ])
        recording = RecordingSpy()
        run = runner(server, recording, repeat=2)

        with self.assertRaisesRegex(RuntimeError, r"Non-comparable prefill.*cache_n, prompt_n"):
            run.measure_point(100)
        self.assertEqual(recording.rows, [])

    def test_repeat_one_never_uses_snapshot_api(self):
        server = MockLlamaServer()
        recording = RecordingSpy()
        run = runner(server, recording, repeat=1)

        run.measure_point(100)
        run.measure_point(200)
        self.assertEqual(server.actions, [])
        self.assertEqual(server.events, ["completion", "completion"])
        self.assertEqual([p["cache_prompt"] for p in server.completion_payloads], [False, True])

    def test_cold_mode_erases_each_repeat_and_keeps_cache_disabled(self):
        server = MockLlamaServer()
        recording = RecordingSpy()
        run = runner(server, recording, cache_mode="cold", repeat=3)

        run.measure_point(100)
        self.assertEqual(server.actions, ["erase", "erase", "erase"])
        self.assertEqual([p["cache_prompt"] for p in server.completion_payloads],
                         [False, False, False])

    def test_cold_mode_aborts_if_slot_erase_fails(self):
        server = MockLlamaServer()
        recording = RecordingSpy()
        run = runner(server, recording, cache_mode="cold", repeat=1)

        with patch.object(benchmark, "reset_slot", return_value=False), \
             self.assertRaisesRegex(RuntimeError, "Cold measurement requires a successful erase"):
            run.measure_point(100)

        self.assertEqual(server.completion_payloads, [])
        self.assertEqual(recording.samples, [])
        self.assertEqual(recording.rows, [])

    def test_prefill_telemetry_and_timing_aggregates_use_all_repeats(self):
        sample = {
            "cache_n": 4,
            "prompt_n": 20,
            "prompt_ms": 0,
            "prefill_tps": 0,
            "decode_tps": 10,
            "predicted_n": 4,
            "predicted_ms": 200,
            "draft_n": 0,
            "draft_acc": 0,
            "wall_s": 1,
            "truncated": False,
            "status": "OK",
        }
        rows = []
        for prompt_ms, prefill_tps, pcie, gpm in ((10, 100, 1, 10), (20, 200, 2, 20), (100, 1000, 100, 100)):
            row = dict(sample)
            row.update(
                prompt_ms=prompt_ms,
                prefill_tps=prefill_tps,
                pcie_prefill_rx_p95_mb_s=pcie,
                gpm_prefill_sm_p95_pct=gpm,
            )
            rows.append(row)

        result = benchmark.aggregate_point(rows, 100, 100, 400,
                                           SimpleNamespace(cache_mode="incremental"))
        self.assertEqual(result["prompt_ms"], 20)
        self.assertEqual(result["prefill_tps"], 200)
        self.assertEqual(result["pcie_prefill_rx_p95_mb_s"], 2)
        self.assertEqual(result["gpm_prefill_sm_p95_pct"], 20)

    def test_snapshot_api_failure_contains_server_body_and_recovery_hint(self):
        server = MockLlamaServer(snapshot_failure=FakeResponse(
            {"error": "slot directory missing"}, status_code=500,
            text="slot directory missing"))

        with self.assertRaisesRegex(RuntimeError, r"Slot save failed \(HTTP 500\).*slot directory missing") as raised:
            benchmark.slot_snapshot("http://mock-server", 7, "save", "snapshot.bin", http=server)
        self.assertIn("--slot-save-path", str(raised.exception))

    def test_csv_schemas_have_unique_columns_and_include_telemetry(self):
        for name in ("VRAM_CSV_FIELDS", "PCIE_CSV_FIELDS", "GPM_CSV_FIELDS",
                     "WINDOWS_CSV_FIELDS", "RESULT_CSV_FIELDS"):
            with self.subTest(name=name):
                fields = tuple(getattr(benchmark, name))
                self.assertEqual(len(fields), len(set(fields)))
                self.assertTrue(all(isinstance(field, str) and field for field in fields))
        self.assertTrue(set(benchmark.telemetry_fields()).issubset(benchmark.RESULT_CSV_FIELDS))


if __name__ == "__main__":
    unittest.main()
