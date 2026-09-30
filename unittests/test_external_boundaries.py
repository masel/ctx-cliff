"""Offline tests for process, HTTP and NVIDIA system boundaries."""

import io
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

from test_csv_recording import benchmark as b


class Response:
    def __init__(self, status=200, data=None, text=""):
        self.status_code = status
        self._data = data
        self.text = text
        self.ok = 200 <= status < 400

    def raise_for_status(self):
        if not self.ok:
            raise b.requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        if isinstance(self._data, BaseException):
            raise self._data
        return self._data


class ManagedServerTests(unittest.TestCase):
    def server(self, **overrides):
        values = dict(command='llama-server --model "model with spaces.gguf"',
                      base="http://127.0.0.1:8080", startup_timeout=1,
                      log_path=None, http=Mock())
        values.update(overrides)
        return b.ManagedLlamaServer(**values)

    def test_popen_command_uses_posix_argv_and_preserves_quoted_groups(self):
        server = self.server()
        with patch.object(b.os, "name", "posix"):
            self.assertEqual(server._popen_command(),
                             ["llama-server", "--model", "model with spaces.gguf"])

    def test_popen_command_keeps_complete_windows_command_line(self):
        server = self.server()
        with patch.object(b.os, "name", "nt"):
            self.assertEqual(server._popen_command(), server.command)

    def test_start_passes_stdio_and_returns_when_health_is_ready(self):
        proc = Mock()
        proc.poll.return_value = None
        server = self.server()
        with patch.object(b.subprocess, "Popen", return_value=proc) as popen, \
             patch.object(b, "server_is_ready", return_value=True):
            server.start()
        popen.assert_called_once_with(server._popen_command(), stdout=None, stderr=None)
        self.assertIs(server.proc, proc)

    def test_immediate_exit_closes_log_and_reports_exit_code(self):
        proc = Mock()
        proc.poll.return_value = 17
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            server = self.server(log_path=str(log))
            with patch.object(b.subprocess, "Popen", return_value=proc), \
                 self.assertRaisesRegex(RuntimeError, r"code 17.*server\.log"):
                server.start()
        self.assertIsNone(server.log_file)
        proc.terminate.assert_not_called()

    def test_start_timeout_stops_process_and_closes_log(self):
        proc = Mock()
        proc.poll.return_value = None
        with tempfile.TemporaryDirectory() as directory:
            server = self.server(startup_timeout=0, log_path=str(Path(directory) / "server.log"))
            with patch.object(b.subprocess, "Popen", return_value=proc), \
                 self.assertRaisesRegex(TimeoutError, "did not become ready"):
                server.start()
        proc.terminate.assert_called_once_with()
        proc.wait.assert_called_once_with(timeout=10)
        self.assertIsNone(server.proc)
        self.assertIsNone(server.log_file)

    def test_stop_escalates_from_terminate_to_kill(self):
        proc = Mock()
        proc.poll.return_value = None
        proc.wait.side_effect = [subprocess.TimeoutExpired("llama-server", 10), None]
        server = self.server()
        server.proc = proc
        server.stop()
        proc.terminate.assert_called_once_with()
        proc.kill.assert_called_once_with()
        self.assertEqual(proc.wait.call_args_list, [call(timeout=10), call(timeout=5)])

    def test_popen_failure_closes_open_log(self):
        with tempfile.TemporaryDirectory() as directory:
            server = self.server(log_path=str(Path(directory) / "server.log"))
            with patch.object(b.subprocess, "Popen", side_effect=OSError("cannot execute")), \
                 self.assertRaisesRegex(OSError, "cannot execute"):
                server.start()
        self.assertIsNone(server.log_file)


class HttpBoundaryTests(unittest.TestCase):
    def test_tokenize_retries_with_legacy_payload(self):
        http = Mock()
        http.post.side_effect = [Response(400), Response(data={"tokens": [1, 2, 3]})]
        self.assertEqual(b.tokenize("http://test", "hello", add_bos=False, http=http), [1, 2, 3])
        self.assertEqual(http.post.call_count, 2)
        first, second = (entry.kwargs["json"] for entry in http.post.call_args_list)
        self.assertEqual(first, {"content": "hello", "add_special": False, "parse_special": True})
        self.assertEqual(second, {"content": "hello", "add_bos": False, "special": True})

    def test_tokenize_raises_last_schema_error_after_both_payloads_fail(self):
        http = Mock()
        http.post.side_effect = [Response(data={}), Response(data={"not_tokens": []})]
        with self.assertRaises(KeyError):
            b.tokenize("http://test", "hello", http=http)
        self.assertEqual(http.post.call_count, 2)

    def test_reset_slot_tries_all_compatible_endpoints(self):
        http = Mock()
        http.request.side_effect = [b.requests.Timeout("slow"), Response(404), Response(200)]
        self.assertTrue(b.reset_slot("http://test", slot_id=4, http=http))
        self.assertEqual(
            [entry.args[:2] for entry in http.request.call_args_list],
            [("post", "http://test/slots/4?action=erase"),
             ("post", "http://test/slots?action=erase&id_slot=4"),
             ("get", "http://test/slots?action=erase&id_slot=4")],
        )

    def test_reset_failure_warning_applies_to_both_cache_modes(self):
        http = Mock()
        http.request.return_value = Response(500)
        with patch.object(sys, "stderr", new_callable=io.StringIO) as output:
            self.assertFalse(b.reset_slot("http://test", slot_id=4, http=http))
        self.assertIn("slot state may be retained", output.getvalue())
        self.assertNotIn("cold measurements", output.getvalue())

    def test_detect_slot_context_prefers_matching_slot(self):
        http = Mock()
        http.get.return_value = Response(data=[{"id": "1", "n_ctx": "8192"},
                                               {"id": 2, "n_ctx": 4096}])
        self.assertEqual(b.detect_slot_n_ctx("http://test", 1, http=http), 8192)
        http.get.assert_called_once_with("http://test/slots", timeout=10)

    def test_detect_slot_context_falls_back_to_props_after_invalid_slots_json(self):
        http = Mock()
        http.get.side_effect = [Response(data=ValueError("bad json")),
                                Response(data={"default_generation_settings": {"n_ctx": "32768"}})]
        self.assertEqual(b.detect_slot_n_ctx("http://test", 0, http=http), 32768)
        self.assertEqual([entry.args[0] for entry in http.get.call_args_list],
                         ["http://test/slots", "http://test/props"])

    def test_completion_builds_deterministic_payload(self):
        http = Mock()
        http.post.return_value = Response(data={"timings": {"predicted_n": 4}})
        result = b.completion("http://test", [1, 2], 4, True, False, 3, True, http=http)
        self.assertEqual(result["timings"]["predicted_n"], 4)
        self.assertEqual(http.post.call_args.kwargs["json"], {
            "prompt": [1, 2], "n_predict": 4, "stream": False,
            "cache_prompt": False, "id_slot": 3, "ignore_eos": True,
            "temperature": 0.0, "top_k": 1,
        })

    def test_completion_extracts_structured_error_metadata(self):
        http = Mock()
        http.post.return_value = Response(
            400,
            {"error": {"message": "context too large", "type": "exceed_context_size",
                       "n_ctx": "4096", "n_prompt_tokens": 5000}},
            "raw response",
        )
        with self.assertRaises(b.CompletionRequestError) as raised:
            b.completion("http://test", "prompt", 4, False, True, 0, False, http=http)
        error = raised.exception
        self.assertEqual((error.status_code, error.message, error.error_type),
                         (400, "context too large", "exceed_context_size"))
        self.assertEqual((error.n_ctx, error.n_prompt_tokens), (4096, 5000))
        self.assertEqual(error.body, "raw response")
        self.assertTrue(error.is_context_overflow)

    def test_completion_preserves_plain_text_error_and_caps_body(self):
        http = Mock()
        http.post.return_value = Response(503, ValueError("not json"), "x" * 5000)
        with self.assertRaises(b.CompletionRequestError) as raised:
            b.completion("http://test", "prompt", 4, False, True, 0, False, http=http)
        self.assertEqual(raised.exception.message, "x" * 4000)
        self.assertEqual(len(raised.exception.body), 4000)


class NvidiaBoundaryTests(unittest.TestCase):
    def monitor(self, **kwargs):
        monitor = b.NvidiaVramMonitor(**kwargs)
        self.addCleanup(monitor.samples.close)
        self.addCleanup(monitor.pcie_samples.close)
        return monitor

    def test_command_adds_gpu_selector_only_when_requested(self):
        self.assertNotIn("-i", self.monitor()._command_for_fields(["index"]))
        command = self.monitor(gpu="0,1")._command_for_fields(["index", "memory.used"])
        self.assertEqual(command[-2:], ["-i", "0,1"])

    def test_probe_keeps_only_supported_optional_fields(self):
        monitor = self.monitor()

        def run(command, **kwargs):
            query = next(value for value in command if value.startswith("--query-gpu="))
            requested = query.split("=", 1)[1]
            supported = requested == ",".join(monitor.BASE_FIELDS) or requested == "index,power.draw"
            return SimpleNamespace(returncode=0 if supported else 1, stdout="", stderr="unsupported")

        with patch.object(b.subprocess, "run", side_effect=run):
            self.assertTrue(monitor._probe_fields())
        self.assertEqual(monitor.fields, monitor.BASE_FIELDS + ["power.draw"])

    def test_probe_reports_base_query_failure(self):
        monitor = self.monitor()
        result = SimpleNamespace(returncode=9, stdout="", stderr="driver unavailable")
        with patch.object(b.subprocess, "run", return_value=result):
            self.assertFalse(monitor._probe_fields())
        self.assertEqual(monitor.error, "driver unavailable")

    def test_reader_skips_malformed_rows_and_preserves_missing_optional_values(self):
        monitor = self.monitor()
        monitor.fields = monitor.BASE_FIELDS + ["power.draw"]
        monitor.current_label = {"phase": "decode", "target_ctx": 2048, "repeat": 2}
        monitor.proc = SimpleNamespace(stdout=io.StringIO(
            "malformed\n"
            "0, N/A, 1000, 50, 60, 70\n"
            "0, 250, 1000, 50, 60, N/A\n"
        ))
        monitor.running = True
        monitor._reader()
        rows = list(monitor.samples.iter_all())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["used_pct"], 25)
        self.assertIsNone(rows[0]["power_draw_w"])
        self.assertEqual((rows[0]["phase"], rows[0]["target_ctx"], rows[0]["repeat"]),
                         ("decode", 2048, 2))

    def test_unexpected_nvidia_smi_output_end_is_reported(self):
        monitor = self.monitor()
        monitor.proc = SimpleNamespace(stdout=io.StringIO(""))
        monitor.running = True
        monitor._reader()
        self.assertEqual(monitor.error, "nvidia-smi monitor output ended unexpectedly")

    def test_start_returns_clear_error_when_nvidia_smi_is_missing(self):
        monitor = self.monitor()
        with patch.object(b.shutil, "which", return_value=None), patch.dict("sys.modules", {"pynvml": None}):
            self.assertFalse(monitor.start())
        # Neither backend works: the error names nvidia-smi and why NVML failed.
        self.assertTrue(monitor.error.startswith("nvidia-smi not found (NVML sampler: pynvml unavailable"))

    def test_early_nvidia_smi_exit_stops_pcie_thread_before_nvml_shutdown(self):
        monitor = self.monitor()
        # Pin the backend: with "auto", a machine with pynvml and a GPU would first
        # run the real NVML sampler and record its shutdown events as well.
        monitor.backend_preference = "nvidia-smi"
        self.addCleanup(monitor.stop)
        events = []

        class FakeThread:
            def __init__(self, target, name, daemon):
                self.name = name
                self.alive = False

            def start(self):
                self.alive = True

            def join(self, timeout=None):
                events.append(f"join:{self.name}")
                self.alive = False

            def is_alive(self):
                return self.alive

        def init_pcie():
            monitor.pcie_source = "nvml"
            monitor._pynvml = SimpleNamespace(nvmlShutdown=lambda: events.append("shutdown"))
            monitor._nvml_initialized = True
            return True

        proc = Mock()
        proc.poll.return_value = 1
        with patch.object(b.shutil, "which", return_value="nvidia-smi"), \
             patch.object(monitor, "_probe_fields", return_value=True), \
             patch.object(monitor, "_init_nvml_pcie", side_effect=init_pcie), \
             patch.object(b.subprocess, "Popen", return_value=proc), \
             patch.object(b.threading, "Thread", FakeThread):
            self.assertFalse(monitor.start())

        self.assertTrue(monitor.stop_event.is_set())
        self.assertEqual(events, ["join:nvml-pcie-monitor", "join:nvidia-telemetry-monitor", "shutdown"])

    def test_nvml_is_shutdown_when_initialization_fails_after_nvml_init(self):
        monitor = self.monitor()
        fake_nvml = SimpleNamespace(
            nvmlInit=Mock(),
            nvmlShutdown=Mock(),
            nvmlDeviceGetCount=Mock(side_effect=RuntimeError("driver failure")),
        )
        with patch.dict(sys.modules, {"pynvml": fake_nvml}):
            self.assertFalse(monitor._init_nvml_pcie())
        fake_nvml.nvmlShutdown.assert_called_once_with()
        self.assertFalse(monitor._nvml_initialized)
        self.assertIsNone(monitor._pynvml)


if __name__ == "__main__":
    unittest.main()
