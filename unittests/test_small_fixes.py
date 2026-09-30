"""Token reuse, input exhaustion, tokenizer timeouts and shared NVML handle lookup."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_csv_recording import SCRIPT, benchmark as b


class TokenReuseTests(unittest.TestCase):
    def test_builder_keeps_token_list_of_chosen_prompt(self):
        calls = []

        def tokenize(prompt):
            calls.append(prompt)
            return [1] + [ord(c) for c in prompt]

        builder = b.PromptBuilder("abcdefghij" * 50, "N:", tokenize, 1.0, 3)
        text, count, chars = builder.build(120)
        self.assertEqual(builder.last_tokens[0], text)
        self.assertEqual(len(builder.last_tokens[1]), count)
        self.assertEqual(count, 120)

    def test_runner_rejects_a_builder_without_token_ids(self):
        builder = Mock(max_prompt_ctx=None, last_tokens=None)
        builder.build.return_value = ("text", 100, 4)
        runner = b.BenchmarkRunner(SimpleNamespace(), "http://x", object(), builder, Mock(), None, None, None)
        with self.assertRaises(b.PromptBuildError):
            runner.build_prompt(100)

    def test_runner_does_not_tokenize_the_final_prompt_again(self):
        builder = b.PromptBuilder("abcdefghij" * 50, "N:", lambda p: [1] + [ord(c) for c in p], 1.0, 3)
        runner = b.BenchmarkRunner(SimpleNamespace(), "http://x", object(), builder, Mock(), None, None, None)
        with patch.object(b, "tokenize") as tokenize:
            tokens, count, _ = runner.build_prompt(120)
        tokenize.assert_not_called()
        self.assertEqual(len(tokens), count)
        # The builder's cached list is not shared with the caller.
        tokens.append(0)
        self.assertEqual(len(builder.last_tokens[1]), count)


class TokenizeTimeoutTests(unittest.TestCase):
    def test_network_errors_are_not_retried_with_the_legacy_payload(self):
        for error in (b.requests.Timeout("slow"), b.requests.ConnectionError("down")):
            http = Mock()
            http.post.side_effect = error
            with self.subTest(error=type(error).__name__), self.assertRaises(type(error)):
                b.tokenize("http://x", "text", http=http)
            self.assertEqual(http.post.call_count, 1)

    def test_api_errors_still_fall_back(self):
        bad = Mock(**{"raise_for_status.side_effect": b.requests.HTTPError("400")})
        good = Mock(**{"raise_for_status.return_value": None, "json.return_value": {"tokens": [1, 2]}})
        http = Mock(**{"post.side_effect": [bad, good]})
        self.assertEqual(b.tokenize("http://x", "text", http=http), [1, 2])


class InputExhaustionTests(unittest.TestCase):
    def run_main(self, text, *extra):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "in.txt").write_text(text, encoding="utf-8")
            argv = [str(SCRIPT), "--file", str(root / "in.txt"), "--csv", str(root / "run.csv"),
                    "--start", "100", "--end", "400", "--step", "100", "--n-predict", "8", "--repeat", "1",
                    "--warmup", "0", "--vram-log", "off", "--gpm-log", "off", "--win-gpu-mem", "off",
                    "--settle", "0", "--no-drift-check", *extra]
            with patch("sys.argv", argv), patch.object(b, "server_is_ready", return_value=True), \
                    patch.object(b, "detect_slot_n_ctx", return_value=10000), \
                    patch.object(b, "tokenize", side_effect=lambda base, text, **kw: [0] * len(text)), \
                    patch.object(b, "reset_slot", return_value=True), \
                    patch.object(b, "fetch_server_props", return_value={}), \
                    patch.object(b, "completion", return_value={"timings": {
                        "predicted_n": 8, "predicted_ms": 80, "prompt_n": 100, "prompt_ms": 50}}), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
                b.main()
            meta = json.loads((root / "run.meta.json").read_text(encoding="utf-8"))
            rows = (root / "run.csv").read_text(encoding="utf-8").splitlines()
        return err.getvalue(), meta, rows

    def test_short_input_warns_and_stops_when_exhausted(self):
        err, meta, rows = self.run_main("y" * 200)
        self.assertIn("holds only about", err)
        self.assertIn("Input file exhausted", err)
        self.assertEqual(meta["stop_reason"], "input_exhausted")
        self.assertEqual(meta["status"], "completed_input_exhausted")
        self.assertEqual(meta["exit_code"], 0)
        self.assertEqual(len(rows), 1 + 3)  # 100, 200, ~230 tokens; the 4th would repeat and is not measured

    def test_long_input_has_no_warning(self):
        err, meta, rows = self.run_main("y" * 5000)
        self.assertNotIn("holds only about", err)
        self.assertEqual(meta["stop_reason"], "completed")
        self.assertEqual(len(rows), 1 + 4)


class NvmlHandleTests(unittest.TestCase):
    def test_indices_uuids_and_all(self):
        nv = SimpleNamespace(
            nvmlDeviceGetCount=lambda: 2,
            nvmlDeviceGetHandleByIndex=lambda i: f"h{i}",
            nvmlDeviceGetHandleByUUID=Mock(side_effect=[TypeError("needs bytes"), "hU"]),
            nvmlDeviceGetIndex=lambda h: 1)
        self.assertEqual(b.resolve_nvml_handles(nv, "all"), [("0", "h0"), ("1", "h1")])
        self.assertEqual(b.resolve_nvml_handles(nv, "0, GPU-abc"), [("0", "h0"), ("1", "hU")])
        nv.nvmlDeviceGetHandleByUUID.assert_called_with(b"GPU-abc")


if __name__ == "__main__":
    unittest.main()


class PromptSearchTests(unittest.TestCase):
    """Regression (26.09.2026): a dense region made the old search give up and halve
    the prompt (target 70000 -> 29147 tokens) on the real 17 MB input file."""

    @staticmethod
    def tokenizer(calls):
        def tokenize(prompt):
            calls.append(1)
            dense = sum(1 for ch in prompt if ch == "一")  # 1 char/token vs 5 elsewhere
            return [0] * (1 + dense + (len(prompt) - dense) // 5)
        return tokenize

    def test_dense_region_does_not_shorten_prompts(self):
        content = ("x" * 47) * 6000 + "一" * 20000 + ("y" * 47) * 40000
        calls = []
        builder = b.PromptBuilder(content, "N: ", self.tokenizer(calls), 4.7, 1, 159936)
        with contextlib.redirect_stderr(io.StringIO()) as err:
            for target in (10000, 30000, 50000, 70000, 90000, 159936):
                count = builder.build(target)[1]
                self.assertEqual(count, target)
        self.assertEqual(err.getvalue(), "")
        self.assertLess(len(calls), 40)

    def test_short_input_returns_whole_file_without_warning(self):
        calls = []
        builder = b.PromptBuilder("z" * 1000, "N: ", self.tokenizer(calls), 5.0, 1)
        with contextlib.redirect_stderr(io.StringIO()) as err:
            prompt, count, chars = builder.build(5000)
        self.assertEqual(chars, 1000)
        self.assertEqual(err.getvalue(), "")

    def test_budget_too_small_for_nonce(self):
        builder = b.PromptBuilder("abc", "a long nonce " * 10, lambda p: [0] * len(p), 1.0, 130)
        with self.assertRaises(b.PromptBuildError):
            builder.build(20)
