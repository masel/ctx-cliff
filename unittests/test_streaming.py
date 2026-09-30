"""Streaming completions: the first generated token anchors the prefill/decode boundary."""
import contextlib
import io
import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_csv_recording import SCRIPT, benchmark as b


def sse(*chunks):
    return [f"data: {json.dumps(c)}".encode() for c in chunks]


class StreamParsingTests(unittest.TestCase):
    def test_first_token_time_content_and_final_chunk(self):
        clock = itertools.count(10.0, 0.5).__next__
        lines = [b"", *sse({"content": "", "stop": False}, {"content": "Hel", "stop": False},
                          {"content": "lo", "stop": False},
                          {"content": "", "stop": True, "timings": {"prompt_ms": 5}, "stop_type": "limit"})]
        result = b.read_completion_stream(lines, clock)
        self.assertEqual(result["content"], "Hello")
        self.assertEqual(result["timings"], {"prompt_ms": 5})
        self.assertEqual(result["stop_type"], "limit")
        self.assertEqual(result["_first_token_mono"], 10.0)

    def test_done_marker_without_final_chunk_is_an_error(self):
        with self.assertRaises(b.CompletionRequestError):
            b.read_completion_stream(sse({"content": "x", "stop": False}) + [b"data: [DONE]"])

    def test_error_chunk_raises_with_message(self):
        with self.assertRaisesRegex(b.CompletionRequestError, "context size exceeded"):
            b.read_completion_stream(sse({"error": {"message": "context size exceeded",
                                                    "type": "exceed_context_size_error"}}))

    def test_plain_json_body_is_accepted(self):
        body = json.dumps({"timings": {"predicted_n": 3}, "content": "abc"})
        result = b.read_completion_stream([body.encode()])
        self.assertEqual(result["content"], "abc")
        self.assertNotIn("_first_token_mono", result)

    def test_completion_posts_stream_request_and_closes(self):
        response = Mock(ok=True, **{"iter_lines.return_value": sse({"content": "a", "stop": False},
                                                                     {"stop": True, "timings": {}})})
        http = Mock(**{"post.return_value": response})
        result = b.completion("http://x", [1, 2], 4, True, cache_prompt=True, slot_id=0, ignore_eos=True,
                              http=http, stream=True)
        _, kwargs = http.post.call_args
        self.assertTrue(kwargs["stream"])
        self.assertTrue(kwargs["json"]["stream"])
        self.assertEqual(result["content"], "a")
        response.close.assert_called_once_with()

    def test_non_streaming_request_is_unchanged(self):
        response = Mock(ok=True, **{"json.return_value": {"content": "z"}})
        http = Mock(**{"post.return_value": response})
        b.completion("http://x", [1], 4, False, cache_prompt=False, slot_id=0, ignore_eos=False, http=http)
        _, kwargs = http.post.call_args
        self.assertNotIn("stream", kwargs)
        self.assertFalse(kwargs["json"]["stream"])


class AnchorTests(unittest.TestCase):
    def test_first_token_anchors_the_boundary(self):
        windows, quality = b.reconstruct_phases(10.0, 13.0, 1000, 1500, first_token=11.2)
        self.assertEqual(windows, (("prefill", 10.2, 11.2), ("decode", 11.2, 13.0)))
        self.assertEqual(quality["phase_method"], "first_token_anchor")
        self.assertAlmostEqual(quality["phase_anchor_offset_ms"], 300.0)  # back-projection: 11.5

    def test_missing_or_implausible_anchor_falls_back(self):
        for first in (None, 9.0, 14.0):
            windows, quality = b.reconstruct_phases(10.0, 13.0, 1000, 1500, first_token=first)
            self.assertEqual(quality["phase_method"], "response_end_backprojection")
            self.assertEqual(windows[1][1], 11.5)
            self.assertIsNone(quality["phase_anchor_offset_ms"])

    def test_aggregate_reports_methods_and_offset(self):
        base = dict(cache_n=0, prompt_n=10, prompt_ms=10, prefill_tps=100, decode_tps=5, predicted_n=8,
                    predicted_ms=100, draft_n=0, draft_acc=0, wall_s=1, truncated=False, status="OK")
        row = b.aggregate_point([dict(base, phase_method="first_token_anchor", phase_anchor_offset_ms=20.0),
                                 dict(base, phase_method="first_token_anchor", phase_anchor_offset_ms=40.0)],
                                1, 1, 1, SimpleNamespace(cache_mode="cold"))
        self.assertEqual(row["phase_method"], "first_token_anchor")
        self.assertEqual(row["phase_anchor_offset_ms_median"], 30.0)
        row = b.aggregate_point([base, dict(base, phase_method="first_token_anchor")], 1, 1, 1,
                                SimpleNamespace(cache_mode="cold"))
        self.assertEqual(row["phase_method"], "first_token_anchor/response_end_backprojection")


class RunnerStreamingTests(unittest.TestCase):
    def run_main(self, *extra):
        seen = []

        def complete(*a, **k):
            seen.append(k.get("stream"))
            import time
            time.sleep(0.01)
            return {"timings": {"predicted_n": 8, "predicted_ms": 5, "prompt_n": 100, "prompt_ms": 5},
                    "_first_token_mono": time.perf_counter() - 0.002 if k.get("stream") else None}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "in.txt").write_text("text " * 1000, encoding="utf-8")
            argv = [str(SCRIPT), "--file", str(root / "in.txt"), "--csv", str(root / "run.csv"),
                    "--start", "100", "--end", "100", "--n-predict", "8", "--repeat", "1", "--warmup", "0",
                    "--vram-log", "off", "--gpm-log", "off", "--win-gpu-mem", "off", "--settle", "0", *extra]
            with patch("sys.argv", argv), patch.object(b, "server_is_ready", return_value=True), \
                    patch.object(b, "detect_slot_n_ctx", return_value=10000), \
                    patch.object(b, "tokenize", side_effect=lambda base, text, **kw: [0] * len(text)), \
                    patch.object(b, "reset_slot", return_value=True), \
                    patch.object(b, "fetch_server_props", return_value={}), \
                    patch.object(b, "completion", side_effect=complete), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                b.main()
            samples = (root / "run.samples.csv").read_text(encoding="utf-8")
        return seen, samples

    def test_streaming_is_default_and_anchors_phases(self):
        seen, samples = self.run_main()
        self.assertTrue(all(seen))
        self.assertIn("first_token_anchor", samples)

    def test_no_stream_option(self):
        seen, samples = self.run_main("--no-stream")
        self.assertEqual(set(seen), {None})
        self.assertNotIn("first_token_anchor", samples)


if __name__ == "__main__":
    unittest.main()
