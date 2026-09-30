"""Token-budget and phase-quality regressions for benchmark comparisons."""
import contextlib
import io
import re
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_csv_recording import benchmark as b
from test_measurement import sample


class PromptQualityTests(unittest.TestCase):
    def test_special_tokens_fit_budget_in_warmup_and_all_measurement_modes(self):
        for mode, repeats in (("cold", 2), ("incremental", 1), ("incremental", 2)):
            with self.subTest(mode=mode, repeats=repeats), tempfile.TemporaryDirectory() as folder:
                source = Path(folder) / "input.txt"
                source.write_text("x" * 1000, encoding="utf-8")
                prompts = []

                def tokens(base, text, add_bos=True, **kwargs):
                    return ([101, 102] if add_bos else []) + [ord(c) for c in text]

                def complete(base, prompt, n_predict, *args, **kwargs):
                    self.assertIsInstance(prompt, list)
                    self.assertEqual(prompt[:2], [101, 102])
                    self.assertLessEqual(len(prompt), 120)
                    prompts.append(prompt)
                    return {"timings": {"cache_n": 0, "prompt_n": len(prompt),
                                        "prompt_ms": 10, "predicted_n": n_predict,
                                        "predicted_ms": 10}}

                argv = ["ctx-cliff.py", "--file", str(source), "--start", "120", "--end", "120",
                        "--n-predict", "8", "--repeat", str(repeats), "--warmup", "1",
                        "--cache-mode", mode, "--settle", "0", "--vram-log", "off",
                        "--gpm-log", "off", "--win-gpu-mem", "off"]
                with patch("sys.argv", argv), patch.object(b, "tokenize", side_effect=tokens), \
                     patch.object(b, "completion", side_effect=complete), \
                     patch.object(b, "server_is_ready", return_value=True), \
                     patch.object(b, "detect_slot_n_ctx", return_value=129), \
                     patch.object(b, "reset_slot", return_value=True), \
                     patch.object(b, "probe_prefill_repeats"), \
                     contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    b.main()
                self.assertEqual(len(prompts), repeats + 1)

    def test_final_tokenization_is_checked_before_completion(self):
        builder = Mock(max_prompt_ctx=100, last_tokens=("text", list(range(101))))
        builder.build.return_value = ("text", 100, 4)
        runner = b.BenchmarkRunner(SimpleNamespace(), "http://test", object(), builder,
                                   Mock(), None, None, None)
        with patch.object(b, "completion") as complete:
            with self.assertRaises(b.PromptBuildError):
                runner.measure_point(100)
        complete.assert_not_called()

    def test_prefill_can_be_valid_when_decode_stops_early(self):
        row = b.aggregate_point([sample(status="STOP@2", predicted_n=2)] * 3,
                                100, 100, 400, SimpleNamespace(cache_mode="cold"))
        self.assertEqual(row["valid_repeats"], 0)
        self.assertEqual(row["prefill_valid_repeats"], 3)
        self.assertEqual(row["prefill_tps"], 1000)

    def test_truncated_and_invalid_prefill_are_excluded(self):
        samples = [sample(), sample(truncated=True, status="TRUNC", prefill_tps=1),
                   sample(prompt_ms=float("nan"), prefill_tps=2), sample(prompt_n=0)]
        row = b.aggregate_point(samples, 100, 100, 400, SimpleNamespace(cache_mode="cold"))
        self.assertEqual(row["prefill_valid_repeats"], 1)
        self.assertEqual(row["prefill_repeats"], 4)
        self.assertEqual(row["prefill_tps"], 1000)
        self.assertEqual(row["prompt_ms"], 100)

    def test_cliff_requires_phase_specific_repeat_quality(self):
        rows = [dict(total_ctx=100, decode_tps_median=100, prefill_tps=100,
                     valid_repeats=3, prefill_valid_repeats=3),
                dict(total_ctx=200, decode_tps_median=10, prefill_tps=10,
                     valid_repeats=1, prefill_valid_repeats=3)]
        self.assertIsNone(b.largest_decode_drop(rows))
        self.assertEqual(b.largest_prefill_drop(rows)[0], 90)
        self.assertEqual(b.largest_decode_drop(rows, min_valid_repeats=1)[0], 90)
        rows[1].update(valid_repeats=3, prefill_valid_repeats=1)
        self.assertIsNone(b.largest_prefill_drop(rows))
        self.assertEqual(b.largest_decode_drop(rows)[0], 90)

    def test_cliff_does_not_bridge_low_quality_or_missing_quality_points(self):
        rows = [dict(total_ctx=100, decode_tps_median=100, valid_repeats=3),
                dict(total_ctx=200, decode_tps_median=50, valid_repeats=1),
                dict(total_ctx=300, decode_tps_median=10, valid_repeats=3)]
        self.assertIsNone(b.largest_decode_drop(rows))
        for row in rows:
            del row["valid_repeats"]
        self.assertIsNone(b.largest_decode_drop(rows))

    def test_no_valid_prefill_has_zero_count_and_no_cliff(self):
        row = b.aggregate_point([sample(prompt_ms=0)] * 3, 100, 100, 400,
                                SimpleNamespace(cache_mode="cold"))
        self.assertEqual(row["prefill_valid_repeats"], 0)
        # Missing is empty, never a fake measured zero.
        for key in ("prefill_tps", "prompt_ms", "prompt_n", "cache_n"):
            self.assertIsNone(row[key])
        self.assertIsNone(b.largest_prefill_drop([row, dict(row, total_ctx=200)]))



class TokenCutTests(unittest.TestCase):
    """Prompts are prefixes of the whole-file tokenization."""
    CONTENT = "def get_user_model():\n    return settings.AUTH_USER_MODEL\n\n" * 400
    NONCE = "[ctx-cliff run=test] "

    @staticmethod
    def word_tokenizer(calls=None):
        # A cut inside a word yields a different (non-canonical) final token, like BPE.
        def tokenize(text):
            if calls is not None:
                calls.append(len(text))
            return [0] + [hash(piece) % 100000 for piece in re.findall(r"\s*\S+|\s+", text)]
        return tokenize

    def builder(self, content=None, cap=None, detok=None, calls=None, tokenizer=None):
        tokenize = tokenizer or self.word_tokenizer(calls)
        nonce_tokens = len(self.word_tokenizer()(self.NONCE))
        return b.PromptBuilder(content or self.CONTENT, self.NONCE, tokenize, 3.0,
                               nonce_tokens, cap, detokenize=detok)

    def test_prompt_is_exact_prefix_of_full_tokenization(self):
        full = self.word_tokenizer()(self.NONCE + self.CONTENT)
        builder = self.builder()
        previous = None
        for target in (50, 333, 1000, 1500):
            text, count, _ = builder.build(target)
            tokens = builder.last_tokens[1]
            self.assertEqual(builder.last_tokens[0], text)
            self.assertEqual(count, target)
            self.assertEqual(tokens, full[:target])
            if previous is not None:  # incremental reuse: each point extends the previous one
                self.assertEqual(tokens[:len(previous)], previous)
            previous = tokens

    def test_short_input_returns_whole_file_without_guard_or_warning(self):
        content = "alpha beta gamma " * 20
        full = self.word_tokenizer()(self.NONCE + content)
        builder = self.builder(content=content)
        with contextlib.redirect_stderr(io.StringIO()) as err:
            _, count, chars = builder.build(5000)
        self.assertEqual(builder.last_tokens[1], full)
        self.assertEqual(count, len(full))
        self.assertEqual(chars, len(content))
        self.assertEqual(err.getvalue(), "")

    def test_cap_limits_token_count(self):
        builder = self.builder(cap=120)
        self.assertEqual(builder.build(1000)[1], 120)

    def test_detokenize_gives_exact_input_characters(self):
        # Fake detokenizer: reconstructs text from a lookup of the full tokenization.
        tokenize = self.word_tokenizer()
        pieces = {hash(p) % 100000: p for p in re.findall(r"\s*\S+|\s+", self.NONCE + self.CONTENT)}
        detok = lambda ids: "<s>" + "".join(pieces[i] for i in ids[1:])
        _, count, chars = self.builder(detok=detok).build(301)
        expected = "".join(pieces[i] for i in tokenize(self.NONCE + self.CONTENT)[1:301])
        self.assertEqual(chars, len(expected) - len(self.NONCE))

    def test_failing_detokenize_falls_back_to_estimate(self):
        def detok(ids):
            raise b.requests.ConnectionError("no /detokenize")
        _, count, chars = self.builder(detok=detok).build(301)
        self.assertEqual(count, 301)
        self.assertGreater(chars, 0)
        self.assertLess(chars, len(self.CONTENT))

    def test_counting_tokenizer_is_rejected(self):
        with self.assertRaises(b.PromptBuildError):
            self.builder(tokenizer=len).build(200)

    def test_nonce_larger_than_budget_fails(self):
        with self.assertRaises(b.PromptBuildError):
            self.builder().build(1)

    def test_runner_submits_truncated_tokens_without_retokenizing(self):
        builder = self.builder()
        runner = b.BenchmarkRunner(SimpleNamespace(), "http://x", object(), builder, Mock(), None, None, None)
        with patch.object(b, "tokenize") as tokenize:
            tokens, count, _ = runner.build_prompt(400)
        tokenize.assert_not_called()
        self.assertEqual((len(tokens), count), (400, 400))


if __name__ == "__main__":
    unittest.main()
