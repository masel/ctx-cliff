"""--scenario agent: chat-template prompt with a fixed suffix, sampler settings."""
import contextlib
import io
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_csv_recording import benchmark as b


def word_tokenizer(text):
    # Special markers like <|end|> are one token each, like parse_special in llama-server.
    return [0] + [hash(piece) % 100000 for piece in re.findall(r"<\|\w+\|>|\s*[^\s<]+|\s+|<", text)]


class SamplingTests(unittest.TestCase):
    def test_payload_contains_only_explicit_settings_and_advances_seed(self):
        args = SimpleNamespace(temperature=0.6, top_p=None, top_k=20, min_p=None, seed=100)
        self.assertEqual(b.sampling_payload(args, 0), {"temperature": 0.6, "top_k": 20, "seed": 100})
        self.assertEqual(b.sampling_payload(args, 2)["seed"], 102)
        self.assertEqual(b.sampling_payload(SimpleNamespace(), 0), {})

    def test_completion_sends_sampling_and_deterministic_wins(self):
        http = Mock()
        http.post.return_value = Mock(ok=True, json=Mock(return_value={}))
        b.completion("http://x", [1, 2], 8, False, True, 0, False, http=http,
                     sampling={"temperature": 0.7, "seed": 5})
        payload = http.post.call_args.kwargs["json"]
        self.assertEqual((payload["temperature"], payload["seed"]), (0.7, 5))
        b.completion("http://x", [1, 2], 8, True, True, 0, False, http=http, sampling={"seed": 5})
        payload = http.post.call_args.kwargs["json"]
        self.assertEqual((payload["temperature"], payload["top_k"], payload["seed"]), (0.0, 1, 5))


class SuffixPromptTests(unittest.TestCase):
    CONTENT = "class Model:\n    def save(self):\n        return self.pk\n\n" * 300
    PREFIX = "<|system|>[ctx-cliff run=t]\nagent<|end|><|user|>Files:\n\n"
    SUFFIX = "<|end|><|assistant|>ok<|end|><|user|>task<|end|><|assistant|>"

    def builder(self, cap=None):
        suffix = word_tokenizer(self.SUFFIX)[1:]
        return b.PromptBuilder(self.CONTENT, self.PREFIX, word_tokenizer, 3.0,
                               len(word_tokenizer(self.PREFIX)), cap, suffix_tokens=suffix), suffix

    def test_suffix_follows_exact_input_prefix_and_fills_budget(self):
        builder, suffix = self.builder()
        full = word_tokenizer(self.PREFIX + self.CONTENT)
        previous = None
        for target in (100, 400, 900):
            _, count, _ = builder.build(target)
            tokens = builder.last_tokens[1]
            self.assertEqual(count, target)
            self.assertEqual(len(tokens), target)
            self.assertEqual(tokens[-len(suffix):], suffix)
            body = tokens[:-len(suffix)]
            self.assertEqual(body, full[:len(body)])
            if previous is not None:  # history grows: the previous excerpt is a prefix again
                self.assertEqual(body[:len(previous)], previous)
            previous = body

    def test_cap_includes_suffix(self):
        builder, _ = self.builder(cap=200)
        self.assertEqual(builder.build(900)[1], 200)

    def test_budget_smaller_than_prefix_and_suffix_fails(self):
        builder, suffix = self.builder()
        with self.assertRaises(b.PromptBuildError):
            builder.build(builder.nonce_tokens + len(suffix))

    def test_suffix_without_token_ids_fails_instead_of_dropping_it(self):
        builder = b.PromptBuilder(self.CONTENT, self.PREFIX, len, 3.0, 10, suffix_tokens=[1, 2])
        with self.assertRaises(b.PromptBuildError):
            builder.build(500)


class AgentTemplateTests(unittest.TestCase):
    @staticmethod
    def http(prompt):
        http = Mock()
        http.post.return_value = Mock(raise_for_status=Mock(), json=Mock(return_value={"prompt": prompt}))
        return http

    def test_parts_split_around_context_and_pass_thinking(self):
        http = self.http("<s>SYS" + b.AGENT_CONTEXT_MARKER + "<e>TASK<a>")
        prefix, suffix = b.agent_prompt_parts("http://x", "[run]", "do it", "off", http=http)
        self.assertEqual((prefix, suffix), ("<s>SYS", "<e>TASK<a>"))
        body = http.post.call_args.kwargs["json"]
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertIn("[run]", body["messages"][0]["content"])
        self.assertEqual(body["messages"][-1], {"role": "user", "content": "do it"})
        b.agent_prompt_parts("http://x", "[run]", "do it", "auto", http=http)
        self.assertNotIn("chat_template_kwargs", http.post.call_args.kwargs["json"])

    def test_template_that_drops_or_repeats_the_placeholder_is_rejected(self):
        for prompt in ("no placeholder", b.AGENT_CONTEXT_MARKER * 2):
            with self.assertRaises(b.PromptBuildError):
                b.agent_prompt_parts("http://x", "[run]", "t", "auto", http=self.http(prompt))


class AgentRunTests(unittest.TestCase):
    def test_every_request_ends_with_the_task_and_carries_the_sampler(self):
        template = ("<|system|>{run}\nagent<|end|><|user|>Files:\n\n" + b.AGENT_CONTEXT_MARKER
                    + "<|end|><|assistant|>ok<|end|><|user|>TASK<|end|><|assistant|>")
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "input.txt"
            source.write_text("def f(x):\n    return x\n\n" * 400, encoding="utf-8")
            requests_seen = []

            def tokens(base, text, add_bos=True, **kwargs):
                return word_tokenizer(text) if add_bos else word_tokenizer(text)[1:]

            def complete(base, prompt, n_predict, *args, **kwargs):
                requests_seen.append((prompt, kwargs.get("sampling")))
                return {"timings": {"cache_n": 0, "prompt_n": len(prompt), "prompt_ms": 10,
                                    "predicted_n": n_predict, "predicted_ms": 10}}

            def render(base, messages, template_kwargs=None, http=None):
                return template.replace("{run}", messages[0]["content"].split("\n")[0])

            argv = ["ctx-cliff.py", "--file", str(source), "--scenario", "agent", "--start", "300",
                    "--end", "600", "--step", "300", "--n-predict", "8", "--repeat", "2", "--warmup", "0",
                    "--cache-mode", "cold", "--settle", "0", "--vram-log", "off", "--gpm-log", "off",
                    "--win-gpu-mem", "off", "--no-drift-check", "--temperature", "0.6", "--seed", "7"]
            with patch("sys.argv", argv), patch.object(b, "tokenize", side_effect=tokens), \
                 patch.object(b, "detokenize", side_effect=RuntimeError("n/a")), \
                 patch.object(b, "apply_template", side_effect=render), \
                 patch.object(b, "completion", side_effect=complete), \
                 patch.object(b, "server_is_ready", return_value=True), \
                 patch.object(b, "detect_slot_n_ctx", return_value=4096), \
                 patch.object(b, "reset_slot", return_value=True), \
                 patch.object(b, "probe_prefill_repeats"), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                b.main()
        suffix = word_tokenizer("<|end|><|assistant|>ok<|end|><|user|>TASK<|end|><|assistant|>")[1:]
        self.assertEqual([len(p) for p, _ in requests_seen], [300, 300, 600, 600])
        for prompt, sampling in requests_seen:
            self.assertEqual(prompt[-len(suffix):], suffix)
        self.assertEqual([s["seed"] for _, s in requests_seen], [7, 8, 7, 8])
        self.assertTrue(all(s["temperature"] == 0.6 for _, s in requests_seen))


class AgentCliTests(unittest.TestCase):
    def rejected(self, *arguments):
        argv = ["ctx-cliff.py", "--file", "input.txt", *arguments]
        with patch.object(sys, "argv", argv), \
             patch.object(b, "termination_handler", return_value=contextlib.nullcontext()), \
             patch.object(b, "CsvRecording", return_value=contextlib.nullcontext()), \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            b.main()
        self.assertEqual(raised.exception.code, 2)

    def invoke(self, *arguments):
        captured = {}
        argv = ["ctx-cliff.py", "--file", "input.txt", *arguments]
        with patch.object(sys, "argv", argv), \
             patch.object(b, "run_benchmark", side_effect=lambda args, *_: captured.update(args=args)), \
             patch.object(b, "termination_handler", return_value=contextlib.nullcontext()), \
             patch.object(b, "CsvRecording", return_value=contextlib.nullcontext()):
            b.main()
        return captured["args"]

    def test_invalid_combinations_are_rejected(self):
        self.rejected("--agent-task", "x")
        self.rejected("--agent-thinking", "off")
        self.rejected("--scenario", "agent", "--agent-task", "  ")
        self.rejected("--deterministic", "--temperature", "0.5")
        self.rejected("--deterministic", "--min_p=0.0")

    def test_underscore_spellings_and_aliases_reach_the_payload(self):
        args = self.invoke("--min_p=0.0", "--presence_penalty=0.0", "--repetition_penalty=1.0",
                           "--top-p", "0.95", "--frequency-penalty", "0.1", "--repeat_last_n", "128")
        self.assertEqual(b.sampling_payload(args, 0), {
            "top_p": 0.95, "min_p": 0.0, "repeat_penalty": 1.0, "repeat_last_n": 128,
            "presence_penalty": 0.0, "frequency_penalty": 0.1})

    def test_generic_sampler_fields_are_parsed_as_json(self):
        args = self.invoke("--sampler", "dry_multiplier=0.8", "--sampler", "xtc_probability=0.5",
                           "--sampler", 'dry_sequence_breakers=["\\n", ":"]', "--sampler", "mirostat=2",
                           "--sampler", "samplers_note=plain text", "--seed", "3")
        self.assertEqual(b.sampling_payload(args, 1), {
            "dry_multiplier": 0.8, "xtc_probability": 0.5, "dry_sequence_breakers": ["\n", ":"],
            "mirostat": 2, "samplers_note": "plain text", "seed": 4})

    def test_penalties_may_accompany_deterministic(self):
        args = self.invoke("--deterministic", "--presence-penalty", "0.5")
        self.assertEqual(b.sampling_payload(args, 0), {"presence_penalty": 0.5})

    def test_generic_sampler_rejects_reserved_and_dedicated_fields(self):
        for setting in ("n_predict=5", "seed=1", "top_p=0.9", "no-equals", "Bad-Key=1"):
            with self.subTest(setting=setting):
                self.rejected("--sampler", setting)


if __name__ == "__main__":
    unittest.main()
