"""--api openai-chat: turn-wise growing conversation, chat requests, Strata-style cache reuse."""
import contextlib
import io
import itertools
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from test_csv_recording import benchmark as b


def render(system, turns, thinking="auto"):
    """Strata's Qwen3.8 chat template (serve/chat_template.jinja) for plain text turns."""
    text = "<|im_start|>system\n" + ("" if thinking == "off" else "Reasoning effort is set to xhigh.\n\n")
    text += system.strip() + "<|im_end|>\n"
    for turn in turns:
        if turn["role"] == "user":
            text += "<|im_start|>user\n" + turn["content"].strip() + "<|im_end|>\n"
        else:
            text += ("<|im_start|>assistant\n<think>\n" + turn.get("reasoning", "").strip() + "\n</think>\n\n"
                     + turn["content"].strip() + "<|im_end|>\n")
    return text + "<|im_start|>assistant\n" + ("<think>\n\n</think>\n\n" if thinking == "off" else "<think>\n")


def count(system, turns, thinking="auto"):
    return len(re.findall(r"<\|\w+\|>|\s*[^\s<]+|\s+|<", render(system, turns, thinking)))


CONTENT = "".join(f"def function_{i}(value):\n    return value * {i}\n\n" for i in range(3000))


class ChatPromptBuilderTests(unittest.TestCase):
    def builder(self, thinking="auto", cap=None, content=CONTENT):
        return b.ChatPromptBuilder(content, "run", "TASK", thinking,
                                   lambda system, turns: count(system, turns, thinking), 3.0, cap)

    def test_points_fill_the_target_end_at_line_breaks_and_extend_the_previous_prompt(self):
        for thinking in ("auto", "on", "off"):
            builder = self.builder(thinking)
            previous = None
            for target in (500, 2000, 6000, 9000):
                ends, tokens, chars = builder.build(target)
                self.assertLessEqual(tokens, target)
                self.assertGreaterEqual(tokens, target - 64)
                self.assertEqual(chars, ends[-1])
                self.assertTrue(all(CONTENT[end - 1] == "\n" for end in ends))
                prompt = render(builder.system("0" * 16), builder.turns(ends), thinking)
                if previous is not None:
                    # Strata resumes from the checkpoint at the previous generation prompt.
                    self.assertTrue(prompt.startswith(previous), thinking)
                previous = prompt
                self.assertEqual(builder.build(target), (ends, tokens, chars))  # warmup/drift: same point
                builder.set_reply(ends, "" if thinking == "off" else f"thinking at {target}", f"answer {target}")

    def test_history_holds_the_first_recorded_reply_per_point(self):
        builder = self.builder()
        builder.set_reply([10], "r1", "a1")
        builder.set_reply([10], "r2", "a2")  # later repeats do not replace it
        turns = builder.turns([10, 20])
        self.assertEqual([t["role"] for t in turns], ["user", "assistant", "user", "assistant", "user"])
        self.assertEqual(turns[1]["content"], b.CHAT_AGENT_ACK)
        self.assertEqual((turns[3]["reasoning"], turns[3]["content"]), ("r1", "a1"))
        with self.assertRaises(b.PromptBuildError):
            builder.turns([10, 20, 30])

    def test_step_task_ends_every_tool_result(self):
        builder = b.ChatPromptBuilder(CONTENT, "run", "TASK", "auto",
                                      lambda system, turns: count(system, turns), 3.0, step_task="STEP")
        builder.set_reply([40], "r", "a")
        results = [t["content"] for t in builder.turns([40, 80]) if t["role"] == "user"][1:]
        self.assertEqual(results, [b.AGENT_CONTEXT_INTRO + CONTENT[:40].rstrip("\n") + "\n\nSTEP",
                                   b.AGENT_CONTEXT_INTRO + CONTENT[40:80].rstrip("\n") + "\n\nSTEP"])
        self.assertTrue(self.builder().turns([40])[-1]["content"].endswith(CONTENT[:40]))  # no step task

    def test_default_step_task_names_lines_and_definitions(self):
        content = "import os\n\nclass Alpha:\n    def inner(self):\n        pass\n\ndef beta():\n    pass\n"
        builder = b.ChatPromptBuilder(content, "run", "TASK", "auto", None, 3.0, step_task=b.AGENT_STEP_TASK)
        self.assertIn("(lines 3-8 of the file, from `Alpha` to `beta`)", builder.step_text(11, len(content)))
        self.assertIn("(lines 1-1 of the file)", builder.step_text(0, 10))
        self.assertIn("around `beta`", builder.step_text(content.index("def beta"), len(content)))

    def test_cap_targets_below_built_points_and_exhausted_input(self):
        builder = self.builder(cap=3000)
        self.assertLessEqual(builder.build(9000)[1], 3000)
        self.assertEqual(builder.build(9000), builder.build(3000))
        with self.assertRaises(b.PromptBuildError):
            builder.build(1000)
        small = self.builder(content=CONTENT[:2000])
        first = small.build(4000)
        self.assertEqual(first[0][-1], 2000)
        self.assertEqual(small.build(8000), first)  # measure_point turns this into InputExhausted

    def test_target_without_room_for_a_turn_fails(self):
        with self.assertRaises(b.PromptBuildError):
            self.builder().build(20)

    def test_request_id_has_fixed_length(self):
        self.assertEqual({len(b.chat_request_id()) for _ in range(50)}, {b.CHAT_REQUEST_ID_DIGITS})


def sse(*chunks):
    return [f"data: {json.dumps(c)}".encode() for c in chunks]


class ChatStreamTests(unittest.TestCase):
    def test_reasoning_and_content_count_and_last_chunk_carries_timings(self):
        clock = itertools.count(5.0, 1.0).__next__
        lines = [b": keep-alive", *sse(
            {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
            {"choices": [{"delta": {"reasoning_content": "hm"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "ok"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "length"}], "usage": {"prompt_tokens": 42},
             "timings": {"cache_n": 7, "prompt_n": 35}}), b"data: [DONE]"]
        result = b.read_chat_stream(lines, clock)
        self.assertEqual((result["content"], result["stop_type"], result["prompt_tokens"]), ("hmok", "length", 42))
        self.assertEqual(result["timings"], {"cache_n": 7, "prompt_n": 35})
        self.assertEqual(result["_first_token_mono"], 5.0)

    def test_errors_and_missing_final_chunk(self):
        with self.assertRaisesRegex(b.CompletionRequestError, "busy"):
            b.read_chat_stream(sse({"error": {"message": "busy"}}))
        with self.assertRaises(b.CompletionRequestError):
            b.read_chat_stream(sse({"choices": [{"delta": {"content": "x"}}]}) + [b"data: [DONE]"])

    def test_non_streamed_response(self):
        data = {"choices": [{"message": {"content": "a", "reasoning_content": "r"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3}, "timings": {"predicted_n": 2}}
        result = b.chat_response(data)
        self.assertEqual((result["content"], result["stop_type"], result["timings"]), ("ra", "stop", {"predicted_n": 2}))


class ChatRequestTests(unittest.TestCase):
    def test_payload_maps_samplers_thinking_and_deterministic(self):
        http = Mock()
        http.post.return_value = Mock(ok=True, json=Mock(return_value={"choices": [{"message": {}}]}))
        turns = [{"role": "user", "content": "t"}, {"role": "assistant", "content": "a", "reasoning": "r"}]
        b.chat_completion("http://x", "sys", turns, 9, True, "off", http=http,
                          sampling={"repeat_penalty": 1.1, "seed": 3, "top_k": 20})
        url, payload = http.post.call_args.args[0], http.post.call_args.kwargs["json"]
        self.assertEqual(url, "http://x/v1/chat/completions")
        self.assertEqual(payload["messages"][0], {"role": "system", "content": "sys"})
        self.assertEqual(payload["messages"][2], {"role": "assistant", "content": "a", "reasoning_content": "r"})
        self.assertEqual((payload["max_tokens"], payload["repetition_penalty"], payload["seed"]), (9, 1.1, 3))
        self.assertEqual((payload["temperature"], payload["top_k"]), (0.0, 1))
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})

    def test_count_tokens_sends_the_same_conversation_in_anthropic_form(self):
        http = Mock()
        http.post.return_value = Mock(ok=True, json=Mock(return_value={"input_tokens": 77}))
        turns = [{"role": "user", "content": "t"}, {"role": "assistant", "content": "a", "reasoning": "r"},
                 {"role": "assistant", "content": "b", "reasoning": ""}]
        self.assertEqual(b.count_chat_tokens("http://x", "sys", turns, "on", http=http), 77)
        body = http.post.call_args.kwargs["json"]
        self.assertEqual(body["system"], "sys")
        self.assertEqual(body["messages"][1]["content"],
                         [{"type": "thinking", "thinking": "r"}, {"type": "text", "text": "a"}])
        self.assertEqual(body["messages"][2]["content"], "b")
        self.assertEqual(body["output_config"], {"effort": "high"})
        self.assertEqual(b.chat_thinking_fields("on", "openai"), {"reasoning_effort": "high"})
        self.assertEqual(b.chat_thinking_fields("off", "anthropic"), {"thinking": {"type": "disabled"}})
        self.assertEqual(b.chat_thinking_fields("auto", "openai"), {})


class ChatRunTests(unittest.TestCase):
    def run_main(self, *extra, reuse=True, same_reply=False):
        seen = []
        state = {"held": set()}  # (system, rendered prompt) the fake server has checkpoints for

        def complete(base, system, turns, n_predict, deterministic, thinking, **kwargs):
            prompt = render(system, turns, thinking)
            tokens = count(system, turns, thinking)
            cached = max((count_text(p) for p in state["held"] if reuse and prompt.startswith(p)), default=0)
            state["held"].add(prompt)
            seen.append((system, tokens, cached, kwargs.get("sampling"), turns))
            # Like Strata, a fully cached prompt still reads its last few tokens, slowly.
            prompt_n, prompt_ms = (5, 100.0) if cached == tokens else (tokens - cached, 10.0)
            return {"content": "x" if same_reply else f"x{len(seen)}", "reasoning": f"r{len(seen)}", "answer": f"a{len(seen)}",
                    "stop_type": "length", "truncated": False, "prompt_tokens": tokens,
                    "timings": {"cache_n": tokens - prompt_n, "prompt_n": prompt_n, "prompt_ms": prompt_ms,
                                "predicted_n": n_predict, "predicted_ms": 20.0}}

        def count_text(prompt):
            return len(re.findall(r"<\|\w+\|>|\s*[^\s<]+|\s+|<", prompt))

        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "input.py"
            source.write_text(CONTENT, encoding="utf-8")
            csv_path = Path(folder) / "run.csv"
            argv = ["ctx-cliff.py", "--file", str(source), "--api", "openai-chat", "--scenario", "agent",
                    "--start", "1000", "--end", "3000", "--step", "1000", "--n-predict", "8",
                    "--vram-log", "off", "--gpm-log", "off", "--win-gpu-mem", "off", "--seed", "4",
                    "--csv", str(csv_path), *extra]
            err = io.StringIO()
            with patch.object(sys, "argv", argv), \
                 patch.object(b, "count_chat_tokens",
                              side_effect=lambda base, system, turns, thinking, http=None: count(system, turns, thinking)), \
                 patch.object(b, "chat_completion", side_effect=complete), \
                 patch.object(b, "completion", side_effect=AssertionError("llama API used")), \
                 patch.object(b, "tokenize", side_effect=AssertionError("llama API used")), \
                 patch.object(b, "reset_slot", side_effect=AssertionError("slot erase used")), \
                 patch.object(b, "server_is_ready", return_value=True), \
                 patch.object(b, "detect_slot_n_ctx", return_value=None), \
                 patch.object(b, "fetch_server_props", return_value=None), \
                 contextlib.redirect_stdout(err), contextlib.redirect_stderr(err):
                b.main()
            meta = json.loads(csv_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        return seen, meta, err.getvalue()

    def test_incremental_reuses_the_previous_point_and_measures_prefill_once(self):
        seen, meta, err = self.run_main("--repeat", "2", "--warmup", "1")
        run_id = meta["prompt"]["agent"]["run_request_id"]
        systems = [system for system, *_ in seen]
        self.assertNotIn(run_id, systems[0])                    # warmup: unseen request id
        self.assertTrue(all(run_id in s for s in systems[1:7]))  # 3 points x 2 repeats
        self.assertNotIn(run_id, systems[7])                    # drift check: unseen again
        first, second = seen[1], seen[3]
        self.assertEqual(first[2], 0)                           # first point: full prefill
        self.assertEqual(second[2], first[1])                   # next point: whole previous prompt cached
        self.assertEqual(seen[2][2], seen[2][1])                # repeat 2: everything cached
        # The next point's history holds the first measured repeat's reply (not warmup, not repeat 2).
        self.assertEqual(seen[3][4][3], {"role": "assistant", "content": "a2", "reasoning": "r2"})
        self.assertIn("CSV rows written: results=3", err)       # no guard stopped the sweep
        self.assertIn("this last tool output (lines 1-", seen[1][4][-1]["content"])
        self.assertEqual(meta["prompt"]["agent"]["step_task"], b.AGENT_STEP_TASK)
        plain, plain_meta, _ = self.run_main("--repeat", "1", "--warmup", "0", "--no-drift-check",
                                             "--agent-step-task", "none")
        self.assertFalse(any("For this step only" in t["content"] for t in plain[0][4]))
        self.assertEqual(plain_meta["prompt"]["agent"]["step_task"], "")
        self.assertEqual([s[3]["seed"] for s in seen[1:7]], [4, 5, 4, 5, 4, 5])
        self.assertEqual(meta["measurement"]["prefill_mode"], "first_repeat_only")
        self.assertEqual(meta["prompt"]["api"], "openai-chat")
        self.assertNotIn("reused only", err)
        self.assertNotIn("the reply is identical", err)

    def test_lost_reuse_and_cold_mode(self):
        _, _, err = self.run_main("--repeat", "1", "--warmup", "0", "--no-drift-check", reuse=False,
                                  same_reply=True)
        self.assertEqual(err.count("the reply is identical to the previous point's"), 1)
        self.assertIn("reused only 0 of the previous point's", err)
        seen, meta, err = self.run_main("--repeat", "2", "--warmup", "0", "--no-drift-check",
                                        "--cache-mode", "cold")
        self.assertEqual(len({system for system, *_ in seen}), len(seen))  # a new request id every time
        self.assertTrue(all(cached == 0 for _, _, cached, *_ in seen))
        self.assertEqual(meta["measurement"]["prefill_mode"], "cold")

    def test_invalid_combinations_are_rejected(self):
        for arguments in (["--api", "openai-chat"], ["--api", "openai-chat", "--scenario", "agent", "--ignore-eos"],
                          ["--scenario", "agent", "--agent-step-task", "x"],
                          ["--api", "openai-chat", "--scenario", "agent", "--agent-step-task", " "]):
            argv = ["ctx-cliff.py", "--file", "input.txt", *arguments]
            with patch.object(sys, "argv", argv), contextlib.redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit) as raised:
                b.main()
            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
