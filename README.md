# ctx-cliff

A benchmark for [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`
that measures how prefill and decode speed change as the prompt context grows,
and records GPU telemetry while it does so. With `--api openai-chat` it also
measures chat servers without llama-server's `/completion` API, such as
[Strata](https://github.com/Niko1221/Strata).

It steps through increasing context sizes (for example 10k, 20k, … 120k tokens),
runs several completion requests at each point and reports:

- prefill and decode throughput (median over repeats),
- draft acceptance and cost per decode step when speculative decoding is active
  (MTP, DFlash, draft model, n-gram, …),
- free VRAM, power, PCIe traffic and GPU engine utilization (NVIDIA NVML/GPM),
- sudden drops ("cliffs") between neighbouring context sizes,
- whether the GPU state drifted during the run.

It can simulate an agentic coding session (a chat conversation whose file
excerpt grows from point to point), compare runs with different builds,
settings or models, and stops runs that have become meaningless (for example
when the driver starts spilling VRAM into system memory).

It does **not** measure answer quality.

## Requirements

- Python 3.10 or newer with `requests`; for NVIDIA telemetry also `nvidia-ml-py`:
  ```bash
  python -m pip install requests nvidia-ml-py
  ```
- A `llama-server` (upstream llama.cpp or a fork), either already running or
  started by the script via `--server-command`; or, with `--api openai-chat`, a
  running Strata server (`/v1/chat/completions` and `/v1/messages/count_tokens`).
- A large UTF-8 text file as prompt source. `data/` contains two: Django's
  source code (`data/django.py`) and *War and Peace* (`data/war_and_peace.txt`).
- Optional: an NVIDIA GPU and driver for telemetry. Missing telemetry sources are
  disabled with a note. On Windows the script also records dedicated/shared GPU
  memory.

## Quick start

All commands are run from the repository directory.

**Against a running server** (default `http://127.0.0.1:8080`, change with `--base-url`):

```bash
python ctx-cliff.py --file data/django.py --start 1000 --end 100000 --step 10000 --repeat 3 --csv
```

**Let the script start and stop the server.** The whole server command goes into
`--server-command`; the script waits for `/health`, reads the buffer sizes from
the server log and stops the server at the end.

Linux / macOS:

```bash
python ctx-cliff.py --file data/django.py --server-command "/opt/llama.cpp/llama-server --model /models/model.gguf --ctx-size 100064 -ngl 99 -np 1 --port 8080" --start 1000 --end 100000 --step 10000 --csv
```

Windows (PowerShell):

```powershell
python ctx-cliff.py --file data/django.py --server-command 'C:\llama.cpp\llama-server.exe --model "D:\My Models\model.gguf" --ctx-size 100064 -ngl 99 -np 1 --port 8080' --start 1000 --end 100000 --step 10000 --csv
```

**Simulate agentic coding, with speculative decoding and realistic sampling:**

```bash
python ctx-cliff.py --file data/django.py --scenario agent \
  --server-command "llama-server --model model.gguf --ctx-size 100064 -ngl 99 -np 1 --spec-type draft-mtp --spec-draft-n-max 2" \
  --temperature 0.6 --top-p 0.95 --top-k 20 --min_p=0.0 --seed 1 \
  --n-predict 512 --repeat 3 --nonce agent-ab --csv
```

Sampler values belong in the script options, not in `--server-command`: the
script sends them with every request and gives each repeat its own seed.

**Chat server without `/completion`, for example Strata.** The conversation grows
by tool-result turns like an agent loop; sampling keeps the model from copying
its earlier replies:

```bash
python ctx-cliff.py --file data/django.py --api openai-chat --scenario agent \
  --start 8000 --end 128000 --step 8000 --repeat 2 --n-predict 256 \
  --temperature 0.6 --top-p 0.95 --top-k 20 --seed 1 --csv
```

**Compare runs** (reference first; works across builds, settings and models):

```bash
python ctx-cliff.py --compare outputs/ctx-cliff-A.csv outputs/ctx-cliff-B.csv
python ctx-cliff.py --file data/django.py --scenario agent --reference outputs/ctx-cliff-A.csv --csv
```

`--csv` without a path writes to `outputs/` with a timestamped name. Server logs
go to `logs/`.

## Reading the output

One row per context point:

```text
     ctx |     new |  prefill |  decode |  draft |   step |   free | power | ... | status
     tok |     tok |    tok/s |   tok/s |      % |     ms |    MiB |     W | ... |
   10000 |    9984 |    848.3 |   39.07 |  61.2% |   38.4 |   1210 |   162 | ... |        OK
```

| Column | Meaning |
|---|---|
| `ctx` | Prompt length in tokens (as counted by the server's tokenizer). |
| `new` | Tokens actually processed in this request (the rest came from the prompt cache). |
| `prefill` / `decode` | Median throughput over the repeats. |
| `draft` | Accepted / proposed draft tokens; only shown when drafting is active. |
| `step` | Median cost of one decode step in ms. With drafting this reflects the context cost regardless of how predictable the generated text is. |
| `free` | Lowest free VRAM seen during the point. |
| `clock` | GPU SM clock median/min in MHz; a low value means the GPU did not run at full boost. |
| `PF/DC …` | Prefill/decode PCIe traffic, link saturation and GPU engine utilization. |
| `status` | `OK`, or for example `2/3 OK` if repeats were invalid. |

After the table the script lists notes collected during the sweep, the drift
check (first point measured again), the cliff analysis and, with
`--reference`, the comparison.

## Most important options

| Option | Default | Effect |
|---|---|---|
| `--start / --end / --step` | 10000 / 120000 / 5000 | Context points in tokens. |
| `--repeat` | 3 | Requests per point. |
| `--n-predict` | 64 | Tokens to generate per request. |
| `--cache-mode` | incremental | `incremental` reuses the common prefix like a growing conversation; `cold` processes the full prompt every time. |
| `--scenario` | file | `file`: continue the input file; `agent`: chat conversation with growing file excerpt and fixed task. |
| `--api` | llama | `llama`: llama-server's `/completion`; `openai-chat`: `/v1/chat/completions` (Strata), needs `--scenario agent`. |
| `--deterministic` | off | Greedy decoding (`temperature=0`, `top_k=1`). |
| `--temperature`, `--top-p`, `--top-k`, `--min-p`, … | server | Sampling settings sent with every request; `--sampler KEY=VALUE` for any other field. |
| `--nonce TEXT` | random | Fixed run marker, so A/B runs use identical prompts. |
| `--reference CSV` / `--compare REF RUN …` | – | Compare with earlier runs. |
| `--sysmem-guard` | abort | Stop when the driver spills VRAM into system memory (`off` with `--api openai-chat`). |
| `--abort-below-pct` | 20 | Stop when prefill drops below this share of the first point. |

`python ctx-cliff.py --help` lists all options. [docs/details.md](docs/details.md)
explains measurement procedure, cache handling, telemetry, CSV files, cliff
detection and the helper scripts `pcie-calibrate.py` and `gpm-probe.py` in detail.

## Limitations

- The agent scenario simulates a single agent turn (system prompt, growing file
  excerpt, fixed task). It does not replay multi-turn tool-call histories. It
  needs a server with a chat template (`/apply-template`).
- Telemetry is sampled on the machine running the script and is not limited to
  the llama-server process.
- Prefill/decode telemetry windows are reconstructed from server timings, not
  measured by a profiler.
- Incremental repeats rely on slot save/restore (`--slot-save-path`); without it
  the script falls back to measuring prefill only once per point.
- `--api openai-chat` measures prefill once per point (no slot snapshots),
  counts tokens through Strata's `/v1/messages/count_tokens` and cannot force
  `n_predict` tokens (`--ignore-eos`). Its decode rate depends on the generated
  text more than with llama-server; see
  [docs/details.md](docs/details.md#chat-servers-without-completion---api-openai-chat).

## Tests

```bash
python -m unittest discover -s unittests
```

The full suite takes a few minutes; `CTX_CLIFF_FAST_TESTS=1` skips the real-time
simulations of the helper scripts.

## Origin and license

Based on `ctx-cliff.py` by cHunter789, distributed with
[cHunter789/Qwen3.8-27B-i1-IQ4_KS_KT-GGUF](https://huggingface.co/cHunter789/Qwen3.8-27B-i1-IQ4_KS_KT-GGUF)
under the Apache License 2.0
([confirmed by the author](https://huggingface.co/cHunter789/Qwen3.8-27B-i1-IQ4_KS_KT-GGUF/discussions/3)),
and substantially modified and extended since. Licensed under the Apache License
2.0, see [LICENSE](LICENSE). The test inputs in `data/` have their own licenses,
see [data/SOURCES.md](data/SOURCES.md).
