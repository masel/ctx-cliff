# ctx-cliff in detail

This document describes how [`ctx-cliff.py`](../ctx-cliff.py) measures, what each
value means and how the helper scripts work. For installation and first steps see
the [README](../README.md).

Commands are run from the repository directory. Examples use `python`; on some
Linux systems it is `python3`.

## Contents

- [Running the benchmark](#running-the-benchmark)
- [Context points](#context-points)
- [Measurement procedure and cache handling](#measurement-procedure-and-cache-handling)
- [How the values are computed](#how-the-values-are-computed)
- [Telemetry sources and time windows](#telemetry-sources-and-time-windows)
- [Files and CSV recording](#files-and-csv-recording)
- [Errors, crashes and run guards](#errors-crashes-and-run-guards)
- [Detecting prefill and decode cliffs](#detecting-prefill-and-decode-cliffs)
- [GPM plausibility checks](#gpm-plausibility-checks)
- [Options](#options)
- [Limitations](#limitations)
- [Tests](#tests)

## Running the benchmark

### Using a running server

```bash
python ctx-cliff.py --file data/django.py --start 1000 --end 100000 --step 10000 --repeat 3 --deterministic --ignore-eos --csv
```

The default address is `http://127.0.0.1:8080`; use `--base-url` for another one.
The server needs `/health`, `/tokenize` and `/completion`.

### Starting and stopping the server automatically

```bash
python ctx-cliff.py --file data/django.py --server-command "/opt/llama.cpp/llama-server --model /models/model.gguf --ctx-size 100064 -ngl 99 -np 1 --port 8080" --csv
```

Use double quotes around paths with spaces inside the server command, for example
`--model "/models/my models/model.gguf"`. The command is executed without a
shell: pipes, redirections, `&&` and variable expansion inside it are not
interpreted. Server port and `--base-url` must match; automatic starts are limited
to local HTTP addresses.

The script waits for `/health` and normally stops only the server it started
itself. On Windows that server is bound to the script with a Job Object: if Python
ends unexpectedly (crash, closed console window), Windows stops llama-server as
well, so no orphaned server blocks VRAM and port for the next run in a series.
`--keep-server` keeps the server running after the run (without that binding). If
a server is already reachable, a requested start is refused by default;
`--reuse-running-server` uses the running server instead.

**Waiting for VRAM release.** A server that has just exited does not release its
VRAM immediately. If the next run of a series starts its server a second later,
that server may get part of its buffers in shared system memory on an almost full
GPU and run much slower (observed: prefill 523 instead of 853 tok/s). Before
starting its own server, the script therefore reads VRAM usage through NVML and
waits until it has dropped by at most 64 MiB for 3 seconds, at most
`--vram-settle-s` seconds (default 30, `0` disables). Normally this costs 3
seconds. If memory of a predecessor was released, the script reports how long it
waited; the result is stored in the metadata under `server.vram_settle`. The GPUs
from `--vram-gpu` are considered.

**VRAM used by other programs.** Before each server start the script reports
`GPU memory before server start: … MiB used by other processes`. Browsers with
hardware acceleration and desktop apps vary by several hundred MiB. With an almost
full GPU this decides whether the server still fits into dedicated memory
(observed: 710 MiB used by others fit, 1079 MiB led to spilling). An abort by
`--sysmem-guard` includes this value.

**Server memory.** After the start the script reads the buffer sizes from the
server log (model, KV cache with type, recurrent state, compute buffers; separately
for the main model and the draft/MTP model), prints them as
`server memory (MiB): …` and stores them under `server.memory` in the metadata.
This requires a server log file (not `--server-log -`). These values are exact;
measured VRAM usage is too noisy for comparisons because it includes other
programs.

### Measuring full prefill on every request

```bash
python ctx-cliff.py --file data/django.py --cache-mode cold --repeat 3 --warmup 0 --deterministic --ignore-eos --csv
```

`cold` means: do not reuse the prompt cache. It does not put GPU caches, clocks,
temperature or OS caches into a defined cold state.

### A/B comparison with identical prompts

```bash
python ctx-cliff.py --file data/django.py --deterministic --ignore-eos --nonce ab-test --csv
```

With the same `--nonce` in all variants, prompts and (deterministic) outputs are
comparable; see [Generated output](#generated-output).

### Comparing runs (`--reference`, `--compare`)

```bash
# Compare finished runs without a server: reference first, then one or more runs
python ctx-cliff.py --compare outputs/run-a.csv outputs/run-b.csv outputs/run-c.csv

# Compare a new run with an earlier one right after the sweep
python ctx-cliff.py --file data/django.py --scenario agent --reference outputs/run-a.csv --csv
```

The reference can come from any configuration: without drafting, with another KV
cache type, another build or model. All points with the same `target_ctx` are
compared; points present in only one run are listed and skipped. Per point the
table shows prefill, decode and `step ms` (cost of one decode step) as
reference → run with the change in percent, plus tokens per step; below it the
median of the changes over all common points. Without drafting `step ms` is simply
ms per token. For older result CSVs without step columns the script derives them
from the matching `.samples.csv`. Result CSVs from before September 2026 named the
draft columns `mtp_*`; they are still read.

If the `.meta.json` files are present, the script shows the differences between
the server commands (for example `--ctx-size`, `--spec-type`) and checks the
settings: differences in scenario, `--cache-mode`, `--n-predict` or input file
give a `WARNING` (different workload); differences in sampler, task, thinking mode
or `--deterministic` give a `NOTE` (different generated text, which matters for
drafting). With several runs a matrix of decode changes follows, one column per
run.

Differences in memory use appear in the line `server memory (MiB)`, for example
`draft KV 356.0 f16 -> 77.1 kvarn_k3v3_g128 (-278.9)`. For older runs the values
are read from the log file their `.meta.json` points to. `--reference` is read at
start, so a typo does not surface only after the sweep; the comparison is also
stored in the metadata under `comparison`. `--compare` ignores all other options.

### Simulating agentic coding (`--scenario agent`)

```bash
python ctx-cliff.py --file data/django.py --scenario agent --temperature 0.6 --top-p 0.95 --top-k 20 --seed 1 --n-predict 512 --repeat 3 --nonce ab-test --csv
```

In the default scenario `file` the model simply continues the input file. With
`--scenario agent` the script renders a conversation in the model's chat template
through the server's `/apply-template`:

1. system prompt of a coding agent, containing the run marker,
2. user message "Output of the file-reading tool …" with the excerpt from `--file`,
3. fixed assistant reply ("I have read the files. What should I change?"),
4. fixed task (`--agent-task`, otherwise a built-in programming task),
5. the template's generation prompt.

Only the file excerpt in message 2 grows from point to point; everything after it
is the same at every point (the token count of a point includes it). The model
therefore always answers the same task after a longer history. In incremental mode
the previous history comes from the cache; the new excerpt and the task are
processed as in a real agent turn.

This is a simplified agent turn: there is no multi-turn history of tool calls and
results. The server needs a chat template (`/apply-template`).

`--agent-thinking on|off` sets `enable_thinking` for the template; `auto` keeps
the template default. Rendering, prefix and suffix tokens are stored in
`.meta.json` under `prompt.agent`.

### Sampling settings

Sampling settings are sent with every request; without them the server settings
apply. There are options for `--temperature`, `--top-p`, `--top-k`, `--min-p`,
`--typical-p`, `--repeat-penalty` (also `--repetition-penalty`),
`--repeat-last-n`, `--presence-penalty` and `--frequency-penalty`, each also with
underscores (`--min_p=0.0`), so recommendations from model cards can be copied
directly. Any other field of the llama-server API goes through
`--sampler KEY=VALUE` (repeatable, VALUE is parsed as JSON where possible), for
example `--sampler dry_multiplier=0.8` or `--sampler xtc_probability=0.5`. The
server silently ignores unknown fields.

Repeat r uses the seed `SEED + r − 1` at every point, so sampled outputs are
reproducible and comparable between A/B runs. `--deterministic` cannot be combined
with `--temperature`, `--top-p`, `--top-k`, `--min-p` and `--typical-p`, but with
penalties. For MTP and other drafting methods sampled decoding is more realistic
than greedy decoding, which inflates acceptance. The values sent are stored in
`.meta.json` under `prompt.sampling_first_repeat`.

Sampler values belong in the script options, not in `--server-command`. Values in
the server command (`--temp`, `--top-k`, `--seed` …) are only server defaults; the
script's values are sent per request and override them. Only through the script
does every repeat get its own seed (a `--seed` in the server command applies to
all requests alike, so all repeats produce the same text), does `--deterministic`
check for contradictions and does `--compare` report differences as
`sampling differs`.

### Running series of runs

Series of runs (for example variants of a server option) can be scripted with a
shell script or batch file. Environment variables such as `GGML_*` set before
the call are inherited by a server the script starts and recorded in the
metadata.

```bash
for threshold in 32768 0; do
  GGML_EXAMPLE_OPTION=$threshold python ctx-cliff.py --file data/django.py \
    --server-command "llama-server --model model.gguf --ctx-size 43000 -ngl 99 -np 1" \
    --start 10000 --end 42000 --step 4000 --nonce series --csv outputs/series-$threshold.csv
done
```

```powershell
foreach ($value in "32768", "0") {
  $env:GGML_EXAMPLE_OPTION = $value
  python ctx-cliff.py --file data/django.py --server-command 'llama-server.exe --model model.gguf --ctx-size 43000 -ngl 99 -np 1' --start 10000 --end 42000 --step 4000 --nonce series --csv "outputs/series-$value.csv"
}
Remove-Item Env:GGML_EXAMPLE_OPTION
```

## Context points

A step is a context point with a target prompt length.
`--start 1000 --end 31000 --step 10000` requests the points 1000, 11000, 21000 and
31000 tokens. Each point consists of `--repeat` completion requests; each request
contains prefill followed by decode.

The prompt consists of a run marker (unique per run, fixed with `--nonce`) and a
growing prefix of the input file. A file that is too short is not repeated
artificially: if it does not reach `--end`, the script warns in advance, and as
soon as a point would not give a longer prompt the run ends regularly
(`stop_reason` `input_exhausted`) without measuring that point.

The script tokenizes a slightly larger text window with the server's tokenizer and
passes its first N tokens. The prompt is thus an exact prefix of the tokenization
of the whole file and hits the target exactly. The last 64 tokens before the end
of the window are discarded because they can differ from the tokenization of the
whole file. `target_chars` is determined exactly through `/detokenize`, otherwise
estimated. Consecutive points are token prefixes of each other.

Cutting the file by characters would usually end the prompt in the middle of a
word or indentation (for example `def get_user_mo`), with a token that never
appears in normally tokenized text. Models react to that erratically, with an
immediate EOS (`STOP@1`) or, with `--deterministic`, with loops.

If the slot's context size is known through `/slots` or `/props`, the script
reserves `--n-predict` + 1 tokens for decode and limits the prompt range
accordingly. The extra token is needed because llama-server reports `truncated` as
soon as prompt and generated tokens together reach `n_ctx`. If the step grid ends
well before the effective limit, an extra end point is inserted.

## Measurement procedure and cache handling

### Incremental with prefill repeats (default)

1. Connect to or start the server, load the prompt source and start telemetry.
2. With `--repeat > 1`, check once that the slot can be saved, before the warmup.
3. Run the warmup; its measurements are not part of the results.
4. At the first context point disable prompt reuse (`cache_prompt=false`) for
   **every** repeat. This forces full prefill even if server-internal checkpoints
   survived a slot reset.
5. Before every later point, determine the common token prefix of the previous and
   the new prompt. Prepare it with `cache_prompt=false` and `n_predict=1`: the
   first output token is not yet part of the cache. Save only this pure prefix
   state. Before **every** repeat, including the first, restore the same snapshot.
6. Process the same prompt with prefix reuse enabled and generate up to
   `--n-predict` tokens.
7. Aggregate the point. Keep the token IDs of the input prompt for the next point;
   the generated continuation is not part of it.

Preparation and measurement use token IDs so that a text interface cannot produce
a different last token. At least one token of the new prompt is left for the
measured prefill. The preparation costs one extra, unmeasured prefix pass per
later point.

The number of saved and restored tokens must match the prepared prefix length
exactly. Every repeat must reuse it completely (`cache_n` equals the prefix
length), and `cache_n` and `prompt_n` must agree within a repeated incremental
point. Otherwise the point is not published as a result row and the run ends there
(see [Errors, crashes and run guards](#errors-crashes-and-run-guards)); points
already completed are kept and evaluated.

Save, restore, prompt preparation and `--settle` are outside the measured
completion requests. Raw telemetry keeps running in between.

### Automatic snapshot folder

If the script actually starts the server through `--server-command` and no
`--slot-save-path` is given there, it creates `slot-snapshots` **next to the
Python file** for incremental repeats. The absolute path is appended to the server
command and printed as a note.

An explicitly given snapshot path is left unchanged; its directory must already
exist. For a server started separately, the option must be set at server start.
When a running server is reused, the local command is not changed and no folder is
created.

Each run uses one uniquely named `ctx-cliff-*.bin` file that is overwritten again
and again. With a local server started by the script this file is **deleted at the
end of the run by default**, also after handled aborts and errors. This applies to
the automatic folder as well as to a snapshot path given explicitly in the started
server command.

`--keep-snapshot` keeps the file for inspection. `--keep-server` alone does not
prevent the deletion. The snapshot folder and files of other runs are kept. A
failed deletion is reported as a warning with the path.

With an external or reused server the actual file path is not reliably known. A
note then shows the file name for manual removal; the slot API has no delete
action. After a hard process kill or power failure no automatic cleanup is
possible either. Depending on model and context the file can become large.

### Fallback without snapshot support

A real save request checks whether the server can save the slot. If it is
rejected, for example because of a missing server option, missing API or write
permissions, the reason and a clear fallback note are printed.

The run then uses the original behaviour:

- The first request per point provides the prefill statistics.
- The other requests use the prompt cache filled in the meantime.
- Decode is still measured `--repeat` times.
- Prefill telemetry also comes only from the first request; memory and power still
  include all requests.

The fallback is shown in the column `prefill_mode` (`first_repeat_only`) of the
result and samples CSV and, with the reason, in the metadata
(`measurement.prefill_repeat_error`). A note also appears at the end of the run.
`prefill_valid_repeats` is then 1; prefill cliff detection then needs
`--cliff-min-repeats 1`.

| `prefill_mode` | Meaning |
|---|---|
| `snapshot` | incremental, every repeat starts from the same prefix snapshot |
| `first_repeat_only` | incremental fallback: only the first repeat measures prefill |
| `incremental` | incremental with `--repeat 1` |
| `cold` | slot reset and `cache_prompt=false` before every repeat |

The probe tests save, not a full restore. Network errors, timeouts and later
save/restore errors after a successful probe abort the run and are not treated as
missing support.

With a single repeat the server may have to truncate its cache between points;
`cache_n` is then rounded down to a multiple of the server's cache block size
(128 tokens with llama.cpp).

### Cold mode and a single repeat

In `cold` mode a slot reset must succeed before every request; otherwise the
context point is aborted. `cache_prompt=false` is set as well. All repeats measure
full prefill; snapshots are not needed. With `--repeat 1` there is no snapshot
probe and no snapshot save either.

## How the values are computed

There are two levels: **server timings per completion request** and **sensor
samples during that request**. Each repeat is evaluated first; the results are
then combined per context point. Not all values are means.

### Console columns

| Column | Meaning and aggregation per context point |
|---|---|
| `ctx` | Prompt length determined by the tokenizer, including the run marker, excluding generated tokens. |
| `new` | Median of the server-reported `prompt_n`: prompt tokens actually processed; in the fallback only the first request. |
| `prefill` | Median of the prefill rates in tokens/s; in the fallback only the first request. |
| `decode` | Median of the decode rates of all valid complete repeats. |
| `draft` | `100 × sum of accepted draft tokens / sum of proposed draft tokens` over all repeats; `n/a` without drafts at the point. Covers every kind of drafting llama-server reports in `timings.draft_n`/`draft_n_accepted` (MTP, DFlash, draft model, n-gram). The column appears when drafting was detected during the warmup or at the first point. CSV: `draft_n`, `draft_acc`, `draft_acc_pct`. |
| `step ms` | Median cost of one verification step, `predicted_ms / (predicted_n − draft_n_accepted)`, over valid decode repeats; only with active drafting. Independent of how predictable the generated text is (see [Decode with drafting](#decode-with-drafting)). |
| `free` | **Lowest free VRAM** during all requests of the point, prefill and decode together; MiB. |
| `power` | Mean of all valid power samples during the requests, weighted by the number of valid samples; W. |
| `PF/DC PCIe` | GPM receive/transmit in MiB/s: first p95 per repeat and phase, then the median of these p95 values. |
| `PF/DC sat` | Share of valid GPU interval time with **at least 90 %** of the theoretical PCIe link rate. |
| `PF/DC BUS` | Mean of valid NVML bus-busy samples of the phase; measures busy time, not bandwidth. |
| `PF/DC GPU` | GPM `SM / Occupancy / Tensor / DRAM`, each a mean in percent weighted by valid GPU interval time; suspicious SM/O/T zeros are excluded by default (`*`, see [GPM plausibility checks](#gpm-plausibility-checks)). |
| `status` | `OK` if all repeats are valid, otherwise for example `2/3 OK`. |

`PF` is prefill, `DC` decode; `R/T` is receive/transmit. DRAM utilization
describes memory bandwidth use, not the share of occupied VRAM. Occupancy is SM
occupancy, not memory fill level either.

Example for `free`: if the lowest values seen in the repeats are 900, 700 and
850 MiB, the table shows **700 MiB**, not a mean.

### Server timings and further CSV values

- Rates come preferably from `prompt_per_second` and `predicted_per_second` of the
  server response. If no valid rate is present, it is computed as
  `tokens / (milliseconds / 1000)`.
- `prompt_ms` is the median of the prefill times. `prefill_tps` is separately the
  median of the rates; it is not recomputed from the rounded median time.
- `cache_n` is the number of tokens the server reports as already cached. The work
  actually reused is easier to see there than in the throughput alone.
- `wall_s_median` is the median of the whole HTTP completion duration. Unlike the
  server timings it includes communication and client overhead.
- `decode_tps_min/max` are the lowest/highest valid decode rate.
- `*_peak_*` and `*_max_*` usually take the maximum over all repeats, `*_min_*`
  the minimum. Temperature peaks, VRAM peaks and maximum clocks are kept as
  extremes.
- Median, p95 and p99 fields are **medians of the per-repeat statistics**, not
  quantiles of all raw samples pooled.
- GPM means and saturation shares are weighted by valid interval length. Power and
  BUS are weighted by valid sample counts.
- `win_shared_delta_mib` is the largest range of shared memory seen within a
  single request, not the end value minus the start of the whole run.
- `pstate_mode` is the most frequent P-state per request and then the most
  frequent of these per-repeat values. Sample counts and valid durations are summed.

Missing or non-finite telemetry values are excluded; real zeros are kept. The
console shows missing values as `n/a`, the CSV as empty cells. Without GPM the GPM
console columns are not replaced by differently defined legacy PCIe values.

### Validity of decode measurements

A repeat counts for decode only if its status is `OK` and the rate is positive.
The individual statuses are listed in `sample_statuses`:

| Status | Meaning |
|---|---|
| `OK` | At least the requested number of decode tokens, no truncation reported. |
| `STOP@N` | Stopped early after N tokens. |
| `EMPTY` | No tokens generated. |
| `TRUNC` | The server reports truncation. |

Invalid repeats are excluded from the decode median and decode-phase telemetry.
They are not generally removed from prefill, memory, power or draft statistics. If
no decode repeat is valid, `decode_tps_median/min/max` stay empty (console `n/a`)
and the point gives no valid cliff comparison. The same applies to `prefill_tps`,
`prompt_ms`, `prompt_n` and `cache_n` without a valid prefill repeat and to
`draft_acc_pct` without draft tokens. A 0 in these columns is therefore always a
real measurement. `--ignore-eos` raises the chance of complete decode windows but
does not guarantee it.

### Decode with drafting

With drafting the decode rate also depends on how predictable the text just
generated is; between repeats it varies by 10 to 30 %. Each verification step
yields the accepted draft tokens plus one sampled token. From this ctx-cliff splits
the decode rate into two quantities:

```text
steps           = predicted_n − draft_n_accepted
tokens per step = predicted_n / steps          (content: predictability)
ms per step     = predicted_ms / steps         (context: cost of one step)
```

Without drafting one step is one token. In measurements with Qwen3.8-27B and MTP,
ms per step varied between repeats by 0.2 to 3 %, the decode rate by 7 to 14 %.
`.samples.csv` contains `decode_steps`, `tokens_per_step` and `ms_per_step`, the
result CSV `tokens_per_step_median` and `ms_per_step_median/min/max`; with active
drafting the table shows the columns `draft` and `step ms`. Cliff detection
treats drafting separately, see
[Detecting prefill and decode cliffs](#detecting-prefill-and-decode-cliffs).

## Telemetry sources and time windows

| Source | Default interval | Content |
|---|---|---|
| NVML (otherwise `nvidia-smi`) | 250 ms | VRAM, utilization, power, temperature, clocks, P-state where available. By default NVML is queried directly with the same time base as the other sensors; without a working `nvidia-ml-py` an `nvidia-smi` process is used (`--vram-backend`). |
| Legacy NVML PCIe | 250 ms polling | RX then TX counter, each with a 20 ms NVML measurement window; the actual spacing is recorded. |
| NVML BUS | 1000 ms | Bus busy time over the previous second. |
| NVML GPM | 250 ms | GPU engines and PCIe rates per measurement interval; the interval must be longer than 100 ms. GPM must be supported by GPU and driver. |
| Windows PDH/WDDM | 1000 ms | Dedicated and shared GPU memory of all captured adapter instances (Windows only). |

Automatic telemetry that is unavailable is disabled with a note. Explicitly
requested backends (`--vram-log nvidia`, `--gpm-log on`, `--win-gpu-mem on`) abort
the run if they are unavailable.

### Time windows for prefill and decode

The server reports durations, but no timestamps synchronized with the client
telemetry. By default ctx-cliff therefore requests a streamed response: the
arrival of the first generated token marks the end of prefill directly.

```text
first token (client clock)  → decode start, decode until end of response
  └─ minus prompt_ms        → prefill start
```

`phase_method` is then `first_token_anchor`. The column `phase_anchor_offset_ms`
(result CSV: median) shows by how many milliseconds a back-projection from the
end of the response would have shifted the boundary. The server timings
(`prompt_ms`, `predicted_ms`, rates) are not affected by streaming.

With `--no-stream`, or if the server does not stream, both phases are projected
back from the end of the response (`response_end_backprojection`): end of response
minus `predicted_ms` is the decode start, minus `prompt_ms` the prefill start. The
windows are clipped to the start of the request. Neither is a GPU profiler
measurement.

GPM only uses positive measurement intervals that lie completely inside the
window. BUS values only count if their preceding 1 s window fits as well. Short
phases may therefore contain no valid GPM/BUS values.

For VRAM and Windows memory a nearby sample outside the request window may be used
if none lies inside. The limit is `max(0.5 s, 2 × polling interval)` for VRAM and
1.5 s for Windows. Such values may reflect activity between requests and are
marked as fallback.

Useful diagnostic fields in the CSV:

- `*_valid_samples`, `*_valid_ms`: valid observations per metric.
- `gpm_*_coverage_pct`: observed share of the phase.
- `gpm_*_boundary_samples`: measurement intervals crossing a phase boundary.
- `vram_fallback_samples`, `win_fallback_samples` and `*_fallback_max_age_ms`:
  fallback samples used and their distance.
- `phase_wall_residual_ms_max`: largest positive remainder between request time
  and the sum of the server phase durations.
- `phase_timing_excess_ms_max`: largest excess of the phase durations over the
  request time.

Residuals and coverage do not quantify the actual error of the phase assignment.
Even 100 % coverage does not prove exactly aligned phases.

### PCIe saturation and units

Link capacity is derived from generation and width. The saturation share uses
`max(RX, TX) / theoretical link rate`, not the sum of both directions. Without a
valid capacity or one of the directions there is no share. The theoretical link
rate is not a guaranteed practical payload rate.

GPM uses **MiB/s** (1024² bytes/s), legacy PCIe **MB/s** (1000² bytes/s). The CSV
names mark the difference with `mib_s` and `mb_s`.

### GPM PCIe and the legacy NVML PCIe sampler

Both measure PCIe traffic, but differently:

- **GPM** counts without gaps and reports the average per measurement interval
  (default 250 ms). The PCIe columns and saturation in the console come from GPM.
- **The legacy sampler** measures 20 ms of receive, then 20 ms of transmit per
  query. It catches short peaks but observes only part of the time.

A legacy query takes over 100 ms on some systems. Each row of the `.pcie.csv`
contains `pcie_query_ms` (duration of the RX/TX query) and `pcie_poll_interval_ms`
(actual spacing to the previous query of the same GPU); the timestamp lies in the
middle of the query. At the end of the run the requested and achieved interval and
the observed time share are printed; the same values are stored in the metadata
under `legacy_pcie_sampler`. If the interval is missed clearly, the script suggests
an achievable value.

In real runs the legacy sampler reported about 1.7 times the GPM values (same
unit). A calibration test (RTX 5060 Ti, driver 616.92, PCIe 3.0 x8, see below)
explains this:

- **GPM** reports 1.08 times the payload for large copies (read as MiB/s). This
  matches counted TLP headers (about 8 % with 256-byte packets); 7.57 GB/s is just
  below the physical maximum of 7.88 GB/s for 3.0 x8. GPM is therefore the reliable
  absolute quantity, also for saturation.
- **The legacy sampler** consistently reports ~1.67 times the payload, for large
  and bursty copies. At 11.7 GB/s it exceeds what a 3.0 x8 link can transfer at
  all; its absolute values and the saturation derived from them are too high by
  this factor. As a relative quantity (trend, comparison) it remains useful.
- **Small transactions** widen the gap: with 64-byte copies the legacy sampler
  reports 2.05 times GPM instead of 1.55. The median of 1.7 in real runs lies in
  between.
- **GPM dropouts also affect PCIe:** with bursty copies GPM reported exactly 0 in
  both directions in 18 of 168 intervals while copying continued (the link
  temporarily fell back to 2.0/1.0). The legacy sampler kept measuring, which makes
  it useful as a cross-check for GPM zeros.
- **GPM lag:** GPM values appear about 70 ms (42–109 ms) after the actual
  traffic. With 250 ms intervals this matters for the prefill/decode assignment.

What ctx-cliff does with this:

- `--gpm-lag-ms` (default 70) shifts every GPM interval back by the lag before
  assigning it to prefill/decode. The `.gpm.csv` contains the corrected times in
  `interval_start/end_mono` and the actual query times in `sample_start/end_mono`.
  After each request the script waits for the lag so the last intervals are
  available; this is outside the measured time.
- `--pcie-legacy-scale` divides all rates and saturation shares computed from the
  legacy sampler by the calibration factor. The raw `.pcie.csv` stays unchanged;
  the factor used is stored in the result CSV (`pcie_legacy_scale`, likewise
  `gpm_lag_ms`). Without a factor the summary at the end points this out.
  `pcie-calibrate.py` prints suitable values of both options for GPU and driver.
- The legacy sampler queries the current PCIe generation and width only once per
  second and the maximum values only once; these queries used to cost about as much
  time as the throughput measurement itself.
- The legacy sampler serves as a cross-check for complete GPM dropouts (see
  [GPM plausibility checks](#gpm-plausibility-checks)).

### Sysmem fallback on Windows

If VRAM runs out, the Windows driver (WDDM) moves buffers into shared system
memory, and every prefill step reads them over PCIe. With all layers on the GPU,
PCIe receive during prefill is normally around 5–30 MB/s; with spilling it is
1–11 GB/s, and prefill becomes 5 to 30 times slower. `--sysmem-guard` uses this to
stop such runs, see [Run guards](#run-guards). Single p95 peaks of around 1000 MB/s
also occur in normal runs at high context, which is why the guard uses the median.

### PCIe calibration test (`pcie-calibrate.py`)

The helper script checks both PCIe sensors against transfers of known size. It
copies data between host memory and GPU through the CUDA driver library
(`nvcuda.dll` / `libcuda`, part of every NVIDIA driver; PyTorch is not needed)
while the legacy sampler and GPM run. Only `nvidia-ml-py` is required.

```bash
python pcie-calibrate.py
```

Stop llama-server and other GPU compute programs first; their traffic would
distort the result. The script warns if it finds such processes. The run takes
about 50 s and temporarily uses 256 MiB of VRAM (`--buffer-mib`). Phases, each
with idle time in between:

1. steady large copies host→GPU and GPU→host (calibration),
2. the same directions in bursts (8 MiB, then a 25 ms pause),
3. very many tiny copies of 64 bytes each (`--tiny-bytes`): hardly any payload,
   but many small PCIe transactions for copy setup, doorbells and
   acknowledgements, similar to the control traffic of kernel launches during
   inference.

The table shows per phase the actual rate and both sensors with the ratio
sensor/actual. GPM is read as MiB/s, the legacy sampler as KB/s (1000 bytes) and
additionally as KiB/s (1024 bytes). Only measurement windows lying completely
inside a phase count. Interpretation:

| Ratio | Meaning |
|---|---|
| 0.97–1.03 | the sensor measures the transferred payload |
| 1.03–1.06 | KiB instead of KB, or slight protocol overhead |
| 1.06–1.40 | probably counts PCIe protocol overhead |
| clearly above | the sensor deviates fundamentally |

For bursty copies each sensor is compared with its own ratio for large copies; if
it deviates by more than 15 %, it is named. GPM intervals reporting exactly 0 in
both directions while copies are running count as dropouts: they are excluded and
reported as `GPM DROPOUT`. The script also estimates from the phase boundaries of
the large copies by how many milliseconds GPM lags behind the actual traffic.

With the tiny copies the payload is negligible; both sensors report almost only
management traffic. If the ratio legacy/GPM is clearly higher here than for large
copies, the two count small transactions or protocol overhead differently. That
would explain the gap in real runs, whose PCIe traffic with a model fully on the
GPU consists mostly of such control traffic.

NVIDIA describes both sensors as byte counters, not as utilization:
`nvmlDeviceGetPcieThroughput` reads "a byte counter over a 20 ms interval" (KB/s),
GPM reports "PCIe traffic to/from this GPU in MiB/sec". Whether headers and
protocol traffic are counted is documented for neither; the related DCGM profiling
counter explicitly says "header and payload". The results are also written to
`outputs/pcie-calibrate-*.json` (summary, GPU, driver, PCIe link) and `.csv` (all
raw measurements).

### Investigating GPM dropouts (`gpm-probe.py`)

From earlier runs (2,576 load periods) it is known that the length of the idle
time before a load period does not matter, but a dropout state persists: after a
period with a dropout, the next one drops out in 73 % of cases, after a clean one
in only 4 %. NVIDIA documents no cause; GPM's "streaming" mode is only available
on Windows in TCC mode, not for GeForce.

`gpm-probe.py` answers the open questions with controlled load cycles (idle 0.5/2/6
s, then 5.5 s of load, either compute kernels or bursty copies):

1. **Does the legacy PCIe sampler cause the dropouts?** Cycles alternate with and
   without it.
2. **Does the idle duration matter?**
3. **Can a dropout be fixed?** If a cycle shows a dropout after 1.5 s, the script
   restarts GPM measurement, first with new sample buffers, then if necessary with
   a full NVML re-initialization. The load continues without a pause, because a
   pause would itself be an idle→load transition and distort the result.

```bash
python gpm-probe.py
```

**Results** (RTX 5060 Ti, Windows, driver 616.92; 360 cycles with restart, 60
without):

- **The legacy PCIe sampler is not the cause:** dropouts with it 19 %, without it
  16 % of the copy cycles; in the comparison run 11 of 30 against 11 of 30.
- **SM zeros could not be reproduced:** in 180 cycles with simple compute load
  (98–99 % utilization) SM = 0 never occurred. These dropouts depend on something
  that only happens with llama-server.
- **Complete dropouts with bursty copies** are patchy (zeros and values
  alternating), and their frequency varies strongly between runs (after a 2 s
  pause once 7 %, once 47 %).
- **A restart is not a reliable fix:** dropouts also fade without a restart (share
  of zeros 70 % → 44 % → 23 %; with restart 75 % → 30 % → 17 %). The benefit is
  small and inconsistent. ctx-cliff therefore detects and filters dropouts instead
  of "repairing" them.

For experiments with real load ctx-cliff has `--gpm-restart realloc|reinit`
(default `off`). If a request shows an SM or complete GPM dropout, ctx-cliff
restarts GPM measurement after that request and before the next one, outside all
measurement windows. Each restart appears as a `NOTE`, in the column
`gpm_restart` of the `.samples.csv`, as the count `gpm_restarts` in the result CSV
and with time and reason in the metadata (`gpm_restart_events`). Since dropouts
otherwise carry over into the next request in 73 % of cases, a comparison with and
without the option shows whether the restart helps. `reinit` re-initializes NVML
completely only if no other NVML user is active (`--vram-log off`), because NVML
counts references.

Stop llama-server first. With the defaults the run takes about 7 minutes (48
cycles; `--repeats` changes that). At the end there is an evaluation with dropout
rates per condition and the result of the restarts, plus `outputs/gpm-probe-*.json`.
The compute kernel is compiled at runtime by the driver; if that fails, the script
uses copy load only.

### Several GPUs and other processes

The sensor values come from the machine running the Python script, even if
`--base-url` points to a remote server. They are not limited to the llama-server
process; other GPU applications can influence them.

`--vram-gpu` limits NVIDIA/GPM to selected GPUs. Windows memory values, however,
include all captured adapter instances.

With several NVIDIA GPUs, `free` is the lowest free memory of a single GPU, not the
sum. `vram_peak_mib` sums the per-GPU maxima per request; these maxima need not
have occurred at the same time. GPM engine means weight valid GPU intervals
together; PCIe quantiles are not a summed rate over all GPUs. Coverage unites the
observed time intervals and therefore does not automatically mean complete
coverage of every GPU.

## Files and CSV recording

Without `--csv` no result CSV is written. `--csv` without a path generates a unique
timestamped name under `outputs`; `--csv-dir` changes that folder. `--csv-export`
is an alias. `--csv path.csv` uses an explicit name; its parent directory must
exist. Explicit output files may be overwritten; automatic names avoid collisions.

With CSV recording enabled, configured telemetry sources get the same file stem.
If a backend is unavailable, a file may contain only the header. Runtime errors of
a successfully started monitor abort the measurement; completed result rows and
flushed raw data are kept.

| File | Content |
|---|---|
| `ctx-cliff-….csv` | One aggregated result row per fully completed context point. |
| `ctx-cliff-….samples.csv` | One row per measured repeat, including invalid ones, with validation error and output fingerprint. |
| `ctx-cliff-….meta.json` | Run metadata, see below. |
| `ctx-cliff-….vram.csv` | Raw NVIDIA memory, clock, power and utilization samples. |
| `ctx-cliff-….pcie.csv` | Raw legacy NVML PCIe and BUS samples. |
| `ctx-cliff-….gpm.csv` | Raw GPM intervals with engine and PCIe values. |
| `ctx-cliff-….win-gpu.csv` | Raw Windows dedicated/shared memory values. |

Individual raw data paths can be given separately with `--vram-csv`,
`--pcie-csv`, `--gpm-csv` and `--win-gpu-mem-csv`, also without a result CSV.

At start the script reports each file path once (`CSV recording: ...`). At the end
it prints a one-line summary of the rows written per file
(`CSV rows written: results=16, samples=16, ...`).

Result rows are written and flushed immediately, raw data about every 250 ms. On
an orderly end the PCIe/GPM files are extended with reconstructed phase
assignments. After a hard process kill these additions and the last unwritten
samples may be missing; previously flushed data is generally kept. Flushing does
not protect against power failure.

For raw data export the program uses temporary archives in the system temp
folder. The active request and recent history stay in RAM. Warmup, pauses and
snapshot I/O can appear in raw data but are not part of the regular result
windows, apart from the nearby fallback samples described above.

Servers started by the script log to `logs/llama-server-YYYYMMDD-HHMMSS.log` by
default. `--server-log-dir` changes the folder, `--server-log PATH` the file name
and `--server-log -` sends the output to the console. Relative log/CSV paths are
relative to the working directory; the automatic snapshot folder is relative to
the script directory.

### Run metadata (`.meta.json`)

With `--csv` a JSON file with the same stem is written next to the result CSV. It
is rewritten atomically after each phase of the run, so an aborted run also leaves
its last known state. Content:

- `argv`, all options, start/end time, `status` (`running`, `completed`,
  `completed_context_limit`, `stopped_sysmem_fallback`, `stopped_prefill_floor`,
  `failed`, `interrupted`), `exit_code`, `stop_reason`, `completed_points`, on
  errors `failed_target_ctx`, `error` and possibly `server_exit_code`.
- Server: the command actually executed including an automatically added
  `--slot-save-path`, log file, buffer sizes from the log (`server.memory`) and a
  compact extract of `/props` (among others `build_info`, `model_path`, generation
  defaults; without chat templates).
- Environment variables with the prefixes `GGML_`, `LLAMA_`, `CUDA_`, `NVIDIA_`,
  `HIP_`, `ROCR_`, `HSA_`, `VK_`, `OMP_`, `MKL_`, `OPENBLAS_`, `KMP_`. With a server
  started by the script these are also the server's values; with an external server
  only those of the benchmark process.
- GPU name, UUID, VBIOS, driver and CUDA driver version, memory, power limit and
  maximum PCIe link (NVML, otherwise `nvidia-smi`), unless NVIDIA telemetry is
  disabled.
- SHA-256 of script and input file, Python/platform version, run marker, tokenizer
  estimate, slot context, planned context points, `prefill_mode` with fallback
  reason, active telemetry sources, whether drafting was active
  (`drafting_active`), notes collected during the sweep (`sweep_notes`), cliff
  summary (`cliffs`), drift check (`drift_check`) and comparison (`comparison`).

Values of `--api-key` in the server command and variables or `/props` fields whose name contains `KEY`,
`TOKEN`, `SECRET`, `PASSW` or `CREDENTIAL` are stored as `<redacted>`. A failure to
write the metadata only produces a warning.

### Generated output

The program does not rate answer quality, but records a fingerprint of the
generated text per repeat in `.samples.csv`:

- `output_sha256`: the first 16 hex characters of the text's SHA-256.
- `output_chars`, `stop_type` (from the server).
- `output_excerpt`: the first 200 characters, line breaks and tabs as `\n`/`\t`.
- `output_loop_pct`: share of the last up to 1000 characters consisting of at
  least three directly consecutive copies of the same string. From 50 % a
  `NOTE … looks degenerate` appears. This is a heuristic for endless loops, for
  example with `--ignore-eos`; such loops can also distort draft acceptance and
  thus the decode rate. Without drafting they have practically no influence on the
  decode rate.

Notes produced during the sweep (early EOS `STOP@…`, `TRUNC`, loops, differing
outputs despite `--deterministic`, GPM restarts) do not appear between the table
rows but collected below the table as `NOTES during the sweep` and in the metadata
under `sweep_notes`. Errors and aborts are still reported immediately.

The result CSV contains per point the `output_sha256` of the first repeat,
`output_variants` (number of different outputs over all repeats) and
`output_loop_pct_max`. If outputs differ within a point despite
`--deterministic`, a note appears.

For A/B comparisons, for example different `GGML_*` settings, start all runs with
`--deterministic` (or the same sampling settings and `--seed`) and the same
`--nonce`. By default the run marker at the start of the prompt is random, so
prompts and outputs differ between runs. With the same marker the prompts are
identical, and equal `output_sha256` per `target_ctx` show identical outputs.
Differences are not automatically errors (other kernels may differ slightly
numerically), but show from which context size variants diverge. With a reused
server a fixed marker can allow cache hits from earlier runs; the `cache_n` check
in snapshot mode detects unexpected reuse.

## Errors, crashes and run guards

If a context point fails (connection lost, timeout, cache validation, telemetry
error, rejected request), the run ends there, but:

- points already completed stay in the CSV and are evaluated for cliffs as usual;
- at the end `RUN ENDED EARLY at target=…` appears with the error and the number
  of completed points;
- the exit code is 1, the metadata contains `status=failed` and the error.

If a llama-server started by the script crashed, the script reports its exit code,
on Windows also in hexadecimal (for example `3221225477 (0xC0000005)`, access
violation), and the path of the server log. If a point reaches the server's
context limit, the run ends regularly instead (exit code 0 if points were
completed). If the warmup or the first point already fails, the run ends with exit
code 1 without a traceback. Unexpected program errors still show the full
traceback. `Ctrl+C` ends with exit code 130 after printing the completed points.

### Run guards

Two guards check every single repeat and end the run as soon as further points
would give nothing useful. In a series of runs the next one then starts without
losing time.

- **Sysmem fallback (`--sysmem-guard`, default `abort`):** if the median PCIe
  receive rate during prefill reaches `--sysmem-guard-mb-s` (default 1000), the run
  is stopped (`warn`: note only, `off`: disabled); see
  [Sysmem fallback on Windows](#sysmem-fallback-on-windows). This also applies to
  the warmup. Setups that stream weights over PCIe on purpose (CPU offload) need
  `warn` or `off`. Without PCIe telemetry the guard is inactive, which a note
  points out.
- **Prefill floor (`--abort-below-pct`, default 20):** if prefill of a repeat drops
  below this share of the prefill median at the first point, the run is stopped.
  This works regardless of the cause and without telemetry. Prefill also drops
  normally with context (example: to about 40 % at 160k); 20 % would only be
  reached at around 400k there. `0` disables the check. The drift check at the end
  is not checked.

The triggering repeat is recorded with the reason (`validation_error`) in
`.samples.csv`; its point does not appear in the result CSV. Completed points are
kept and evaluated. Exit code 1, `stop_reason` `sysmem_fallback` or
`prefill_floor`, `status` `stopped_sysmem_fallback` or `stopped_prefill_floor`;
the drift check is skipped.

## Detecting prefill and decode cliffs

For prefill and decode separately, neighbouring result rows with positive median
rates and increasing actual context length are compared:

```text
drop in % = 100 × (previous rate − current rate) / previous rate
```

Points only count if they have at least `--cliff-min-repeats` valid repeats for the
phase (default 2; `--repeat 1` needs `--cliff-min-repeats 1`). Invalid neighbours
are not skipped to compare points further apart.

A drop of at least `--cliff-pct` (default 15 %) only counts as a
`PREFILL/DECODE CLIFF CANDIDATE` if in addition

- the value ranges of the repeats do not overlap (lowest rate of the previous point
  > highest rate of the current one), so a single outlier does not create a
  candidate, and
- for prefill both points processed a comparable number of new tokens (`prompt_n`
  differs by at most a factor of 1.5); small batches are naturally slower.

All confirmed candidates are printed, each with a suggestion for a finer follow-up
run. Drops above the threshold that miss one of the conditions appear as
`not counted as a cliff` with the reason. In addition the overall change from the
first to the last usable point is shown; a decline of at least `--cliff-pct`
without a single candidate is followed by `(gradual, no single step >= …%)`,
meaning a gradual decline over many small steps. The result CSV also contains
`prefill_tps_min/max`, the metadata a summary under `cliffs`.

A candidate is not a proven cause: VRAM shortage, transfers, clocks, temperature
and other load should be checked in the telemetry.

With active drafting the following also applies:

- A decode drop of at least `--cliff-pct` for which the step rate (1000 / ms per
  step) drops by less than `--cliff-pct` appears as `not counted as a cliff
  (content effect: tokens/step … step cost …)`: the model generated less
  predictable text at that point.
- A separate evaluation `DECODE STEP RATE` (metadata: `cliffs.decode_step_rate`)
  looks for drops in the step rate itself, i.e. pure context effects that more
  tokens per step can hide in the decode rate.

### Drift check

Context points always run in ascending order. If temperature, clocks or background
load change during the run, that would look like a context effect. After a
complete sweep (or on reaching the context limit) ctx-cliff therefore measures the
first point again, exactly as at the start: slot emptied, full prefill for every
repeat. The result appears directly below `drift check: re-measuring the first
point …` and in the metadata (`drift_check`), not in the CSV files:

```text
drift check: re-measuring the first point (target=10000) ...
decode 39.07 -> 38.81 tok/s (-0.7%); prefill 848.28 -> 844.21 tok/s (-0.5%)
-> stable: no relevant drift between start and end of the run
```

After that the measurement is finished, and a server started by the script is
stopped immediately (unless `--keep-server`), before the evaluation is printed. This
frees VRAM earlier, so the next run of a series can start sooner.

A warning appears if prefill or decode rate changed by more than 5 % and the new
value lies outside the original range of the repeats. The check costs the time of
one point at the smallest context; `--no-drift-check` disables it. It is skipped
after an abort or error.

## GPM plausibility checks

GPM at times reports SM, occupancy and tensor as 0 % although the GPU is under full
load. In the runs so far, nvidia-smi showed 98 % utilization at about 160 W in such
phases, GPM graphics was at ~97 % and DRAM at 20–50 %. The dropouts last up to
several minutes, practically always start after an idle phase and end with the next
one; they therefore affect whole requests. Graphics, DRAM and the GPM PCIe values
are not affected.

**Detection:** a request counts as affected for a GPU if anywhere in it (prefill or
decode) at least four consecutive GPM samples spanning at least one second show
graphics ≥ 25 % while SM, occupancy and tensor are each between 0 and 0.1 %.
Missing values, gaps and normal samples interrupt such a series. The 25 % threshold
is based on recorded data: from about 20 % graphics, SM is either practically 0
(dropout) or at least ~7 %; below that, genuinely very small SM values are normal.

**Handling (`--gpm-suspect exclude`, default):** in an affected request all samples
of this GPU with SM, occupancy and tensor ≤ 0.1 % count as suspicious, in both
phases, also at low graphics activity. They are removed from the
SM/occupancy/tensor statistics. SM values clearly above 0 are kept. Graphics, DRAM
and PCIe always use all samples. If no clean sample remains, the columns show `n/a`
or stay empty in the CSV. Over several repeats the remaining valid time is used as
weight, so clean repeats determine the value.

`--gpm-suspect keep` restores the earlier behaviour: suspicious zeros are included
in the means and only marked. The result CSV names the handling used in
`gpm_suspect_handling`. The raw GPM CSV always stays unchanged.

A `*` after PF GPU or DC GPU means: at least one repeat of this phase had
suspicious samples (in the default mode, excluded ones). This is a plausibility
hint, not proof of a driver bug; a missing asterisk does not guarantee error-free
telemetry. `status` still describes benchmark success, not telemetry reliability.

**Complete dropouts:** GPM can also at times report exactly 0 for *all* values,
including PCIe, while the GPU is working (seen in the calibration test with bursty
copies). This can only be detected with a second sensor: if a GPM interval reports
exactly 0 everywhere while the legacy PCIe sampler shows at least 20 MB/s on
average or nvidia-smi at least 25 % utilization, it counts as a dropout. Such
intervals are removed from all GPM statistics (including graphics, DRAM, PCIe and
coverage) and counted as `dropout_samples`/`dropout_ms`; the console marks the
phase with `!`. Real zeros at idle are kept, because then neither sensor shows
activity.

The result CSV adds under `gpm_prefill_` and `gpm_decode_`:

- `dropout_samples`, `dropout_ms`: complete GPM dropouts (see above).
- `suspect_phases`: number of repeat phases with suspicious samples.
- `suspect_samples`: number of suspicious (excluded) samples.
- `suspect_ms`: their total duration, summed as GPU time with several GPUs.
- `suspect_max_run_ms`: longest qualifying series within the phase; 0 if the
  request was detected as affected only through the other phase.

## Options

The complete reference:

```bash
python ctx-cliff.py --help
```

| Option | Default | Effect |
|---|---|---|
| `--file` | required | UTF-8 prompt source (not with `--compare`). |
| `--reference CSV` | – | Compare with this earlier run after the sweep. |
| `--compare REF RUN …` | – | Only compare finished runs, without server and measurement. |
| `--start / --end / --step` | 10000 / 120000 / 5000 | Target prompt range in tokens. |
| `--repeat` | 3 | Completion requests per point; prefill repeats depend on cache mode and snapshot support. |
| `--n-predict` | 64 | Decode tokens requested per request. |
| `--warmup` | 1 | Warmup requests before the measurement. |
| `--settle` | 0.25 s | Pause after planned resets or before measurements after a snapshot restore. |
| `--cache-mode` | incremental | Prefix reuse or `cold`. |
| `--deterministic` | off | Sets `temperature=0` and `top_k=1`; does not guarantee identical timings. |
| `--nonce TEXT` | random | Fixed run marker at the start of the prompt for comparable A/B runs (1–64 printable characters). |
| `--ignore-eos` | off | Requests continuation despite EOS. |
| `--scenario` | file | `file`: continue the file; `agent`: chat conversation with growing file excerpt and fixed task. |
| `--agent-task TEXT` | built-in | Fixed task at the end of the agent prompt. |
| `--agent-thinking` | auto | `enable_thinking` of the chat template: `on`, `off` or template default. |
| `--temperature / --top-p / --top-k / --min-p / --typical-p` | server | Sampling settings for every request (also with underscores). |
| `--repeat-penalty / --repeat-last-n / --presence-penalty / --frequency-penalty` | server | Penalties for every request; `--repetition-penalty` is an alias. |
| `--sampler KEY=VALUE` | – | Any other llama-server sampling field, repeatable. |
| `--seed` | server | Seed; repeat r uses `SEED + r − 1`. |
| `--base-url` | `http://127.0.0.1:8080` | Server address. |
| `--server-command` | – | Start this server and stop it at the end. |
| `--reuse-running-server` | off | Use a server that is already running instead of refusing to start. |
| `--keep-server` | off | Keep a server started by the script running after the run. |
| `--slot-id` | 0 | Slot for all benchmark requests. |
| `--server-start-timeout` | 300 s | Time to wait for a server started by the script. |
| `--vram-settle-s` | 30 s | Before a server start, wait at most this long for VRAM of a predecessor to be released; `0` = off. |
| `--keep-snapshot` | off | Keep the snapshot of a local server started by the script instead of deleting it. |
| `--cliff-pct` | 15 | Threshold for drops between neighbouring points. |
| `--cliff-min-repeats` | 2 | Minimum valid repeats per phase and point for cliff comparisons. |
| `--no-drift-check` | off | Do not measure the first point again at the end. |
| `--sysmem-guard` | abort | On sysmem fallback abort (`abort`), warn (`warn`) or do nothing (`off`). |
| `--sysmem-guard-mb-s` | 1000 | Median PCIe receive during prefill that triggers the sysmem guard. |
| `--abort-below-pct` | 20 | Abort when prefill drops below this share of the first point; `0` = off. |
| `--vram-log` | auto | `auto`, `off` or forced `nvidia`. |
| `--vram-backend` | auto | Source of VRAM/clock/power values: `nvml`, `nvidia-smi` or `auto` (NVML, else nvidia-smi). |
| `--vram-interval-ms` | 250 | NVIDIA sampling interval. |
| `--vram-gpu` | all | NVIDIA GPU selection, for example `0` or `0,1`. |
| `--no-stream` | off | Do not stream responses; phase boundary by back-projection. |
| `--pcie-interval-ms` | 250 | Legacy NVML PCIe polling interval; if a query is slower, the achieved spacing is reported. |
| `--pcie-legacy-scale` | 1.0 | Divide legacy PCIe sampler values in all evaluations by this factor (for example 1.55 from the calibration test). |
| `--gpm-log` | auto | `auto` follows `--vram-log`; otherwise `on`/`off`. |
| `--gpm-interval-ms` | 250 | GPM interval, must be longer than 100 ms. |
| `--gpm-suspect` | exclude | Exclude suspicious GPM SM/O/T zeros from the means, or include them with `keep`. |
| `--gpm-lag-ms` | 70 | GPM values describe traffic this many ms before they are read; intervals are shifted back. `0` disables. |
| `--gpm-restart` | off | Experimental: restart GPM measurement after a request with a GPM dropout (`realloc` or `reinit`). |
| `--win-gpu-mem` | auto | Windows memory telemetry, otherwise `on`/`off`. |
| `--win-gpu-mem-interval-ms` | 1000 | Windows sampling interval, at least 1000 ms. |

All sensors can be disabled for a comparison run with
`--vram-log off --gpm-log off --win-gpu-mem off`.

## Limitations

- Snapshot support and content depend on the server and model type. Some
  implementations do not store the list of their internal prompt checkpoints in
  slot files. Therefore pure prefix states without a decode tail are prepared and
  their reuse is checked. Matching token counts do not prove that all internal
  states are equal.
- Save/restore is outside the timing, but can influence hardware caches, memory
  residency and thus subsequent measurements.
- Some server builds lose the slot snapshot when speculative decoding is active;
  incremental runs then abort with "prefix reuse lost". Use `--cache-mode cold`
  in that case.
- Sensor intervals can miss short peaks. More frequent polling can itself affect
  performance.
- Prefill/decode telemetry is based on estimated time windows. Exact phase analysis
  requires server-side instrumentation.
- The program uses and modifies the selected server slot. For comparable results no
  other client should use it at the same time.
- The agent scenario simulates a single agent turn, not a multi-turn tool-call
  history.

## Tests

Run the whole offline test suite from the repository directory:

```bash
python -m unittest discover -s unittests -v
```

The simulations of the calibration and probe scripts run in real time; the whole
suite takes a few minutes. For a quick run they can be skipped:

```bash
CTX_CLIFF_FAST_TESTS=1 python -m unittest discover -s unittests
```

```powershell
$env:CTX_CLIFF_FAST_TESTS = "1"; python -m unittest discover -s unittests
```

All tests load `ctx-cliff.py` from the repository directory by default.
`CTX_CLIFF_SCRIPT` can point to the absolute path of another copy of the script.

Among other things the tests cover CSV recording, measurement phases and
aggregation, GPM plausibility, prefill repeats, snapshot fallback and cleanup, the
agent scenario and sampling options, run guards, drafting step statistics, run
comparison, VRAM settling, run metadata, output fingerprints, empty instead of 0
values and behaviour on errors and server crashes, as well as the calibration and
probe scripts with a simulated GPU.
