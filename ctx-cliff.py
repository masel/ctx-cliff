#!/usr/bin/env python3

# Copyright 2026 cHunter789 (original version)
# Copyright 2026 masel (modifications and extensions)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
ctx-cliff.py — repeated prefill and decode measurements.

The benchmark measures server-side llama.cpp timings at progressively larger
prompt contexts. It separates two cache behaviours explicitly:

  incremental  keep one slot and reuse the common prompt prefix between steps
  cold         erase the slot before every sample and disable prompt caching

At each context point it records:
  - cache_n / prompt_n and prefill tok/s
  - median decode tok/s across repeated samples
  - draft acceptance (MTP, DFlash, draft model, ...)
  - wall time and stop status

In incremental mode, each repeat starts from the same saved prompt-prefix state.
Later points prepare that exact token prefix without a cached decode tail. Prefill
time and throughput are medians across repeats, as is decode throughput.
Managed launches automatically add --slot-save-path beside this script if absent.
A rejected save probe disables prefill repeats with a warning (decode still repeats).
One uniquely named snapshot file is reused throughout the run. Managed local
launches delete it on exit by default; --keep-snapshot preserves it. External
servers require manual deletion because the API has no file-delete action.
Prefix preparation, snapshot I/O and --settle are outside the measured request windows.

Origin: based on ctx-cliff.py by cHunter789, distributed with
  https://huggingface.co/cHunter789/Qwen3.8-27B-i1-IQ4_KS_KT-GGUF
  under the Apache License 2.0 (confirmed by the author in
  https://huggingface.co/cHunter789/Qwen3.8-27B-i1-IQ4_KS_KT-GGUF/discussions/3).
  This version has been substantially modified and extended.
"""

import argparse
import bisect
import math
import pickle
import signal
import collections
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import re
import traceback
import secrets
import shlex
import shutil
import statistics
import subprocess
import sys
import threading
import tempfile
from contextlib import ExitStack, contextmanager
import time
from typing import Any, Dict, List, Optional, Protocol
from urllib.parse import urlparse

import requests


class HttpResponse(Protocol):
    @property
    def ok(self) -> bool: ...

    @property
    def status_code(self) -> int: ...

    @property
    def text(self) -> str: ...

    def json(self) -> Any: ...

    def raise_for_status(self) -> None: ...


class GetClient(Protocol):
    def get(self, url: str, **kwargs: Any) -> HttpResponse: ...


class PostClient(Protocol):
    def post(self, url: str, **kwargs: Any) -> HttpResponse: ...


class RequestClient(Protocol):
    def request(self, method: str, url: str, **kwargs: Any) -> HttpResponse: ...


class BenchmarkHttpClient(PostClient, RequestClient, Protocol):
    """HTTP methods needed by the runner, including test doubles."""


VRAM_CSV_FIELDS = (
    'timestamp',
    'gpu_index',
    'phase',
    'target_ctx',
    'repeat',
    'used_mib',
    'total_mib',
    'used_pct',
    'gpu_util_pct',
    'mem_util_pct',
    'pstate',
    'gpu_clock_mhz',
    'mem_clock_mhz',
    'power_draw_w',
    'power_limit_w',
    'temperature_c',
    'pcie_link_gen',
    'pcie_link_width',
)

PCIE_CSV_FIELDS = (
    'timestamp',
    'gpu_index',
    'phase',
    'subphase',
    'bus_subphase',
    'target_ctx',
    'repeat',
    'pcie_rx_mb_s',
    'pcie_tx_mb_s',
    'pcie_bus_util_pct',
    'pcie_bus_window_s',
    'pcie_link_gen',
    'pcie_link_width',
    'pcie_link_gen_max',
    'pcie_link_width_max',
)

GPM_CSV_FIELDS = (
    'timestamp',
    'gpu_index',
    'target_ctx',
    'repeat',
    'subphase',
    'interval_start_mono',
    'interval_end_mono',
    'sample_start_mono',
    'sample_end_mono',
    'interval_ms',
    'graphics_util_pct',
    'sm_util_pct',
    'sm_occupancy_pct',
    'tensor_util_pct',
    'dram_bw_util_pct',
    'pcie_rx_mib_s',
    'pcie_tx_mib_s',
    'pcie_link_gen',
    'pcie_link_width',
    'pcie_theoretical_mib_s',
)

WINDOWS_CSV_FIELDS = (
    'timestamp',
    'phase',
    'target_ctx',
    'repeat',
    'dedicated_mib',
    'shared_mib',
    'adapter_instances',
)

RESULT_CSV_FIELDS = (
    'target_ctx',
    'total_ctx',
    'target_chars',
    'cache_mode',
    'cache_n',
    'prompt_n',
    'prompt_ms',
    'prefill_tps',
    'decode_tps_median',
    'decode_tps_min',
    'decode_tps_max',
    'valid_repeats',
    'repeat',
    'draft_n',
    'draft_acc',
    'draft_acc_pct',
    'tokens_per_step_median',
    'ms_per_step_median',
    'ms_per_step_min',
    'ms_per_step_max',
    'wall_s_median',
    'vram_samples',
    'vram_peak_mib',
    'vram_total_mib',
    'vram_peak_pct',
    'vram_free_min_mib',
    'gpu_util_peak_pct',
    'mem_util_peak_pct',
    'gpu_clock_min_mhz',
    'gpu_clock_median_mhz',
    'gpu_clock_max_mhz',
    'mem_clock_min_mhz',
    'mem_clock_median_mhz',
    'mem_clock_max_mhz',
    'pstate_mode',
    'power_avg_w',
    'power_peak_w',
    'power_limit_w',
    'temperature_peak_c',
    'pcie_samples',
    'pcie_rx_median_mb_s',
    'pcie_rx_p95_mb_s',
    'pcie_rx_p99_mb_s',
    'pcie_rx_peak_mb_s',
    'pcie_tx_median_mb_s',
    'pcie_tx_p95_mb_s',
    'pcie_tx_p99_mb_s',
    'pcie_tx_peak_mb_s',
    'pcie_link_gen',
    'pcie_link_width',
    'pcie_theoretical_mb_s',
    'pcie_p95_pct_theoretical',
    'pcie_p99_pct_theoretical',
    'pcie_peak_pct_theoretical',
    'pcie_over80_pct',
    'pcie_over90_pct',
    'pcie_bus_samples',
    'pcie_bus_avg_pct',
    'pcie_bus_median_pct',
    'pcie_bus_p95_pct',
    'pcie_bus_peak_pct',
    'pcie_prefill_samples',
    'pcie_prefill_rx_median_mb_s',
    'pcie_prefill_rx_p95_mb_s',
    'pcie_prefill_rx_p99_mb_s',
    'pcie_prefill_rx_peak_mb_s',
    'pcie_prefill_tx_median_mb_s',
    'pcie_prefill_tx_p95_mb_s',
    'pcie_prefill_tx_p99_mb_s',
    'pcie_prefill_tx_peak_mb_s',
    'pcie_prefill_p95_pct_theoretical',
    'pcie_prefill_p99_pct_theoretical',
    'pcie_prefill_peak_pct_theoretical',
    'pcie_prefill_over80_pct',
    'pcie_prefill_over90_pct',
    'pcie_prefill_bus_samples',
    'pcie_prefill_bus_avg_pct',
    'pcie_prefill_bus_median_pct',
    'pcie_prefill_bus_p95_pct',
    'pcie_prefill_bus_peak_pct',
    'pcie_decode_samples',
    'pcie_decode_rx_median_mb_s',
    'pcie_decode_rx_p95_mb_s',
    'pcie_decode_rx_p99_mb_s',
    'pcie_decode_rx_peak_mb_s',
    'pcie_decode_tx_median_mb_s',
    'pcie_decode_tx_p95_mb_s',
    'pcie_decode_tx_p99_mb_s',
    'pcie_decode_tx_peak_mb_s',
    'pcie_decode_p95_pct_theoretical',
    'pcie_decode_p99_pct_theoretical',
    'pcie_decode_peak_pct_theoretical',
    'pcie_decode_over80_pct',
    'pcie_decode_over90_pct',
    'pcie_decode_bus_samples',
    'pcie_decode_bus_avg_pct',
    'pcie_decode_bus_median_pct',
    'pcie_decode_bus_p95_pct',
    'pcie_decode_bus_peak_pct',
    'gpm_prefill_samples',
    'gpm_prefill_graphics_avg_pct',
    'gpm_prefill_sm_avg_pct',
    'gpm_prefill_occupancy_avg_pct',
    'gpm_prefill_tensor_avg_pct',
    'gpm_prefill_dram_avg_pct',
    'gpm_prefill_graphics_p95_pct',
    'gpm_prefill_sm_p95_pct',
    'gpm_prefill_occupancy_p95_pct',
    'gpm_prefill_tensor_p95_pct',
    'gpm_prefill_dram_p95_pct',
    'gpm_prefill_pcie_rx_median_mib_s',
    'gpm_prefill_pcie_rx_p95_mib_s',
    'gpm_prefill_pcie_rx_p99_mib_s',
    'gpm_prefill_pcie_rx_peak_mib_s',
    'gpm_prefill_pcie_tx_median_mib_s',
    'gpm_prefill_pcie_tx_p95_mib_s',
    'gpm_prefill_pcie_tx_p99_mib_s',
    'gpm_prefill_pcie_tx_peak_mib_s',
    'gpm_prefill_pcie_theoretical_mib_s',
    'gpm_prefill_pcie_p95_pct_theoretical',
    'gpm_prefill_pcie_p99_pct_theoretical',
    'gpm_prefill_pcie_peak_pct_theoretical',
    'gpm_prefill_pcie_over80_pct',
    'gpm_prefill_pcie_over90_pct',
    'gpm_decode_samples',
    'gpm_decode_graphics_avg_pct',
    'gpm_decode_sm_avg_pct',
    'gpm_decode_occupancy_avg_pct',
    'gpm_decode_tensor_avg_pct',
    'gpm_decode_dram_avg_pct',
    'gpm_decode_graphics_p95_pct',
    'gpm_decode_sm_p95_pct',
    'gpm_decode_occupancy_p95_pct',
    'gpm_decode_tensor_p95_pct',
    'gpm_decode_dram_p95_pct',
    'gpm_decode_pcie_rx_median_mib_s',
    'gpm_decode_pcie_rx_p95_mib_s',
    'gpm_decode_pcie_rx_p99_mib_s',
    'gpm_decode_pcie_rx_peak_mib_s',
    'gpm_decode_pcie_tx_median_mib_s',
    'gpm_decode_pcie_tx_p95_mib_s',
    'gpm_decode_pcie_tx_p99_mib_s',
    'gpm_decode_pcie_tx_peak_mib_s',
    'gpm_decode_pcie_theoretical_mib_s',
    'gpm_decode_pcie_p95_pct_theoretical',
    'gpm_decode_pcie_p99_pct_theoretical',
    'gpm_decode_pcie_peak_pct_theoretical',
    'gpm_decode_pcie_over80_pct',
    'gpm_decode_pcie_over90_pct',
    'win_gpu_samples',
    'win_dedicated_peak_mib',
    'win_shared_peak_mib',
    'win_shared_delta_mib',
    'win_gpu_adapter_instances',
    'status',
    'sample_statuses',
)

VRAM_DEFAULTS = {'vram_samples': 0,
 'vram_peak_mib': None,
 'vram_total_mib': None,
 'vram_peak_pct': None,
 'vram_free_min_mib': None,
 'gpu_util_peak_pct': None,
 'mem_util_peak_pct': None,
 'gpu_clock_min_mhz': None,
 'gpu_clock_median_mhz': None,
 'gpu_clock_max_mhz': None,
 'mem_clock_min_mhz': None,
 'mem_clock_median_mhz': None,
 'mem_clock_max_mhz': None,
 'pstate_mode': '',
 'power_avg_w': None,
 'power_peak_w': None,
 'power_limit_w': None,
 'temperature_peak_c': None,
 'pcie_samples': 0,
 'pcie_rx_median_mb_s': None,
 'pcie_rx_p95_mb_s': None,
 'pcie_rx_p99_mb_s': None,
 'pcie_rx_peak_mb_s': None,
 'pcie_tx_median_mb_s': None,
 'pcie_tx_p95_mb_s': None,
 'pcie_tx_p99_mb_s': None,
 'pcie_tx_peak_mb_s': None,
 'pcie_link_gen': None,
 'pcie_link_width': None,
 'pcie_theoretical_mb_s': None,
 'pcie_p95_pct_theoretical': None,
 'pcie_p99_pct_theoretical': None,
 'pcie_peak_pct_theoretical': None,
 'pcie_over80_pct': None,
 'pcie_over90_pct': None,
 'pcie_bus_samples': 0,
 'pcie_bus_avg_pct': None,
 'pcie_bus_median_pct': None,
 'pcie_bus_p95_pct': None,
 'pcie_bus_peak_pct': None,
 'power_valid_samples': 0}

def finite_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def polling_ticks(stop: threading.Event, interval_s: float, immediate: bool = False) -> Any:
    """Interruptible cadence without bursts of catch-up polls after a driver stall."""
    next_tick = time.perf_counter() + (0.0 if immediate else interval_s)
    while not stop.wait(max(0.0, next_tick - time.perf_counter())):
        yield time.perf_counter()
        next_tick += interval_s
        if next_tick < time.perf_counter():
            next_tick = time.perf_counter() + interval_s


@contextmanager
def termination_handler() -> Any:
    """Route a catchable SIGTERM through normal context-manager cleanup."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGTERM)
    def terminate(signum: int, frame: Any) -> None:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def rounded(value: Any, digits: int = 2) -> Optional[float]:
    value = finite_number(value)
    return round(value, digits) if value is not None else None


def percentile(values: Any, pct: float) -> Optional[float]:
    """Linear interpolation over finite observations; missing is distinct from zero."""
    if not 0 <= pct <= 100:
        raise ValueError("percentile must be between 0 and 100")
    values = sorted(v for x in values if (v := finite_number(x)) is not None)
    if not values:
        return None
    position = (len(values) - 1) * pct / 100.0
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def reduce_values(values: Any, operation: str = "median") -> Any:
    values = [v for x in values if (v := finite_number(x)) is not None]
    if operation == "sum":
        return sum(values)
    if not values:
        return None
    return {"median": statistics.median, "min": min, "max": max, "mean": statistics.mean}[operation](values)


def weighted_mean(pairs: Any) -> Optional[float]:
    valid = [(v, w) for value, weight in pairs
             if (v := finite_number(value)) is not None
             and (w := finite_number(weight)) is not None and w > 0]
    return sum(v * w for v, w in valid) / sum(w for _, w in valid) if valid else None


class SampleStore:
    """Indexed active-request buffer with a buffered, disk-backed raw archive.

    Idle history is limited to five seconds. A request pins its start until all
    phase summaries are complete. CSV consumers read the archive incrementally;
    exports stream it, so neither needs to retain the entire run in RAM.
    """
    def __init__(self, time_key: str = "t_mono") -> None:
        self.time_key = time_key
        self.rows: List[Dict[str, Any]] = []
        self.times: List[float] = []
        self.first = 0
        self.pin: Optional[float] = None
        self.archive: Any = None
        self.archive_enabled = True
        self.error: Optional[Exception] = None
        self.closed = False
        self.count = 0
        self.csv_count = 0
        self.csv_offset = 0
        self.lock = threading.RLock()

    def __len__(self) -> int:
        return self.count

    def append(self, row: Dict[str, Any]) -> None:
        with self.lock:
            if self.closed:
                raise RuntimeError("telemetry store is closed")
            now = float(row.get(self.time_key, 0.0))
            if self.times and now < self.times[-1]:
                raise ValueError("telemetry timestamps must be nondecreasing")
            if self.archive_enabled:
                try:
                    if self.archive is None:
                        self.archive = tempfile.TemporaryFile(mode="w+b")
                    self.archive.seek(0, os.SEEK_END)
                    pickle.dump(row, self.archive, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception as error:
                    self.error = error
                    raise
            self.count += 1
            self.rows.append(row)
            self.times.append(now)
            cutoff = min(self.pin, now - 5.0) if self.pin is not None else now - 5.0
            self.first = bisect.bisect_left(self.times, cutoff, lo=self.first)
            # Amortize compaction and promptly release references to old rows.
            if self.first >= 1024 and self.first >= len(self.rows) // 2:
                del self.rows[:self.first]
                del self.times[:self.first]
                self.first = 0

    def extend(self, rows: Any) -> None:
        for row in rows:
            self.append(row)

    def begin(self, start: float) -> None:
        with self.lock:
            self.pin = start

    def end(self) -> None:
        with self.lock:
            self.pin = None

    def window(self, start: float, end: float, nearest_s: float = 0.0) -> tuple[List[Dict[str, Any]], float]:
        with self.lock:
            lo = bisect.bisect_left(self.times, start, lo=self.first)
            hi = bisect.bisect_right(self.times, end, lo=lo)
            selected = list(self.rows[lo:hi])
            if not selected and nearest_s and self.first < len(self.rows):
                at = bisect.bisect_left(self.times, end, lo=self.first)
                choices = [i for i in (at - 1, at) if self.first <= i < len(self.rows)]
                closest = min(choices, key=lambda i: abs(self.times[i] - end))
                age = abs(self.times[closest] - end)
                if age <= nearest_s:
                    return [self.rows[closest]], age * 1000.0
            return selected, 0.0

    def read_since(self, count: int, limit: int = 4096) -> List[Dict[str, Any]]:
        with self.lock:
            if self.error is not None:
                raise RuntimeError(f"telemetry archive failed: {self.error}") from self.error
            if self.archive is None:
                return []
            if count != self.csv_count:
                raise ValueError("CSV archive consumer must advance sequentially")
            self.archive.seek(self.csv_offset)
            rows = [pickle.load(self.archive) for _ in range(min(limit, self.count - count))]
            self.csv_offset = self.archive.tell()
            self.csv_count += len(rows)
            return rows

    def iter_all(self) -> Any:
        offset = 0
        with self.lock:
            if self.error is not None:
                raise RuntimeError(f"telemetry archive failed: {self.error}") from self.error
            if not self.archive_enabled:
                raise RuntimeError("raw archive is disabled for this monitor")
            remaining = self.count
        while remaining:
            with self.lock:
                self.archive.seek(offset)
                batch = [pickle.load(self.archive) for _ in range(min(1024, remaining))]
                offset = self.archive.tell()
            remaining -= len(batch)
            yield from batch

    def close(self) -> None:
        with self.lock:
            self.closed = True
            if self.archive is not None:
                self.archive.close()
                self.archive = None


class PhaseIndex:
    """Lookup disjoint reconstructed windows in logarithmic time."""
    def __init__(self, windows: Any) -> None:
        self.windows = sorted(windows, key=lambda w: w[3])
        self.starts = [w[3] for w in self.windows]

    def find(self, start: float, end: float) -> Any:
        index = bisect.bisect_right(self.starts, start) - 1
        if index >= 0:
            window = self.windows[index]
            if start <= end <= window[4]:
                return window
        return None


class PromptBuildError(ValueError):
    pass


class InputExhausted(RuntimeError):
    """The input file cannot produce a longer prompt than the previous point."""


class PromptBuilder:
    """Prompts of an exact token count: the first N tokens of the tokenized input.

    A character prefix usually ends inside a word or an indentation run, so its
    final token is one the model never sees in training (e.g. `get_user_mo`);
    greedy decoding then often jumps to EOS or loops. Instead, a window with at
    least TOKEN_GUARD spare tokens is tokenized and the token list truncated, so
    every submitted token matches the tokenization of the whole file and
    successive points are token prefixes of each other.
    """
    # Tokens this close to the character end of the tokenized window may differ
    # from the tokenization of the full file and are dropped.
    TOKEN_GUARD = 64
    MAX_TOKENIZE_CALLS = 40

    def __init__(self, content: str, nonce: str, tokenize: Any, chars_per_token: float, nonce_tokens: int,
                 max_prompt_ctx: Optional[int] = None, detokenize: Any = None,
                 suffix_tokens: Optional[List[int]] = None) -> None:
        if not content:
            raise PromptBuildError("the prompt source is empty")
        self.content, self.nonce, self.tokenize = content, nonce, tokenize
        self.initial_density, self.nonce_tokens = chars_per_token, nonce_tokens
        self.max_prompt_ctx, self.detokenize = max_prompt_ctx, detokenize
        # Fixed tokens after the input excerpt (--scenario agent: end of the context
        # message, pseudo reply, task and generation prompt).
        self.suffix_tokens = list(suffix_tokens or [])
        self.last_tokens: Optional[tuple[str, List[int]]] = None
        self.reset()

    def reset(self) -> None:
        self.density = self.initial_density

    def build(self, ctx: int) -> tuple[str, int, int]:
        """Return (tokenized text window, prompt tokens, covered input characters).

        The submitted token IDs are kept in `last_tokens` = (window, IDs).
        """
        self.last_tokens = None
        if self.max_prompt_ctx is not None:
            ctx = min(ctx, self.max_prompt_ctx)
        suffix = self.suffix_tokens
        budget = ctx - len(suffix)  # prefix and input excerpt
        if budget <= self.nonce_tokens:
            raise PromptBuildError(f"prompt prefix and suffix exceed the {ctx}-token budget")
        need = budget + self.TOKEN_GUARD
        chars = min(len(self.content), int((need - self.nonce_tokens) * self.density * 1.02) + 64)
        tokens: List[int] = []
        for _ in range(self.MAX_TOKENIZE_CALLS):
            tokens = self.tokenize(self.nonce + self.content[:chars])
            if not isinstance(tokens, list):
                raise PromptBuildError("the tokenizer must return token IDs")
            if len(tokens) >= need or chars >= len(self.content):
                break
            rate = chars / max(1, len(tokens) - self.nonce_tokens)
            chars = min(len(self.content), int(chars + (need - len(tokens)) * rate * 1.05) + 64)
        exhausted = chars >= len(self.content)
        # The end of the file is a natural end: no guard needed there.
        count = min(budget, len(tokens) if exhausted else len(tokens) - self.TOKEN_GUARD)
        if count <= self.nonce_tokens:
            raise PromptBuildError(f"prompt prefix and minimum input exceed the {ctx}-token budget")
        if count < budget and not exhausted:
            print(f"WARNING: prompt for target {ctx} has only {count + len(suffix)} tokens after "
                  f"{self.MAX_TOKENIZE_CALLS} tokenizer calls", file=sys.stderr)
        self.density = chars / max(1, len(tokens) - self.nonce_tokens)
        window = self.nonce + self.content[:chars]
        self.last_tokens = (window, tokens[:count] + suffix)
        return window, count + len(suffix), self._content_chars(tokens[:count], count, chars, len(tokens))

    def _content_chars(self, tokens: List[int], count: int, chars: int, window_tokens: int) -> int:
        """Input characters covered by the truncated prompt (exact via detokenize if possible)."""
        if self.detokenize is not None:
            try:
                text = self.detokenize(tokens)
                # A leading BOS may render as text; the nonce locates the input start.
                start = text.find(self.nonce) if isinstance(text, str) else -1
                if start >= 0:
                    covered = text[start + len(self.nonce):]
                    if self.content.startswith(covered):
                        return len(covered)
            except Exception:
                pass
        # Estimate: share of the tokenized window, which spans `chars` input characters.
        content_tokens = max(1, window_tokens - self.nonce_tokens)
        return min(chars, int(chars * max(0, count - self.nonce_tokens) / content_tokens))


def reconstruct_phases(start: float, end: float, prompt_ms: float, decode_ms: float,
                       first_token: Optional[float] = None) -> tuple[Any, Dict[str, Any]]:
    """Prefill/decode windows on the client clock.

    With streaming, the arrival of the first generated token marks the end of
    prefill directly (method first_token_anchor); prefill is then placed before
    it using prompt_ms. Without it, both phases are back-projected from the end
    of the response (response_end_backprojection). phase_anchor_offset_ms shows
    how far the back-projected boundary would have been off.
    """
    prompt_ms = max(0.0, finite_number(prompt_ms) or 0.0)
    decode_ms = max(0.0, finite_number(decode_ms) or 0.0)
    backprojected = max(start, end - decode_ms / 1000.0)
    anchored = first_token is not None and start < first_token <= end
    decode_start = first_token if anchored else backprojected
    prefill_start = max(start, decode_start - prompt_ms / 1000.0)
    residual = (end - start) * 1000.0 - prompt_ms - decode_ms
    return (("prefill", prefill_start, decode_start), ("decode", decode_start, end)), {
        "phase_method": "first_token_anchor" if anchored else "response_end_backprojection",
        "phase_wall_residual_ms": max(0.0, residual),
        "phase_timing_excess_ms": max(0.0, -residual),
        "phase_anchor_offset_ms": (backprojected - first_token) * 1000.0 if anchored else None,
    }


def covered_ms(rows: Any) -> float:
    intervals = sorted((float(r["interval_start_mono"]), float(r["interval_end_mono"])) for r in rows)
    total, previous_end = 0.0, float("-inf")
    for start, end in intervals:
        total += max(0.0, end - max(start, previous_end))
        previous_end = max(previous_end, end)
    return total * 1000.0


def interval_ms(row: Dict[str, Any]) -> float:
    return max(0.0, (float(row["interval_end_mono"]) - float(row["interval_start_mono"])) * 1000.0)


def metric_stats(rows: Any, key: str, weight: Any = None) -> Dict[str, Any]:
    valid = [(value, weight(row) if weight else 1.0) for row in rows
             if (value := finite_number(row.get(key))) is not None]
    values = [v for v, _ in valid]
    return {"samples": len(valid), "valid_ms": sum(w for _, w in valid) if weight else 0.0,
            "avg": weighted_mean(valid), "median": percentile(values, 50),
            "p95": percentile(values, 95), "p99": percentile(values, 99),
            "peak": max(values) if values else None}


GPM_ENGINES = {
    "graphics": "graphics_util_pct", "sm": "sm_util_pct", "occupancy": "sm_occupancy_pct",
    "tensor": "tensor_util_pct", "dram": "dram_bw_util_pct",
}


def pcie_defaults(prefix: str = "pcie_", unit: str = "mb_s") -> Dict[str, Any]:
    values = {prefix + "samples": 0, prefix + "ratio_samples": 0, prefix + "ratio_valid_ms": 0.0}
    for direction in ("rx", "tx"):
        values[prefix + direction + "_valid_samples"] = 0
        values[prefix + direction + "_valid_ms"] = 0.0
        for statistic in ("median", "p95", "p99", "peak"):
            values[f"{prefix}{direction}_{statistic}_{unit}"] = None
    for key in ("p95_pct_theoretical", "p99_pct_theoretical", "peak_pct_theoretical", "over80_pct", "over90_pct"):
        values[prefix + key] = None
    values[prefix + "theoretical_" + unit] = None
    return values


def summarize_pcie(rows: Any, prefix: str, unit: str, capacity: Any, weight: Any = None) -> Dict[str, Any]:
    result = pcie_defaults(prefix, unit)
    result[prefix + "samples"] = len(rows)
    for direction in ("rx", "tx"):
        stats = metric_stats(rows, f"pcie_{direction}_{unit}", weight)
        for statistic in ("median", "p95", "p99", "peak"):
            result[f"{prefix}{direction}_{statistic}_{unit}"] = stats[statistic]
        result[prefix + direction + "_valid_samples"] = stats["samples"]
        result[prefix + direction + "_valid_ms"] = stats["valid_ms"]
    ratios, capacities = [], []
    for row in rows:
        cap = finite_number(capacity(row))
        rx, tx = (finite_number(row.get(f"pcie_{direction}_{unit}")) for direction in ("rx", "tx"))
        if cap is not None and cap > 0:
            capacities.append(cap)
            if rx is not None and tx is not None:
                ratios.append((100.0 * max(rx, tx) / cap, weight(row) if weight else 1.0))
    result[prefix + "theoretical_" + unit] = percentile(capacities, 50)
    result[prefix + "ratio_samples"] = len(ratios)
    result[prefix + "ratio_valid_ms"] = sum(w for _, w in ratios) if weight else 0.0
    for statistic, quantile in (("p95", 95), ("p99", 99)):
        result[prefix + statistic + "_pct_theoretical"] = percentile([v for v, _ in ratios], quantile)
    result[prefix + "peak_pct_theoretical"] = max((v for v, _ in ratios), default=None)
    for threshold in (80, 90):
        result[prefix + f"over{threshold}_pct"] = weighted_mean([(100.0 if v >= threshold else 0.0, w) for v, w in ratios])
    return result


def empty_gpm() -> Dict[str, Any]:
    result = {"gpm_samples": 0, "gpm_phase_ms": 0.0, "gpm_covered_ms": 0.0,
              "gpm_coverage_pct": None, "gpm_boundary_samples": 0,
              "gpm_dropout_samples": 0, "gpm_dropout_ms": 0.0,
              "gpm_suspect_samples": 0, "gpm_suspect_ms": 0.0,
              "gpm_suspect_phases": 0, "gpm_suspect_max_run_ms": 0.0}
    for engine in GPM_ENGINES:
        result.update({f"gpm_{engine}_avg_pct": None, f"gpm_{engine}_p95_pct": None,
                       f"gpm_{engine}_valid_samples": 0, f"gpm_{engine}_valid_ms": 0.0})
    result.update(pcie_defaults("gpm_pcie_", "mib_s"))
    return result


def empty_vram() -> Dict[str, Any]:
    result = dict(VRAM_DEFAULTS)
    result.update(pcie_defaults())
    result.update({"vram_fallback_samples": 0, "vram_fallback_max_age_ms": None})
    return result


def empty_windows() -> Dict[str, Any]:
    return {"win_gpu_samples": 0, "win_dedicated_peak_mib": None, "win_shared_peak_mib": None,
            "win_shared_delta_mib": None, "win_gpu_adapter_instances": 0,
            "win_fallback_samples": 0, "win_fallback_max_age_ms": None}


# Graphics activity above which SM/occupancy/tensor <= 0.1 % cannot be real.
# In the recorded traces, samples with graphics >= 20 % either show SM ~0 (dropout)
# or SM >= ~7 %; below 20 % genuinely tiny SM values are common. 25 % keeps a margin.
GPM_SUSPECT_GRAPHICS_PCT = 25.0
GPM_ZERO_PCT = 0.1
# Engines affected by the observed GPM dropout. Graphics, DRAM and GPM PCIe keep
# reporting plausible values during such runs and are never excluded.
GPM_SUSPECT_ENGINES = ("sm", "occupancy", "tensor")
GPM_SUSPECT_KEYS = ("sm_util_pct", "sm_occupancy_pct", "tensor_util_pct")


def gpm_zero_like(row: Dict[str, Any]) -> bool:
    """SM, occupancy and tensor all present and <= 0.1 %."""
    values = [finite_number(row.get(key)) for key in GPM_SUSPECT_KEYS]
    return all(v is not None and 0.0 <= v <= GPM_ZERO_PCT for v in values)


def gpm_plausibility(rows: Any) -> Dict[str, Any]:
    """Flag sustained near-zero SM counters under load; never correct raw values.

    Conservative heuristic, not proof of a driver fault: at least four adjacent
    samples spanning >=1 s, graphics >=25%, SM/occupancy/tensor each <=0.1%.
    Missing values, idle samples and gaps break runs. Evaluate each GPU separately.
    Durations/counts include only qualifying runs (GPU-time for multiple GPUs).
    """
    return plausibility_summary(find_suspect_runs(rows))


def find_suspect_runs(rows: Any) -> List[List[Dict[str, Any]]]:
    """Qualifying runs of adjacent suspect GPM samples, evaluated per GPU."""
    by_gpu: Dict[Any, Any] = {}
    for row in rows:
        by_gpu.setdefault(row.get("gpu_index", 0), []).append(row)
    runs: List[List[Dict[str, Any]]] = []
    for gpu_rows in by_gpu.values():
        run: List[Dict[str, Any]] = []
        duration, previous_end = 0.0, None
        for row in sorted(gpu_rows, key=lambda r: float(r["interval_start_mono"])) + [None]:
            suspect = False
            if row is not None:
                start, end = float(row["interval_start_mono"]), float(row["interval_end_mono"])
                graphics = finite_number(row.get("graphics_util_pct"))
                suspect = (end > start and graphics is not None
                           and graphics >= GPM_SUSPECT_GRAPHICS_PCT and gpm_zero_like(row))
            adjacent = row is not None and (previous_end is None or abs(start - previous_end) <= 0.001)
            if not suspect or not adjacent:
                if len(run) >= 4 and duration >= 1000.0 - 1e-6:
                    runs.append(run)
                run, duration = [], 0.0
            if suspect:
                run.append(row)
                duration += (end - start) * 1000.0
            previous_end = end if row is not None else None
    return runs


def plausibility_summary(runs: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    durations = [sum(interval_ms(row) for row in run) for run in runs]
    return {"gpm_suspect_samples": sum(len(run) for run in runs), "gpm_suspect_ms": sum(durations),
            "gpm_suspect_phases": int(bool(runs)),
            "gpm_suspect_max_run_ms": max(durations, default=0.0)}


GPM_VALUE_KEYS = (*GPM_ENGINES.values(), "pcie_rx_mib_s", "pcie_tx_mib_s")


def gpm_all_zero(row: Dict[str, Any]) -> bool:
    """Every reported GPM value (engines and PCIe) is exactly 0."""
    values = [v for key in GPM_VALUE_KEYS if (v := finite_number(row.get(key))) is not None]
    return len(values) >= 2 and all(v == 0.0 for v in values)


def gpm_window_rows(store: SampleStore, start: float, end: float) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(candidates by end time, intervals fully inside [start, end])."""
    candidates, _ = store.window(start, end)
    selected = [r for r in candidates if float(r["interval_start_mono"]) >= start and interval_ms(r) > 0]
    return candidates, selected


def gpm_dropout_gpus(store: SampleStore, start: float, end: float) -> set:
    """GPUs whose SM/occupancy/tensor counters dropped out during one request.

    The dropout persists for a whole busy period, so a request is judged as a
    unit: one qualifying run anywhere in it (usually the decode) marks the GPU.
    """
    _, selected = gpm_window_rows(store, start, end)
    return {run[0].get("gpu_index", 0) for run in find_suspect_runs(selected)}


def summarize_gpm(store: SampleStore, start: float, end: float,
                  exclude_suspect: bool = True, dropout_gpus: Optional[set] = None,
                  activity: Optional[Any] = None) -> Dict[str, Any]:
    """Summarize one phase window.

    A GPU is in dropout if a qualifying suspect run lies in this window or, when
    given, in the whole request (dropout_gpus). For such a GPU, every sample with
    SM/occupancy/tensor <= 0.1 % counts as suspect: these zeros cannot be trusted,
    while clearly nonzero SM values remain valid. With exclude_suspect, suspect
    samples are left out of the SM/occupancy/tensor statistics only; valid_ms
    shrinks accordingly, so repeat aggregation weights the remaining clean time.
    gpm_suspect_max_run_ms still reports the longest qualifying run in the window.

    Complete GPM dropouts (every value exactly 0) are only recognisable with a
    second sensor: activity(gpu, start, end) returns evidence (legacy PCIe
    traffic or nvidia-smi utilization) that the GPU was busy. Such intervals are
    excluded from all GPM statistics and counted as gpm_dropout_*.
    """
    candidates, selected = gpm_window_rows(store, start, end)
    result = empty_gpm()
    dropped = [r for r in selected if activity is not None and gpm_all_zero(r)
               and activity(r.get("gpu_index", 0), float(r["interval_start_mono"]),
                            float(r["interval_end_mono"]))]
    result["gpm_dropout_samples"] = len(dropped)
    result["gpm_dropout_ms"] = sum(interval_ms(r) for r in dropped)
    all_selected = selected
    if dropped:
        dropped_ids = {id(r) for r in dropped}
        selected = [r for r in selected if id(r) not in dropped_ids]
    runs = find_suspect_runs(selected)
    flagged = {run[0].get("gpu_index", 0) for run in runs} | set(dropout_gpus or ())
    suspect = [r for r in selected if r.get("gpu_index", 0) in flagged and gpm_zero_like(r)]
    result.update({"gpm_suspect_samples": len(suspect),
                   "gpm_suspect_ms": sum(interval_ms(r) for r in suspect),
                   "gpm_suspect_phases": int(bool(suspect)),
                   "gpm_suspect_max_run_ms": plausibility_summary(runs)["gpm_suspect_max_run_ms"]})
    suspect_ids = {id(row) for row in suspect} if exclude_suspect else set()
    clean = [r for r in selected if id(r) not in suspect_ids]
    result.update({"gpm_samples": len(all_selected), "gpm_phase_ms": max(0.0, end - start) * 1000.0,
                   "gpm_covered_ms": covered_ms(selected),
                   "gpm_boundary_samples": sum(float(r["interval_start_mono"]) < start < float(r["interval_end_mono"]) for r in candidates)})
    # Include right-edge crossings in the diagnostic, but never in the metrics.
    after, _ = store.window(end, float("inf"))
    result["gpm_boundary_samples"] += sum(float(r["interval_start_mono"]) < end < float(r["interval_end_mono"]) for r in after)
    if end > start:
        result["gpm_coverage_pct"] = 100.0 * result["gpm_covered_ms"] / result["gpm_phase_ms"]
    for engine, source in GPM_ENGINES.items():
        stats = metric_stats(clean if engine in GPM_SUSPECT_ENGINES else selected, source, interval_ms)
        for out, stat in (("avg_pct", "avg"), ("p95_pct", "p95"), ("valid_samples", "samples"), ("valid_ms", "valid_ms")):
            result[f"gpm_{engine}_{out}"] = stats[stat]
    result.update(summarize_pcie(selected, "gpm_pcie_", "mib_s", lambda r: r.get("pcie_theoretical_mib_s"), interval_ms))
    return result


def aggregate_telemetry(rows: Any, defaults: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Aggregate repeat summaries: quantiles = median of per-repeat quantiles.

    Means use valid observation counts, or valid GPU-interval milliseconds for
    GPM. Never substitute zero for an unavailable value or denominator.
    """
    result = dict(defaults)
    for key, default in defaults.items():
        actual = prefix + key
        values = [row.get(actual) for row in rows]
        if key in ("gpm_suspect_phases", "gpm_suspect_ms", "gpm_dropout_ms"):
            result[key] = reduce_values(values, "sum")
            if key == "gpm_suspect_phases":
                result[key] = int(result[key])
            continue
        if key == "pstate_mode":
            strings = [v for v in values if v]
            result[key] = collections.Counter(strings).most_common(1)[0][0] if strings else ""
            continue
        if key.endswith("samples") or key.endswith("valid_ms") or key.endswith(("phase_ms", "covered_ms")):
            result[key] = reduce_values(values, "sum")
            if key.endswith("samples"):
                result[key] = int(result[key])
            continue
        if key.endswith("coverage_pct"):
            duration = sum(row.get(prefix + "gpm_phase_ms", 0.0) or 0.0 for row in rows)
            result[key] = 100.0 * sum(row.get(prefix + "gpm_covered_ms", 0.0) or 0.0 for row in rows) / duration if duration else None
            continue
        operation = "median"
        if "_peak_" in key or "_max_" in key or key in ("vram_total_mib", "win_gpu_adapter_instances", "win_shared_delta_mib"):
            operation = "max"
        elif "_min_" in key:
            operation = "min"
        weight_key = None
        if key.startswith("gpm_") and key.endswith("_avg_pct"):
            weight_key = key.replace("_avg_pct", "_valid_ms")
        elif key in ("gpm_pcie_over80_pct", "gpm_pcie_over90_pct"):
            weight_key = "gpm_pcie_ratio_valid_ms"
        elif key in ("pcie_over80_pct", "pcie_over90_pct"):
            weight_key = "pcie_ratio_samples"
        elif key == "pcie_bus_avg_pct":
            weight_key = "pcie_bus_samples"
        elif key == "power_avg_w":
            weight_key = "power_valid_samples"
        if weight_key:
            result[key] = weighted_mean([(r.get(actual), r.get(prefix + weight_key)) for r in rows])
        else:
            result[key] = reduce_values(values, operation)
        if isinstance(default, int):
            result[key] = int(result[key]) if result[key] is not None else default
    return result


def phase_fields(summary: Dict[str, Any], family: str, phase: str) -> Dict[str, Any]:
    return {family + phase + "_" + key[len(family):]: value for key, value in summary.items() if key.startswith(family)}


def summarize_vram(store: SampleStore, pcie_store: SampleStore, start: float, end: float,
                   sample_interval_ms: int, pcie_scale: float = 1.0) -> Dict[str, Any]:
    selected, age = store.window(start, end, max(0.5, 2 * sample_interval_ms / 1000.0))
    pcie_rows, _ = pcie_store.window(start, end)
    if pcie_scale != 1.0:
        # Calibrated legacy correction: raw NVML throughput / scale = GPM-equivalent
        # link traffic (see pcie-calibrate.py). Raw pcie.csv values stay unchanged.
        pcie_rows = [dict(row, **{key: (row[key] / pcie_scale if finite_number(row.get(key)) is not None
                                        else row.get(key)) for key in ("pcie_rx_mb_s", "pcie_tx_mb_s")})
                     for row in pcie_rows]
    result = empty_vram()
    result["vram_samples"] = len(selected)
    fallback = sum(not start <= float(r["t_mono"]) <= end for r in selected)
    result["vram_fallback_samples"] = fallback
    result["vram_fallback_max_age_ms"] = age if fallback else None
    if selected:
        by_gpu: Dict[str, Any] = {}
        for row in selected:
            by_gpu.setdefault(str(row["gpu_index"]), []).append(row)
        # Preserve the existing multi-GPU interpretation: sum of per-GPU peaks
        # (an upper bound, not necessarily a simultaneously observed total).
        result["vram_peak_mib"] = sum(max(r["used_mib"] for r in rows) for rows in by_gpu.values())
        result["vram_total_mib"] = sum(max(r["total_mib"] for r in rows) for rows in by_gpu.values())
        result["vram_peak_pct"] = max(r["used_pct"] for r in selected)
        result["vram_free_min_mib"] = min(r["total_mib"] - r["used_mib"] for r in selected)
    fields = {
        "gpu_util_peak_pct": ("gpu_util_pct", "max"), "mem_util_peak_pct": ("mem_util_pct", "max"),
        "power_avg_w": ("power_draw_w", "mean"), "power_peak_w": ("power_draw_w", "max"),
        "power_limit_w": ("power_limit_w", "median"), "temperature_peak_c": ("temperature_c", "max"),
    }
    for kind in ("gpu", "mem"):
        for operation, label in (("min", "min"), ("median", "median"), ("max", "max")):
            fields[f"{kind}_clock_{label}_mhz"] = (f"{kind}_clock_mhz", operation)
    for key, (source, operation) in fields.items():
        result[key] = reduce_values([r.get(source) for r in selected], operation)
    result["power_valid_samples"] = sum(finite_number(r.get("power_draw_w")) is not None for r in selected)
    states = [str(r["pstate"]) for r in selected if r.get("pstate")]
    result["pstate_mode"] = collections.Counter(states).most_common(1)[0][0] if states else ""
    result.update(summarize_pcie(pcie_rows, "pcie_", "mb_s", lambda r:
        NvidiaVramMonitor._pcie_theoretical_mb_s(r.get("pcie_link_gen") or 0.0, r.get("pcie_link_width") or 0.0)))
    for field in ("pcie_link_gen", "pcie_link_width"):
        result[field] = reduce_values([r.get(field) for r in pcie_rows])
    bus = [r.get("pcie_bus_util_pct") for r in pcie_rows
           if float(r["t_mono"]) - NvidiaVramMonitor.BUS_WINDOW_S >= start
           and finite_number(r.get("pcie_bus_util_pct")) is not None]
    result["pcie_bus_samples"] = len(bus)
    result["pcie_bus_avg_pct"] = reduce_values(bus, "mean")
    result["pcie_bus_median_pct"] = percentile(bus, 50)
    result["pcie_bus_p95_pct"] = percentile(bus, 95)
    result["pcie_bus_peak_pct"] = reduce_values(bus, "max")
    return result


def summarize_windows(store: SampleStore, start: float, end: float) -> Dict[str, Any]:
    selected, age = store.window(start, end, 1.5)
    result = empty_windows()
    result["win_gpu_samples"] = len(selected)
    fallback = sum(not start <= float(r["t_mono"]) <= end for r in selected)
    result["win_fallback_samples"] = fallback
    result["win_fallback_max_age_ms"] = age if fallback else None
    if selected:
        shared = [r["shared_mib"] for r in selected]
        result.update({"win_dedicated_peak_mib": max(r["dedicated_mib"] for r in selected),
                       "win_shared_peak_mib": max(shared), "win_shared_delta_mib": max(shared) - min(shared),
                       "win_gpu_adapter_instances": max(r["adapter_instances"] for r in selected)})
    return result


def telemetry_fields() -> Dict[str, Any]:
    values = {**empty_vram(), **empty_windows()}
    for phase in ("prefill", "decode"):
        values.update(phase_fields(empty_vram(), "pcie_", phase))
        values.update(phase_fields(empty_gpm(), "gpm_", phase))
    return values


def prefill_mode(args: Any) -> str:
    """How prefill repeats were actually measured for this run (for the result CSV).

    snapshot           incremental, every repeat restores the same prefix snapshot
    first_repeat_only  incremental fallback: snapshots unavailable, only repeat 1 is prefill
    incremental        incremental with --repeat 1 (no snapshots needed)
    cold               slot erased and prompt cache disabled before every repeat
    """
    if args.cache_mode == "cold":
        return "cold"
    if getattr(args, "repeat", 1) <= 1:
        return "incremental"
    return "snapshot" if getattr(args, "prefill_repeat_enabled", True) else "first_repeat_only"


OUTPUT_EXCERPT_CHARS = 200
OUTPUT_LOOP_WINDOW_CHARS = 1000
OUTPUT_LOOP_WARN_PCT = 50.0
OUTPUT_LOOP_MIN_CHARS = 20


def output_hash(text: str) -> str:
    """Short, stable fingerprint of the generated text (compare across runs/variants)."""
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def output_excerpt(text: str, limit: int = OUTPUT_EXCERPT_CHARS) -> str:
    """Single-line excerpt: control characters escaped so CSV rows stay one line."""
    excerpt = text[:limit]
    return (excerpt.replace("\\", "\\\\").replace("\r", "\\r")
            .replace("\n", "\\n").replace("\t", "\\t"))


def output_loop_pct(text: str, window: int = OUTPUT_LOOP_WINDOW_CHARS) -> Optional[float]:
    """Share of the output tail covered by >=3 consecutive copies of one unit.

    Only the last `window` characters are analysed. A greedy model that falls
    into a loop (e.g. with --ignore-eos) ends in a periodic tail; normal text
    or code scores low. Heuristic only: it flags degenerate output, it does not
    judge quality. Returns None for empty output.
    """
    if not text:
        return None
    s = text[-window:]
    n = len(s)
    best = 0
    for period in range(1, n // 3 + 1):
        index = n - period - 1
        while index >= 0 and s[index] == s[index + period]:
            index -= 1
        covered = n - 1 - index  # period + matching run
        if covered >= 3 * period:
            best = max(best, covered)
    return 100.0 * best / n


def analyze_output(content: Any) -> Dict[str, Any]:
    text = content if isinstance(content, str) else ""
    return {
        "output_sha256": output_hash(text) if isinstance(content, str) else None,
        "output_chars": len(text) if isinstance(content, str) else None,
        "output_loop_pct": rounded(output_loop_pct(text), 1),
        "output_excerpt": output_excerpt(text) if isinstance(content, str) else None,
    }


def decode_step_stats(predicted_n: int, predicted_ms: float, draft_acc: int) -> Dict[str, Any]:
    """Split decode speed into tokens per target step and cost per step.

    With speculative/MTP decoding every verification step yields its accepted
    draft tokens plus one sampled token, so steps ~= predicted_n - draft_acc
    (without drafting: one token per step). Tokens per step depends on how
    predictable the generated text is; ms per step on the model and context,
    which makes it the content-independent measure for context effects.
    """
    steps = predicted_n - max(0, draft_acc)
    if predicted_n <= 0 or predicted_ms <= 0 or steps <= 0:
        return {"decode_steps": None, "tokens_per_step": None, "ms_per_step": None}
    return {"decode_steps": steps, "tokens_per_step": predicted_n / steps, "ms_per_step": predicted_ms / steps}


def summarize_outputs(samples: Any) -> Dict[str, Any]:
    """Per-point output fingerprint: first repeat's hash plus consistency checks."""
    hashes = [s.get("output_sha256") for s in samples if s.get("output_sha256")]
    loops = [v for s in samples if (v := finite_number(s.get("output_loop_pct"))) is not None]
    return {
        "output_sha256": hashes[0] if hashes else None,
        "output_variants": len(set(hashes)) if hashes else None,
        "output_loop_pct_max": max(loops) if loops else None,
    }


def aggregate_point(samples: Any, ctx: int, total_ctx: int, target_chars: int, args: Any) -> Dict[str, Any]:
    if not samples:
        raise ValueError("cannot aggregate an empty benchmark point")
    valid = [s for s in samples if s["status"] == "OK"
             and (finite_number(s["decode_tps"]) or 0) > 0]
    decode = [s["decode_tps"] for s in valid]
    per_step = [v for s in valid if (v := finite_number(s.get("tokens_per_step"))) is not None]
    step_ms = [v for s in valid if (v := finite_number(s.get("ms_per_step"))) is not None]
    prefill = (samples[:1] if args.cache_mode == "incremental"
               and not getattr(args, "prefill_repeat_enabled", True) else samples)
    prefill_repeats = len(prefill)
    # An early EOS invalidates decode, but not an otherwise complete prefill.
    prefill = [s for s in prefill if not s.get("truncated", False)
               and all((finite_number(s.get(key)) or 0) > 0
                       for key in ("prompt_n", "prompt_ms", "prefill_tps"))]
    draft = sum(s["draft_n"] for s in samples)
    accepted = sum(s["draft_acc"] for s in samples)
    # Missing measurements stay empty (None) instead of 0: a zero rate or a 0 %
    # draft acceptance would be indistinguishable from a real measured value.
    row = {
        "target_ctx": ctx, "total_ctx": total_ctx, "target_chars": target_chars, "cache_mode": args.cache_mode,
        "prefill_mode": prefill_mode(args),
        "cache_n": int(statistics.median(s["cache_n"] for s in prefill)) if prefill else None,
        "prompt_n": int(statistics.median(s["prompt_n"] for s in prefill)) if prefill else None,
        "prompt_ms": round(statistics.median(s["prompt_ms"] for s in prefill), 3) if prefill else None,
        "prefill_tps": round(statistics.median(s["prefill_tps"] for s in prefill), 2) if prefill else None,
        "prefill_tps_min": round(min(s["prefill_tps"] for s in prefill), 2) if prefill else None,
        "prefill_tps_max": round(max(s["prefill_tps"] for s in prefill), 2) if prefill else None,
        "prefill_valid_repeats": len(prefill), "prefill_repeats": prefill_repeats,
        "decode_tps_median": round(statistics.median(decode), 3) if decode else None,
        "decode_tps_min": round(min(decode), 3) if decode else None,
        "decode_tps_max": round(max(decode), 3) if decode else None,
        "valid_repeats": len(valid), "repeat": len(samples), "draft_n": draft, "draft_acc": accepted,
        "draft_acc_pct": round(100.0 * accepted / draft, 2) if draft else None,
        "tokens_per_step_median": round(statistics.median(per_step), 3) if per_step else None,
        "ms_per_step_median": round(statistics.median(step_ms), 3) if step_ms else None,
        "ms_per_step_min": round(min(step_ms), 3) if step_ms else None,
        "ms_per_step_max": round(max(step_ms), 3) if step_ms else None,
        **summarize_outputs(samples),
        "wall_s_median": round(statistics.median(s["wall_s"] for s in samples), 3),
        "status": "OK" if len(valid) == len(samples) else f"{len(valid)}/{len(samples)} OK",
        "sample_statuses": ";".join(s["status"] for s in samples),
        "telemetry_quantiles": "median_of_repeat_quantiles",
        "gpm_average_weighting": "valid_gpu_interval_ms",
        "gpm_suspect_handling": ("kept" if getattr(args, "gpm_suspect", "exclude") == "keep"
                                 else "excluded_from_sm_occupancy_tensor"),
        "gpm_lag_ms": getattr(args, "gpm_lag_ms", 0.0),
        "gpm_restarts": sum(1 for s in samples if s.get("gpm_restart")),
        "pcie_legacy_scale": getattr(args, "pcie_legacy_scale", 1.0),
        "phase_method": ("/".join(sorted({s.get("phase_method") or "response_end_backprojection" for s in samples}))),
        "phase_anchor_offset_ms_median": rounded(reduce_values(s.get("phase_anchor_offset_ms") for s in samples), 3),
        "phase_wall_residual_ms_max": rounded(max(s.get("phase_wall_residual_ms", 0.0) for s in samples), 3),
        "phase_timing_excess_ms_max": rounded(max(s.get("phase_timing_excess_ms", 0.0) for s in samples), 3),
    }
    telemetry = aggregate_telemetry(samples, {**empty_vram(), **empty_windows()})
    for phase, repeats in (("prefill", prefill), ("decode", valid)):
        for family, defaults in (("pcie_", empty_vram()), ("gpm_", empty_gpm())):
            # Normalize phase keys before applying the same explicit aggregation rules.
            normalized = [{family + key[len(family + phase + '_'):]: value for key, value in sample.items()
                           if key.startswith(family + phase + '_')} for sample in repeats]
            summary = aggregate_telemetry(normalized, {k: v for k, v in defaults.items() if k.startswith(family)})
            telemetry.update(phase_fields(summary, family, phase))
    for key, value in telemetry.items():
        if isinstance(value, float):
            value = rounded(value, 3 if key.endswith("_ms") else 2)
        row[key] = value
    return row


def format_completion_error(error: Any, ctx: int) -> str:
    details = [f"{name}={value}" for name, value in (("type", error.error_type),
               ("prompt", error.n_prompt_tokens), ("n_ctx", error.n_ctx)) if value is not None and value != ""]
    suffix = f" ({', '.join(details)})" if details else ""
    return f"\nHTTP {error.status_code} from llama-server at target={ctx}: {error.message}{suffix}"


def format_exit_code(code: Optional[int]) -> str:
    """Show Windows NTSTATUS-style codes in hex too (e.g. 3221225477 = 0xC0000005)."""
    if code is None:
        return "unknown"
    if code < 0 or code > 255:
        return f"{code} (0x{code & 0xFFFFFFFF:08X})"
    return str(code)


def managed_server_exit_code(managed_server: Any, wait_s: float = 0.0) -> Optional[int]:
    """Exit code of a managed server that has died, else None (running or not managed)."""
    proc = getattr(managed_server, "proc", None)
    if proc is None:
        return None
    code = proc.poll()
    if code is None and wait_s > 0:
        # A crash often closes the socket slightly before the process has exited.
        try:
            code = proc.wait(timeout=wait_s)
        except subprocess.TimeoutExpired:
            code = None
    return code


class GuardAbortError(RuntimeError):
    """A run guard stopped the sweep because further points would not be meaningful."""
    stop_reason = "guard_abort"


class SysmemFallbackError(GuardAbortError):
    """Prefill PCIe traffic shows that GPU memory overflows into shared system memory."""
    stop_reason = "sysmem_fallback"


class PrefillFloorError(GuardAbortError):
    """Prefill fell below --abort-below-pct of the first point."""
    stop_reason = "prefill_floor"


SYSMEM_GUARD_DEFAULT_MB_S = 1000.0


def sysmem_fallback_signal(sample: Dict[str, Any]) -> Optional[float]:
    """Highest prefill PCIe receive median (GPM or legacy sampler); None without PCIe telemetry.

    With all layers on the GPU, prefill barely touches PCIe (~20 MB/s). When the
    Windows driver spills buffers into shared system memory, every prefill step
    reads them over PCIe (observed: 8-11 GB/s, prefill 5-30x slower).
    """
    values = [finite_number(sample.get(key)) for key in
              ("gpm_prefill_pcie_rx_median_mib_s", "pcie_prefill_rx_median_mb_s")]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def is_expected_point_error(error: BaseException) -> bool:
    """Runtime failures get a one-line message; anything else also a traceback."""
    return isinstance(error, (CompletionRequestError, requests.RequestException, PromptBuildError,
                              RuntimeError, OSError, ValueError))


def report_point_failure(error: BaseException, ctx: int, managed_server: Any,
                         log_path: Optional[str]) -> Dict[str, Any]:
    """Print a concise diagnosis for a failed context point; return metadata fields.

    Must be called from inside the `except` block so unexpected errors (programming
    bugs) can still show their full traceback.
    """
    fields: Dict[str, Any] = {"failed_target_ctx": ctx,
                              "error": f"{type(error).__name__}: {error}"[:2000]}
    if isinstance(error, CompletionRequestError):
        print(format_completion_error(error, ctx), file=sys.stderr)
    else:
        print(f"\nERROR at target={ctx}: {type(error).__name__}: {error}", file=sys.stderr)
        if not is_expected_point_error(error):
            traceback.print_exc()
    connection_lost = isinstance(error, (requests.ConnectionError, requests.Timeout))
    code = managed_server_exit_code(managed_server, wait_s=3.0 if connection_lost else 0.5)
    if code is not None:
        fields["server_exit_code"] = code
        hint = (f"; see {os.path.abspath(log_path)}" if log_path and log_path != "-" else "")
        print(f"llama-server CRASHED/EXITED with code {format_exit_code(code)} during target={ctx}{hint}",
              file=sys.stderr)
    elif connection_lost and managed_server is not None:
        print("llama-server process is still running; the connection failed or timed out.",
              file=sys.stderr)
    return fields


def display_number(value: Any, digits: int = 0) -> str:
    return f"{value:.{digits}f}" if finite_number(value) is not None else "n/a"


def display_group(values: Any) -> str:
    values = list(values)
    if all(finite_number(value) is None for value in values):
        return "n/a"
    return "/".join(display_number(value) for value in values)


def print_live_row(row: Dict[str, Any], drafting_on: bool) -> None:
    cells = []
    for phase in ("prefill", "decode"):
        g = "gpm_" + phase + "_"
        pcie = display_group(row.get(g + f"pcie_{direction}_p95_mib_s") for direction in ("rx", "tx"))
        gpu = display_group(row.get(g + engine + "_avg_pct") for engine in ("sm", "occupancy", "tensor", "dram"))
        if row.get(g + "suspect_phases", 0):
            gpu += "*"
        if row.get(g + "dropout_samples", 0):
            gpu += "!"
        cells.append(f"{pcie:>14} | {display_number(row.get(g + 'pcie_over90_pct'), 1):>6} | "
                     f"{display_number(row.get('pcie_' + phase + '_bus_avg_pct')):>6} | {gpu:>13}")
    draft_pct = row.get('draft_acc_pct')
    draft = (f" | {display_number(draft_pct, 1) + ('%' if finite_number(draft_pct) is not None else ''):>6}"
           f" | {display_number(row.get('ms_per_step_median'), 1):>6}"
           if drafting_on else "")
    print(f"{row['total_ctx']:>8} | {display_number(row.get('prompt_n')):>7} | "
          f"{display_number(row.get('prefill_tps'), 1):>8} | "
          f"{display_number(row.get('decode_tps_median'), 2):>7}{draft} | {display_number(row.get('vram_free_min_mib')):>6} | "
          f"{display_number(row.get('power_avg_w')):>5} | {' | '.join(cells)} | {row['status']:>9}", flush=True)


# Keep existing column order and add explicit validity/coverage metadata.
RESULT_CSV_FIELDS = tuple(dict.fromkeys((*RESULT_CSV_FIELDS, *telemetry_fields(),
    "prefill_valid_repeats", "prefill_repeats",
    "telemetry_quantiles", "gpm_average_weighting", "phase_method",
    "phase_wall_residual_ms_max", "phase_timing_excess_ms_max",
    "prefill_mode", "output_sha256", "output_variants", "output_loop_pct_max",
    "gpm_suspect_handling", "gpm_lag_ms", "pcie_legacy_scale", "gpm_restarts", "phase_anchor_offset_ms_median",
    "prefill_tps_min", "prefill_tps_max")))
PCIE_CSV_FIELDS = (*PCIE_CSV_FIELDS, "t_mono", "pcie_query_ms", "pcie_poll_interval_ms")
SAMPLE_CSV_FIELDS = tuple(dict.fromkeys((
    "target_ctx", "total_ctx", "target_chars", "cache_mode", "prefill_mode", "repeat",
    "cache_n", "prompt_n", "prompt_ms", "prefill_tps", "decode_tps",
    "predicted_n", "predicted_ms", "draft_n", "draft_acc", "decode_steps", "tokens_per_step", "ms_per_step",
    "wall_s",
    "truncated", "status", "stop_type", "gpm_restart", "validation_error",
    "output_sha256", "output_chars", "output_loop_pct", "output_excerpt", "phase_method",
    "phase_wall_residual_ms", "phase_timing_excess_ms", "phase_anchor_offset_ms", *telemetry_fields(),
)))


def validate_distinct_paths(paths: Any) -> None:
    """Reject aliases before opening any input, log or CSV for writing."""
    seen = []
    for path in paths:
        if path is None:
            continue
        resolved = os.path.realpath(path)
        if any(os.path.normcase(resolved) == os.path.normcase(previous) or
               (os.path.exists(resolved) and os.path.exists(previous)
                and os.path.samefile(resolved, previous)) for previous in seen):
            raise ValueError(f"Path collides with an input, log or another output: {path}")
        seen.append(resolved)


def allocate_csv_path(directory: str) -> str:
    """Reserve a timestamped result name without reusing an existing trace stem."""
    os.makedirs(directory, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = 0
    while True:
        name = f"ctx-cliff-{stamp}" + (f"-{suffix}" if suffix else "")
        stem = os.path.join(directory, name)
        suffix += 1
        if any(os.path.exists(stem + ext) for ext in
               (".csv", ".samples.csv", ".vram.csv", ".pcie.csv", ".gpm.csv", ".win-gpu.csv",
                ".meta.json")):
            continue
        try:
            # Exclusive creation also protects concurrent benchmark launches.
            with open(stem + ".csv", "x", encoding="utf-8"):
                pass
            return stem + ".csv"
        except FileExistsError:
            continue


def write_csv_atomic(path: str, fields: Any, rows: Any) -> None:
    """Preserve the live trace until its phase-enriched replacement is complete."""
    fd, temporary = tempfile.mkstemp(prefix=".ctx-cliff-", suffix=".csv.tmp",
                                     dir=os.path.dirname(os.path.abspath(path)))
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fields})
            f.flush()
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


METADATA_ENV_PREFIXES = ("GGML_", "LLAMA_", "CUDA_", "NVIDIA_", "HIP_", "ROCR_", "HSA_",
                         "VK_", "OMP_", "MKL_", "OPENBLAS_", "KMP_")
SENSITIVE_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSW|CREDENTIAL", re.IGNORECASE)
REDACTED = "<redacted>"


def redact_command(command: str) -> str:
    """Hide llama-server --api-key values; everything else stays reproducible."""
    return re.sub(r'(--api-key(?:=|\s+))("[^"]*"|\'[^\']*\'|\S+)', r"\1" + REDACTED, command)


def relevant_environment(environ: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Backend-related variables (e.g. GGML_*) that change server behaviour."""
    environ = os.environ if environ is None else environ
    return {key: (REDACTED if SENSITIVE_NAME.search(key) else environ[key])
            for key in sorted(environ) if key.upper().startswith(METADATA_ENV_PREFIXES)}


def file_fingerprint(path: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {"path": os.path.abspath(path)}
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                digest.update(block)
        stat = os.stat(path)
        info.update(bytes=stat.st_size, sha256=digest.hexdigest(),
                    modified=dt.datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"))
    except OSError as error:
        info["error"] = str(error)
    return info


def sanitize_props(value: Any, depth: int = 0) -> Any:
    """Keep /props compact: drop chat templates, shorten long strings and lists."""
    if depth > 6:
        return None
    if isinstance(value, dict):
        return {str(k): sanitize_props(v, depth + 1) for k, v in value.items()
                if not str(k).startswith("chat_template") and not SENSITIVE_NAME.search(str(k))}
    if isinstance(value, list):
        return [sanitize_props(v, depth + 1) for v in value[:64]]
    if isinstance(value, str) and len(value) > 1000:
        return value[:1000] + "..."
    return value


def fetch_server_props(base: str, http: Any) -> Optional[Dict[str, Any]]:
    """Best-effort /props snapshot (model path, build info, generation defaults)."""
    try:
        response = http.get(f"{base}/props", timeout=10)
        if not response.ok:
            return None
        data = response.json()
    except Exception:
        return None
    return sanitize_props(data) if isinstance(data, dict) else None


def collect_gpu_info() -> Optional[Dict[str, Any]]:
    """Static GPU/driver description for the metadata file; never raises."""
    def text(value: Any) -> str:
        return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)

    def attempt(function: Any, *call_args: Any) -> Any:
        try:
            return function(*call_args)
        except Exception:
            return None

    try:
        import pynvml as nv
        nv.nvmlInit()
    except Exception:
        nv = None
    if nv is not None:
        try:
            info: Dict[str, Any] = {"source": "nvml"}
            for key, name in (("driver_version", "nvmlSystemGetDriverVersion"),
                              ("nvml_version", "nvmlSystemGetNVMLVersion")):
                value = attempt(getattr(nv, name, lambda: None))
                if value is not None:
                    info[key] = text(value)
            cuda = attempt(getattr(nv, "nvmlSystemGetCudaDriverVersion", lambda: None))
            if cuda:
                info["cuda_driver_version"] = f"{int(cuda) // 1000}.{int(cuda) % 1000 // 10}"
            gpus = []
            for index in range(int(nv.nvmlDeviceGetCount())):
                handle = nv.nvmlDeviceGetHandleByIndex(index)
                entry: Dict[str, Any] = {"index": index}
                for key, name in (("name", "nvmlDeviceGetName"), ("uuid", "nvmlDeviceGetUUID"),
                                  ("vbios", "nvmlDeviceGetVbiosVersion")):
                    value = attempt(getattr(nv, name, lambda h: None), handle)
                    if value is not None:
                        entry[key] = text(value)
                memory = attempt(getattr(nv, "nvmlDeviceGetMemoryInfo", lambda h: None), handle)
                if memory is not None:
                    entry["memory_total_mib"] = round(int(memory.total) / 1048576.0)
                limit = attempt(getattr(nv, "nvmlDeviceGetPowerManagementLimit", lambda h: None), handle)
                if limit is not None:
                    entry["power_limit_w"] = round(int(limit) / 1000.0, 1)
                for key, name in (("pcie_link_gen_max", "nvmlDeviceGetMaxPcieLinkGeneration"),
                                  ("pcie_link_width_max", "nvmlDeviceGetMaxPcieLinkWidth")):
                    value = attempt(getattr(nv, name, lambda h: None), handle)
                    if value is not None:
                        entry[key] = int(value)
                gpus.append(entry)
            info["gpus"] = gpus
            return info
        except Exception:
            pass
        finally:
            attempt(nv.nvmlShutdown)
    if shutil.which("nvidia-smi") is None:
        return None
    fields = ("index", "name", "uuid", "driver_version", "vbios_version", "memory.total",
              "power.limit", "pcie.link.gen.max", "pcie.link.width.max")
    try:
        result = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(fields)}",
                                 "--format=csv,noheader,nounits"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
    except Exception:
        return None
    if result.returncode != 0:
        return None
    gpus = [dict(zip(fields, (part.strip() for part in line.split(","))))
            for line in result.stdout.splitlines() if line.strip()]
    return {"source": "nvidia-smi", "gpus": gpus}


class RunMetadata:
    """`<stem>.meta.json`: everything needed to tell runs apart and reproduce them.

    Rewritten atomically after each stage, so an interrupted run still leaves the
    last known state. A disabled instance (path=None) accepts updates silently.
    Metadata failures only warn; they never abort a measurement.
    """

    def __init__(self, path: Optional[str], args: Any) -> None:
        self.path = path
        self.warned = False
        self.data: Dict[str, Any] = {}
        if path is None:
            return
        arguments = {key: (redact_command(value) if isinstance(value, str) else value)
                     for key, value in sorted(vars(args).items())}
        self.data = {
            "schema_version": 1,
            "status": "running",
            "started_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "finished_at": None,
            "argv": [redact_command(item) for item in sys.argv],
            "arguments": arguments,
            "script": file_fingerprint(os.path.abspath(__file__)),
            "input_file": file_fingerprint(args.file) if getattr(args, "file", None) else None,
            "host": {"hostname": platform.node(), "platform": platform.platform(),
                     "machine": platform.machine(), "python": sys.version.split()[0],
                     "requests": getattr(requests, "__version__", None)},
            "environment": relevant_environment(),
        }
        self.write()

    def update(self, **fields: Any) -> None:
        if self.path is None:
            return
        self.data.update(fields)
        self.write()

    def write(self) -> None:
        if self.path is None:
            return
        temporary = None
        try:
            directory = os.path.dirname(os.path.abspath(self.path))
            fd, temporary = tempfile.mkstemp(prefix=".ctx-cliff-", suffix=".json.tmp", dir=directory)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False, default=str)
                f.write("\n")
            os.replace(temporary, self.path)
            temporary = None
        except Exception as error:
            if not self.warned:
                print(f"WARNING: could not write run metadata {self.path}: {error}", file=sys.stderr)
                self.warned = True
        finally:
            if temporary is not None and os.path.exists(temporary):
                try:
                    os.unlink(temporary)
                except OSError:
                    pass

    def exit_hook(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        """ExitStack hook: record how the run ended, including unexpected exits."""
        if self.path is None:
            return False
        fields: Dict[str, Any] = {
            "finished_at": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
        if exc_type is None:
            fields.setdefault("exit_code", self.data.get("exit_code", 0))
            if self.data.get("status") == "running":
                fields["status"] = "completed"
        elif issubclass(exc_type, SystemExit):
            code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
            fields["exit_code"] = code
            if self.data.get("status") == "running":
                fields["status"] = "completed" if code == 0 else "failed"
        elif issubclass(exc_type, KeyboardInterrupt):
            fields.update(status="interrupted", exit_code=130)
        else:
            fields.update(status="failed", exit_code=1,
                          error=f"{exc_type.__name__}: {exc}")
        self.update(**fields)
        if not self.warned:
            print(f"saved: {self.path} (run metadata)", file=sys.stderr)
        return False


class CsvRecording:
    """Flush new raw rows independently of HTTP requests; finalize after monitors stop.

    PCIe/GPM phase labels require response timings and remain provisional in the
    live files. Cleanup replaces those files atomically using the existing phase
    exporters. A killed process still leaves its already-flushed raw telemetry.
    flush() protects against process termination, not power loss or disk failure.
    """

    def __init__(self, args: Any) -> None:
        self.files: Dict[str, Any] = {}
        self.writers: Dict[str, Any] = {}
        self.sources: Dict[str, Any] = {}
        self.active_monitors: set[int] = set()
        self.paths: Dict[str, str] = {}
        self.counts: Dict[str, int] = {}
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.error: Optional[Exception] = None
        self.closed = False

        def trace_path(explicit: Optional[str], suffix: str) -> Optional[str]:
            if explicit is not None:
                return explicit
            if not args.csv:
                return None
            stem = args.csv[:-4] if args.csv.lower().endswith(".csv") else args.csv
            return stem + suffix

        nvidia = args.vram_log != "off"
        gpm = args.gpm_log == "on" or (args.gpm_log == "auto" and nvidia)
        windows = args.win_gpu_mem != "off" and sys.platform == "win32"
        specs = [
            ("results", args.csv, RESULT_CSV_FIELDS),
            ("samples", trace_path(None, ".samples.csv"), SAMPLE_CSV_FIELDS),
            ("vram", trace_path(args.vram_csv, ".vram.csv") if nvidia else None, VRAM_CSV_FIELDS),
            ("pcie", trace_path(args.pcie_csv, ".pcie.csv") if nvidia else None, PCIE_CSV_FIELDS),
            ("gpm", trace_path(args.gpm_csv, ".gpm.csv") if gpm else None, GPM_CSV_FIELDS),
            ("windows", trace_path(args.win_gpu_mem_csv, ".win-gpu.csv") if windows else None, WINDOWS_CSV_FIELDS),
        ]
        # Run metadata accompanies the result CSV (same stem, .meta.json).
        self.meta_path: Optional[str] = trace_path(None, ".meta.json")
        # Early creation must not truncate the benchmark input or another output.
        protected = [args.file]
        if args.server_log and args.server_log != "-":
            protected.append(args.server_log)
        validate_distinct_paths([*protected, *(path for _, path, _ in specs), self.meta_path])
        self.specs = specs

    def __enter__(self) -> "CsvRecording":
        try:
            for name, path, fields in self.specs:
                if path is None:
                    continue
                f = open(path, "w", newline="", encoding="utf-8")
                self.files[name] = f
                self.paths[name] = path
                self.counts[name] = 0
                writer = csv.DictWriter(f, fieldnames=fields)
                self.writers[name] = writer
                writer.writeheader()
                f.flush()
                print(f"CSV recording: {path}", file=sys.stderr)
            if any(name not in {"results", "samples"} for name in self.files):
                self.thread = threading.Thread(target=self._run, name="csv-recording", daemon=True)
                self.thread.start()
            return self
        except BaseException:
            for f in self.files.values():
                f.close()
            raise

    def attach(self, name: str, monitor: Any, attribute: str, finalize: Any = None) -> None:
        with self.lock:
            store = getattr(monitor, attribute)
            if isinstance(store, SampleStore):
                store.archive_enabled = name in self.files
            self.sources[name] = (monitor, attribute, finalize)

    def activate(self, monitor: Any) -> None:
        """Check runtime errors only for monitors that started successfully."""
        with self.lock:
            self.active_monitors.add(id(monitor))

    def _drain(self) -> None:
        with self.lock:
            sources = list(self.sources.items())
        for name, (monitor, attribute, _) in sources:
            if name not in self.files:
                continue
            while True:
                with monitor.lock:
                    store = getattr(monitor, attribute)
                    rows = store.read_since(self.counts[name]) if isinstance(store, SampleStore) else list(store[self.counts[name]:])
                if not rows:
                    break
                writer = self.writers[name]
                writer.writerows({key: row.get(key) for key in writer.fieldnames} for row in rows)
                self.files[name].flush()
                self.counts[name] += len(rows)

    def _run(self) -> None:
        while not self.done.wait(0.25):
            try:
                self._drain()
            except Exception as e:
                self.error = e
                print(f"ERROR: CSV recording failed: {e}", file=sys.stderr)
                return

    def check(self) -> None:
        if self.error is not None:
            raise RuntimeError(f"CSV recording failed: {self.error}") from self.error
        with self.lock:
            sources = list(self.sources.values())
            active_monitors = set(self.active_monitors)
        for monitor, attribute, _ in sources:
            store = getattr(monitor, attribute)
            if isinstance(store, SampleStore) and store.error is not None:
                raise RuntimeError(f"telemetry archive failed: {store.error}") from store.error
            if id(monitor) in active_monitors:
                if getattr(monitor, "error", None):
                    raise RuntimeError(f"telemetry monitor failed: {monitor.error}")
                # A missing optional PCIe backend also sets pcie_error at startup.
                # Only errors from an active NVML backend invalidate the trace.
                if getattr(monitor, "pcie_source", None) == "nvml" and getattr(monitor, "pcie_error", None):
                    raise RuntimeError(f"PCIe telemetry failed: {monitor.pcie_error}")

    def write_result(self, row: Dict[str, Any]) -> None:
        self.check()
        if "results" in self.writers:
            missing = set(RESULT_CSV_FIELDS) - row.keys()
            if missing:
                raise ValueError("result row is missing CSV fields: " + ", ".join(sorted(missing)))
            self.writers["results"].writerow(row)
            self.files["results"].flush()
            self.counts["results"] += 1

    def write_sample(self, row: Dict[str, Any]) -> None:
        self.check()
        if "samples" in self.writers:
            writer = self.writers["samples"]
            writer.writerow({key: row.get(key) for key in writer.fieldnames})
            self.files["samples"].flush()
            self.counts["samples"] += 1

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.closed:
            return
        self.closed = True
        self.done.set()
        if self.thread is not None:
            self.thread.join()
        failures = [self.error] if self.error is not None else []
        try:
            if self.error is None:
                self._drain()
        except Exception as e:
            failures.append(e)
        finally:
            for f in self.files.values():
                try:
                    f.close()
                except Exception as e:
                    failures.append(e)
        for name, (_, _, finalize) in self.sources.items():
            if finalize is not None and name in self.paths:
                try:
                    self.counts[name] = finalize(self.paths[name])
                except Exception as e:
                    failures.append(e)
                    print(f"WARNING: phase export failed; live CSV retained at {self.paths[name]}: {e}", file=sys.stderr)
        for monitor, attribute, _ in self.sources.values():
            store = getattr(monitor, attribute)
            if isinstance(store, SampleStore):
                try:
                    store.close()
                except Exception as e:
                    failures.append(e)
        if self.paths:
            # Paths were already announced by "CSV recording: ..." at startup;
            # only the final row counts are new information here.
            counts = ", ".join(f"{name}={self.counts[name]}" for name in self.paths)
            print(f"CSV rows written: {counts}", file=sys.stderr)
        if failures:
            if exc_type is None:
                raise RuntimeError(f"CSV recording/finalization failed: {failures[0]}") from failures[0]
            print(f"WARNING: CSV cleanup also failed: {failures[0]}", file=sys.stderr)


def server_is_ready(base: str, timeout: float = 2.0, http: Optional[GetClient] = None) -> bool:
    """Return True only when llama-server's health endpoint is ready."""
    try:
        return (http if http is not None else requests).get(f'{base}/health', timeout=timeout).ok
    except requests.RequestException:
        return False


JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9


def bind_to_kill_job(process_handle: int, kernel32: Any = None) -> Optional[int]:
    """Windows: put a process into a job that is killed when this script exits.

    The job handle is only held by this Python process. If Python ends in any
    way (crash, closed console, Task Manager), Windows closes the handle and
    terminates the server, so no orphaned llama-server keeps VRAM and the port
    busy for the next run. Returns the job handle, or None when not possible.
    """
    import ctypes
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class BASIC_LIMIT(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class EXTENDED_LIMIT(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC_LIMIT), ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    if kernel32 is None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                                     wintypes.DWORD]
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return None
    info = EXTENDED_LIMIT()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if (not kernel32.SetInformationJobObject(job, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
                                             ctypes.byref(info), ctypes.sizeof(info))
            or not kernel32.AssignProcessToJobObject(job, process_handle)):
        kernel32.CloseHandle(job)
        return None
    return job


class ManagedLlamaServer:
    """Start a local llama-server and own its lifetime."""

    def __init__(
        self,
        command: str,
        base: str,
        startup_timeout: float,
        log_path: Optional[str],
        http: Optional[GetClient] = None,
        kill_with_script: bool = False,
    ) -> None:
        self.http = http
        # Windows: terminate the server even if this script dies abruptly.
        self.kill_with_script = kill_with_script
        self.job: Optional[int] = None
        self.command = command
        self.base = base
        self.startup_timeout = startup_timeout
        self.log_path = log_path
        self.proc: Optional[subprocess.Popen[Any]] = None
        self.log_file: Optional[Any] = None

    def _display_command(self) -> str:
        return self.command

    def _popen_command(self) -> Any:
        # Windows CreateProcess accepts a complete command line. POSIX Popen
        # requires an argv list, so preserve quoted groups with shlex there.
        if os.name != "nt":
            return shlex.split(self.command)
        return self.command

    def _log_hint(self) -> str:
        if self.log_path and self.log_path != "-":
            return f"; see {self.log_path}"
        return ""

    def start(self) -> None:
        stdout: Any = None
        stderr: Any = None
        if self.log_path and self.log_path != "-":
            self.log_file = open(self.log_path, "w", encoding="utf-8", buffering=1)
            stdout = self.log_file
            stderr = subprocess.STDOUT

        print(f"starting llama-server: {self._display_command()}", file=sys.stderr)
        if self.log_path and self.log_path != "-":
            print(f"llama-server output: {os.path.abspath(self.log_path)}", file=sys.stderr)
        try:
            self.proc = subprocess.Popen(
                self._popen_command(),
                stdout=stdout,
                stderr=stderr,
            )
        except Exception:
            if self.log_file is not None:
                self.log_file.close()
                self.log_file = None
            raise
        if self.kill_with_script and os.name == "nt":
            try:
                self.job = bind_to_kill_job(int(self.proc._handle))  # type: ignore[attr-defined]
            except Exception:
                self.job = None
            if self.job is None:
                print("WARNING: could not bind llama-server to this script; if the script is killed, "
                      "stop llama-server manually before the next run", file=sys.stderr)

        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            assert self.proc is not None
            return_code = self.proc.poll()
            if return_code is not None:
                self.stop()
                raise RuntimeError(
                    f"llama-server exited during startup with code {return_code}"
                    f"{self._log_hint()}"
                )
            if server_is_ready(self.base, http=self.http):
                print(f"llama-server ready at {self.base}", file=sys.stderr)
                return
            time.sleep(0.25)

        self.stop()
        raise TimeoutError(
            f"llama-server did not become ready within {self.startup_timeout:.1f}s"
            f"{self._log_hint()}"
        )

    def stop(self) -> None:
        proc = self.proc
        self.proc = None
        if proc is not None and proc.poll() is None:
            print("stopping managed llama-server", file=sys.stderr)
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            except Exception as e:
                print(f"WARNING: failed to stop llama-server cleanly: {e}", file=sys.stderr)
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None
        if self.job is not None:
            try:
                import ctypes
                ctypes.WinDLL("kernel32").CloseHandle(self.job)
            except Exception:
                pass
            self.job = None


def tokenize(base: str, text: str, add_bos: bool = True, http: Optional[PostClient] = None) -> List[int]:
    payloads = [{'content': text, 'add_special': add_bos, 'parse_special': True}, {'content': text, 'add_bos': add_bos, 'special': True}]
    last_error: Optional[Exception] = None
    for payload in payloads:
        try:
            r = (http if http is not None else requests).post(f'{base}/tokenize', json=payload, timeout=120)
            r.raise_for_status()
            return r.json()['tokens']
        except (requests.Timeout, requests.ConnectionError):
            # The alternative payload only helps with an older API, not with an
            # unreachable or hanging server: do not wait a second time.
            raise
        except (requests.RequestException, KeyError, TypeError, ValueError) as e:
            last_error = e
    assert last_error is not None
    raise last_error


def detokenize(base: str, tokens: List[int], http: Optional[PostClient] = None) -> str:
    r = (http if http is not None else requests).post(f'{base}/detokenize', json={'tokens': tokens}, timeout=120)
    r.raise_for_status()
    return r.json()['content']


# llama-server /completion sampler fields with a dedicated CLI option (--top-p and --top_p).
SAMPLER_OPTIONS: Dict[str, Any] = {
    "temperature": float, "top_p": float, "top_k": int, "min_p": float, "typical_p": float,
    "repeat_penalty": float, "repeat_last_n": int, "presence_penalty": float, "frequency_penalty": float,
}
SAMPLER_ALIASES = {"repeat_penalty": ("--repetition-penalty", "--repetition_penalty")}
# Meaningless or contradictory next to --deterministic (temperature=0, top_k=1).
DETERMINISTIC_CONFLICTS = ("temperature", "top_p", "top_k", "min_p", "typical_p")
# Request fields the script controls itself; --sampler must not override them.
RESERVED_REQUEST_FIELDS = {"prompt", "n_predict", "stream", "cache_prompt", "id_slot", "ignore_eos", "seed"}


def parse_sampler_setting(text: str) -> tuple[str, Any]:
    """--sampler KEY=VALUE: VALUE is read as JSON (numbers, true/false, lists), otherwise as a string."""
    key, sep, raw = text.partition("=")
    key = key.strip()
    if not sep or not re.fullmatch(r"[a-z_][a-z0-9_]*", key):
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE with a llama-server field name, got {text!r}")
    if key in RESERVED_REQUEST_FIELDS:
        raise argparse.ArgumentTypeError(f"{key} is set by the script itself (use its own option)")
    if key in SAMPLER_OPTIONS:
        raise argparse.ArgumentTypeError(f"use --{key.replace('_', '-')} instead of --sampler {key}=...")
    try:
        value = json.loads(raw)
    except ValueError:
        value = raw
    return key, value


def sampling_payload(args: Any, repeat_idx: int) -> Dict[str, Any]:
    """Explicit sampler settings; the seed advances per repeat so repeat r uses the same seed at every point."""
    payload = {key: getattr(args, key) for key in SAMPLER_OPTIONS if getattr(args, key, None) is not None}
    payload.update(dict(getattr(args, "sampler", None) or {}))
    if getattr(args, "seed", None) is not None:
        payload["seed"] = args.seed + repeat_idx
    return payload


AGENT_CONTEXT_MARKER = "@@CTX_CLIFF_CONTEXT@@"
AGENT_SYSTEM_PROMPT = (
    "You are an autonomous coding agent working in a large Python code base. You read files "
    "with tools, reason about the code and reply with precise, complete code changes.")
AGENT_CONTEXT_INTRO = "Output of the file-reading tool for the relevant part of the repository:\n\n"
AGENT_ACKNOWLEDGEMENT = "I have read the files. What should I change?"
AGENT_DEFAULT_TASK = (
    "Add a new function `describe_public_api(module_source: str) -> dict` to the code above. "
    "It must parse the given module source with the `ast` module and return a mapping of every "
    "public top-level class name to the sorted list of its public method names. Use type hints, "
    "write a docstring and handle syntax errors gracefully. Reply with the complete implementation.")


def apply_template(base: str, messages: List[Dict[str, str]], template_kwargs: Optional[Dict[str, Any]] = None,
                   http: Optional[PostClient] = None) -> str:
    body: Dict[str, Any] = {"messages": messages}
    if template_kwargs:
        body["chat_template_kwargs"] = template_kwargs
    r = (http if http is not None else requests).post(f"{base}/apply-template", json=body, timeout=120)
    r.raise_for_status()
    return r.json()["prompt"]


def agent_prompt_parts(base: str, run_marker: str, task: str, thinking: str,
                       http: Optional[PostClient] = None) -> tuple[str, str]:
    """Render the agent conversation with the model's chat template around the input excerpt.

    Returns (prefix, suffix): prefix = system prompt with run marker and the start of
    the context message; suffix = end of that message, a fixed assistant reply, the
    fixed task and the generation prompt. The excerpt grows between them, so every
    point asks the same question after a longer history.
    """
    messages = [
        {"role": "system", "content": f"{run_marker}\n{AGENT_SYSTEM_PROMPT}"},
        {"role": "user", "content": AGENT_CONTEXT_INTRO + AGENT_CONTEXT_MARKER},
        {"role": "assistant", "content": AGENT_ACKNOWLEDGEMENT},
        {"role": "user", "content": task},
    ]
    kwargs = {"enable_thinking": thinking == "on"} if thinking in ("on", "off") else None
    rendered = apply_template(base, messages, kwargs, http=http)
    parts = rendered.split(AGENT_CONTEXT_MARKER)
    if len(parts) != 2:
        raise PromptBuildError("the chat template did not keep the context placeholder exactly once")
    return parts[0], parts[1]


def reset_slot(base: str, slot_id: int = 0, quiet: bool = False, http: Optional[RequestClient] = None) -> bool:
    """Erase one llama-server slot. The first endpoint is the documented one."""
    attempts = [('post', f'{base}/slots/{slot_id}?action=erase'), ('post', f'{base}/slots?action=erase&id_slot={slot_id}'), ('get', f'{base}/slots?action=erase&id_slot={slot_id}')]
    for method, url in attempts:
        try:
            r = (http if http is not None else requests).request(method, url, timeout=60)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
    if not quiet:
        print(f'WARNING: failed to erase slot {slot_id} at {base}; slot state may be retained.', file=sys.stderr)
    return False


def prepare_snapshot_command(command: str, args: Any) -> str:
    """Add a local snapshot directory only when starting a server that needs it."""
    if args.cache_mode != "incremental" or args.repeat <= 1:
        return command
    tokens = shlex.split(command, posix=os.name != "nt")
    if any(token.strip('\"\'').split("=", 1)[0] == "--slot-save-path" for token in tokens):
        return command
    directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "slot-snapshots")
    os.makedirs(directory, exist_ok=True)
    option = (subprocess.list2cmdline(["--slot-save-path", directory]) if os.name == "nt"
              else shlex.join(["--slot-save-path", directory]))
    print(f"NOTE: no --slot-save-path specified; using snapshot directory {directory}", file=sys.stderr)
    return f"{command} {option}"


def probe_prefill_repeats(args: Any, base: str, filename: str, http: Optional[PostClient] = None) -> None:
    """Probe save support before warmup; a rejected save does not alter the slot."""
    args.prefill_repeat_enabled = True
    args.prefill_repeat_error = None
    if args.cache_mode != "incremental" or args.repeat <= 1:
        return
    try:
        slot_snapshot(base, args.slot_id, "save", filename, http=http)
    except RuntimeError as error:
        args.prefill_repeat_enabled = False
        args.prefill_repeat_error = str(error)
        print(f"WARNING: prefill repeats disabled: {error}\n"
              f"Continuing with one prefill measurement per context point and "
              f"{args.repeat} decode samples (original cache behaviour).", file=sys.stderr)
    else:
        print(f"NOTE: slot snapshots available; prefill and decode use {args.repeat} samples per point.",
              file=sys.stderr)
        print("NOTE: each later point prepares a pure token-prefix snapshot in an extra "
              "unmeasured pass; every repeat must reuse that entire prefix.", file=sys.stderr)


def snapshot_directory_from_command(command: str) -> Optional[str]:
    """Resolve the explicit directory in the command actually used for a local launch."""
    lexer = shlex.shlex(command, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    if os.name == "nt":
        lexer.escape = ""  # Windows paths contain literal backslashes.
    tokens = list(lexer)
    directory = None
    for index, token in enumerate(tokens):
        if token == "--slot-save-path" and index + 1 < len(tokens):
            directory = tokens[index + 1]
        elif token.startswith("--slot-save-path="):
            directory = token.split("=", 1)[1]
    return os.path.abspath(directory) if directory else None


def cleanup_snapshot(directory: str, filename: str) -> None:
    """Remove only this run's generated file; never remove the directory or other runs."""
    if os.path.basename(filename) != filename or not filename.startswith("ctx-cliff-") or not filename.endswith(".bin"):
        raise ValueError("invalid benchmark snapshot filename")
    path = os.path.join(directory, filename)
    try:
        os.unlink(path)
    except FileNotFoundError:
        return
    except OSError as error:
        print(f"WARNING: could not remove snapshot {path}: {error}", file=sys.stderr)
    else:
        print(f"NOTE: removed snapshot {path}", file=sys.stderr)


def slot_snapshot(base: str, slot_id: int, action: str, filename: str, http: Optional[PostClient] = None) -> int:
    """Save/restore server-side state; fail rather than silently measure a warm cache."""
    client = http if http is not None else requests
    response = client.post(f"{base}/slots/{slot_id}?action={action}",
                           json={"filename": filename}, timeout=3600)
    hint = "Start llama-server with --slot-save-path pointing to an existing writable directory."
    if not response.ok:
        raise RuntimeError(f"Slot {action} failed (HTTP {response.status_code}): "
                           f"{response.text[:1000]}. {hint}")
    data = response.json()
    count = data.get("n_saved" if action == "save" else "n_restored")
    if type(count) is not int or count < 0:
        raise RuntimeError(f"Slot {action} did not return a valid token count. {hint}")
    return count


class CompletionRequestError(RuntimeError):
    def __init__(
        self,
        status_code: int,
        message: str,
        error_type: str = "",
        n_ctx: Optional[int] = None,
        n_prompt_tokens: Optional[int] = None,
        body: str = "",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.error_type = error_type
        self.n_ctx = n_ctx
        self.n_prompt_tokens = n_prompt_tokens
        self.body = body

    @property
    def is_context_overflow(self) -> bool:
        t = self.error_type.lower()
        m = self.message.lower()
        return (
            "context" in t
            or "exceed_context" in t
            or ("context" in m and ("exceed" in m or "too large" in m or "increasing" in m))
        )


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def detect_slot_n_ctx(base: str, slot_id: int, http: Optional[GetClient] = None) -> Optional[int]:
    """Best-effort discovery of the actual per-slot context limit."""
    try:
        r = (http if http is not None else requests).get(f'{base}/slots', timeout=10)
        if r.ok:
            data = r.json()
            if isinstance(data, list):
                for slot in data:
                    if isinstance(slot, dict) and _int_or_none(slot.get('id')) == slot_id:
                        n_ctx = _int_or_none(slot.get('n_ctx'))
                        if n_ctx and n_ctx > 0:
                            return n_ctx
    except (requests.RequestException, ValueError, TypeError):
        pass
    try:
        r = (http if http is not None else requests).get(f'{base}/props', timeout=10)
        if r.ok:
            data = r.json()
            if isinstance(data, dict):
                settings = data.get('default_generation_settings', {})
                if isinstance(settings, dict):
                    n_ctx = _int_or_none(settings.get('n_ctx'))
                    if n_ctx and n_ctx > 0:
                        return n_ctx
    except (requests.RequestException, ValueError, TypeError):
        pass
    return None


def read_completion_stream(lines: Any, clock: Any = time.perf_counter) -> Dict[str, Any]:
    """Parse llama-server's /completion SSE stream ("data: {...}" lines).

    Returns the final chunk (timings, stop_type, truncated, ...) with the full
    generated text in "content" and the client time of the first chunk that
    carried generated output in "_first_token_mono". A body that is plain JSON
    (server without streaming) is returned as is, without a first-token time.
    """
    first: Optional[float] = None
    parts: List[str] = []
    final: Optional[Dict[str, Any]] = None
    raw: List[str] = []
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8", "replace")
        line = line.strip()
        if not line:
            continue
        if not line.startswith("data:"):
            raw.append(line)
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        if isinstance(chunk, dict) and chunk.get("error"):
            error = chunk["error"] if isinstance(chunk["error"], dict) else {"message": str(chunk["error"])}
            raise CompletionRequestError(500, str(error.get("message") or error), str(error.get("type") or ""))
        if first is None and (chunk.get("content") or chunk.get("tokens")):
            first = clock()
        if chunk.get("content"):
            parts.append(chunk["content"])
        if chunk.get("stop"):
            final = chunk
            break
    if final is None:
        if raw:
            return json.loads("\n".join(raw))
        raise CompletionRequestError(502, "streamed completion ended without a final chunk")
    result = dict(final)
    result["content"] = "".join(parts)
    result["_first_token_mono"] = first
    return result


def completion(base: str, prompt: str | List[int], n_predict: int, deterministic: bool, cache_prompt: bool, slot_id: int, ignore_eos: bool, http: Optional[PostClient] = None, stream: bool = False, sampling: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {'prompt': prompt, 'n_predict': n_predict, 'stream': stream, 'cache_prompt': cache_prompt, 'id_slot': slot_id, 'ignore_eos': ignore_eos}
    payload.update(sampling or {})
    if deterministic:
        payload['temperature'] = 0.0
        payload['top_k'] = 1
    client = http if http is not None else requests
    if stream:
        r = client.post(f'{base}/completion', json=payload, timeout=1800, stream=True)
    else:
        r = client.post(f'{base}/completion', json=payload, timeout=1800)
    if not r.ok:
        body = r.text[:4000]
        message = body or f'HTTP {r.status_code}'
        error_type = ''
        n_ctx = None
        n_prompt_tokens = None
        try:
            data = r.json()
            err = data.get('error', data) if isinstance(data, dict) else {}
            if isinstance(err, dict):
                message = str(err.get('message') or message)
                error_type = str(err.get('type') or err.get('err_type') or '')
                n_ctx = _int_or_none(err.get('n_ctx'))
                n_prompt_tokens = _int_or_none(err.get('n_prompt_tokens'))
            if isinstance(data, dict):
                n_ctx = n_ctx or _int_or_none(data.get('n_ctx'))
                n_prompt_tokens = n_prompt_tokens or _int_or_none(data.get('n_prompt_tokens'))
        except (ValueError, TypeError):
            pass
        raise CompletionRequestError(r.status_code, message, error_type, n_ctx, n_prompt_tokens, body)
    if stream:
        try:
            return read_completion_stream(r.iter_lines())
        finally:
            close = getattr(r, "close", None)
            if close is not None:
                close()
    return r.json()


def rate_from_timing(t: Dict[str, Any], count_key: str, ms_key: str, rate_key: str) -> float:
    """Prefer a finite server rate, then derive it from a valid count and duration."""
    reported = finite_number(t.get(rate_key))
    if reported is not None and reported >= 0:
        return reported
    count, ms = finite_number(t.get(count_key)), finite_number(t.get(ms_key))
    if count is not None and ms is not None and count > 0 and ms > 0:
        return finite_number(count / (ms / 1000.0)) or 0.0
    return 0.0


def sample_status(predicted_n: int, n_predict: int, truncated: bool) -> str:
    if truncated:
        return "TRUNC"
    if predicted_n <= 0:
        return "EMPTY"
    if predicted_n < n_predict:
        return f"STOP@{predicted_n}"
    return "OK"


def largest_rate_drop(
    results: List[Dict[str, Any]], rate_key: str, min_valid_repeats: int = 2
) -> Optional[tuple[float, Dict[str, Any], Dict[str, Any]]]:
    """Compare adjacent increasing contexts with enough valid phase repeats."""
    valid_key = "prefill_valid_repeats" if rate_key == "prefill_tps" else "valid_repeats"
    candidates = []
    for prev, cur in zip(results, results[1:]):
        if any((_int_or_none(row.get(valid_key)) or 0) < min_valid_repeats for row in (prev, cur)):
            continue
        prev_tps = finite_number(prev.get(rate_key))
        cur_tps = finite_number(cur.get(rate_key))
        prev_ctx = _int_or_none(prev.get("total_ctx"))
        cur_ctx = _int_or_none(cur.get("total_ctx"))
        if (
            prev_tps is None or cur_tps is None
            or prev_tps <= 0 or cur_tps <= 0
            or prev_ctx is None or cur_ctx is None or cur_ctx <= prev_ctx
        ):
            continue
        drop_pct = 100.0 * (prev_tps - cur_tps) / prev_tps
        if drop_pct > 0:
            candidates.append((drop_pct, prev, cur))
    return max(candidates, key=lambda item: item[0]) if candidates else None


def largest_decode_drop(results: List[Dict[str, Any]], min_valid_repeats: int = 2) -> Optional[tuple[float, Dict[str, Any], Dict[str, Any]]]:
    """Return the largest adjacent drop in median decode throughput."""
    return largest_rate_drop(results, "decode_tps_median", min_valid_repeats)


def largest_prefill_drop(results: List[Dict[str, Any]], min_valid_repeats: int = 2) -> Optional[tuple[float, Dict[str, Any], Dict[str, Any]]]:
    """Return the largest adjacent drop in median prefill throughput."""
    return largest_rate_drop(results, "prefill_tps", min_valid_repeats)


# Prefill rates are only compared between points whose newly processed token
# counts (prompt_n) differ by at most this factor: small batches are slower.
PREFILL_COMPARABLE_RATIO = 1.5


def step_rate_rows(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Result rows with the decode step rate (steps/s = 1000 / ms per step) for cliff analysis."""
    def rate(ms: Any) -> Optional[float]:
        value = finite_number(ms)
        return 1000.0 / value if value is not None and value > 0 else None
    return [{**row, "decode_step_rate": rate(row.get("ms_per_step_median")),
             # The fastest repeat has the smallest step cost.
             "decode_step_rate_min": rate(row.get("ms_per_step_max")),
             "decode_step_rate_max": rate(row.get("ms_per_step_min"))} for row in results]


def drafting_active(*rows: Dict[str, Any]) -> bool:
    return any((finite_number(row.get("tokens_per_step_median")) or 1.0) > 1.0 for row in rows)


def analyze_cliffs(results: List[Dict[str, Any]], rate_key: str, min_valid_repeats: int,
                   cliff_pct: float) -> Dict[str, Any]:
    """All adjacent drops of one phase, split into confirmed and unconfirmed.

    A drop >= cliff_pct is a confirmed candidate only if
      * the repeat ranges do not overlap (previous minimum > current maximum), so
        a single noisy repeat cannot create it,
      * for prefill, both points processed a comparable number of new tokens, and
      * for decode with MTP/drafting, the cost per verification step rose as well;
        otherwise fewer tokens per step (less predictable text) caused the drop.
    The overall change from the first to the last usable point shows gradual
    declines that no single step reveals.
    """
    prefill = rate_key == "prefill_tps"
    valid_key = "prefill_valid_repeats" if prefill else "valid_repeats"
    low_key, high_key = ((f"{rate_key}_min", f"{rate_key}_max") if rate_key == "decode_step_rate"
                         else ("prefill_tps_min", "prefill_tps_max") if prefill
                         else ("decode_tps_min", "decode_tps_max"))
    usable = [row for row in results
              if (_int_or_none(row.get(valid_key)) or 0) >= min_valid_repeats
              and (finite_number(row.get(rate_key)) or 0) > 0 and _int_or_none(row.get("total_ctx")) is not None]
    drops = []
    for prev, cur in zip(results, results[1:]):
        if prev not in usable or cur not in usable or int(cur["total_ctx"]) <= int(prev["total_ctx"]):
            continue
        prev_rate, cur_rate = float(prev[rate_key]), float(cur[rate_key])
        drop_pct = 100.0 * (prev_rate - cur_rate) / prev_rate
        if drop_pct <= 0:
            continue
        prev_low, cur_high = finite_number(prev.get(low_key)), finite_number(cur.get(high_key))
        separated = None if prev_low is None or cur_high is None else prev_low > cur_high
        comparable = True
        if prefill:
            a, b = _int_or_none(prev.get("prompt_n")) or 0, _int_or_none(cur.get("prompt_n")) or 0
            comparable = a > 0 and b > 0 and max(a, b) / min(a, b) <= PREFILL_COMPARABLE_RATIO
        reasons = []
        if separated is False:
            reasons.append("repeat ranges overlap")
        if not comparable:
            reasons.append("prompt_n differs")
        if rate_key == "decode_tps_median" and drafting_active(prev, cur):
            prev_ms, cur_ms = finite_number(prev.get("ms_per_step_median")), finite_number(cur.get("ms_per_step_median"))
            if prev_ms and cur_ms and 100.0 * (1.0 - prev_ms / cur_ms) < cliff_pct:
                reasons.append(
                    f"content effect: tokens/step {display_number(prev.get('tokens_per_step_median'), 2)}"
                    f"->{display_number(cur.get('tokens_per_step_median'), 2)}, "
                    f"step cost {100.0 * (cur_ms / prev_ms - 1.0):+.1f}%")
        drops.append({"drop_pct": drop_pct, "absolute": prev_rate - cur_rate, "prev": prev, "cur": cur,
                      "separated": separated, "comparable": comparable, "reasons": reasons})
    strong = sorted((d for d in drops if d["drop_pct"] >= cliff_pct and not d["reasons"]),
                    key=lambda d: -d["drop_pct"])
    unconfirmed = sorted((d for d in drops if d["drop_pct"] >= cliff_pct and d["reasons"]),
                         key=lambda d: -d["drop_pct"])
    overall = None
    if len(usable) >= 2:
        first, last = usable[0], usable[-1]
        overall = {"first_ctx": first["total_ctx"], "last_ctx": last["total_ctx"],
                   "change_pct": 100.0 * (float(last[rate_key]) - float(first[rate_key])) / float(first[rate_key])}
    return {"drops": drops, "candidates": strong, "unconfirmed": unconfirmed,
            "largest": max(drops, key=lambda d: d["drop_pct"]) if drops else None,
            "overall": overall, "usable_points": len(usable)}


DRIFT_WARN_PCT = 5.0


def compare_drift(original: Dict[str, Any], again: Dict[str, Any]) -> Dict[str, Any]:
    """Change of the first point between start and end of the run.

    A change counts as drift when it exceeds DRIFT_WARN_PCT and the re-measured
    median lies outside the original point's repeat range.
    """
    result: Dict[str, Any] = {"target_ctx": original.get("target_ctx")}
    for name, key, low, high in (("decode", "decode_tps_median", "decode_tps_min", "decode_tps_max"),
                                 ("prefill", "prefill_tps", "prefill_tps_min", "prefill_tps_max")):
        before, after = finite_number(original.get(key)), finite_number(again.get(key))
        change = 100.0 * (after - before) / before if before and after is not None else None
        lo, hi = finite_number(original.get(low)), finite_number(original.get(high))
        outside = None if after is None or lo is None or hi is None else not lo <= after <= hi
        result[name] = {"before": before, "after": after, "change_pct": change,
                        "outside_original_range": outside,
                        "drift": bool(change is not None and abs(change) > DRIFT_WARN_PCT and outside is not False)}
    result["drift"] = result["decode"]["drift"] or result["prefill"]["drift"]
    return result


def print_drift(drift: Optional[Dict[str, Any]]) -> None:
    if not drift:
        return
    # Printed right below "drift check: re-measuring the first point ...".
    if drift.get("error"):
        print(f"drift check failed: {drift['error']}")
        return
    parts = []
    for name in ("decode", "prefill"):
        d = drift[name]
        if d["before"] is not None and d["after"] is not None:
            parts.append(f"{name} {d['before']:.2f} -> {d['after']:.2f} tok/s ({d['change_pct']:+.1f}%)")
    print("; ".join(parts) if parts else "no comparable values")
    if drift["drift"]:
        print(f"-> WARNING: more than {DRIFT_WARN_PCT:.0f}% and outside the original repeat range: conditions "
              "changed during the run (temperature, clocks, background load). Compare points with care; "
              "gradual trends may be drift rather than context effects.")
    elif parts:
        print("-> stable: no relevant drift between start and end of the run")


def print_cliff_report(phase: str, analysis: Dict[str, Any], args: Any) -> None:
    def between(d: Dict[str, Any]) -> str:
        unit = "steps/s" if "STEP" in phase else "tok/s"
        return (f"{d['drop_pct']:.1f}% ({d['absolute']:.2f} {unit}) between ctx={d['prev']['total_ctx']} "
                f"and ctx={d['cur']['total_ctx']}")

    if not analysis["drops"] and not analysis["overall"]:
        print(f"\n{phase}: no drop between adjacent increasing contexts with "
              f"at least {args.cliff_min_repeats} valid repeats each.")
        if phase == "PREFILL" and prefill_mode(args) == "first_repeat_only" and args.cliff_min_repeats > 1:
            print("-> prefill repeats were disabled (snapshot fallback): each point has one "
                  "prefill sample; use --cliff-min-repeats 1 to compare them anyway")
        return
    candidates = analysis["candidates"]
    if candidates:
        for d in candidates:
            print(f"\n{phase} CLIFF CANDIDATE: {between(d)}")
            finer_step = max(250, (int(d["cur"]["total_ctx"]) - int(d["prev"]["total_ctx"])) // 10)
            print(f"-> verify with: --start {d['prev']['total_ctx']} --end {d['cur']['total_ctx']} "
                  f"--step {finer_step} --repeat {max(5, args.repeat)} --deterministic")
    elif analysis["largest"] is not None:
        print(f"\n{phase}: largest relative drop {between(analysis['largest'])}")
        if not analysis["unconfirmed"]:
            print(f"-> below --cliff-pct {args.cliff_pct:.1f}%; no strong cliff detected")
    else:
        print(f"\n{phase}: no drop between adjacent points")
    for d in analysis["unconfirmed"]:
        print(f"-> not counted as a cliff ({', '.join(d['reasons'])}): {between(d)}")
    overall = analysis["overall"]
    if overall is not None:
        gradual = (f" (gradual, no single step >= {args.cliff_pct:g}%)"
                   if not candidates and -overall["change_pct"] >= args.cliff_pct else "")
        print(f"-> overall {phase.lower()} change ctx {overall['first_ctx']} -> {overall['last_ctx']}: "
              f"{overall['change_pct']:+.1f}%{gradual}")


def cliff_summary(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """JSON-friendly digest for the metadata file."""
    def pair(d: Dict[str, Any]) -> Dict[str, Any]:
        return {"from_ctx": d["prev"]["total_ctx"], "to_ctx": d["cur"]["total_ctx"],
                "drop_pct": round(d["drop_pct"], 2), "reasons": d["reasons"]}
    return {"candidates": [pair(d) for d in analysis["candidates"]],
            "unconfirmed": [pair(d) for d in analysis["unconfirmed"]],
            "overall_change_pct": (round(analysis["overall"]["change_pct"], 2)
                                   if analysis["overall"] else None)}


def resolve_nvml_handles(nv: Any, gpu: str) -> List[tuple[str, Any]]:
    """NVML handles for 'all' or a comma list of indices/UUIDs, labelled by NVML index."""
    if gpu == "all":
        return [(str(i), nv.nvmlDeviceGetHandleByIndex(i)) for i in range(int(nv.nvmlDeviceGetCount()))]
    handles: List[tuple[str, Any]] = []
    for token in (x.strip() for x in gpu.split(",")):
        if not token:
            continue
        try:
            idx = int(token)
            handles.append((str(idx), nv.nvmlDeviceGetHandleByIndex(idx)))
        except ValueError:
            # `nvidia-smi -i` accepts UUIDs too. NVML's Python wrapper may expect
            # either str or bytes depending on the package version.
            try:
                handle = nv.nvmlDeviceGetHandleByUUID(token)
            except Exception:
                handle = nv.nvmlDeviceGetHandleByUUID(token.encode("utf-8"))
            try:
                label = str(int(nv.nvmlDeviceGetIndex(handle)))
            except Exception:
                label = token
            handles.append((label, handle))
    return handles


VRAM_SETTLE_WINDOW_S = 3.0
VRAM_SETTLE_TOLERANCE_MIB = 64.0


def nvml_used_mib_reader(gpu: str) -> tuple[Any, Any]:
    """(read, close) for the summed used VRAM of the selected GPUs; (None, None) without NVML."""
    try:
        import pynvml as nv
        nv.nvmlInit()
    except Exception:
        return None, None

    def close() -> None:
        try:
            nv.nvmlShutdown()
        except Exception:
            pass

    try:
        handles = [handle for _, handle in resolve_nvml_handles(nv, gpu)]
    except Exception:
        handles = []
    if not handles:
        close()
        return None, None

    def read() -> Optional[float]:
        try:
            return sum(float(nv.nvmlDeviceGetMemoryInfo(handle).used) for handle in handles) / 1048576.0
        except Exception:
            return None
    return read, close


def wait_for_vram_settle(read: Any, timeout_s: float, window_s: float = VRAM_SETTLE_WINDOW_S,
                         tolerance_mib: float = VRAM_SETTLE_TOLERANCE_MIB, interval_s: float = 0.5,
                         clock: Any = time.monotonic, sleep: Any = time.sleep) -> Optional[Dict[str, Any]]:
    """Wait until used VRAM has not dropped by more than tolerance_mib for window_s.

    A process that just ended (e.g. the llama-server of the previous run in a
    batch file) may still be releasing its memory. A server started meanwhile
    can get part of its buffers in shared system memory (Windows sysmem
    fallback) and run far slower. Returns None when VRAM cannot be read.
    """
    start = clock()
    first = read()
    if first is None:
        return None
    samples = [(start, first)]
    while True:
        now = clock()
        recent = [used for moment, used in samples if now - moment <= window_s] or [samples[-1][1]]
        settled = now - start >= window_s and max(recent) - samples[-1][1] <= tolerance_mib
        if settled or now - start >= timeout_s:
            return {"initial_mib": round(first, 1), "final_mib": round(samples[-1][1], 1),
                    "released_mib": round(first - samples[-1][1], 1), "waited_s": round(now - start, 2),
                    "settled": settled}
        sleep(interval_s)
        used = read()
        if used is not None:
            samples.append((clock(), used))


def settle_vram_before_server_start(args: Any) -> Optional[Dict[str, Any]]:
    """--vram-settle-s: wait for released GPU memory before launching a managed server."""
    settle_s = getattr(args, "vram_settle_s", 0.0)
    if not settle_s or settle_s <= 0:
        return None
    read, close = nvml_used_mib_reader(getattr(args, "vram_gpu", "all"))
    if read is None:
        return None
    try:
        result = wait_for_vram_settle(read, settle_s)
    finally:
        close()
    if result and result["released_mib"] > VRAM_SETTLE_TOLERANCE_MIB:
        print(f"GPU memory: waited {result['waited_s']:.1f} s while {result['released_mib']:.0f} MiB "
              "were released by a previous process", file=sys.stderr)
    if result:
        # Browsers, desktop apps etc. vary by hundreds of MiB between runs; with a
        # nearly full GPU that decides whether the server fits into dedicated VRAM.
        print(f"GPU memory before server start: {result['final_mib']:.0f} MiB used by other processes",
              file=sys.stderr)
    if result and not result["settled"]:
        print(f"WARNING: GPU memory was still being released after {settle_s:g} s "
              f"({result['final_mib']:.0f} MiB used); starting llama-server anyway", file=sys.stderr)
    return result


class NvidiaVramMonitor:
    """NVIDIA sampler: nvidia-smi + legacy NVML PCIe snapshots and BUS busy telemetry."""

    BASE_FIELDS = [
        "index",
        "memory.used",
        "memory.total",
        "utilization.gpu",
        "utilization.memory",
    ]
    OPTIONAL_FIELDS = [
        "pstate",
        "clocks.current.graphics",
        "clocks.current.memory",
        "power.draw",
        "power.limit",
        "temperature.gpu",
        # Link state is also sampled directly through NVML when available, but
        # these are useful as a low-rate fallback/reference in the VRAM trace.
        "pcie.link.gen.current",
        "pcie.link.width.current",
    ]

    NVML_INSTALL = "py -m pip install -U nvidia-ml-py"
    NVML_URL = "https://pypi.org/project/nvidia-ml-py/"
    # Dynamic Pstates BUS utilization is defined by NVML over the trailing 1-second window.
    BUS_INTERVAL_MS = 1000
    BUS_WINDOW_S = 1.0
    # Current PCIe link generation/width is refreshed at most this often.
    LINK_REFRESH_S = 1.0

    def __init__(
        self,
        interval_ms: int = 250,
        gpu: str = "all",
        pcie_interval_ms: int = 250,
    ) -> None:
        self.interval_ms = interval_ms
        self.gpu = gpu
        self.pcie_interval_ms = pcie_interval_ms
        self.fields: List[str] = list(self.BASE_FIELDS)
        self.samples = SampleStore('t_mono')
        self.pcie_samples = SampleStore()
        # Retrospective phase windows reconstructed from llama-server timing data.
        # Each tuple: (target_ctx, repeat_idx, subphase, start_mono, end_mono).
        self.pcie_phase_windows: List[tuple[Any, Any, str, float, float]] = []
        self.lock = threading.Lock()
        self.proc: Optional[subprocess.Popen[str]] = None
        self.thread: Optional[threading.Thread] = None
        self.pcie_thread: Optional[threading.Thread] = None
        self.running = False
        self.stop_event = threading.Event()
        self.current_label: Dict[str, Any] = {
            "phase": "idle",
            "target_ctx": None,
            "repeat": None,
        }
        self.error: Optional[str] = None
        self.pcie_error: Optional[str] = None
        self.pcie_source: str = "none"
        self.bus_error: Optional[str] = None
        self.bus_supported: bool = False
        self._bus_domain_index: int = 3
        self._pynvml: Any = None
        self._nvml_handles: List[tuple[str, Any]] = []
        self._nvml_initialized = False
        # Achieved legacy PCIe timing (all GPUs): RX+TX query duration and the
        # spacing between consecutive polls of the same GPU, in milliseconds.
        self.pcie_query_ms: List[float] = []
        self.pcie_poll_ms: List[float] = []
        self.pcie_probe_query_ms: Optional[float] = None
        # Divide legacy throughput by this calibration factor (see pcie-calibrate.py).
        self.pcie_scale = 1.0
        # Telemetry backend: 'auto' prefers direct NVML, falls back to nvidia-smi.
        self.backend_preference = "auto"
        self.backend = "none"
        self.sampler_error: Optional[str] = None
        self._sampler_nv: Any = None
        self._sampler_initialized = False
        self._sampler_handles: List[tuple[str, Any]] = []

    # Evidence that a GPU was busy, used to recognise complete GPM dropouts.
    # Legacy raw values in idle stay around 1 MB/s on the calibrated system.
    ACTIVITY_PCIE_MB_S = 20.0
    ACTIVITY_GPU_UTIL_PCT = 25.0

    def activity_evidence(self, gpu_index: Any, start: float, end: float) -> Optional[str]:
        """'legacy_pcie' or 'nvidia_smi_util' if the GPU was demonstrably active."""
        gpu = str(gpu_index)
        rows, _ = self.pcie_samples.window(start, end)
        traffic = [rx + tx for r in rows if str(r.get("gpu_index")) == gpu
                   and (rx := finite_number(r.get("pcie_rx_mb_s"))) is not None
                   and (tx := finite_number(r.get("pcie_tx_mb_s"))) is not None]
        if traffic and statistics.mean(traffic) >= self.ACTIVITY_PCIE_MB_S:
            return "legacy_pcie"
        rows, _ = self.samples.window(start, end)
        util = [u for r in rows if str(r.get("gpu_index")) == gpu
                and (u := finite_number(r.get("gpu_util_pct"))) is not None]
        if util and max(util) >= self.ACTIVITY_GPU_UTIL_PCT:
            return "nvidia_smi_util"
        return None

    def pcie_timing_summary(self) -> Optional[Dict[str, Any]]:
        """Requested vs achieved legacy PCIe sampling, or None without samples."""
        with self.lock:
            query, poll = list(self.pcie_query_ms), list(self.pcie_poll_ms)
        if not query:
            return None
        return {
            "requested_interval_ms": self.pcie_interval_ms,
            "achieved_interval_median_ms": rounded(percentile(poll, 50), 1),
            "achieved_interval_p90_ms": rounded(percentile(poll, 90), 1),
            "rx_tx_query_median_ms": rounded(percentile(query, 50), 1),
            "rx_tx_query_p90_ms": rounded(percentile(query, 90), 1),
            # Two back-to-back 20 ms NVML windows per poll (RX, then TX).
            "scale": self.pcie_scale,
            "time_coverage_pct_per_direction": (
                rounded(100.0 * 20.0 / percentile(poll, 50), 1) if poll else None),
            "polls": len(query),
        }

    @staticmethod
    def _number(value: str) -> Optional[float]:
        return finite_number(value.strip())

    def _command_for_fields(self, fields: List[str]) -> List[str]:
        cmd = [
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ]
        if self.gpu != "all":
            cmd.extend(["-i", self.gpu])
        return cmd

    def _probe_fields(self) -> bool:
        # Base telemetry is required. Optional diagnostics are probed separately
        # so one unsupported field never disables the whole monitor.
        try:
            probe = subprocess.run(
                self._command_for_fields(self.BASE_FIELDS),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
                check=False,
            )
        except Exception as e:
            self.error = str(e)
            return False
        if probe.returncode != 0:
            self.error = (probe.stderr or probe.stdout or "nvidia-smi probe failed").strip()
            return False

        supported = list(self.BASE_FIELDS)
        for field in self.OPTIONAL_FIELDS:
            try:
                q = subprocess.run(
                    self._command_for_fields(["index", field]),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=10,
                    check=False,
                )
                if q.returncode == 0:
                    supported.append(field)
            except Exception:
                pass
        self.fields = supported
        return True

    def _init_nvml_pcie(self) -> bool:
        """Initialise direct NVML PCIe polling. Never disables normal VRAM telemetry."""
        try:
            import pynvml  # provided by NVIDIA's `nvidia-ml-py` package
        except ModuleNotFoundError:
            self.pcie_error = (
                "official NVIDIA NVML Python bindings not installed; install with "
                f"`{self.NVML_INSTALL}` ({self.NVML_URL})"
            )
            return False
        except Exception as e:
            self.pcie_error = f"failed to import pynvml: {e}"
            return False

        # Keep the module reference before initialisation so a later probe error
        # can still pair a successful nvmlInit() with nvmlShutdown().
        self._pynvml = pynvml
        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True
            handles = resolve_nvml_handles(pynvml, self.gpu)
            if not handles:
                raise RuntimeError("no NVML GPU handles resolved")

            # Probe the exact counters we need. NVML reports KB/s over an internal
            # 20 ms measurement window; convert to decimal MB/s in the sampler.
            test_handle = handles[0][1]
            probe_start = time.perf_counter()
            pynvml.nvmlDeviceGetPcieThroughput(test_handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)
            pynvml.nvmlDeviceGetPcieThroughput(test_handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)
            self.pcie_probe_query_ms = (time.perf_counter() - probe_start) * 1000.0

            # BUS is a utilization/busy-time domain, not a bandwidth percentage.
            # NVIDIA defines its percentage over the trailing 1-second interval.
            self._bus_domain_index = int(getattr(pynvml, "NVML_GPU_UTILIZATION_DOMAIN_BUS", 3))
            try:
                dyn = pynvml.nvmlDeviceGetDynamicPstatesInfo(test_handle)
                bus = dyn.utilization[self._bus_domain_index]
                self.bus_supported = bool(int(bus.bIsPresent))
                if not self.bus_supported:
                    self.bus_error = "NVML BUS utilization domain is not present on this GPU"
            except Exception as e:
                self.bus_supported = False
                self.bus_error = f"NVML BUS utilization unavailable: {e}"

            self._nvml_handles = handles
            self.pcie_source = "nvml"
            return True
        except Exception as e:
            self.pcie_error = f"NVML PCIe throughput unavailable: {e}"
            self._shutdown_nvml()
            return False

    def _shutdown_nvml(self) -> None:
        if self._nvml_initialized and self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
        self._nvml_initialized = False
        self._pynvml = None
        self._nvml_handles = []

    def _nvml_optional_int(self, fn_name: str, handle: Any) -> Optional[float]:
        nvml = self._pynvml
        if nvml is None:
            return None
        fn = getattr(nvml, fn_name, None)
        if fn is None:
            return None
        try:
            return float(fn(handle))
        except Exception:
            return None

    def _pcie_reader(self) -> None:
        nvml = self._pynvml
        if nvml is None:
            return
        interval_s = max(0.020, self.pcie_interval_ms / 1000.0)
        bus_interval_s = self.BUS_INTERVAL_MS / 1000.0
        next_bus_tick = time.perf_counter()
        previous_query: Dict[str, float] = {}
        # Link state queries cost as much as the throughput query itself on some
        # systems; refresh the current link at most once per LINK_REFRESH_S and the
        # (fixed) maximum link only once.
        link_cache: Dict[str, tuple] = {}
        for now_mono in polling_ticks(self.stop_event, interval_s, immediate=True):
            bus_due = self.bus_supported and now_mono >= next_bus_tick
            now_iso = dt.datetime.now().astimezone().isoformat(timespec="milliseconds")
            with self.lock:
                label = dict(self.current_label)

            for idx, handle in list(self._nvml_handles):
                query_start = time.perf_counter()
                try:
                    # NVML returns KB/s from a byte counter sampled over 20 ms.
                    # Use decimal MB/s to match PCIe line-rate calculations.
                    rx_mb_s = float(
                        nvml.nvmlDeviceGetPcieThroughput(handle, nvml.NVML_PCIE_UTIL_RX_BYTES)
                    ) / 1000.0
                    tx_mb_s = float(
                        nvml.nvmlDeviceGetPcieThroughput(handle, nvml.NVML_PCIE_UTIL_TX_BYTES)
                    ) / 1000.0
                except Exception as e:
                    self.pcie_error = f"NVML PCIe polling failed: {e}"
                    continue
                query_end = time.perf_counter()
                # Honest timing: RX and TX are two consecutive NVML windows inside
                # [query_start, query_end]; time-stamp the sample at their centre
                # and record the achieved spacing between polls of this GPU.
                query_ms = (query_end - query_start) * 1000.0
                poll_ms = ((query_start - previous_query[idx]) * 1000.0
                           if idx in previous_query else None)
                previous_query[idx] = query_start
                sample_mono = (query_start + query_end) / 2.0

                bus_util_pct: Optional[float] = None
                if bus_due:
                    try:
                        dyn = nvml.nvmlDeviceGetDynamicPstatesInfo(handle)
                        bus = dyn.utilization[self._bus_domain_index]
                        if int(bus.bIsPresent):
                            bus_util_pct = float(bus.percentage)
                    except Exception as e:
                        # Keep RX/TX alive if BUS utilization is unsupported or transiently fails.
                        self.bus_error = f"NVML BUS utilization polling failed: {e}"

                cached = link_cache.get(idx)
                if cached is None or query_end - cached[0] >= self.LINK_REFRESH_S:
                    gen = self._nvml_optional_int("nvmlDeviceGetCurrPcieLinkGeneration", handle)
                    width = self._nvml_optional_int("nvmlDeviceGetCurrPcieLinkWidth", handle)
                    if cached is None:
                        gen_max = self._nvml_optional_int("nvmlDeviceGetMaxPcieLinkGeneration", handle)
                        width_max = self._nvml_optional_int("nvmlDeviceGetMaxPcieLinkWidth", handle)
                    else:
                        gen_max, width_max = cached[3], cached[4]
                    link_cache[idx] = cached = (time.perf_counter(), gen, width, gen_max, width_max)
                _, gen, width, gen_max, width_max = cached
                with self.lock:
                    self.pcie_query_ms.append(query_ms)
                    if poll_ms is not None:
                        self.pcie_poll_ms.append(poll_ms)
                    self.pcie_samples.append(
                        {
                            "timestamp": now_iso,
                            "t_mono": sample_mono,
                            "gpu_index": idx,
                            "pcie_rx_mb_s": rx_mb_s,
                            "pcie_tx_mb_s": tx_mb_s,
                            "pcie_query_ms": query_ms,
                            "pcie_poll_interval_ms": poll_ms,
                            "pcie_bus_util_pct": bus_util_pct,
                            "pcie_bus_window_s": self.BUS_WINDOW_S,
                            "pcie_link_gen": gen,
                            "pcie_link_width": width,
                            "pcie_link_gen_max": gen_max,
                            "pcie_link_width_max": width_max,
                            **label,
                        }
                    )

            if bus_due:
                # Dynamic Pstates BUS is already a trailing 1-second metric; polling it
                # faster would only create heavily overlapping windows.
                next_bus_tick = now_mono + bus_interval_s

    def start(self) -> bool:
        """Prefer direct NVML sampling; fall back to an nvidia-smi subprocess."""
        preference = getattr(self, "backend_preference", "auto")
        if preference in ("auto", "nvml"):
            if self._start_nvml_sampler():
                return True
            if preference == "nvml":
                self.error = f"NVML sampler unavailable: {self.sampler_error}"
                return False
        return self._start_nvidia_smi()

    # --- direct NVML sampling ---------------------------------------------------

    def _nvml_sample(self, nv: Any, handle: Any) -> Optional[Dict[str, Any]]:
        """One telemetry row via NVML (same fields/units as the nvidia-smi query)."""
        def attempt(name: str, *call_args: Any) -> Any:
            function = getattr(nv, name, None)
            if function is None:
                return None
            try:
                return function(handle, *call_args)
            except Exception:
                return None

        memory = attempt("nvmlDeviceGetMemoryInfo")
        if memory is None or not getattr(memory, "total", 0):
            return None
        used, total = float(memory.used) / 1048576.0, float(memory.total) / 1048576.0
        rates = attempt("nvmlDeviceGetUtilizationRates")
        pstate = attempt("nvmlDeviceGetPerformanceState")
        power, limit = attempt("nvmlDeviceGetPowerUsage"), attempt("nvmlDeviceGetPowerManagementLimit")

        def number(value: Any, scale: float = 1.0) -> Optional[float]:
            number_ = finite_number(value)
            return number_ / scale if number_ is not None else None

        return {
            "used_mib": used, "total_mib": total, "used_pct": 100.0 * used / total,
            "gpu_util_pct": number(getattr(rates, "gpu", None)),
            "mem_util_pct": number(getattr(rates, "memory", None)),
            "pstate": f"P{int(pstate)}" if pstate is not None and int(pstate) < 32 else None,
            "gpu_clock_mhz": number(attempt("nvmlDeviceGetClockInfo", getattr(nv, "NVML_CLOCK_GRAPHICS", 0))),
            "mem_clock_mhz": number(attempt("nvmlDeviceGetClockInfo", getattr(nv, "NVML_CLOCK_MEM", 2))),
            "power_draw_w": number(power, 1000.0), "power_limit_w": number(limit, 1000.0),
            "temperature_c": number(attempt("nvmlDeviceGetTemperature", getattr(nv, "NVML_TEMPERATURE_GPU", 0))),
            "pcie_link_gen": number(attempt("nvmlDeviceGetCurrPcieLinkGeneration")),
            "pcie_link_width": number(attempt("nvmlDeviceGetCurrPcieLinkWidth")),
        }

    def _start_nvml_sampler(self) -> bool:
        self.sampler_error = None
        try:
            import pynvml as nv
            nv.nvmlInit()
        except Exception as error:
            self.sampler_error = f"pynvml unavailable: {error}"
            return False
        self._sampler_nv, self._sampler_initialized = nv, True
        try:
            handles = resolve_nvml_handles(nv, self.gpu)
            if not handles:
                raise RuntimeError("no NVML GPU handles resolved")
            if self._nvml_sample(nv, handles[0][1]) is None:
                raise RuntimeError("NVML memory query failed")
        except Exception as error:
            self.sampler_error = str(error)
            self._shutdown_sampler_nvml()
            return False
        self._sampler_handles = handles
        self.backend = "nvml"
        self._init_nvml_pcie()
        self.stop_event.clear()
        self.running = True
        self.thread = threading.Thread(target=self._nvml_reader, name="nvml-telemetry-monitor", daemon=True)
        self.thread.start()
        if self.pcie_source == "nvml":
            self.pcie_thread = threading.Thread(target=self._pcie_reader, name="nvml-pcie-monitor", daemon=True)
            self.pcie_thread.start()
        deadline = time.perf_counter() + max(2.0, self.interval_ms / 1000.0 * 3.0)
        while time.perf_counter() < deadline:
            with self.lock:
                if self.samples:
                    return True
            time.sleep(0.02)
        self.error = "NVML sampler produced no telemetry samples"
        self.stop()
        return False

    def _nvml_reader(self) -> None:
        nv = self._sampler_nv
        failures = 0
        for _ in polling_ticks(self.stop_event, self.interval_ms / 1000.0, immediate=True):
            got_any = False
            for idx, handle in list(self._sampler_handles):
                values = self._nvml_sample(nv, handle)
                if values is None:
                    continue
                got_any = True
                now_mono = time.perf_counter()
                now_iso = dt.datetime.now().astimezone().isoformat(timespec="milliseconds")
                with self.lock:
                    label = dict(self.current_label)
                    self.samples.append({"timestamp": now_iso, "t_mono": now_mono, "gpu_index": idx,
                                         **values, **label})
            failures = 0 if got_any else failures + 1
            if failures >= 20:
                self.error = "NVML telemetry queries keep failing"
                return

    def _shutdown_sampler_nvml(self) -> None:
        if getattr(self, "_sampler_initialized", False):
            try:
                self._sampler_nv.nvmlShutdown()
            except Exception:
                pass
        self._sampler_initialized = False

    # --- nvidia-smi fallback -----------------------------------------------------

    def _start_nvidia_smi(self) -> bool:
        if shutil.which("nvidia-smi") is None:
            self.error = "nvidia-smi not found" + (
                f" (NVML sampler: {self.sampler_error})" if getattr(self, "sampler_error", None) else "")
            return False
        if not self._probe_fields():
            return False
        self.backend = "nvidia-smi"

        # Direct NVML is deliberately the only PCIe throughput backend. dmon is
        # not used because its 1 s sampling is too coarse for these sweeps.
        self._init_nvml_pcie()

        cmd = self._command_for_fields(self.fields) + [f"--loop-ms={self.interval_ms}"]
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            self.error = str(e)
            self._shutdown_nvml()
            return False

        self.stop_event.clear()
        self.running = True
        self.thread = threading.Thread(target=self._reader, name="nvidia-telemetry-monitor", daemon=True)
        self.thread.start()
        if self.pcie_source == "nvml":
            self.pcie_thread = threading.Thread(target=self._pcie_reader, name="nvml-pcie-monitor", daemon=True)
            self.pcie_thread.start()

        deadline = time.perf_counter() + max(2.0, self.interval_ms / 1000.0 * 3.0)
        while time.perf_counter() < deadline:
            with self.lock:
                if self.samples:
                    return True
            if self.proc.poll() is not None:
                self.error = "nvidia-smi monitor exited immediately"
                self.stop()
                return False
            time.sleep(0.02)
        self.error = "nvidia-smi produced no telemetry samples"
        self.stop()
        return False

    def _reader(self) -> None:
        proc = self.proc
        if proc is None or proc.stdout is None:
            raise RuntimeError("nvidia-smi reader started without a stdout pipe")
        while self.running:
            raw = proc.stdout.readline()
            if raw == "":
                if self.running:
                    self.error = "nvidia-smi monitor output ended unexpectedly"
                break
            line = raw.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != len(self.fields):
                continue
            values = dict(zip(self.fields, parts))

            idx = values.get("index", "")
            used = self._number(values.get("memory.used", ""))
            total = self._number(values.get("memory.total", ""))
            if used is None or total is None or total <= 0:
                continue

            now_mono = time.perf_counter()
            now_iso = dt.datetime.now().astimezone().isoformat(timespec="milliseconds")
            with self.lock:
                label = dict(self.current_label)
                self.samples.append(
                    {
                        "timestamp": now_iso,
                        "t_mono": now_mono,
                        "gpu_index": idx,
                        "used_mib": used,
                        "total_mib": total,
                        "used_pct": 100.0 * used / total,
                        "gpu_util_pct": self._number(values.get("utilization.gpu", "")),
                        "mem_util_pct": self._number(values.get("utilization.memory", "")),
                        "pstate": values.get("pstate") or None,
                        "gpu_clock_mhz": self._number(values.get("clocks.current.graphics", "")),
                        "mem_clock_mhz": self._number(values.get("clocks.current.memory", "")),
                        "power_draw_w": self._number(values.get("power.draw", "")),
                        "power_limit_w": self._number(values.get("power.limit", "")),
                        "temperature_c": self._number(values.get("temperature.gpu", "")),
                        "pcie_link_gen": self._number(values.get("pcie.link.gen.current", "")),
                        "pcie_link_width": self._number(values.get("pcie.link.width.current", "")),
                        **label,
                    }
                )

    def set_label(self, phase: str, target_ctx: Optional[int], repeat_idx: Optional[int]) -> None:
        with self.lock:
            self.current_label = {
                "phase": phase,
                "target_ctx": target_ctx,
                "repeat": None if repeat_idx is None else repeat_idx + 1,
            }

    @staticmethod
    def _pcie_theoretical_mb_s(gen: float, width: float) -> float:
        """Approximate one-direction PCIe payload line rate, decimal MB/s."""
        gen_i = int(round(gen)) if gen else 0
        width_i = int(round(width)) if width else 0
        gt_s = {1: 2.5, 2: 5.0, 3: 8.0, 4: 16.0, 5: 32.0, 6: 64.0}.get(gen_i, 0.0)
        if gt_s <= 0 or width_i <= 0:
            return 0.0
        encoding_eff = 0.8 if gen_i <= 2 else (128.0 / 130.0)
        return gt_s * 1000.0 * encoding_eff * width_i / 8.0

    def summarize(self, start_mono: float, end_mono: float) -> Dict[str, Any]:
        return summarize_vram(self.samples, self.pcie_samples, start_mono, end_mono, self.interval_ms,
                              self.pcie_scale)

    def stop(self) -> None:
        self.running = False
        self.stop_event.set()
        if self.pcie_thread is not None:
            self.pcie_thread.join(timeout=max(1.0, self.pcie_interval_ms / 1000.0 * 4.0))
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        if self.thread is not None:
            self.thread.join(timeout=1)
        if self.pcie_thread is not None and self.pcie_thread.is_alive():
            raise RuntimeError("PCIe monitor did not stop; NVML resources are still in use")
        self._shutdown_nvml()
        self._shutdown_sampler_nvml()

    def write_csv(self, path: str) -> int:
        write_csv_atomic(path, VRAM_CSV_FIELDS, self.samples.iter_all())
        return len(self.samples)

    def register_pcie_phase_window(
        self, target_ctx: Any, repeat_idx: Any, subphase: str, start_mono: float, end_mono: float
    ) -> None:
        if end_mono <= start_mono:
            return
        with self.lock:
            self.pcie_phase_windows.append((target_ctx, repeat_idx, subphase, start_mono, end_mono))

    @staticmethod
    def _pcie_phase_fields(summary: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        return {prefix + key[5:]: value for key, value in summary.items() if key.startswith("pcie_")}

    def write_pcie_csv(self, path: str) -> int:
        with self.lock:
            index = PhaseIndex(self.pcie_phase_windows)
        def rows():
            for source in self.pcie_samples.iter_all():
                row = dict(source)
                row["subphase"] = row["bus_subphase"] = ""
                when = float(row.get("t_mono", 0.0))
                window = index.find(when, when)
                if window is not None:
                    row["subphase"] = window[2]
                    if row.get("pcie_bus_util_pct") is not None and when - self.BUS_WINDOW_S >= window[3]:
                        row["bus_subphase"] = window[2]
                yield row
        write_csv_atomic(path, PCIE_CSV_FIELDS, rows())
        return len(self.pcie_samples)


class NvidiaGpmMonitor:
    """NVML GPM interval telemetry for GPU engines and PCIe throughput.

    GPM metrics are calculated from two NVML samples. NVIDIA requires the samples
    to be more than 100 ms apart. Every stored row therefore represents a real
    interval [start_mono, end_mono], which lets PF/DC summaries reject intervals
    that cross a phase boundary.
    """

    NVML_INSTALL = "py -m pip install -U nvidia-ml-py"
    NVML_URL = "https://pypi.org/project/nvidia-ml-py/"

    METRIC_DEFS = (
        ("graphics_util_pct", "NVML_GPM_METRIC_GRAPHICS_UTIL"),
        ("sm_util_pct", "NVML_GPM_METRIC_SM_UTIL"),
        ("sm_occupancy_pct", "NVML_GPM_METRIC_SM_OCCUPANCY"),
        ("tensor_util_pct", "NVML_GPM_METRIC_ANY_TENSOR_UTIL"),
        ("dram_bw_util_pct", "NVML_GPM_METRIC_DRAM_BW_UTIL"),
        ("pcie_rx_mib_s", "NVML_GPM_METRIC_PCIE_RX_PER_SEC"),
        ("pcie_tx_mib_s", "NVML_GPM_METRIC_PCIE_TX_PER_SEC"),
    )

    def __init__(self, interval_ms: int = 250, gpu: str = "all") -> None:
        self.interval_ms = interval_ms
        self.gpu = gpu
        self.samples = SampleStore('interval_end_mono')
        self.phase_windows: List[tuple[Any, Any, str, float, float]] = []
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.stop_event = threading.Event()
        self.error: Optional[str] = None
        self.supported = False
        self.metric_ids: List[tuple[str, int]] = []
        self._pynvml: Any = None
        self._nvml_initialized = False
        self._handles: List[tuple[str, Any]] = []
        self._sample_pairs: Dict[str, tuple[Any, Any]] = {}
        self._prev_times: Dict[str, float] = {}
        # Leave suspect SM/occupancy/tensor dropouts out of phase statistics.
        self.exclude_suspect = True
        # GPM values describe traffic this much earlier than they are read
        # (measured with pcie-calibrate.py). Intervals are shifted back by it.
        self.lag_s = 0.0
        # Optional activity_evidence(gpu, start, end) of a second sensor.
        self.activity: Optional[Any] = None
        # Experimental restart of GPM sampling (--gpm-restart), run in the reader thread.
        self.restart_events: List[Dict[str, Any]] = []
        self._restart_request: Optional[str] = None
        self._restart_done = threading.Event()

    def _resolve_handles(self, nv: Any) -> List[tuple[str, Any]]:
        return resolve_nvml_handles(nv, self.gpu)

    @staticmethod
    def _pcie_theoretical_mib_s(gen: float, width: float) -> float:
        # Same payload line-rate approximation as the legacy monitor, converted
        # from decimal MB/s to MiB/s so it can be compared directly with GPM.
        mb_s = NvidiaVramMonitor._pcie_theoretical_mb_s(gen, width)
        return mb_s / 1.048576 if mb_s > 0 else 0.0

    @staticmethod
    def _optional_link_int(nv: Any, fn_name: str, handle: Any) -> Optional[float]:
        fn = getattr(nv, fn_name, None)
        if fn is None:
            return None
        try:
            return float(fn(handle))
        except Exception:
            return None

    def start(self) -> bool:
        try:
            import pynvml as nv
        except ModuleNotFoundError:
            self.error = (
                "official NVIDIA NVML Python bindings not installed; install with "
                f"`{self.NVML_INSTALL}` ({self.NVML_URL})"
            )
            return False
        except Exception as e:
            self.error = f"failed to import pynvml: {e}"
            return False

        required = (
            "nvmlGpmQueryDeviceSupport", "nvmlGpmSampleAlloc", "nvmlGpmSampleGet",
            "nvmlGpmMetricsGet", "nvmlGpmSampleFree", "c_nvmlGpmMetricsGet_t",
            "NVML_GPM_METRICS_GET_VERSION",
        )
        missing = [name for name in required if not hasattr(nv, name)]
        if missing:
            self.error = "installed pynvml lacks GPM API: " + ", ".join(missing)
            return False

        try:
            nv.nvmlInit()
            self._nvml_initialized = True
            self._pynvml = nv
            self._handles = self._resolve_handles(nv)
            if not self._handles:
                raise RuntimeError("no NVML GPU handles resolved")

            metric_ids: List[tuple[str, int]] = []
            for key, constant in self.METRIC_DEFS:
                if hasattr(nv, constant):
                    metric_ids.append((key, int(getattr(nv, constant))))
            if not metric_ids:
                raise RuntimeError("no requested GPM metric constants are available")
            self.metric_ids = metric_ids

            unsupported: List[str] = []
            for idx, handle in self._handles:
                support = nv.nvmlGpmQueryDeviceSupport(handle)
                if not int(support.isSupportedDevice):
                    unsupported.append(idx)
            if unsupported:
                raise RuntimeError("GPM not supported on GPU(s): " + ",".join(unsupported))

            now = time.perf_counter()
            for idx, handle in self._handles:
                a = nv.nvmlGpmSampleAlloc()
                b = nv.nvmlGpmSampleAlloc()
                nv.nvmlGpmSampleGet(handle, a)
                self._sample_pairs[idx] = (a, b)
                self._prev_times[idx] = now

            self.stop_event.clear()
            self.running = True
            self.supported = True
            self.thread = threading.Thread(target=self._reader, name="nvml-gpm-monitor", daemon=True)
            self.thread.start()
            return True
        except Exception as e:
            self.error = f"GPM unavailable: {e}"
            self.stop()
            return False

    def _reader(self) -> None:
        nv = self._pynvml
        if nv is None:
            return
        interval_s = self.interval_ms / 1000.0
        success = int(getattr(nv, "NVML_SUCCESS", 0))
        for now in polling_ticks(self.stop_event, interval_s):
            if self._restart_request is not None:
                kind, self._restart_request = self._restart_request, None
                self._apply_restart(kind)
                self._restart_done.set()
                nv = self._pynvml
                if nv is None:
                    return
                continue
            now_iso = dt.datetime.now().astimezone().isoformat(timespec="milliseconds")

            for idx, handle in list(self._handles):
                pair = self._sample_pairs.get(idx)
                if pair is None:
                    continue
                prev_sample, cur_sample = pair
                start_mono = float(self._prev_times.get(idx, now - interval_s))
                try:
                    nv.nvmlGpmSampleGet(handle, cur_sample)
                    get = nv.c_nvmlGpmMetricsGet_t()
                    get.version = nv.NVML_GPM_METRICS_GET_VERSION
                    get.numMetrics = len(self.metric_ids)
                    get.sample1 = prev_sample
                    get.sample2 = cur_sample
                    for i, (_, metric_id) in enumerate(self.metric_ids):
                        get.metrics[i].metricId = metric_id
                    nv.nvmlGpmMetricsGet(get)

                    row: Dict[str, Any] = {
                        "timestamp": now_iso,
                        "gpu_index": idx,
                        # Lag-corrected interval (what the values describe) ...
                        "interval_start_mono": start_mono - self.lag_s,
                        "interval_end_mono": now - self.lag_s,
                        "interval_ms": (now - start_mono) * 1000.0,
                        # ... and the raw sampling times.
                        "sample_start_mono": start_mono,
                        "sample_end_mono": now,
                    }
                    for i, (key, _) in enumerate(self.metric_ids):
                        m = get.metrics[i]
                        row[key] = finite_number(m.value) if int(m.nvmlReturn) == success else None

                    gen = self._optional_link_int(nv, "nvmlDeviceGetCurrPcieLinkGeneration", handle)
                    width = self._optional_link_int(nv, "nvmlDeviceGetCurrPcieLinkWidth", handle)
                    row["pcie_link_gen"] = gen
                    row["pcie_link_width"] = width
                    row["pcie_theoretical_mib_s"] = self._pcie_theoretical_mib_s(gen or 0.0, width or 0.0)
                    with self.lock:
                        self.samples.append(row)

                    # Reuse the two allocated sample objects without allocating in the hot loop.
                    self._sample_pairs[idx] = (cur_sample, prev_sample)
                    self._prev_times[idx] = now
                except Exception as e:
                    self.error = f"GPM polling failed: {e}"
                    # Reset the previous sample after a transient failure so the next
                    # interval does not accidentally span a long gap.
                    try:
                        nv.nvmlGpmSampleGet(handle, prev_sample)
                        self._prev_times[idx] = now
                    except Exception:
                        pass

    def summarize(self, start_mono: float, end_mono: float,
                  dropout_gpus: Optional[set] = None) -> Dict[str, Any]:
        return summarize_gpm(self.samples, start_mono, end_mono, self.exclude_suspect, dropout_gpus,
                             self.activity)

    def request_restart(self, kind: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Restart GPM sampling at the reader's next tick and wait for it.

        realloc: free and allocate new sample buffers for every GPU.
        reinit:  additionally nvmlShutdown() + nvmlInit() and resolve handles again.
                 NVML is reference-counted: while the NVIDIA VRAM/PCIe monitor
                 keeps its own reference, this does not fully tear NVML down.
        The interval spanning the restart is skipped. Returns the logged event.
        """
        if kind not in ("realloc", "reinit"):
            raise ValueError(f"unknown GPM restart kind: {kind}")
        if not self.running or self.thread is None or not self.thread.is_alive():
            return None
        self._restart_done.clear()
        self._restart_request = kind
        if not self._restart_done.wait(timeout if timeout is not None else 3 * self.interval_ms / 1000.0 + 1.0):
            return None
        return self.restart_events[-1] if self.restart_events else None

    def _apply_restart(self, kind: str) -> None:
        nv = self._pynvml
        started = time.perf_counter()
        event: Dict[str, Any] = {"kind": kind, "t_mono": started,
                                 "timestamp": dt.datetime.now().astimezone().isoformat(timespec="milliseconds")}
        try:
            for a, b in list(self._sample_pairs.values()):
                for sample in (a, b):
                    try:
                        nv.nvmlGpmSampleFree(sample)
                    except Exception:
                        pass
            self._sample_pairs = {}
            if kind == "reinit":
                nv.nvmlShutdown()
                nv.nvmlInit()
                self._handles = self._resolve_handles(nv)
            now = time.perf_counter()
            for idx, handle in self._handles:
                a, b = nv.nvmlGpmSampleAlloc(), nv.nvmlGpmSampleAlloc()
                nv.nvmlGpmSampleGet(handle, a)
                self._sample_pairs[idx] = (a, b)
                self._prev_times[idx] = now
            event["ok"] = True
        except Exception as error:
            event.update(ok=False, error=str(error))
            self.error = f"GPM {kind} restart failed: {error}"
        event["duration_ms"] = (time.perf_counter() - started) * 1000.0
        self.restart_events.append(event)

    def wait_for_lagged(self, end_mono: float) -> None:
        """With a lag correction, intervals describing time up to end_mono arrive
        up to lag_s later. Wait for them (outside all measured windows)."""
        if self.lag_s <= 0 or not self.running:
            return
        remaining = end_mono + self.lag_s + 0.02 - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)

    def dropout_gpus(self, start_mono: float, end_mono: float) -> set:
        return gpm_dropout_gpus(self.samples, start_mono, end_mono)

    @staticmethod
    def phase_fields(summary: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        return {prefix + key[4:]: value for key, value in summary.items() if key.startswith("gpm_")}

    def register_phase_window(
        self, target_ctx: Any, repeat_idx: Any, subphase: str, start_mono: float, end_mono: float
    ) -> None:
        if end_mono <= start_mono:
            return
        with self.lock:
            self.phase_windows.append((target_ctx, repeat_idx, subphase, start_mono, end_mono))

    def write_csv(self, path: str) -> int:
        with self.lock:
            index = PhaseIndex(self.phase_windows)
        def rows():
            for source in self.samples.iter_all():
                row = dict(source)
                row["subphase"] = ""
                window = index.find(float(row["interval_start_mono"]), float(row["interval_end_mono"]))
                if window is not None:
                    row["target_ctx"], row["repeat"], row["subphase"] = window[:3]
                yield row
        write_csv_atomic(path, GPM_CSV_FIELDS, rows())
        return len(self.samples)

    def stop(self) -> None:
        self.running = False
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(1.0, self.interval_ms / 1000.0 * 3.0))
            if self.thread.is_alive():
                raise RuntimeError("monitor did not stop; driver resources are still in use")
        nv = self._pynvml
        if nv is not None:
            for a, b in list(self._sample_pairs.values()):
                for sample in (a, b):
                    try:
                        nv.nvmlGpmSampleFree(sample)
                    except Exception:
                        pass
        self._sample_pairs = {}
        if self._nvml_initialized and nv is not None:
            try:
                nv.nvmlShutdown()
            except Exception:
                pass
        self._nvml_initialized = False
        self._pynvml = None
        self._handles = []


class WindowsGpuMemoryMonitor:
    """Sample WDDM adapter dedicated/shared residency using Windows PDH."""

    PDH_FMT_LARGE = 0x00000400
    ERROR_SUCCESS = 0
    PDH_CSTATUS_NEW_DATA = 1
    PDH_MORE_DATA = 0x800007D2

    def __init__(self, interval_ms: int = 1000) -> None:
        self.interval_ms = interval_ms
        self.samples = SampleStore('t_mono')
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.stop_event = threading.Event()
        self.error: Optional[str] = None
        self.current_label: Dict[str, Any] = {
            "phase": "idle",
            "target_ctx": None,
            "repeat": None,
        }
        self._pdh = None
        self._query = None
        self._counters: Dict[str, Any] = {}
        self._ctypes = None
        self._wintypes = None
        self._ITEM = None

    def _init_pdh(self) -> bool:
        if sys.platform != "win32":
            self.error = "Windows PDH telemetry is only available on Windows"
            return False
        try:
            import ctypes
            from ctypes import wintypes

            class PDH_FMT_COUNTERVALUE_UNION(ctypes.Union):
                _fields_ = [
                    ("longValue", wintypes.LONG),
                    ("doubleValue", ctypes.c_double),
                    ("largeValue", ctypes.c_longlong),
                    ("AnsiStringValue", ctypes.c_char_p),
                    ("WideStringValue", wintypes.LPWSTR),
                ]

            class PDH_FMT_COUNTERVALUE(ctypes.Structure):
                _anonymous_ = ("u",)
                _fields_ = [("CStatus", wintypes.DWORD), ("u", PDH_FMT_COUNTERVALUE_UNION)]

            class PDH_FMT_COUNTERVALUE_ITEM_W(ctypes.Structure):
                _fields_ = [("szName", wintypes.LPWSTR), ("FmtValue", PDH_FMT_COUNTERVALUE)]

            pdh = ctypes.WinDLL("pdh.dll")
            pdh.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            pdh.PdhOpenQueryW.restype = wintypes.LONG
            pdh.PdhAddEnglishCounterW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
            pdh.PdhAddEnglishCounterW.restype = wintypes.LONG
            pdh.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
            pdh.PdhCollectQueryData.restype = wintypes.LONG
            pdh.PdhGetFormattedCounterArrayW.argtypes = [ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
            pdh.PdhGetFormattedCounterArrayW.restype = wintypes.LONG
            pdh.PdhCloseQuery.argtypes = [ctypes.c_void_p]
            pdh.PdhCloseQuery.restype = wintypes.LONG

            query = ctypes.c_void_p()
            status = pdh.PdhOpenQueryW(None, None, ctypes.byref(query))
            if status != self.ERROR_SUCCESS:
                self.error = f"PdhOpenQueryW failed: 0x{status & 0xffffffff:08x}"
                return False

            counters: Dict[str, Any] = {}
            for key, path in {
                "dedicated": r"\GPU Adapter Memory(*)\Dedicated Usage",
                "shared": r"\GPU Adapter Memory(*)\Shared Usage",
            }.items():
                counter = ctypes.c_void_p()
                status = pdh.PdhAddEnglishCounterW(query, path, None, ctypes.byref(counter))
                if status != self.ERROR_SUCCESS:
                    pdh.PdhCloseQuery(query)
                    self.error = f"PdhAddEnglishCounterW({path}) failed: 0x{status & 0xffffffff:08x}"
                    return False
                counters[key] = counter

            self._ctypes = ctypes
            self._wintypes = wintypes
            self._pdh = pdh
            self._query = query
            self._counters = counters
            self._ITEM = PDH_FMT_COUNTERVALUE_ITEM_W
            return True
        except Exception as e:
            self.error = str(e)
            return False

    def _read_array(self, counter: Any) -> Dict[str, int]:
        assert self._ctypes is not None and self._wintypes is not None and self._pdh is not None and self._ITEM is not None
        ctypes = self._ctypes
        wintypes = self._wintypes
        size = wintypes.DWORD(0)
        count = wintypes.DWORD(0)
        status = self._pdh.PdhGetFormattedCounterArrayW(
            counter, self.PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count), None
        )
        status_u32 = int(status) & 0xffffffff
        if status_u32 not in (self.PDH_MORE_DATA, self.ERROR_SUCCESS) or size.value == 0:
            return {}
        buf = ctypes.create_string_buffer(size.value)
        status = self._pdh.PdhGetFormattedCounterArrayW(
            counter, self.PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count), ctypes.cast(buf, ctypes.c_void_p)
        )
        if status != self.ERROR_SUCCESS:
            return {}
        arr = ctypes.cast(buf, ctypes.POINTER(self._ITEM))
        out: Dict[str, int] = {}
        for i in range(count.value):
            item = arr[i]
            if int(item.FmtValue.CStatus) not in (self.ERROR_SUCCESS, self.PDH_CSTATUS_NEW_DATA):
                continue
            out[str(item.szName or i)] = max(0, int(item.FmtValue.largeValue))
        return out

    def _capture(self) -> None:
        dedicated = self._read_array(self._counters["dedicated"])
        shared = self._read_array(self._counters["shared"])
        if not dedicated and not shared:
            return
        now_mono = time.perf_counter()
        now_iso = dt.datetime.now().astimezone().isoformat(timespec="milliseconds")
        with self.lock:
            label = dict(self.current_label)
            self.samples.append(
                {
                    "timestamp": now_iso,
                    "t_mono": now_mono,
                    "dedicated_mib": sum(dedicated.values()) / (1024.0 * 1024.0),
                    "shared_mib": sum(shared.values()) / (1024.0 * 1024.0),
                    "adapter_instances": max(len(dedicated), len(shared)),
                    **label,
                }
            )

    def start(self) -> bool:
        if not self._init_pdh():
            return False
        assert self._pdh is not None and self._query is not None
        status = self._pdh.PdhCollectQueryData(self._query)
        if status != self.ERROR_SUCCESS:
            self.error = f"PdhCollectQueryData failed: 0x{status & 0xffffffff:08x}"
            self.stop()
            return False
        self._capture()
        self.stop_event.clear()
        self.running = True
        self.thread = threading.Thread(target=self._run, name="windows-gpu-memory-monitor", daemon=True)
        self.thread.start()
        return True

    def _run(self) -> None:
        assert self._pdh is not None and self._query is not None
        interval = max(1.0, self.interval_ms / 1000.0)
        for _ in polling_ticks(self.stop_event, interval):
            status = self._pdh.PdhCollectQueryData(self._query)
            if status != self.ERROR_SUCCESS:
                continue
            self._capture()

    def set_label(self, phase: str, target_ctx: Optional[int], repeat_idx: Optional[int]) -> None:
        with self.lock:
            self.current_label = {
                "phase": phase,
                "target_ctx": target_ctx,
                "repeat": None if repeat_idx is None else repeat_idx + 1,
            }

    def summarize(self, start_mono: float, end_mono: float) -> Dict[str, Any]:
        return summarize_windows(self.samples, start_mono, end_mono)

    def stop(self) -> None:
        self.running = False
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(1.2, self.interval_ms / 1000.0 + 0.2))
            if self.thread.is_alive():
                raise RuntimeError("monitor did not stop; driver resources are still in use")
        if self._pdh is not None and self._query is not None:
            try:
                self._pdh.PdhCloseQuery(self._query)
            except Exception:
                pass
            self._query = None

    def write_csv(self, path: str) -> int:
        write_csv_atomic(path, WINDOWS_CSV_FIELDS, self.samples.iter_all())
        return len(self.samples)

class BenchmarkRunner:
    """Run requests against explicit client, prompt, telemetry and output dependencies."""
    def __init__(self, args: Any, base: str, http: BenchmarkHttpClient, builder: PromptBuilder,
                 recording: CsvRecording, vram_monitor: Any, gpm_monitor: Any, win_gpu_monitor: Any) -> None:
        self.args, self.base, self.http, self.builder = args, base, http, builder
        self.recording = recording
        self.vram_monitor, self.gpm_monitor, self.win_gpu_monitor = vram_monitor, gpm_monitor, win_gpu_monitor
        self.measurement_started = False
        self.previous_prompt_tokens: Optional[List[int]] = None
        self.snapshot_filename = f"ctx-cliff-{secrets.token_hex(12)}.bin"

    def note(self, message: str) -> None:
        """Print a NOTE now, or keep it for below the live table while the sweep runs."""
        deferred = getattr(self, "deferred_notes", None)
        if deferred is not None:
            deferred.append(message)
        else:
            print(message, file=sys.stderr)

    def take_sample(self, prompt: str | List[int], target_ctx: int, repeat_idx: int, phase: str = "measure") -> Dict[str, Any]:
        stores = [getattr(monitor, attribute) for monitor in
                  (self.vram_monitor, self.gpm_monitor, self.win_gpu_monitor) if monitor is not None
                  for attribute in ("samples", "pcie_samples") if hasattr(monitor, attribute)]
        for store in stores:
            store.begin(time.perf_counter())
        try:
            return self._take_sample(prompt, target_ctx, repeat_idx, phase)
        finally:
            for store in stores:
                store.end()

    def build_prompt(self, ctx: int) -> tuple[List[int], int, int]:
        text, _, target_chars = self.builder.build(ctx)
        cached = getattr(self.builder, "last_tokens", None)
        if not (isinstance(cached, tuple) and len(cached) == 2 and cached[0] == text):
            raise PromptBuildError(f"prompt builder returned no token IDs for target {ctx}")
        tokens = list(cached[1])  # a copy: the caller must not change the builder's list
        limit = min(ctx, getattr(self.builder, "max_prompt_ctx", None) or ctx)
        if not tokens or len(tokens) > limit:
            raise PromptBuildError(f"Final prompt has {len(tokens)} tokens; budget is {limit}")
        return tokens, len(tokens), target_chars

    def measure_point(self, ctx: int, record: bool = True) -> Dict[str, Any]:
        prompt, total_ctx, target_chars = self.build_prompt(ctx)
        last = getattr(self, "last_total_ctx", None)
        if record and last is not None and total_ctx <= last:
            raise InputExhausted(f"target={ctx} gives a {total_ctx}-token prompt, not longer than the "
                                 f"previous point's {last} tokens")
        repeated_incremental = (self.args.cache_mode == "incremental" and self.args.repeat > 1
                                and getattr(self.args, "prefill_repeat_enabled", True))
        first_point = not self.measurement_started
        saved_tokens = None
        # All modes submit the checked IDs, including model special tokens.
        prompt_tokens = prompt
        if repeated_incremental and not first_point:
            prefix_n = 0
            for previous, current in zip(self.previous_prompt_tokens or [], prompt_tokens):
                if previous != current:
                    break
                prefix_n += 1
            prefix_n = min(prefix_n, len(prompt_tokens) - 1)
            if prefix_n <= 0:
                raise RuntimeError(f"No reusable token prefix at context {ctx}; "
                                   "cannot measure incremental prefill.")
            self.recording.check()
            # A snapshot of the previous decode tail cannot roll back recurrent
            # state: slot restore discards the server's intermediate checkpoints.
            # Rebuild the exact common prefix outside all measured windows. The
            # first sampled token is returned but not evaluated into the cache.
            prepared = completion(self.base, prompt_tokens[:prefix_n], 1,
                                  self.args.deterministic, cache_prompt=False,
                                  slot_id=self.args.slot_id, ignore_eos=True, http=self.http)
            if prepared.get("truncated", False):
                raise RuntimeError(f"Prefix preparation truncated at context {ctx}")
            saved_tokens = slot_snapshot(self.base, self.args.slot_id, "save",
                                         self.snapshot_filename, http=self.http)
            if saved_tokens != prefix_n:
                raise RuntimeError(f"Prefix snapshot token mismatch at context {ctx}: "
                                   f"{saved_tokens} != {prefix_n}; "
                                   "server did not save the exact prepared prefix.")
        samples = []
        for repeat in range(self.args.repeat):
            self.recording.check()
            if repeated_incremental:
                if first_point:
                    # Disable prompt reuse for EVERY repeat, including any warmup checkpoints.
                    reset_slot(self.base, self.args.slot_id, quiet=True, http=self.http)
                    self.measurement_started = False
                else:
                    restored = slot_snapshot(self.base, self.args.slot_id, "restore",
                                             self.snapshot_filename, http=self.http)
                    if restored != saved_tokens:
                        raise RuntimeError(f"Slot restore token mismatch: {restored} != {saved_tokens}")
                if self.args.settle:
                    time.sleep(self.args.settle)
            sample = self.take_sample(prompt, ctx, repeat)
            validation_error = ""
            if saved_tokens is not None and sample["cache_n"] != saved_tokens:
                validation_error = (
                    f"Incremental prefix reuse lost at context {ctx}, repeat {repeat + 1}: "
                    f"cache_n={sample['cache_n']}, expected {saved_tokens}. "
                    "Refusing to publish a full or partial cache rebuild as incremental prefill.")
            if not validation_error and samples and repeated_incremental:
                expected = (samples[0]["cache_n"], samples[0]["prompt_n"])
                actual = (sample["cache_n"], sample["prompt_n"])
                if actual != expected:
                    validation_error = (
                        f"Non-comparable prefill at context {ctx}, repeat {repeat + 1}: "
                        f"(cache_n, prompt_n)={actual}, expected {expected}. "
                        "The server did not reproduce the same prefix work after restore.")
            guard_error: Optional[GuardAbortError] = None
            if not validation_error:
                if message := self.check_sysmem_fallback(sample, ctx, repeat):
                    guard_error = SysmemFallbackError(message)
                elif record and (message := self.check_prefill_floor(sample, ctx, repeat)):
                    guard_error = PrefillFloorError(message)
                if guard_error is not None:
                    validation_error = str(guard_error)
            if record:
                self.recording.write_sample({
                    **sample, "target_ctx": ctx, "total_ctx": total_ctx,
                    "target_chars": target_chars, "cache_mode": self.args.cache_mode,
                    "prefill_mode": prefill_mode(self.args),
                    "repeat": repeat + 1, "validation_error": validation_error,
                })
            if guard_error is not None:
                raise guard_error
            if validation_error:
                raise RuntimeError(validation_error)
            samples.append(sample)
        row = aggregate_point(samples, ctx, total_ctx, target_chars, self.args)
        if (row.get("output_variants") or 0) > 1 and getattr(self.args, "deterministic", False):
            self.note(f"NOTE target={ctx}: {row['output_variants']} different outputs across "
                      f"{len(samples)} repeats despite --deterministic (see .samples.csv)")
        if record:
            self.recording.write_result(row)
            self.last_total_ctx = total_ctx
            if getattr(self, "reference_prefill_tps", None) is None:
                self.reference_prefill_tps = finite_number(row.get("prefill_tps"))
        if repeated_incremental:
            self.previous_prompt_tokens = prompt_tokens
        return row

    def check_sysmem_fallback(self, sample: Dict[str, Any], ctx: int, repeat_idx: int) -> str:
        """Return an abort reason for --sysmem-guard abort, print a NOTE for warn, else ''."""
        mode = getattr(self.args, "sysmem_guard", "off")
        if mode == "off" or not (sample.get("prompt_n") or 0) > 0:
            return ""
        rx = sysmem_fallback_signal(sample)
        if rx is None:
            if not getattr(self, "sysmem_guard_inactive_noted", False):
                self.sysmem_guard_inactive_noted = True
                self.note("NOTE: --sysmem-guard is inactive: no prefill PCIe telemetry "
                          "(needs --vram-log or --gpm-log)")
            return ""
        limit = getattr(self.args, "sysmem_guard_mb_s", SYSMEM_GUARD_DEFAULT_MB_S)
        if rx < limit:
            return ""
        message = (f"prefill PCIe receive median {rx:.0f} MB/s >= {limit:.0f} MB/s at target={ctx}, "
                   f"repeat {repeat_idx + 1} (prefill {display_number(sample.get('prefill_tps'), 1)} tok/s): "
                   "GPU memory is probably overflowing into shared system memory (sysmem fallback). "
                   + (f"Before the server started, other processes already used "
                      f"{self.vram_before_start_mib:.0f} MiB of VRAM (browsers, desktop apps); "
                      if getattr(self, "vram_before_start_mib", None) is not None else "")
                   + "Close GPU-heavy programs or reduce --ctx-size or draft/batch settings; use "
                   "--sysmem-guard warn/off for setups that stream weights over PCIe on purpose (CPU offload)")
        if mode == "warn":
            self.note(f"NOTE: {message}")
            return ""
        return message

    def check_prefill_floor(self, sample: Dict[str, Any], ctx: int, repeat_idx: int) -> str:
        """Abort reason when prefill drops below --abort-below-pct of the first point, else ''."""
        pct = getattr(self.args, "abort_below_pct", None)
        reference = getattr(self, "reference_prefill_tps", None)
        rate = finite_number(sample.get("prefill_tps"))
        if not pct or reference is None or rate is None or rate <= 0 or sample.get("truncated"):
            return ""
        if 100.0 * rate / reference >= pct:
            return ""
        return (f"prefill {rate:.1f} tok/s at target={ctx}, repeat {repeat_idx + 1} is "
                f"{100.0 * rate / reference:.0f}% of the first point's {reference:.1f} tok/s "
                f"(--abort-below-pct {pct:g})")

    def remeasure_first_point(self, ctx: int) -> Dict[str, Any]:
        """Measure a context point again exactly like the first point of the sweep
        (slot erased, full prefill for every repeat), without writing CSV rows."""
        self.measurement_started = False
        self.previous_prompt_tokens = None
        if self.args.cache_mode == "incremental":
            reset_slot(self.base, self.args.slot_id, quiet=True, http=self.http)
        return self.measure_point(ctx, record=False)

    def _maybe_restart_gpm(self, dropout_gpus: Optional[set], gpm_phase: Dict[str, Any],
                           target_ctx: int, repeat_idx: int, phase: str) -> str:
        """Experimental (--gpm-restart): restart GPM sampling after a request that
        showed an SM/occupancy/tensor dropout or a complete GPM dropout. Runs after
        the request, outside all measured windows; the next request shows whether
        it helped (dropouts are otherwise sticky across requests)."""
        kind = getattr(self.args, "gpm_restart", "off")
        monitor = self.gpm_monitor
        if kind == "off" or monitor is None or not hasattr(monitor, "request_restart"):
            return ""
        reasons = []
        if dropout_gpus:
            reasons.append("sm_zero")
        if any((gpm_phase.get(f"gpm_{name}_dropout_samples") or 0) > 0 for name in ("prefill", "decode")):
            reasons.append("all_zero")
        if not reasons:
            return ""
        event = monitor.request_restart(kind)
        ok = bool(event and event.get("ok"))
        if event is not None:
            event.update(target_ctx=target_ctx, repeat=repeat_idx + 1, phase=phase, reason="+".join(reasons))
        self.note(f"NOTE target={target_ctx} repeat={repeat_idx + 1}: GPM dropout ({'+'.join(reasons)}); "
                  f"GPM {kind} restart {'done' if ok else 'FAILED'}")
        return kind if ok else f"{kind}_failed"

    def _take_sample(self,
        prompt: str | List[int], target_ctx: int, repeat_idx: int, phase: str = "measure"
    ) -> Dict[str, Any]:
        self.recording.check()
        if self.args.cache_mode == "cold":
            if not reset_slot(self.base, self.args.slot_id, http=self.http):
                raise RuntimeError(f"Cold measurement requires a successful erase of slot {self.args.slot_id}")
            if self.args.settle:
                time.sleep(self.args.settle)

        if self.vram_monitor is not None:
            self.vram_monitor.set_label(phase, target_ctx, repeat_idx)
        if self.win_gpu_monitor is not None:
            self.win_gpu_monitor.set_label(phase, target_ctx, repeat_idx)
        monitor_start = time.perf_counter()
        t0 = monitor_start
        # Slot erase alone may leave restorable server-side prompt checkpoints.
        # Warmup must not consume the first real point's full-prefill measurement.
        first_measurement = phase == "measure" and not self.measurement_started
        reuse_prompt = self.args.cache_mode == "incremental" and not first_measurement
        sampling = sampling_payload(self.args, repeat_idx)
        try:
            resp = completion(
                self.base,
                prompt,
                self.args.n_predict,
                self.args.deterministic,
                cache_prompt=reuse_prompt,
                slot_id=self.args.slot_id,
                ignore_eos=self.args.ignore_eos,
                http=self.http,
                **({"stream": True} if getattr(self.args, "stream", False) else {}),
                **({"sampling": sampling} if sampling else {}),
            )
        finally:
            monitor_end = time.perf_counter()
            if self.vram_monitor is not None:
                self.vram_monitor.set_label("idle", None, None)
            if self.win_gpu_monitor is not None:
                self.win_gpu_monitor.set_label("idle", None, None)
        if phase == "measure":
            self.measurement_started = True
        wall = monitor_end - t0
        vram = self.vram_monitor.summarize(monitor_start, monitor_end) if self.vram_monitor is not None else empty_vram()
        win_gpu = self.win_gpu_monitor.summarize(monitor_start, monitor_end) if self.win_gpu_monitor is not None else empty_windows()
        t = resp.get("timings", {}) or {}

        cache_n = int(t.get("cache_n", 0) or 0)
        prompt_n = int(t.get("prompt_n", 0) or 0)
        prompt_ms = float(t.get("prompt_ms", 0.0) or 0.0)
        predicted_n = int(t.get("predicted_n", 0) or 0)
        predicted_ms = float(t.get("predicted_ms", 0.0) or 0.0)
        draft_n = int(t.get("draft_n", 0) or 0)
        draft_acc = int(t.get("draft_n_accepted", 0) or 0)

        windows, quality = reconstruct_phases(monitor_start, monitor_end, prompt_ms, predicted_ms,
                                              resp.get("_first_token_mono"))
        pcie_phase, gpm_phase = {}, {}
        if self.gpm_monitor is not None and hasattr(self.gpm_monitor, "wait_for_lagged"):
            self.gpm_monitor.wait_for_lagged(monitor_end)
        # GPM counter dropouts span a whole request: judge it once, apply to both phases.
        dropout_gpus = (self.gpm_monitor.dropout_gpus(monitor_start, monitor_end)
                        if self.gpm_monitor is not None and hasattr(self.gpm_monitor, "dropout_gpus") else None)
        for name, start, end in windows:
            if end <= start:
                continue
            if self.vram_monitor is not None:
                pcie_phase.update(phase_fields(self.vram_monitor.summarize(start, end), "pcie_", name))
                self.vram_monitor.register_pcie_phase_window(target_ctx, repeat_idx + 1, name, start, end)
            if self.gpm_monitor is not None:
                gpm_phase.update(phase_fields(
                    self.gpm_monitor.summarize(start, end, dropout_gpus) if dropout_gpus is not None
                    else self.gpm_monitor.summarize(start, end), "gpm_", name))
                self.gpm_monitor.register_phase_window(target_ctx, repeat_idx + 1, name, start, end)

        gpm_restart = self._maybe_restart_gpm(dropout_gpus, gpm_phase, target_ctx, repeat_idx, phase)

        prefill_tps = rate_from_timing(t, "prompt_n", "prompt_ms", "prompt_per_second")
        decode_tps = rate_from_timing(t, "predicted_n", "predicted_ms", "predicted_per_second")
        truncated = bool(resp.get("truncated", False))
        status = sample_status(predicted_n, self.args.n_predict, truncated)

        if status != "OK":
            self.note(f"NOTE target={target_ctx} repeat={repeat_idx + 1}: {status}; "
                      f"decode sample excluded from median "
                      f"(stop_type={resp.get('stop_type')!r}, truncated={truncated})")

        output = analyze_output(resp.get("content"))
        loop_pct = finite_number(output["output_loop_pct"])
        if (phase == "measure" and loop_pct is not None and loop_pct >= OUTPUT_LOOP_WARN_PCT
                and (output["output_chars"] or 0) >= OUTPUT_LOOP_MIN_CHARS):
            self.note(f"NOTE target={target_ctx} repeat={repeat_idx + 1}: generated text looks degenerate "
                      f"(repeating tail covers {loop_pct:.0f}%): {output['output_excerpt'][-80:]!r}")

        return {
            **output,
            "stop_type": resp.get("stop_type"),
            "gpm_restart": gpm_restart,
            "cache_n": cache_n,
            "prompt_n": prompt_n,
            "prompt_ms": prompt_ms,
            "prefill_tps": prefill_tps,
            "decode_tps": decode_tps,
            "predicted_n": predicted_n,
            "predicted_ms": predicted_ms,
            "draft_n": draft_n,
            "draft_acc": draft_acc,
            **decode_step_stats(predicted_n, predicted_ms, draft_acc),
            "wall_s": wall,
            "truncated": truncated,
            "status": status,
            **quality,
            **vram,
            **pcie_phase,
            **gpm_phase,
            **win_gpu,
        }


# --- comparison against a reference run (--reference, --compare) ---------------

COMPARE_METRICS = (
    # key, label, unit, higher is better
    ("prefill_tps", "prefill", "tok/s", True),
    ("decode_tps_median", "decode", "tok/s", True),
    ("ms_per_step_median", "step", "ms", False),
)
# Different values here mean a different workload; the other settings only change
# the generated text (relevant for MTP/drafting acceptance).
WORKLOAD_SETTINGS = ("scenario", "cache_mode", "n_predict", "input_sha256")


def describe_setting_difference(reference: Any, run: Any) -> str:
    if isinstance(reference, dict) and isinstance(run, dict):
        keys = [key for key in dict.fromkeys((*reference, *run)) if reference.get(key) != run.get(key)]
        return ", ".join(f"{key} {reference.get(key, 'default')!s} -> {run.get(key, 'default')!s}" for key in keys)
    return f"{reference!s} -> {run!s}"


def step_values_from_samples(path: str) -> Dict[int, Dict[str, float]]:
    """Per-point tokens/ms per decode step from an older .samples.csv without these columns."""
    by_ctx: Dict[int, List[Dict[str, Any]]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ctx = _int_or_none(row.get("target_ctx"))
            predicted_n, predicted_ms = _int_or_none(row.get("predicted_n")), finite_number(row.get("predicted_ms"))
            if ctx is None or row.get("status") != "OK" or predicted_n is None or predicted_ms is None:
                continue
            stats = decode_step_stats(predicted_n, predicted_ms, _int_or_none(row.get("draft_acc")) or 0)
            if stats["ms_per_step"] is not None:
                by_ctx.setdefault(ctx, []).append(stats)
    return {ctx: {"tokens_per_step_median": statistics.median(s["tokens_per_step"] for s in values),
                  "ms_per_step_median": statistics.median(s["ms_per_step"] for s in values)}
            for ctx, values in by_ctx.items()}


def comparison_points(rows: Any, samples_path: Optional[str] = None) -> Dict[int, Dict[str, Any]]:
    """target_ctx -> numeric comparison values; fills step values missing in older result CSVs."""
    points: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        ctx = _int_or_none(row.get("target_ctx"))
        if ctx is None:
            continue
        point = {key: finite_number(row.get(key)) for key in
                 ("total_ctx", "prefill_tps", "decode_tps_median", "ms_per_step_median",
                  "tokens_per_step_median", "draft_acc_pct", "draft_n")}
        for key, old_key in (("draft_acc_pct", "mtp_acc_pct"), ("draft_n", "mtp_draft_n")):
            if point[key] is None:
                # Result CSVs before the rename called these columns mtp_*.
                point[key] = finite_number(row.get(old_key))
        if point["ms_per_step_median"] is None and not point["draft_n"] and point["decode_tps_median"]:
            # Without drafting a step is one token.
            point["ms_per_step_median"] = 1000.0 / point["decode_tps_median"]
            point["tokens_per_step_median"] = 1.0
        points[ctx] = point
    if samples_path and os.path.exists(samples_path) and any(
            p["ms_per_step_median"] is None for p in points.values()):
        for ctx, values in step_values_from_samples(samples_path).items():
            if ctx in points and points[ctx]["ms_per_step_median"] is None:
                points[ctx].update(values)
    return points


_MIB = r" *([0-9]+(?:\.[0-9]+)?) MiB"
_HOST_BUFFERS = ("CPU", "CUDA_Host")


def parse_server_memory(lines: Any) -> Optional[Dict[str, Any]]:
    """Exact buffer sizes from a llama-server log (independent of other GPU users).

    Each non-empty KV cache starts a context: the first is the main model, later
    ones are draft/MTP models. Recurrent state and the first device compute
    buffer after a KV cache belong to that context.
    """
    model: Dict[str, float] = {}
    contexts: List[Dict[str, Any]] = []
    for line in lines:
        if m := re.search(r"load_tensors:\s+(\S+) model buffer size = " + _MIB, line):
            model[m.group(1)] = model.get(m.group(1), 0.0) + float(m.group(2))
        elif m := re.search(r"llama_kv_cache_kvarn: type = (\S+), layers = (\d+),.* KVarN = " + _MIB, line):
            contexts.append({"kv_type": m.group(1), "kv_layers": int(m.group(2)), "kv_mib": float(m.group(3))})
        elif m := re.search(r"llama_kv_cache: size = +" + _MIB + r" \( *\d+ cells, +(\d+) layers.*K \(([^)]+)\).*V \(([^)]+)\)", line):
            if float(m.group(1)) > 0:
                types = m.group(3) if m.group(3) == m.group(4) else f"{m.group(3)}/{m.group(4)}"
                contexts.append({"kv_type": types, "kv_layers": int(m.group(2)), "kv_mib": float(m.group(1))})
        elif contexts and (m := re.search(r"llama_memory_recurrent:\s+(\S+) RS buffer size = " + _MIB, line)):
            if m.group(1) not in _HOST_BUFFERS:
                contexts[-1]["recurrent_mib"] = contexts[-1].get("recurrent_mib", 0.0) + float(m.group(2))
        elif contexts and (m := re.search(r"sched_reserve:\s+(\S+) compute buffer size = " + _MIB, line)):
            if m.group(1) not in _HOST_BUFFERS:
                contexts[-1].setdefault("compute_mib", float(m.group(2)))
    if not model and not contexts:
        return None
    for index, context in enumerate(contexts):
        context["name"] = "main" if index == 0 else ("draft" if len(contexts) == 2 else f"draft{index}")
    device_model = sum(size for device, size in model.items() if not device.startswith(_HOST_BUFFERS))
    total = device_model + sum(c.get("kv_mib", 0.0) + c.get("recurrent_mib", 0.0) + c.get("compute_mib", 0.0)
                               for c in contexts)
    return {"model_mib": model, "contexts": contexts, "device_total_mib": round(total, 2)}


def read_server_memory(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path or path == "-" or not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return parse_server_memory(f)
    except OSError:
        return None


def describe_server_memory(memory: Dict[str, Any]) -> str:
    parts = [f"model {sum(v for k, v in memory['model_mib'].items() if not k.startswith(_HOST_BUFFERS)):.0f}"]
    for context in memory["contexts"]:
        prefix = "" if context["name"] == "main" else context["name"] + " "
        parts.append(f"{prefix}KV {context['kv_mib']:.1f} ({context['kv_type']})")
        if "recurrent_mib" in context:
            parts.append(f"{prefix}recurrent {context['recurrent_mib']:.1f}")
        if "compute_mib" in context:
            parts.append(f"{prefix}compute {context['compute_mib']:.1f}")
    return ", ".join(parts) + f" | device total {memory['device_total_mib']:.0f} MiB"


def memory_differences(reference: Optional[Dict[str, Any]], run: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Buffers whose size or type differ between two runs."""
    if not reference or not run:
        return []
    def flat(memory: Dict[str, Any]) -> Dict[str, Any]:
        values: Dict[str, Any] = {}
        for context in memory["contexts"]:
            prefix = "" if context["name"] == "main" else context["name"] + " "
            values[prefix + "KV"] = (context.get("kv_mib"), context.get("kv_type"))
            for key, label in (("recurrent_mib", "recurrent"), ("compute_mib", "compute")):
                if key in context:
                    values[prefix + label] = (context[key], None)
        values["model"] = (sum(v for k, v in memory["model_mib"].items() if not k.startswith(_HOST_BUFFERS)), None)
        values["device total"] = (memory["device_total_mib"], None)
        return values
    ref, cur = flat(reference), flat(run)
    def same(a: Any, b: Any) -> bool:  # sizes are logged with 2 decimals; ignore rounding noise
        return a is not None and b is not None and a[1] == b[1] and abs(a[0] - b[0]) < 0.5
    return [{"buffer": key, "reference": ref.get(key), "run": cur.get(key)}
            for key in dict.fromkeys((*ref, *cur)) if not same(ref.get(key), cur.get(key))]


def run_settings(arguments: Dict[str, Any], prompt: Dict[str, Any], input_sha256: Optional[str]) -> Dict[str, Any]:
    """Settings that decide whether two runs measure the same workload."""
    agent = prompt.get("agent") or {}
    return {
        "scenario": prompt.get("scenario") or "file",
        "cache_mode": arguments.get("cache_mode"),
        "n_predict": arguments.get("n_predict"),
        "deterministic": arguments.get("deterministic"),
        "sampling": prompt.get("sampling_first_repeat"),
        "agent_task": agent.get("task"),
        "agent_thinking": agent.get("thinking"),
        "input_sha256": input_sha256,
    }


def load_run(path: str) -> Dict[str, Any]:
    """Result CSV (+ .meta.json / .samples.csv next to it, if present) of an earlier run."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "target_ctx" not in rows[0]:
        raise ValueError(f"{path} is not a ctx-cliff result CSV (no target_ctx rows)")
    stem = path[:-4] if path.lower().endswith(".csv") else path
    meta: Dict[str, Any] = {}
    if os.path.exists(stem + ".meta.json"):
        try:
            with open(stem + ".meta.json", encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, ValueError):
            meta = {}
    server = meta.get("server") or {}
    return {
        "name": os.path.basename(stem),
        "points": comparison_points(rows, stem + ".samples.csv"),
        "settings": (run_settings(meta.get("arguments") or {}, meta.get("prompt") or {},
                                  (meta.get("input_file") or {}).get("sha256")) if meta else None),
        "server_command": server.get("command") or server.get("requested_command")
        or (meta.get("arguments") or {}).get("server_command"),
        # Older runs: read the buffer sizes from the server log the metadata points to.
        "memory": server.get("memory") or read_server_memory(server.get("log")),
    }


def command_items(command: Optional[str]) -> List[str]:
    """Server command as comparable items: the program and 'option value' pairs."""
    tokens = (command or "").split()
    items: List[str] = []
    index = 0
    if tokens and not tokens[0].startswith("-"):
        items.append(tokens[0])
        index = 1
    while index < len(tokens):
        token = tokens[index]
        value = tokens[index + 1] if index + 1 < len(tokens) else None
        if token.startswith("-") and value is not None and (
                not value.startswith("-") or finite_number(value) is not None):
            items.append(f"{token} {value}")
            index += 2
        else:
            items.append(token)
            index += 1
    return items


def compare_runs(reference: Dict[str, Any], run: Dict[str, Any]) -> Dict[str, Any]:
    """Per common target_ctx: reference value, run value and change in % for each metric."""
    points = []
    for ctx in sorted(set(reference["points"]) & set(run["points"])):
        ref, cur = reference["points"][ctx], run["points"][ctx]
        point: Dict[str, Any] = {"target_ctx": ctx}
        for key, _, _, _ in COMPARE_METRICS:
            a, b = ref.get(key), cur.get(key)
            point[key] = {"reference": a, "run": b,
                          "change_pct": 100.0 * (b / a - 1.0) if a and b is not None else None}
        point["tokens_per_step"] = {"reference": ref.get("tokens_per_step_median"),
                                    "run": cur.get("tokens_per_step_median")}
        points.append(point)
    differences = []
    if reference.get("settings") and run.get("settings"):
        for key, value in reference["settings"].items():
            other = run["settings"].get(key)
            # A missing sampler dict (older runs) means server defaults: {}.
            if key == "sampling":
                value, other = value or {}, other or {}
            if value is not None and other is not None and value != other:
                differences.append({"setting": key, "reference": value, "run": other})
    ref_items, run_items = command_items(reference.get("server_command")), command_items(run.get("server_command"))
    summary = {}
    for key, _, _, _ in COMPARE_METRICS:
        changes = [p[key]["change_pct"] for p in points if p[key]["change_pct"] is not None]
        summary[key] = ({"median_change_pct": statistics.median(changes),
                         "min_change_pct": min(changes), "max_change_pct": max(changes)} if changes else None)
    return {
        "reference": reference["name"], "run": run["name"], "points": points, "summary": summary,
        "setting_differences": differences,
        "memory_differences": memory_differences(reference.get("memory"), run.get("memory")),
        "server_reference_only": [item for item in ref_items if item not in run_items],
        "server_run_only": [item for item in run_items if item not in ref_items],
        "unmatched_reference_ctx": sorted(set(reference["points"]) - set(run["points"])),
        "unmatched_run_ctx": sorted(set(run["points"]) - set(reference["points"])),
    }


def print_comparison(comparison: Dict[str, Any]) -> None:
    print(f"\nCOMPARISON {comparison['run']} vs reference {comparison['reference']}")
    if comparison["server_reference_only"] or comparison["server_run_only"]:
        print("server command: reference only: " + (" ".join(comparison["server_reference_only"]) or "-"))
        print("                run only:       " + (" ".join(comparison["server_run_only"]) or "-"))
    if comparison.get("memory_differences"):
        parts = []
        for difference in comparison["memory_differences"]:
            def show(value: Any) -> str:
                if value is None:
                    return "-"
                size, kind = value
                return f"{size:.1f}" + (f" {kind}" if kind else "")
            ref, run = difference["reference"], difference["run"]
            delta = (f" ({run[0] - ref[0]:+.1f})" if ref and run and ref[0] is not None and run[0] is not None
                     else "")
            parts.append(f"{difference['buffer']} {show(ref)} -> {show(run)}{delta}")
        print("server memory (MiB): " + "; ".join(parts))
    for difference in comparison["setting_differences"]:
        change = describe_setting_difference(difference["reference"], difference["run"])
        if difference["setting"] in WORKLOAD_SETTINGS:
            print(f"WARNING: {difference['setting']} differs ({change}); the runs measure different workloads")
        else:
            print(f"NOTE: {difference['setting']} differs ({change}); generated text may differ, "
                  "which matters for MTP/drafting")
    if not comparison["points"]:
        print("-> no common target contexts")
        return
    header = f"{'ctx':>8}"
    for _, label, unit, _ in COMPARE_METRICS:
        header += f" | {label + ' ' + unit + ' ref -> run':>24} {'change':>7}"
    print(header + f" | {'tok/step':>12}")
    for point in comparison["points"]:
        line = f"{point['target_ctx']:>8}"
        for key, _, _, _ in COMPARE_METRICS:
            value = point[key]
            digits = 1 if key != "decode_tps_median" else 2
            pair = f"{display_number(value['reference'], digits)} -> {display_number(value['run'], digits)}"
            change = f"{value['change_pct']:+.1f}%" if value["change_pct"] is not None else "n/a"
            line += f" | {pair:>24} {change:>7}"
        steps = point["tokens_per_step"]
        line += f" | {display_number(steps['reference'], 2) + ' -> ' + display_number(steps['run'], 2):>12}"
        print(line)
    parts = []
    for key, label, _, higher_better in COMPARE_METRICS:
        s = comparison["summary"][key]
        if s:
            parts.append(f"{label} {s['median_change_pct']:+.1f}% ({s['min_change_pct']:+.1f} .. "
                         f"{s['max_change_pct']:+.1f})")
    print("-> median change over common points: " + "; ".join(parts))
    if comparison["unmatched_reference_ctx"] or comparison["unmatched_run_ctx"]:
        print(f"-> not compared (only in one run): reference {comparison['unmatched_reference_ctx'] or '-'}, "
              f"run {comparison['unmatched_run_ctx'] or '-'}")


def print_comparison_matrix(comparisons: List[Dict[str, Any]]) -> None:
    """Decode change of several runs against the same reference, one column per run."""
    contexts = sorted({p["target_ctx"] for c in comparisons for p in c["points"]})
    names = [c["run"][-15:] for c in comparisons]
    print(f"\nDECODE CHANGE vs {comparisons[0]['reference']}")
    print(f"{'ctx':>8} | " + " | ".join(f"{name:>15}" for name in names))
    for ctx in contexts:
        cells = []
        for comparison in comparisons:
            point = next((p for p in comparison["points"] if p["target_ctx"] == ctx), None)
            change = point["decode_tps_median"]["change_pct"] if point else None
            cells.append(f"{change:+.1f}%" if change is not None else "")
        print(f"{ctx:>8} | " + " | ".join(f"{cell:>15}" for cell in cells))


def run_compare(paths: List[str]) -> int:
    """--compare REF RUN [RUN ...]: offline comparison of finished runs."""
    try:
        reference = load_run(paths[0])
        runs = [load_run(path) for path in paths[1:]]
    except (OSError, ValueError, csv.Error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    comparisons = [compare_runs(reference, run) for run in runs]
    for comparison in comparisons:
        print_comparison(comparison)
    if len(comparisons) > 1:
        print_comparison_matrix(comparisons)
    return 0



def main() -> None:
    epilog = """\
examples:
  # Pass one complete llama-server command line (Windows batch)
  set "LLAMA_SERVER=C:/llama.cpp/llama-server.exe --model D:/models/model.gguf --ctx-size 120064 --batch-size 2048 --ubatch-size 512 --flash-attn on"
  python ctx-cliff.py --file data/django.py ^
      --server-command "%LLAMA_SERVER%"

  # Prefix-cache sweep; recommended for locating the context/decode cliff
  python ctx-cliff.py --file data/django.py

  # Finer sweep around a suspected cliff, 5 prefill/decode samples per point
  python ctx-cliff.py --file data/django.py --start 60000 --end 100000 --step 2500 --repeat 5 --deterministic

  # Force full cold prefill for every sample
  python ctx-cliff.py --file data/war_and_peace.txt --cache-mode cold --repeat 3 --warmup 0

  # A/B runs (e.g. different GGML_* settings) with identical prompts and comparable outputs
  python ctx-cliff.py --file data/django.py --deterministic --nonce ab-test --csv

  # Force the full decode window even if the model emits EOS
  python ctx-cliff.py --file data/django.py --ignore-eos

  # Compare finished runs (any configuration) against a reference, without a server
  python ctx-cliff.py --compare outputs/reference.csv outputs/run-a.csv outputs/run-b.csv

  # Coding-agent turn: growing file excerpt in a chat conversation, same task at every point
  python ctx-cliff.py --file data/django.py --scenario agent --temperature 0.6 --top-k 20 --seed 1 --n-predict 512

  # NVIDIA clocks/power/VRAM + Windows dedicated/shared GPU memory
  python ctx-cliff.py --file data/django.py --vram-log nvidia --vram-interval-ms 250 --vram-csv outputs/nvidia-trace.csv --win-gpu-mem-csv outputs/wddm-trace.csv

notes:
  * --server-command starts a local llama-server, waits for /health, and stops
    only the process it started when the benchmark exits
  * shell operators such as &&, pipes, redirects and variable expansion inside
    --server-command are intentionally not interpreted
  * --base-url remains the health/benchmark endpoint and must use the same port
    that the complete server command starts
  * managed server output goes to logs/llama-server-YYYYMMDD-HHMMSS.log by
    default so runs do not clutter the working directory; change the directory
    with --server-log-dir, use --server-log PATH for an explicit file, or
    --server-log - to mix both outputs in the console
  * CSV headers are written before server startup; complete result rows are
    flushed immediately and raw telemetry is flushed every 250 ms
  * --csv also writes .samples.csv with each completed measurement repeat,
    including invalid samples and validation errors; warmup and prefix preparation
    are excluded. Completed repeats survive an interrupted or failed later repeat.
  * prompt budgets include model special tokens; warmup and measurements submit
    the final checked token IDs instead of retokenizing text inside completion
  * cliff detection requires --cliff-min-repeats valid repeats per phase and point
    (default 2). Use 1 explicitly for exploratory single-repeat runs. Prefill can
    remain valid after early EOS, but truncated or invalid prefill is excluded.
  * completion rejection exits with code 1; a context limit reached after completed
    points ends successfully. An interruption during the sweep exits with code 130.
  * any other failed point (connection loss, timeout, cache validation, telemetry
    failure) stops the sweep but keeps and summarizes completed points (exit code 1);
    a crashed managed llama-server is reported with its exit code and log file
  * --csv also writes <stem>.meta.json: argv, server command (API key redacted),
    GGML_/CUDA_/LLAMA_... environment, /props (build, model), GPU/driver, script and
    input hashes, prefill_mode incl. snapshot-fallback reason, final status
  * each repeat records output_sha256/output_excerpt/output_loop_pct of the generated
    text in .samples.csv; with --deterministic and the same --nonce, A/B runs can be
    compared by output_sha256. A NOTE flags degenerate (looping) output
  * missing rates (no valid decode/prefill repeat, no drafts) are empty CSV
    cells and n/a in the console, never a fake 0
  * live PCIe/GPM phase columns are provisional; orderly cleanup atomically
    replaces these traces with phase-enriched CSVs, including on exceptions
  * missing telemetry is an empty CSV cell (console: n/a); a measured zero stays zero
  * repeat quantiles are medians of per-repeat quantiles, not pooled quantiles;
    GPM means and saturation fractions use valid GPU-interval duration
  * PF/DC windows are estimated from response timings; CSV coverage and timing
    residuals expose sampling gaps but do not measure the true phase alignment error
  * raw telemetry is archived in the system temporary directory; RAM retains the
    active request plus recent history, and raw exports stream the archive
  * --csv (alias --csv-export) without PATH creates a unique timestamped name
    in --csv-dir (default: outputs); telemetry CSVs share that name's stem
  * timings come from llama-server's response; decode throughput therefore does
    not include HTTP/client overhead
  * incremental mode pins requests to --slot-id and erases that slot once before
    the sweep; later context points reuse the common prefix
  * with --repeat > 1, incremental mode prepares the exact common token prefix
    of consecutive prompts before each later point, then saves it without a
    cached decode tail and restores it before EVERY repeat; the first point
    repeats full prefill. Preparation adds one unmeasured prefix pass per point
  * this requires llama-server --slot-save-path DIR (existing writable directory);
    one unique ctx-cliff-*.bin snapshot is reused throughout the run
  * managed local snapshots are deleted on exit, including orderly interruption;
    --keep-snapshot preserves them; external/reused servers require manual deletion
  * managed launches without --slot-save-path create slot-snapshots beside this
    script and append the option automatically; explicit paths are left unchanged
  * before warmup a save probe checks support; if rejected, a warning is printed
    and only decode repeats, with prefill statistics taken from the first sample
  * prefix preparation, snapshot I/O and settling are excluded from timings and PF/DC windows;
    raw telemetry still includes this activity outside the measurement windows
  * prefill time/rate and PF telemetry use all repeats; differing cache_n/prompt_n
    abort the point instead of publishing a misleading median
  * restored token counts must match the prepared prefix, and every measured
    repeat must reuse that entire prefix; cache loss aborts the point
  * cold mode requires a successful slot erase before every repeat and sets cache_prompt=false
  * cache_n is logged explicitly instead of inferring cache behaviour from tok/s
  * cliff detection separately checks relative drops between adjacent median
    prefill and decode rates
  * STOP/EMPTY samples are shown but excluded from decode medians and cliff detection
  * --vram-log auto starts one long-lived nvidia-smi sampler when available;
    it also records clocks, P-state, power and temperature when supported
  * PCIe RX/TX uses direct NVML polling (no dmon fallback). Install NVIDIA's
    official Python bindings with: py -m pip install -U nvidia-ml-py
    https://pypi.org/project/nvidia-ml-py/
  * GPM is the primary PF/DC GPU + PCIe telemetry when supported. It records
    graphics/SM/occupancy/tensor/DRAM utilization plus interval-average PCIe RX/TX;
    --gpm-interval-ms defaults to 250 ms and must be >100 ms per NVML requirements
  * --pcie-interval-ms controls the legacy NVML 20 ms PCIe snapshot sampler; it is
    kept as a raw peak/reference trace, while live PF/DC PCIe + saturation use GPM.
    Samples are time-stamped at the centre of their RX+TX query; pcie.csv records
    pcie_query_ms and the achieved pcie_poll_interval_ms. The requested-vs-achieved
    numbers always go to .meta.json; a console NOTE only appears if the sampler
    could not keep up or is uncalibrated (--pcie-legacy-scale)
  * GPM samples with implausible SM/occupancy/tensor = 0 under load are excluded
    from SM/O/T phase statistics by default (--gpm-suspect keep restores averaging);
    graphics, DRAM and GPM PCIe are unaffected and always use all samples
  * NVML BUS utilization is sampled at 1 Hz because it already reports busy time
    over the trailing 1-second interval; PF/DC BUS summaries exclude boundary-crossing windows
  * prefill/decode telemetry windows are reconstructed from llama-server's
    prompt_ms/predicted_ms timings; raw GPM/PCIe CSVs include subphase labels
  * on Windows, --win-gpu-mem auto samples WDDM adapter Dedicated/Shared Usage;
    these counters are system-wide across GPU adapter instances and sampled at 1 s
  * the benchmark queries /slots (falling back to /props) for per-slot n_ctx,
    reserves n_predict tokens, and automatically caps --end at the safe prompt limit
  * HTTP errors include llama-server's response body instead of a bare requests traceback
  * high-frequency nvidia-smi polling can perturb performance slightly; use
    --vram-log off for a clean A/B control run and 250-500 ms for diagnostics
"""

    ap = argparse.ArgumentParser(
        description=(
            "Locate prefill- and decode-performance cliffs as prompt context grows. "
            "Measures cache/prefill, decode speed, draft acceptance, and GPU/GPM telemetry."
        ),
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g = ap.add_argument_group("server")
    g.add_argument(
        "--base-url",
        default="http://127.0.0.1:8080",
        help="llama-server base URL (default: %(default)s)",
    )
    g.add_argument(
        "--slot-id",
        type=int,
        default=0,
        help="llama-server slot to pin the benchmark to (default: %(default)s)",
    )
    g.add_argument(
        "--server-command",
        default=None,
        metavar="COMMAND",
        help="start and manage this complete llama-server command line",
    )
    g.add_argument(
        "--server-start-timeout",
        type=float,
        default=300.0,
        help="seconds to wait for a managed server to become healthy (default: %(default)s)",
    )
    g.add_argument(
        "--vram-settle-s",
        type=float,
        default=30.0,
        help=("before starting a managed server, wait up to this many seconds until GPU memory "
              "released by a previous process (e.g. the last run of a batch file) stops dropping; "
              "otherwise the new server may land partly in shared system memory. 0 disables "
              "(default: %(default)s)"),
    )
    g.add_argument(
        "--server-log",
        default=None,
        help=(
            "managed server stdout/stderr log "
            "(default: --server-log-dir/llama-server-YYYYMMDD-HHMMSS.log; '-' means console)"
        ),
    )
    g.add_argument(
        "--server-log-dir",
        default="logs",
        help="directory for automatic managed-server logs (default: %(default)s)",
    )
    g.add_argument(
        "--reuse-running-server",
        action="store_true",
        help="with a managed launch, use an already healthy server instead of failing",
    )
    g.add_argument(
        "--keep-server",
        action="store_true",
        help="leave a successfully started managed server running after the benchmark",
    )
    g.add_argument(
        "--keep-snapshot",
        action="store_true",
        help="keep this run's snapshot file (default: delete after a managed local run)",
    )

    g = ap.add_argument_group("context range")
    g.add_argument("--start", type=int, default=10000, help="starting prompt context in tokens")
    g.add_argument("--end", type=int, default=120000, help="ending prompt context in tokens")
    g.add_argument("--step", type=int, default=5000, help="context increment in tokens")

    g = ap.add_argument_group("measurement")
    g.add_argument(
        "--n-predict",
        type=int,
        default=64,
        help="tokens to generate per decode sample (default: %(default)s)",
    )
    g.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="prefill and decode samples per context point; falls back to decode-only repeats if slot snapshots are unavailable (default: %(default)s)",
    )
    g.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="warmup samples before the real sweep (default: %(default)s)",
    )
    g.add_argument(
        "--settle",
        type=float,
        default=0.25,
        help="pause after a slot erase, in seconds (default: %(default)s)",
    )
    g.add_argument(
        "--deterministic",
        action="store_true",
        help="request temperature=0/top_k=1 for more repeatable output",
    )
    g.add_argument(
        "--ignore-eos",
        action="store_true",
        help="ask llama-server to ignore EOS so samples are more likely to reach n_predict",
    )
    for key, kind in SAMPLER_OPTIONS.items():
        names = ["--" + key.replace("_", "-")] + (["--" + key] if "_" in key else [])
        names += SAMPLER_ALIASES.get(key, ())
        g.add_argument(*names, dest=key, type=kind, default=None,
                       help=f"llama-server {key} sent with every request (default: server setting)")
    g.add_argument(
        "--sampler", action="append", type=parse_sampler_setting, default=None, metavar="KEY=VALUE",
        help=("any other llama-server sampling field sent with every request, e.g. dry_multiplier=0.8 "
              "or xtc_probability=0.5; VALUE is parsed as JSON if possible; repeatable"),
    )
    g.add_argument(
        "--seed", type=int, default=None,
        help=("sampling seed; repeat r uses SEED+r-1 at every point, so sampled outputs are reproducible "
              "and comparable across A/B runs (default: server setting, usually random)"),
    )
    g.add_argument(
        "--cliff-pct",
        type=float,
        default=15.0,
        help="minimum adjacent median prefill or decode drop to flag as a cliff candidate (default: %(default)s%%)",
    )
    g.add_argument(
        "--no-stream", dest="stream", action="store_false",
        help=("do not stream completions; prefill/decode windows are then back-projected from the "
              "response end instead of anchored at the first generated token"),
    )
    g.add_argument(
        "--no-drift-check", dest="drift_check", action="store_false",
        help="skip re-measuring the first point after the sweep (drift check against heat/clock changes)",
    )
    g.add_argument(
        "--sysmem-guard", choices=("abort", "warn", "off"), default="abort",
        help=("stop the sweep (abort), print a NOTE (warn) or do nothing (off) when the prefill PCIe "
              "receive median reaches --sysmem-guard-mb-s, i.e. GPU memory overflows into shared "
              "system memory (default: %(default)s; use off for intentional CPU offload)"),
    )
    g.add_argument(
        "--sysmem-guard-mb-s", type=float, default=SYSMEM_GUARD_DEFAULT_MB_S, metavar="MB_S",
        help="prefill PCIe receive median that triggers --sysmem-guard (default: %(default)s)",
    )
    g.add_argument(
        "--abort-below-pct", type=float, default=20.0, metavar="PCT",
        help=("stop the sweep when a sample's prefill tok/s falls below PCT%% of the first point's "
              "median; 0 disables it (default: %(default)s; prefill normally declines with context, "
              "e.g. to ~40%% at 160k)"),
    )
    g.add_argument(
        "--cliff-min-repeats", type=int, default=2,
        help="minimum valid repeats per phase and point for cliff detection (default: %(default)s; use 1 for exploratory single-repeat runs)",
    )

    g = ap.add_argument_group("cache / prompt handling")
    g.add_argument(
        "--cache-mode",
        choices=("incremental", "cold"),
        default="incremental",
        help="prefix-cache behaviour (default: %(default)s)",
    )
    g.add_argument(
        "--scenario",
        choices=("file", "agent"),
        default="file",
        help=("file: the prompt is a plain prefix of --file and the model continues it (default); "
              "agent: chat-template conversation like a coding agent - the --file excerpt is a "
              "tool result that grows per point, followed by the same fixed task at every point"),
    )
    g.add_argument(
        "--agent-task",
        default=None,
        metavar="TEXT",
        help="with --scenario agent: fixed instruction after the context (default: a built-in coding task)",
    )
    g.add_argument(
        "--agent-thinking",
        choices=("auto", "on", "off"),
        default="auto",
        help=("with --scenario agent: enable_thinking for the chat template; auto keeps the template "
              "default (default: %(default)s)"),
    )
    g.add_argument(
        "--nonce",
        default=None,
        metavar="TEXT",
        help=("fixed run marker at the start of every prompt (default: random per run). "
              "Use the same value for A/B runs so prompts and, with --deterministic, "
              "generated outputs (output_sha256) are directly comparable"),
    )

    g = ap.add_argument_group("VRAM telemetry")
    g.add_argument(
        "--vram-log",
        choices=("auto", "off", "nvidia"),
        default="auto",
        help="live VRAM telemetry backend (default: %(default)s)",
    )
    g.add_argument(
        "--vram-backend",
        choices=("auto", "nvml", "nvidia-smi"),
        default="auto",
        help=("source of VRAM/clock/power/temperature samples: direct NVML (same clock as the other "
              "sensors), an nvidia-smi subprocess, or auto = NVML if pynvml works, else nvidia-smi "
              "(default: %(default)s)"),
    )
    g.add_argument(
        "--vram-interval-ms",
        type=int,
        default=250,
        help="nvidia-smi sampling interval in milliseconds (default: %(default)s)",
    )
    g.add_argument(
        "--vram-gpu",
        default="all",
        help="GPU index/UUID or comma-separated list passed to nvidia-smi -i (default: all)",
    )
    g.add_argument(
        "--vram-csv",
        default=None,
        help="save raw timestamped VRAM/utilization samples to CSV",
    )
    g.add_argument(
        "--pcie-interval-ms",
        type=int,
        default=250,
        help=("legacy NVML PCIe RX/TX polling interval in milliseconds (default: %(default)s). "
              "Each poll measures RX then TX over 20 ms each; one poll can take >100 ms, so "
              "the achieved interval is recorded per sample and reported at the end"),
    )
    g.add_argument(
        "--pcie-csv",
        default=None,
        help="save raw legacy high-resolution NVML PCIe RX/TX samples to CSV",
    )
    g.add_argument(
        "--gpm-log",
        choices=("auto", "off", "on"),
        default="auto",
        help="NVML GPM engine/PCIe telemetry (auto follows --vram-log; default: %(default)s)",
    )
    g.add_argument(
        "--gpm-interval-ms",
        type=int,
        default=250,
        help="NVML GPM interval in milliseconds; must be >100 (default: %(default)s)",
    )
    g.add_argument(
        "--gpm-lag-ms",
        type=float,
        default=70.0,
        help=("GPM values describe traffic this many ms before they are read; GPM intervals are "
              "shifted back by it before phase assignment (measured with pcie-calibrate.py; "
              "0 disables; default: %(default)s)"),
    )
    g.add_argument(
        "--pcie-legacy-scale",
        type=float,
        default=1.0,
        help=("divide legacy NVML PCIe throughput by this factor in all summaries (legacy/GPM ratio "
              "reported by pcie-calibrate.py, e.g. 1.55); raw pcie.csv stays unchanged "
              "(default: %(default)s = uncorrected)"),
    )
    g.add_argument(
        "--gpm-restart",
        choices=("off", "realloc", "reinit"),
        default="off",
        help=("EXPERIMENTAL: after a request with a GPM dropout, restart GPM sampling before the next "
              "request: 'realloc' = new sample buffers, 'reinit' = also NVML shutdown/init. Every "
              "restart is logged (NOTE, .samples.csv gpm_restart, .meta.json). Default: %(default)s"),
    )
    g.add_argument(
        "--gpm-suspect",
        choices=("exclude", "keep"),
        default="exclude",
        help=("GPM samples with implausible SM/occupancy/tensor=0 under load (>=4 adjacent "
              "samples, >=1 s, graphics >=25%%) mark the whole request; its SM/O/T zero samples are "
              "then left out of SM/O/T phase statistics ('exclude') or averaged in ('keep'; "
              "default: %(default)s). Raw GPM CSV is unchanged"),
    )
    g.add_argument(
        "--gpm-csv",
        default=None,
        help="save raw NVML GPM interval samples to CSV",
    )
    g.add_argument(
        "--win-gpu-mem",
        choices=("auto", "off", "on"),
        default="auto",
        help="Windows WDDM dedicated/shared GPU-memory telemetry (default: %(default)s)",
    )
    g.add_argument(
        "--win-gpu-mem-interval-ms",
        type=int,
        default=1000,
        help="Windows GPU-memory sampling interval; >=1000 ms recommended (default: %(default)s)",
    )
    g.add_argument(
        "--win-gpu-mem-csv",
        default=None,
        help="save raw Windows dedicated/shared GPU-memory samples to CSV",
    )

    g = ap.add_argument_group("input / output")
    g.add_argument("--file", default=None, help="large UTF-8 test file (required unless --compare)")
    g.add_argument(
        "--reference", default=None, metavar="CSV",
        help=("result CSV of an earlier run (any configuration); after the sweep, prefill, decode and "
              "decode step cost are compared per common target context"),
    )
    g.add_argument(
        "--compare", nargs="+", default=None, metavar="CSV",
        help=("only compare finished runs, without a server: REFERENCE.csv RUN.csv [RUN.csv ...]; "
              "all other options are ignored"),
    )
    g.add_argument(
        "--csv", "--csv-export", dest="csv", nargs="?", const="", default=None,
        metavar="PATH",
        help="stream CSV results; omit PATH for a timestamped file in --csv-dir",
    )
    g.add_argument(
        "--csv-dir", default="outputs",
        help="directory for automatically named CSVs (default: %(default)s); explicit PATH takes precedence",
    )

    args = ap.parse_args()

    if args.compare is not None:
        if len(args.compare) < 2:
            ap.error("--compare needs a reference CSV and at least one run CSV")
        sys.exit(run_compare(args.compare))
    if not args.file:
        ap.error("--file is required")
    if args.reference is not None:
        try:
            load_run(args.reference)  # fail now, not after a long sweep
        except (OSError, ValueError, csv.Error) as error:
            ap.error(f"--reference: {error}")
    if args.start <= 0 or args.end < args.start or args.step <= 0:
        ap.error("require 0 < --start <= --end and --step > 0")
    if args.n_predict <= 0:
        ap.error("--n-predict must be > 0")
    if args.repeat <= 0:
        ap.error("--repeat must be > 0")
    if args.warmup < 0:
        ap.error("--warmup must be >= 0")
    if args.cliff_pct < 0:
        ap.error("--cliff-pct must be >= 0")
    if args.cliff_min_repeats < 1:
        ap.error("--cliff-min-repeats must be >= 1")
    if args.vram_interval_ms < 100:
        ap.error("--vram-interval-ms must be >= 100")
    if args.pcie_interval_ms < 20:
        ap.error("--pcie-interval-ms must be >= 20 (NVML throughput uses a 20 ms internal window)")
    if args.gpm_interval_ms <= 100:
        ap.error("--gpm-interval-ms must be > 100 (NVML GPM requires samples >100 ms apart)")
    if args.win_gpu_mem_interval_ms < 1000:
        ap.error("--win-gpu-mem-interval-ms must be >= 1000")
    if args.server_command is not None and not args.server_command.strip():
        ap.error("--server-command must not be empty")
    if args.vram_settle_s < 0:
        ap.error("--vram-settle-s must be >= 0")
    if args.server_start_timeout <= 0:
        ap.error("--server-start-timeout must be > 0")
    if not 0 <= args.gpm_lag_ms <= 1000:
        ap.error("--gpm-lag-ms must be between 0 and 1000")
    if not 0.1 <= args.pcie_legacy_scale <= 10:
        ap.error("--pcie-legacy-scale must be between 0.1 and 10")
    if args.nonce is not None and (not args.nonce.strip() or len(args.nonce) > 64
                                   or not args.nonce.isprintable()):
        ap.error("--nonce must be 1-64 printable characters")
    if args.scenario != "agent" and (args.agent_task is not None or args.agent_thinking != "auto"):
        ap.error("--agent-task/--agent-thinking require --scenario agent")
    if args.agent_task is not None and not args.agent_task.strip():
        ap.error("--agent-task must not be empty")
    args.sampler = dict(args.sampler or [])
    if args.sysmem_guard_mb_s <= 0:
        ap.error("--sysmem-guard-mb-s must be > 0")
    if not 0 <= args.abort_below_pct < 100:
        ap.error("--abort-below-pct must be >= 0 and < 100 (0 disables it)")
    if args.deterministic and any(getattr(args, key) is not None for key in DETERMINISTIC_CONFLICTS):
        ap.error("--deterministic sets temperature=0/top_k=1; do not combine it with "
                 "--temperature/--top-p/--top-k/--min-p/--typical-p")

    with ExitStack() as resources:
        resources.enter_context(termination_handler())
        if args.csv == "":
            args.csv = allocate_csv_path(args.csv_dir)
        prepare_server_log(args)
        recording = resources.enter_context(CsvRecording(args))
        run_benchmark(args, ap, resources, recording)


def print_pcie_timing(summary: Optional[Dict[str, Any]]) -> None:
    """Flag legacy PCIe sampler issues (stdout, with the summary).

    "Legacy" is the direct-NVML RX/TX sampler announced at startup as
    "PCIe telemetry: direct NVML RX/TX ..."; it is separate from GPM. The
    full requested-vs-achieved numbers always go to .meta.json regardless
    of whether anything is printed here; this stays quiet when the sampler
    kept up, since a matching requested/achieved interval is the expected,
    unremarkable case.
    """
    if not summary or summary.get("achieved_interval_median_ms") is None:
        return
    requested = summary["requested_interval_ms"]
    achieved = summary["achieved_interval_median_ms"]
    if achieved > 1.2 * requested:
        print(f"\nNOTE: the legacy PCIe sampler (the direct-NVML RX/TX counters from 'PCIe telemetry' "
              f"above, distinct from GPM) could not keep up: requested {requested} ms, achieved median "
              f"{achieved:.0f} ms (p90 {summary['achieved_interval_p90_ms']:.0f} ms); RX+TX query median "
              f"{summary['rx_tx_query_median_ms']:.0f} ms; only ~{summary['time_coverage_pct_per_direction']:.0f}% "
              "of the time is captured per direction (full numbers in .meta.json)")
        if achieved > 1.5 * requested:
            print(f"-> the requested interval was not reached; use --pcie-interval-ms "
                  f"{int(math.ceil(achieved / 10.0)) * 10} or larger for an evenly spaced trace")
    if summary.get("scale", 1.0) == 1.0:
        print("\nNOTE: the legacy PCIe sampler is uncalibrated on this GPU; its raw MB/s and saturation "
              "values read higher than the actual transfer rate (~1.55x GPM on the tested RTX 5060 Ti). "
              "Measure with pcie-calibrate.py and pass --pcie-legacy-scale to correct them.")


def prepare_server_log(args: Any) -> None:
    # Preserve managed-server logs in a dedicated directory by default instead
    # of filling the benchmark working directory. Explicit --server-log PATH
    # (including '-') is always respected.
    if args.server_command and args.server_log is None:
        log_dir = os.path.abspath(args.server_log_dir)
        os.makedirs(log_dir, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        candidate = os.path.join(log_dir, f"llama-server-{stamp}.log")
        if os.path.exists(candidate):
            stem, ext = os.path.splitext(candidate)
            suffix = 1
            while os.path.exists(f"{stem}-{suffix}{ext}"):
                suffix += 1
            candidate = f"{stem}-{suffix}{ext}"
        args.server_log = candidate


def run_benchmark(args: Any, ap: Any, resources: ExitStack, recording: CsvRecording) -> None:
    http = resources.enter_context(requests.Session())
    # Registered before the server/monitors so cleanup runs after they stop.
    snapshot_cleanup = resources.enter_context(ExitStack())
    prepare_server_log(args)
    base = args.base_url.rstrip("/")
    managed_server: Optional[ManagedLlamaServer] = None
    # Registered before the server so its exit hook records the final outcome
    # after the managed server has been stopped.
    meta = RunMetadata(getattr(recording, "meta_path", None), args)
    resources.push(meta.exit_hook)
    reused_running_server = False
    vram_settle: Optional[Dict[str, Any]] = None
    server_memory: Optional[Dict[str, Any]] = None

    if args.server_command:
        parsed_base = urlparse(base)
        server_host = parsed_base.hostname
        if parsed_base.scheme != "http" or not server_host:
            ap.error("a managed llama-server requires an absolute http --base-url")
        if server_host.lower() not in {"localhost", "127.0.0.1", "::1"}:
            ap.error("managed server launch may only be used with a local --base-url")

        if server_is_ready(base, http=http):
            if not args.reuse_running_server:
                print(
                    f"ERROR: a server is already healthy at {base}; refusing to ignore the "
                    "requested server command. Stop it or add --reuse-running-server.",
                    file=sys.stderr,
                )
                sys.exit(2)
            print(
                f"using already running server at {base}; --server-command was not started",
                file=sys.stderr,
            )
            reused_running_server = True
        else:
            vram_settle = settle_vram_before_server_start(args)
            managed_server = ManagedLlamaServer(
                command=prepare_snapshot_command(args.server_command, args),
                base=base,
                startup_timeout=args.server_start_timeout,
                log_path=args.server_log,
                http=http,
                kill_with_script=not args.keep_server,
            )
            if not args.keep_server:
                resources.callback(managed_server.stop)
            try:
                managed_server.start()
            except Exception as e:
                print(f"ERROR: could not start llama-server: {e}", file=sys.stderr)
                sys.exit(1)
            server_memory = read_server_memory(args.server_log)
            if server_memory:
                print(f"server memory (MiB): {describe_server_memory(server_memory)}", file=sys.stderr)
    elif not server_is_ready(base, timeout=5.0, http=http):
        print(f"ERROR: server unavailable at {base}", file=sys.stderr)
        sys.exit(1)

    if meta.path is not None:
        meta.update(server={
            "base_url": base,
            "managed": managed_server is not None,
            "reused_running_server": reused_running_server,
            "vram_settle": vram_settle,
            "memory": server_memory,
            "requested_command": redact_command(args.server_command) if args.server_command else None,
            "command": redact_command(managed_server.command) if managed_server is not None else None,
            "log": (os.path.abspath(args.server_log)
                    if managed_server is not None and args.server_log and args.server_log != "-" else None),
            "environment_scope": ("benchmark process, inherited by the managed server"
                                  if managed_server is not None else
                                  "benchmark process only; the server may run with a different environment"),
            "props": fetch_server_props(base, http),
        })

    try:
        with open(args.file, "r", encoding="utf-8") as f:
            file_content = f.read()
    except Exception as e:
        print(f"ERROR while reading {args.file}: {e}", file=sys.stderr)
        sys.exit(1)

    if not file_content:
        print(f"ERROR: {args.file} is empty", file=sys.stderr)
        sys.exit(1)

    # A random run marker keeps prompts unique per run. A fixed --nonce makes
    # prompts (and, with --deterministic, generated text) comparable across A/B runs.
    nonce_value = getattr(args, "nonce", None) or secrets.token_hex(8)
    nonce = f"[ctx-cliff run={nonce_value}] "
    scenario = getattr(args, "scenario", "file")
    prompt_prefix, suffix_tokens, add_bos = nonce, [], True
    agent_meta: Optional[Dict[str, Any]] = None
    if scenario == "agent":
        task = getattr(args, "agent_task", None) or AGENT_DEFAULT_TASK
        try:
            prompt_prefix, suffix = agent_prompt_parts(base, nonce.strip(), task,
                                                       getattr(args, "agent_thinking", "auto"), http=http)
        except (requests.RequestException, KeyError, TypeError, ValueError) as e:
            print(f"ERROR: --scenario agent could not render the chat template via /apply-template: {e}",
                  file=sys.stderr)
            sys.exit(1)
        # A template that renders BOS itself must not get a second one from add_special.
        with_bos = tokenize(base, prompt_prefix, add_bos=True, http=http)
        without_bos = tokenize(base, prompt_prefix, add_bos=False, http=http)
        add_bos = not (len(with_bos) == len(without_bos) + 1 and len(with_bos) > 1 and with_bos[0] == with_bos[1])
        suffix_tokens = tokenize(base, suffix, add_bos=False, http=http)
        agent_meta = {"task": task, "thinking": getattr(args, "agent_thinking", "auto"),
                      "prefix_text": prompt_prefix, "suffix_text": suffix,
                      "suffix_tokens": len(suffix_tokens), "add_bos": add_bos}
    nonce_tokens = len(tokenize(base, prompt_prefix, add_bos=add_bos, http=http))
    chunk = file_content[:10000]
    chunk_tokens = len(tokenize(base, chunk, add_bos=False, http=http))
    chars_per_token = len(chunk) / max(1, chunk_tokens)

    print(
        f"input loaded: {args.file} ({len(file_content)} chars) | "
        f"estimate {chars_per_token:.2f} chars/token | "
        + (f"agent prefix {nonce_tokens} tok, suffix {len(suffix_tokens)} tok" if scenario == "agent"
           else f"nonce {nonce_tokens} tok"),
        file=sys.stderr,
    )
    print(
        f"test mode={args.cache_mode} slot={args.slot_id} repeat={args.repeat} "
        f"n_predict={args.n_predict} deterministic={args.deterministic} "
        f"ignore_eos={args.ignore_eos} scenario={scenario}"
        + "".join(f" {key}={value}" for key, value in sampling_payload(args, 0).items()),
        file=sys.stderr,
    )

    server_n_ctx = detect_slot_n_ctx(base, args.slot_id, http=http)
    max_prompt_ctx: Optional[int] = None
    if server_n_ctx:
        # Reserve the requested decode window so every valid sample can actually
        # produce n_predict tokens without running off the end of the slot. The
        # extra token: llama-server flags `truncated` once prompt + generated
        # tokens reach n_ctx, even if the n_predict-th token was still produced.
        max_prompt_ctx = server_n_ctx - args.n_predict - 1
        if max_prompt_ctx <= 0:
            ap.error("--n-predict leaves no prompt budget in the server context")
        print(
            f"server context: slot {args.slot_id} n_ctx={server_n_ctx}; "
            f"max benchmark prompt={max_prompt_ctx} (reserving {args.n_predict} decode tokens)",
            file=sys.stderr,
        )
        if args.start > max_prompt_ctx:
            print(
                f"ERROR: --start {args.start} exceeds safe prompt limit {max_prompt_ctx} "
                f"for slot n_ctx={server_n_ctx}",
                file=sys.stderr,
            )
            sys.exit(2)
        if args.end > max_prompt_ctx:
            print(
                f"NOTE: --end {args.end} exceeds the slot limit; sweep will stop at "
                f"{max_prompt_ctx} prompt tokens.",
                file=sys.stderr,
            )

    prompt_meta = {
        "nonce": nonce, "nonce_fixed": bool(getattr(args, "nonce", None)),
        "nonce_tokens": nonce_tokens, "chars_per_token_estimate": round(chars_per_token, 4),
        "input_chars": len(file_content), "server_slot_n_ctx": server_n_ctx,
        "max_prompt_ctx": max_prompt_ctx,
        "scenario": scenario, "agent": agent_meta, "sampling_first_repeat": sampling_payload(args, 0),
    }
    meta.update(prompt=prompt_meta)

    vram_monitor: Optional[NvidiaVramMonitor] = None
    if args.vram_log != "off":
        candidate = NvidiaVramMonitor(args.vram_interval_ms, args.vram_gpu, args.pcie_interval_ms)
        candidate.pcie_scale = getattr(args, "pcie_legacy_scale", 1.0)
        candidate.backend_preference = getattr(args, "vram_backend", "auto")
        resources.callback(candidate.stop)
        recording.attach("vram", candidate, "samples")
        recording.attach("pcie", candidate, "pcie_samples", candidate.write_pcie_csv)
        if candidate.start():
            vram_monitor = candidate
            recording.activate(candidate)
            print(
                f"VRAM telemetry: NVIDIA via {candidate.backend} every {args.vram_interval_ms} ms "
                f"(gpu={args.vram_gpu})",
                file=sys.stderr,
            )
            if candidate.pcie_source == "nvml":
                print(
                    f"PCIe telemetry: direct NVML RX/TX every {args.pcie_interval_ms} ms requested "
                    "(NVML counter window = 20 ms per direction)",
                    file=sys.stderr,
                )
                probe_ms = candidate.pcie_probe_query_ms
                if probe_ms is not None and probe_ms * len(candidate._nvml_handles) > args.pcie_interval_ms:
                    print(
                        f"NOTE: one RX+TX PCIe query took {probe_ms:.0f} ms; the legacy sampler "
                        f"cannot reach {args.pcie_interval_ms} ms and will poll back-to-back. "
                        "The achieved interval is recorded per sample and summarized at the end.",
                        file=sys.stderr,
                    )
                if candidate.bus_supported:
                    print(
                        "PCIe BUS telemetry: NVML busy-time every 1000 ms "
                        "(metric window = trailing 1 s)",
                        file=sys.stderr,
                    )
                elif candidate.bus_error:
                    print(
                        f"PCIe BUS telemetry: unavailable ({candidate.bus_error}); RX/TX continues",
                        file=sys.stderr,
                    )
            else:
                print(
                    f"PCIe telemetry: unavailable ({candidate.pcie_error or 'unknown reason'}); "
                    "normal VRAM telemetry continues",
                    file=sys.stderr,
                )
                if candidate.pcie_error and "not installed" in candidate.pcie_error:
                    print(
                        "NVML install: py -m pip install -U nvidia-ml-py  "
                        "https://pypi.org/project/nvidia-ml-py/",
                        file=sys.stderr,
                    )
            host = (urlparse(base).hostname or "").lower()
            if host not in {"", "localhost", "127.0.0.1", "::1"}:
                print(
                    f"WARNING: llama-server is at {host}; VRAM telemetry samples the machine "
                    "running this benchmark, not the remote server.",
                    file=sys.stderr,
                )
        elif args.vram_log == "nvidia":
            print(f"ERROR: NVIDIA VRAM telemetry unavailable: {candidate.error}", file=sys.stderr)
            sys.exit(2)
        else:
            print(f"VRAM telemetry: disabled ({candidate.error})", file=sys.stderr)

    gpm_monitor: Optional[NvidiaGpmMonitor] = None
    gpm_should_start = args.gpm_log == "on" or (args.gpm_log == "auto" and args.vram_log != "off")
    if gpm_should_start:
        gpm_candidate = NvidiaGpmMonitor(args.gpm_interval_ms, args.vram_gpu)
        gpm_candidate.exclude_suspect = getattr(args, "gpm_suspect", "exclude") != "keep"
        gpm_candidate.lag_s = getattr(args, "gpm_lag_ms", 0.0) / 1000.0
        if vram_monitor is not None:
            gpm_candidate.activity = vram_monitor.activity_evidence
        resources.callback(gpm_candidate.stop)
        recording.attach("gpm", gpm_candidate, "samples", gpm_candidate.write_csv)
        if gpm_candidate.start():
            gpm_monitor = gpm_candidate
            recording.activate(gpm_candidate)
            print(
                f"GPM telemetry: NVML every {args.gpm_interval_ms} ms "
                "(primary PF/DC SM/occupancy/tensor/DRAM + PCIe interval metrics)",
                file=sys.stderr,
            )
        elif args.gpm_log == "on":
            print(f"ERROR: GPM telemetry unavailable: {gpm_candidate.error}", file=sys.stderr)
            sys.exit(2)
        else:
            print(f"GPM telemetry: unavailable ({gpm_candidate.error}); legacy telemetry continues", file=sys.stderr)

    win_gpu_monitor: Optional[WindowsGpuMemoryMonitor] = None
    if args.win_gpu_mem != "off":
        if sys.platform == "win32":
            candidate_win = WindowsGpuMemoryMonitor(args.win_gpu_mem_interval_ms)
            resources.callback(candidate_win.stop)
            recording.attach("windows", candidate_win, "samples")
            if candidate_win.start():
                win_gpu_monitor = candidate_win
                recording.activate(candidate_win)
                print(
                    f"Windows GPU memory: WDDM/PDH dedicated+shared every "
                    f"{args.win_gpu_mem_interval_ms} ms (all adapter instances)",
                    file=sys.stderr,
                )
            elif args.win_gpu_mem == "on":
                print(f"ERROR: Windows GPU-memory telemetry unavailable: {candidate_win.error}", file=sys.stderr)
                sys.exit(2)
            else:
                print(f"Windows GPU memory: disabled ({candidate_win.error})", file=sys.stderr)
        elif args.win_gpu_mem == "on":
            print("ERROR: --win-gpu-mem on requires Windows", file=sys.stderr)
            sys.exit(2)

    if meta.path is not None:
        meta.update(telemetry={
            "nvidia_smi": vram_monitor is not None,
            "vram_backend": vram_monitor.backend if vram_monitor is not None else None,
            "nvml_pcie": vram_monitor is not None and vram_monitor.pcie_source == "nvml",
            "nvml_bus": vram_monitor is not None and vram_monitor.bus_supported,
            "gpm": gpm_monitor is not None,
            "windows_pdh": win_gpu_monitor is not None,
        }, gpu=(collect_gpu_info() if args.vram_log != "off" or args.gpm_log == "on" else None))

    effective_end = min(args.end, max_prompt_ctx) if max_prompt_ctx else args.end
    ctxs = list(range(args.start, effective_end + 1, args.step))
    # If the regular step grid does not land near the safe end, include the exact
    # final point. This is useful when e.g. n_ctx=102400 and --step=5000.
    if ctxs and ctxs[-1] < effective_end and (effective_end - ctxs[-1]) >= max(128, args.step // 5):
        ctxs.append(effective_end)

    estimated_file_tokens = int(len(file_content) / max(chars_per_token, 1e-9)) + nonce_tokens + len(suffix_tokens)
    if ctxs and estimated_file_tokens < ctxs[-1]:
        reachable = [c for c in ctxs if c <= estimated_file_tokens]
        print(f"WARNING: {args.file} holds only about {estimated_file_tokens} tokens, but the sweep goes to "
              f"{ctxs[-1]}; points above that cannot grow and the sweep stops when the input is exhausted "
              f"({len(reachable)} of {len(ctxs)} points reachable). Use a larger input file.", file=sys.stderr)
    meta.update(input_estimated_tokens=estimated_file_tokens)

    builder = PromptBuilder(file_content, prompt_prefix,
                            lambda prompt: tokenize(base, prompt, add_bos=add_bos, http=http),
                            chars_per_token, nonce_tokens, max_prompt_ctx,
                            detokenize=lambda ids: detokenize(base, ids, http=http),
                            suffix_tokens=suffix_tokens)
    runner = BenchmarkRunner(args, base, http, builder, recording, vram_monitor, gpm_monitor, win_gpu_monitor)
    runner.vram_before_start_mib = vram_settle["final_mib"] if vram_settle else None
    uses_snapshots = args.cache_mode == "incremental" and args.repeat > 1
    if uses_snapshots and not getattr(args, "keep_snapshot", False):
        directory = snapshot_directory_from_command(managed_server.command) if managed_server is not None else None
        if directory is not None:
            # Register before the probe, including cleanup of partially written saves.
            snapshot_cleanup.callback(cleanup_snapshot, directory, runner.snapshot_filename)
        else:
            print(f"NOTE: automatic snapshot deletion unavailable for this server; "
                  f"if created, remove {runner.snapshot_filename} from its slot-save directory manually. "
                  "The server API has no file-delete action.", file=sys.stderr)
    probe_prefill_repeats(args, base, runner.snapshot_filename, http=http)
    meta.update(measurement={
        "prefill_mode": prefill_mode(args),
        "prefill_repeat_enabled": getattr(args, "prefill_repeat_enabled", None),
        "prefill_repeat_error": getattr(args, "prefill_repeat_error", None),
        "snapshot_filename": runner.snapshot_filename if uses_snapshots else None,
        "planned_contexts": ctxs,
    })

    def abort_before_results(error: BaseException, ctx: int, stage: str) -> None:
        """Diagnose a failure before any result row exists, record it and exit 1."""
        failure = report_point_failure(error, ctx, managed_server, args.server_log)
        meta.update(status="failed", stop_reason=f"{stage}_failed", completed_points=0,
                    exit_code=1, **failure)
        sys.exit(1)

    # Warm up separately. Besides warming clocks/caches, use the already-paid-for
    # warmup response to detect whether speculative/MTP drafting is actually active.
    # No extra request is needed in the normal (--warmup >= 1) case.
    drafting_on = False
    if args.warmup:
        try:
            warm_prompt, _, _ = runner.build_prompt(ctxs[0])
            print(f"warmup: {args.warmup} sample(s) @ ~{ctxs[0]} tok", file=sys.stderr)
            for i in range(args.warmup):
                if args.cache_mode == "incremental" and i == 0:
                    reset_slot(base, args.slot_id, quiet=True, http=http)
                warm_sample = runner.take_sample(warm_prompt, ctxs[0], i, phase="warmup")
                if message := runner.check_sysmem_fallback(warm_sample, ctxs[0], i):
                    raise SysmemFallbackError(message)
                if int(warm_sample.get("draft_n", 0) or 0) > 0:
                    drafting_on = True
        except Exception as e:
            print("ERROR: warmup failed; no measurement was started.", file=sys.stderr)
            abort_before_results(e, ctxs[0], "warmup")

    # Best-effort reset before the sweep. BenchmarkRunner also disables reuse for
    # the first real request, so a retained warmup checkpoint cannot shrink prefill.
    if args.cache_mode == "incremental":
        reset_slot(base, args.slot_id, http=http)
        if args.settle:
            time.sleep(args.settle)

    # Reset density estimate too; warmup should not influence target sizing.
    builder.reset()

    results: List[Dict[str, Any]] = []

    # If warmup was explicitly disabled, use the first real point as the drafting probe
    # before printing the table header. This avoids an extra request and still lets
    # us omit the draft column when drafting is inactive.
    first_row: Optional[Dict[str, Any]] = None
    first_ctx: Optional[int] = None
    if not args.warmup and ctxs:
        first_ctx = ctxs[0]
        try:
            first_row = runner.measure_point(first_ctx)
        except Exception as e:
            abort_before_results(e, first_ctx, "first_point")
        if first_row is not None and int(first_row.get("draft_n", 0) or 0) > 0:
            drafting_on = True
    meta.update(drafting_active=drafting_on)

    # Print one compact live table. GPM is primary for PF/DC PCIe throughput,
    # saturation and GPU-engine metrics. Legacy NVML PCIe remains available in
    # the raw/aggregate CSV as a short-window peak/reference sensor.
    draft_header_1 = f" | {'draft':>6} | {'step':>6}" if drafting_on else ""
    draft_header_2 = f" | {'%':>6} | {'ms':>6}" if drafting_on else ""
    header_1 = (
        f"{'ctx':>8} | {'new':>7} | {'prefill':>8} | {'decode':>7}{draft_header_1} | {'free':>6} | "
        f"{'power':>5} | {'PF PCIe':>14} | {'PF sat':>6} | {'PF BUS':>6} | {'PF GPU':>13} | "
        f"{'DC PCIe':>14} | {'DC sat':>6} | {'DC BUS':>6} | {'DC GPU':>13} | {'status':>9}"
    )
    header_2 = (
        f"{'tok':>8} | {'tok':>7} | {'tok/s':>8} | {'tok/s':>7}{draft_header_2} | {'MiB':>6} | "
        f"{'W':>5} | {'p95 R/T MiB/s':>14} | {'>90%':>6} | {'avg %':>6} | {'SM/O/T/D %':>13} | "
        f"{'p95 R/T MiB/s':>14} | {'>90%':>6} | {'avg %':>6} | {'SM/O/T/D %':>13} | {'':>9}"
    )
    table_width = max(len(header_1), len(header_2))
    print("\nPF = prefill, DC = decode, R/T = PCIe receive/transmit")
    print("GPM PCIe = median of repeat p95s; sat = valid GPU-interval time >=90% link rate; BUS = busy time (1 s)")
    print("PF/DC windows are reconstructed estimates; CSV includes coverage, validity and fallback diagnostics")
    print("GPU = GPM SM utilization / SM occupancy / tensor utilization / DRAM bandwidth utilization")
    if drafting_on:
        print("draft % = accepted draft tokens (MTP, DFlash, draft model, ...); step ms = decode cost per verification step, "
              "independent of how predictable the generated text is")
    if getattr(args, "gpm_suspect", "exclude") == "keep":
        print("GPU * = suspect SM/O/T counters in at least one repeat; averages retain these values; benchmark status is separate")
    else:
        print("GPU * = suspect SM/O/T=0 dropouts found and excluded from SM/O/T (n/a if nothing clean remains); "
              "graphics/DRAM/PCIe unaffected")
    print("GPU ! = complete GPM dropout (all values 0 while legacy PCIe/nvidia-smi showed activity); "
          "those intervals are excluded from all GPM columns")
    print("=" * table_width)
    print(header_1)
    print(header_2)
    print("-" * table_width)



    exit_code = 0
    stop_reason = "completed"
    failure: Dict[str, Any] = {}
    # NOTEs from the sweep go below the table instead of between its rows.
    runner.deferred_notes = []
    try:
        for idx, ctx in enumerate(ctxs):
            if idx == 0 and first_row is not None:
                row = first_row
            else:
                try:
                    row = runner.measure_point(ctx)
                except InputExhausted as e:
                    print(f"\nInput file exhausted: {e}; stopping the sweep without measuring it.",
                          file=sys.stderr)
                    stop_reason = "input_exhausted"
                    break
                except CompletionRequestError as e:
                    failure = report_point_failure(e, ctx, managed_server, args.server_log)
                    if e.is_context_overflow and "server_exit_code" not in failure:
                        exit_code = 0 if results else 1
                        stop_reason = "context_limit"
                        print(
                            "Reached the server context limit; stopping the sweep cleanly "
                            "and reporting completed points.",
                            file=sys.stderr,
                        )
                    else:
                        exit_code, stop_reason = 1, "failed"
                        print(
                            "Stopping the sweep after the server rejected the completion request. "
                            "The response above is the actual llama-server error.",
                            file=sys.stderr,
                        )
                    break
                except GuardAbortError as e:
                    failure = report_point_failure(e, ctx, managed_server, args.server_log)
                    exit_code, stop_reason = 1, e.stop_reason
                    print("Stopping the sweep; completed points are kept and summarized below.",
                          file=sys.stderr)
                    break
                except Exception as e:
                    # One failed point (lost connection, server crash, cache validation,
                    # telemetry failure, ...) must not discard the completed points.
                    failure = report_point_failure(e, ctx, managed_server, args.server_log)
                    exit_code, stop_reason = 1, "failed"
                    print("Stopping the sweep; completed points are kept and summarized below.",
                          file=sys.stderr)
                    break

            results.append(row)
            print_live_row(row, drafting_on)
    except KeyboardInterrupt:
        print("\nInterrupted; reporting completed points.", file=sys.stderr)
        exit_code, stop_reason = 130, "interrupted"
    if results:
        # Close the live table before anything else (drift check, reports) is printed.
        print("=" * table_width, flush=True)
    sweep_notes, runner.deferred_notes = runner.deferred_notes, None
    if sweep_notes:
        print(f"\nNOTES during the sweep ({len(sweep_notes)}):")
        for message in sweep_notes:
            print("  - " + re.sub(r"^NOTE:?\s*", "", message))
        meta.update(sweep_notes=sweep_notes)

    run_status = {"completed": "completed", "context_limit": "completed_context_limit",
                  "input_exhausted": "completed_input_exhausted",
                  "interrupted": "interrupted"}.get(stop_reason, "failed")
    if exit_code == 1:
        run_status = {"sysmem_fallback": "stopped_sysmem_fallback",
                      "prefill_floor": "stopped_prefill_floor"}.get(stop_reason, "failed")
    drift: Optional[Dict[str, Any]] = None
    if getattr(args, "drift_check", True) and len(results) >= 2 and stop_reason in (
            "completed", "context_limit", "input_exhausted"):
        first = results[0]
        print(f"\ndrift check: re-measuring the first point (target={first['target_ctx']}) ...", flush=True)
        try:
            drift = compare_drift(first, runner.remeasure_first_point(int(first["target_ctx"])))
        except KeyboardInterrupt:
            print("drift check interrupted", file=sys.stderr)
        except Exception as e:
            drift = {"error": f"{type(e).__name__}: {e}"}
            report_point_failure(e, int(first["target_ctx"]), managed_server, args.server_log)
        print_drift(drift)
    # All measurements are done: free the GPU now instead of after the reports
    # (the next run of a batch file can start sooner).
    if managed_server is not None and not getattr(args, "keep_server", False):
        managed_server.stop()

    pcie_timing = vram_monitor.pcie_timing_summary() if vram_monitor is not None else None
    meta.update(status=run_status, stop_reason=stop_reason, exit_code=exit_code,
                completed_points=len(results),
                last_total_ctx=results[-1]["total_ctx"] if results else None,
                legacy_pcie_sampler=pcie_timing, drift_check=drift,
                gpm_restart_events=(gpm_monitor.restart_events if gpm_monitor is not None else []),
                **failure)

    if not results:
        print_pcie_timing(pcie_timing)
        if exit_code:
            sys.exit(exit_code)
        return

    # Relative cliff detection on adjacent valid aggregate points. Keep the two
    # phases separate: a prefill discontinuity must not hide a decode one (or
    # vice versa).
    cliffs = {}
    phases = [("PREFILL", "prefill_tps", results), ("DECODE", "decode_tps_median", results)]
    if drafting_on:
        # Context effect on decode without the text-dependent tokens per step.
        phases.append(("DECODE STEP RATE", "decode_step_rate", step_rate_rows(results)))
    for phase, rate_key, rows in phases:
        analysis = analyze_cliffs(rows, rate_key, args.cliff_min_repeats, args.cliff_pct)
        print_cliff_report(phase, analysis, args)
        cliffs[phase.lower().replace(" ", "_")] = cliff_summary(analysis)
    meta.update(cliffs=cliffs)

    print_pcie_timing(pcie_timing)
    if prefill_mode(args) == "first_repeat_only":
        print("\nNOTE: slot snapshots were unavailable (prefill_mode=first_repeat_only); prefill "
              "statistics come from the first repeat of each point only. Reason: "
              f"{getattr(args, 'prefill_repeat_error', None) or 'unknown'}")
    if getattr(args, "reference", None):
        try:
            current = {
                "name": os.path.basename(args.csv[:-4]) if args.csv else "this run",
                "points": comparison_points(results),
                "settings": run_settings(vars(args), prompt_meta,
                                         (meta.data.get("input_file") or {}).get("sha256")
                                         or file_fingerprint(args.file).get("sha256")),
                "server_command": (redact_command(managed_server.command) if managed_server is not None
                                   else getattr(args, "server_command", None)),
                "memory": server_memory,
            }
            comparison = compare_runs(load_run(args.reference), current)
            print_comparison(comparison)
            meta.update(comparison=comparison)
        except (OSError, ValueError, csv.Error) as error:
            print(f"\nWARNING: comparison with --reference {args.reference} failed: {error}")
    if failure and stop_reason != "context_limit":
        print(f"\nRUN ENDED EARLY at target={failure.get('failed_target_ctx')}: "
              f"{failure.get('error')}"
              + (f" (llama-server exit code {format_exit_code(failure['server_exit_code'])})"
                 if failure.get("server_exit_code") is not None else ""))
        print(f"   {len(results)} of {len(ctxs)} planned points completed.")

    if exit_code:
        sys.exit(exit_code)



if __name__ == "__main__":
    main()
