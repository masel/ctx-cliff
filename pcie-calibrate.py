#!/usr/bin/env python3
# Copyright 2026 masel
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
pcie-calibrate.py — check the NVML PCIe throughput sensors against known transfers.

ctx-cliff records PCIe traffic from two NVIDIA sensors that disagree in real runs:

  legacy  nvmlDeviceGetPcieThroughput: RX, then TX, each a 20 ms snapshot (KB/s)
  GPM     NVML_GPM_METRIC_PCIE_RX/TX_PER_SEC: gap-free interval average (MiB/s)

This script moves a known number of bytes between host and GPU with the CUDA
driver API (nvcuda.dll / libcuda, installed with every NVIDIA driver; no PyTorch
needed) while both sensors run, then compares each sensor with the true rate:

  idle          baseline background traffic
  h2d / d2h     steady pinned-memory copies (host->device = GPU RX, device->host = GPU TX)
  h2d_bursty /  short copies with pauses, similar to bursty inference traffic;
  d2h_bursty    shows whether 20 ms snapshots are biased for bursty traffic
  h2d_tiny /    back-to-back 64-byte copies: almost no payload, but lots of small
  d2h_tiny      PCIe transactions (copy setup, doorbells, completions), similar to
                the control traffic of kernel launches during inference. Shows how
                much each sensor reports beyond the payload for small transactions

Run it while no other program uses the GPU (stop llama-server first). The result
table and a JSON/CSV report go to --out-dir. Nothing on the GPU is changed apart
from a temporary buffer (--buffer-mib).
"""

import argparse
import csv
import ctypes
import datetime as dt
import json
import math
import os
import statistics
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional

MIB = 1024 * 1024
MB = 1000 * 1000


# --------------------------------------------------------------------------- CUDA

class CudaError(RuntimeError):
    pass


def load_cuda() -> Any:
    if sys.platform == "win32":
        return ctypes.WinDLL("nvcuda.dll")
    last: Optional[OSError] = None
    for name in ("libcuda.so.1", "libcuda.so"):
        try:
            return ctypes.CDLL(name)
        except OSError as error:
            last = error
    raise OSError(f"CUDA driver library not found: {last}")


_INT_P = ctypes.POINTER(ctypes.c_int)
_VOID_PP = ctypes.POINTER(ctypes.c_void_p)
_U64_P = ctypes.POINTER(ctypes.c_uint64)
CUDA_PROTOTYPES = {
    "cuInit": [ctypes.c_uint],
    "cuDeviceGet": [_INT_P, ctypes.c_int],
    "cuDeviceGetName": [ctypes.c_char_p, ctypes.c_int, ctypes.c_int],
    "cuDeviceGetPCIBusId": [ctypes.c_char_p, ctypes.c_int, ctypes.c_int],
    "cuDevicePrimaryCtxRetain": [_VOID_PP, ctypes.c_int],
    "cuDevicePrimaryCtxRelease_v2": [ctypes.c_int],
    "cuDevicePrimaryCtxRelease": [ctypes.c_int],
    "cuCtxSetCurrent": [ctypes.c_void_p],
    "cuCtxSynchronize": [],
    "cuMemAlloc_v2": [_U64_P, ctypes.c_size_t],
    "cuMemFree_v2": [ctypes.c_uint64],
    "cuMemAllocHost_v2": [_VOID_PP, ctypes.c_size_t],
    "cuMemFreeHost": [ctypes.c_void_p],
    "cuMemcpyHtoD_v2": [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t],
    "cuMemcpyDtoH_v2": [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t],
    "cuGetErrorName": [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)],
}


class CudaCopier:
    """Pinned host buffer + device buffer; synchronous copies in the calling thread."""

    def __init__(self, ordinal: int, buffer_bytes: int, lib: Any = None) -> None:
        self.lib = lib if lib is not None else load_cuda()
        for name, argtypes in CUDA_PROTOTYPES.items():
            function = getattr(self.lib, name, None)
            if function is not None:
                function.argtypes = argtypes
                function.restype = ctypes.c_int
        self.buffer_bytes = buffer_bytes
        self.device = ctypes.c_int(0)
        self.ctx = ctypes.c_void_p()
        self.host = ctypes.c_void_p()
        self.dev_ptr = ctypes.c_uint64(0)
        self._retained = False
        try:
            self._check(self.lib.cuInit(0), "cuInit")
            self._check(self.lib.cuDeviceGet(ctypes.byref(self.device), ordinal), "cuDeviceGet")
            self._check(self.lib.cuDevicePrimaryCtxRetain(ctypes.byref(self.ctx), self.device),
                        "cuDevicePrimaryCtxRetain")
            self._retained = True
            self._check(self.lib.cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent")
            self._check(self.lib.cuMemAlloc_v2(ctypes.byref(self.dev_ptr), buffer_bytes), "cuMemAlloc")
            self._check(self.lib.cuMemAllocHost_v2(ctypes.byref(self.host), buffer_bytes),
                        "cuMemAllocHost (pinned host memory)")
        except Exception:
            self.close()
            raise

    def _check(self, status: int, what: str) -> None:
        if status == 0:
            return
        name = ctypes.c_char_p()
        try:
            self.lib.cuGetErrorName(status, ctypes.byref(name))
            label = name.value.decode() if name.value else str(status)
        except Exception:
            label = str(status)
        raise CudaError(f"{what} failed: {label} ({status})")

    def name(self) -> str:
        buffer = ctypes.create_string_buffer(256)
        self._check(self.lib.cuDeviceGetName(buffer, 256, self.device), "cuDeviceGetName")
        return buffer.value.decode(errors="replace")

    def pci_bus_id(self) -> str:
        buffer = ctypes.create_string_buffer(64)
        self._check(self.lib.cuDeviceGetPCIBusId(buffer, 64, self.device), "cuDeviceGetPCIBusId")
        return buffer.value.decode(errors="replace")

    def h2d(self, nbytes: int) -> None:
        # Synchronous with respect to the host for pinned source memory.
        self._check(self.lib.cuMemcpyHtoD_v2(self.dev_ptr, self.host, nbytes), "cuMemcpyHtoD")

    def d2h(self, nbytes: int) -> None:
        self._check(self.lib.cuMemcpyDtoH_v2(self.host, self.dev_ptr, nbytes), "cuMemcpyDtoH")

    def sync(self) -> None:
        self._check(self.lib.cuCtxSynchronize(), "cuCtxSynchronize")

    def close(self) -> None:
        lib = self.lib
        if self.dev_ptr.value:
            lib.cuMemFree_v2(self.dev_ptr)
            self.dev_ptr = ctypes.c_uint64(0)
        if self.host.value:
            lib.cuMemFreeHost(self.host)
            self.host = ctypes.c_void_p()
        if self._retained:
            release = getattr(lib, "cuDevicePrimaryCtxRelease_v2", None) or lib.cuDevicePrimaryCtxRelease
            release(self.device)
            self._retained = False


# --------------------------------------------------------------------------- NVML

def normalize_bus_id(value: Any) -> str:
    """'0000:01:00.0' and '00000000:01:00.0' compare equal."""
    text = value.decode(errors="replace") if isinstance(value, bytes) else str(value)
    text = text.strip().strip("\x00").lower()
    parts = text.split(":")
    if len(parts) == 3:
        try:
            return f"{int(parts[0], 16):x}:{parts[1]}:{parts[2]}"
        except ValueError:
            pass
    return text


def find_nvml_handle(nv: Any, bus_id: Optional[str], fallback_index: int) -> tuple:
    """NVML device matching the CUDA device's PCI bus id (CUDA and NVML order may differ)."""
    if bus_id:
        wanted = normalize_bus_id(bus_id)
        for index in range(int(nv.nvmlDeviceGetCount())):
            handle = nv.nvmlDeviceGetHandleByIndex(index)
            try:
                info = nv.nvmlDeviceGetPciInfo(handle)
                candidates = [getattr(info, "busId", ""), getattr(info, "busIdLegacy", "")]
            except Exception:
                continue
            if any(candidate and normalize_bus_id(candidate) == wanted for candidate in candidates):
                return index, handle
    return fallback_index, nv.nvmlDeviceGetHandleByIndex(fallback_index)


def nvml_text(value: Any) -> str:
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)


def optional_call(function: Any, *args: Any) -> Any:
    try:
        return function(*args)
    except Exception:
        return None


class LegacySampler(threading.Thread):
    """nvmlDeviceGetPcieThroughput RX then TX, back-to-back or at an interval.

    Each row keeps the exact wall-clock span of both calls, so the analysis can
    use only snapshots that lie completely inside a phase.
    """

    def __init__(self, nv: Any, handle: Any, interval_s: float = 0.0,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        super().__init__(name="legacy-pcie", daemon=True)
        self.nv, self.handle, self.interval_s, self.clock = nv, handle, interval_s, clock
        self.rows: List[Dict[str, Any]] = []
        self.stop_event = threading.Event()
        self.error: Optional[str] = None
        # pause(): no NVML calls until resume(); parked is set once the thread
        # is outside any NVML call (needed before an NVML re-initialisation).
        self.paused = threading.Event()
        self.parked = threading.Event()

    def pause(self, timeout: float = 2.0) -> bool:
        self.paused.set()
        return self.parked.wait(timeout) if self.is_alive() else True

    def resume(self) -> None:
        self.paused.clear()

    def run(self) -> None:
        nv = self.nv
        while not self.stop_event.is_set():
            if self.paused.is_set():
                self.parked.set()
                self.stop_event.wait(0.01)
                continue
            self.parked.clear()
            started = self.clock()
            try:
                rx = nv.nvmlDeviceGetPcieThroughput(self.handle, nv.NVML_PCIE_UTIL_RX_BYTES)
                rx_end = self.clock()
                tx = nv.nvmlDeviceGetPcieThroughput(self.handle, nv.NVML_PCIE_UTIL_TX_BYTES)
                tx_end = self.clock()
            except Exception as error:
                self.error = f"nvmlDeviceGetPcieThroughput failed: {error}"
                return
            self.rows.append({"sensor": "legacy", "rx_start": started, "rx_end": rx_end,
                              "tx_start": rx_end, "tx_end": tx_end,
                              "rx_raw": float(rx), "tx_raw": float(tx)})
            remaining = self.interval_s - (self.clock() - started)
            if remaining > 0:
                self.stop_event.wait(remaining)


PCIE_GPM_METRICS = (("rx_mib_s", "NVML_GPM_METRIC_PCIE_RX_PER_SEC"),
                    ("tx_mib_s", "NVML_GPM_METRIC_PCIE_TX_PER_SEC"))


class GpmSampler(threading.Thread):
    """GPM metrics over consecutive, gap-free intervals (default: PCIe RX/TX).

    request_remedy('realloc' | 'reinit') restarts GPM sampling at the next tick:
    'realloc' frees and allocates new sample buffers, 'reinit' additionally shuts
    NVML down and initialises it again (other NVML users must be paused first;
    the new device handle is in self.handle afterwards). Each remedy is logged
    in self.events and the first interval after it is skipped.
    """

    def __init__(self, nv: Any, handle: Any, interval_s: float = 0.2,
                 clock: Callable[[], float] = time.perf_counter,
                 metrics: tuple = PCIE_GPM_METRICS, nvml_index: int = 0) -> None:
        super().__init__(name="gpm-sampler", daemon=True)
        self.nv, self.handle, self.interval_s, self.clock = nv, handle, interval_s, clock
        self.metrics, self.nvml_index = metrics, nvml_index
        self.rows: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
        self.stop_event = threading.Event()
        self.error: Optional[str] = None
        self.samples: List[Any] = []
        self.metric_ids: List[int] = []
        self._remedy: Optional[str] = None
        self._remedy_done = threading.Event()

    def prepare(self) -> None:
        nv = self.nv
        for name in ("nvmlGpmQueryDeviceSupport", "nvmlGpmSampleAlloc", "nvmlGpmSampleGet",
                     "nvmlGpmMetricsGet", "c_nvmlGpmMetricsGet_t",
                     *(constant for _, constant in self.metrics)):
            if not hasattr(nv, name):
                raise RuntimeError(f"installed pynvml lacks {name}; update with: "
                                   "py -m pip install -U nvidia-ml-py")
        support = nv.nvmlGpmQueryDeviceSupport(self.handle)
        if not int(support.isSupportedDevice):
            raise RuntimeError("GPM is not supported on this GPU/driver")
        self.metric_ids = [int(getattr(nv, constant)) for _, constant in self.metrics]
        self.samples = [nv.nvmlGpmSampleAlloc(), nv.nvmlGpmSampleAlloc()]

    def request_remedy(self, kind: str, timeout: float = 3.0, wait: bool = True) -> Optional[Dict[str, Any]]:
        """Ask the sampler thread to restart GPM sampling at its next tick.

        With wait=False the call returns at once; remedy_done() tells when it ran.
        """
        if kind not in ("realloc", "reinit"):
            raise ValueError(kind)
        self._remedy_done.clear()
        self._remedy = kind
        if not wait:
            return None
        if not self._remedy_done.wait(timeout):
            return None
        return self.events[-1] if self.events else None

    def remedy_done(self) -> bool:
        return self._remedy_done.is_set()

    def _apply_remedy(self, kind: str) -> None:
        nv = self.nv
        started = self.clock()
        event: Dict[str, Any] = {"event": kind, "time": started}
        try:
            self.free()
            if kind == "reinit":
                nv.nvmlShutdown()
                nv.nvmlInit()
                self.handle = nv.nvmlDeviceGetHandleByIndex(self.nvml_index)
            self.samples = [nv.nvmlGpmSampleAlloc(), nv.nvmlGpmSampleAlloc()]
            event["ok"] = True
        except Exception as error:
            event.update(ok=False, error=str(error))
        event["duration_ms"] = (self.clock() - started) * 1000.0
        self.events.append(event)

    def run(self) -> None:
        nv = self.nv
        try:
            previous, current = self.samples
            nv.nvmlGpmSampleGet(self.handle, previous)
            previous_time = self.clock()
            success = int(getattr(nv, "NVML_SUCCESS", 0))
            while not self.stop_event.wait(self.interval_s):
                if self._remedy is not None:
                    kind, self._remedy = self._remedy, None
                    self._apply_remedy(kind)
                    self._remedy_done.set()
                    if not self.events[-1].get("ok"):
                        self.error = f"GPM {kind} failed: {self.events[-1].get('error')}"
                        return
                    previous, current = self.samples
                    nv.nvmlGpmSampleGet(self.handle, previous)
                    previous_time = self.clock()
                    continue
                nv.nvmlGpmSampleGet(self.handle, current)
                now = self.clock()
                get = nv.c_nvmlGpmMetricsGet_t()
                get.version = nv.NVML_GPM_METRICS_GET_VERSION
                get.numMetrics = len(self.metric_ids)
                get.sample1, get.sample2 = previous, current
                for index, metric_id in enumerate(self.metric_ids):
                    get.metrics[index].metricId = metric_id
                nv.nvmlGpmMetricsGet(get)
                row: Dict[str, Any] = {"sensor": "gpm", "start": previous_time, "end": now}
                for index, (key, _) in enumerate(self.metrics):
                    metric = get.metrics[index]
                    row[key] = float(metric.value) if int(metric.nvmlReturn) == success else None
                self.rows.append(row)
                previous, current = current, previous
                previous_time = now
        except Exception as error:
            self.error = f"GPM polling failed: {error}"
        finally:
            self._remedy_done.set()

    def free(self) -> None:
        for sample in self.samples:
            optional_call(self.nv.nvmlGpmSampleFree, sample)
        self.samples = []


# ------------------------------------------------------------------------- phases

PHASE_KINDS = {
    "idle": (None, None),
    "h2d": ("h2d", "steady"), "d2h": ("d2h", "steady"),
    "h2d_bursty": ("h2d", "bursty"), "d2h_bursty": ("d2h", "bursty"),
    "h2d_tiny": ("h2d", "tiny"), "d2h_tiny": ("d2h", "tiny"),
}
TRANSFER_KINDS = ("h2d", "d2h", "h2d_bursty", "d2h_bursty", "h2d_tiny", "d2h_tiny")


def default_plan(seconds: float, idle_s: float) -> List[tuple]:
    plan: List[tuple] = [("idle", idle_s)]
    for kind in TRANSFER_KINDS:
        plan += [(kind, seconds), ("idle", idle_s)]
    return plan


def run_phase(kind: str, seconds: float, copier: Any, chunk_bytes: int, burst_bytes: int,
              burst_gap_s: float, clock: Callable[[], float] = time.perf_counter,
              sleep: Callable[[float], None] = time.sleep,
              probe: Optional[Callable[[], Dict[str, Any]]] = None,
              tiny_bytes: int = 64) -> Dict[str, Any]:
    """Run one phase; bytes are counted exactly, times cover first copy to final sync."""
    direction, mode = PHASE_KINDS[kind]
    bursty = mode == "bursty"
    result: Dict[str, Any] = {"phase": kind, "h2d_bytes": 0, "d2h_bytes": 0, "copies": 0}
    started = clock()
    if direction is None:
        sleep(seconds)
    else:
        copy = copier.h2d if direction == "h2d" else copier.d2h
        size = {"steady": chunk_bytes, "bursty": burst_bytes, "tiny": tiny_bytes}[mode]
        deadline = started + seconds
        probed = probe is None
        while clock() < deadline:
            copy(size)
            result[f"{direction}_bytes"] += size
            result["copies"] += 1
            if not probed and clock() >= started + seconds / 2:
                result["link"] = probe()
                probed = True
            if bursty:
                sleep(burst_gap_s)
        copier.sync()
    ended = clock()
    result.update(start=started, end=ended, seconds=ended - started)
    return result


# ----------------------------------------------------------------------- analysis

def mean(values: List[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def ratio(value: Optional[float], truth: Optional[float]) -> Optional[float]:
    if value is None or truth is None or truth <= 0:
        return None
    return value / truth


def interpret(value: Optional[float]) -> str:
    """Plain-language reading of sensor/true ratio for a steady transfer."""
    if value is None:
        return "no data"
    if 0.97 <= value <= 1.03:
        return "matches the transferred payload"
    if 1.03 < value <= 1.06:
        return "about +2.4-5 %: KiB-vs-KB unit or small protocol overhead"
    if 1.06 < value <= 1.40:
        return "above payload: likely counts PCIe protocol overhead (TLP headers, completions)"
    if 0.90 <= value < 0.97:
        return "slightly below payload"
    return "far off the transferred payload"


def analyze_phase(phase: Dict[str, Any], legacy_rows: List[Dict[str, Any]],
                  gpm_rows: List[Dict[str, Any]], edge_s: float = 0.05) -> Dict[str, Any]:
    """Compare both sensors with the true byte rate inside one phase.

    Units: truth in bytes/s. Legacy raw values are reported by NVML in 'KB/s';
    both 1000 and 1024 are shown. GPM is documented as MiB/s.
    Only snapshots/intervals completely inside [start+edge, end-edge] are used.
    """
    lo, hi = phase["start"] + edge_s, phase["end"] - edge_s
    seconds = max(phase["end"] - phase["start"], 1e-9)
    transfer = phase["phase"] != "idle"
    # GPM can report exactly 0 in both directions while copies are running (seen
    # under light, bursty load). Such intervals are dropouts, not measurements.
    in_phase = [row for row in gpm_rows if row["start"] >= lo and row["end"] <= hi]
    dropouts = [row for row in in_phase if transfer and row.get("rx_mib_s") == 0 and row.get("tx_mib_s") == 0]
    gpm_rows = [row for row in in_phase if not any(row is d for d in dropouts)]
    truth = {"rx": phase["h2d_bytes"] / seconds, "tx": phase["d2h_bytes"] / seconds}
    out: Dict[str, Any] = {"phase": phase["phase"], "seconds": round(seconds, 3),
                           "copies_per_s": phase.get("copies", 0) / seconds,
                           "true_h2d_MB_s": truth["rx"] / MB, "true_d2h_MB_s": truth["tx"] / MB,
                           "gpm_zero_intervals": len(dropouts), "gpm_intervals": len(in_phase)}
    for direction in ("rx", "tx"):
        legacy = [row[f"{direction}_raw"] for row in legacy_rows
                  if row[f"{direction}_start"] >= lo and row[f"{direction}_end"] <= hi]
        raw = mean(legacy)
        out[f"legacy_{direction}_samples"] = len(legacy)
        out[f"legacy_{direction}_raw_kb_s"] = raw
        out[f"legacy_{direction}_MB_s_if_KB"] = None if raw is None else raw * 1000 / MB
        out[f"legacy_{direction}_MB_s_if_KiB"] = None if raw is None else raw * 1024 / MB
        inside = [row for row in gpm_rows if row["start"] >= lo and row["end"] <= hi
                  and row.get(f"{direction}_mib_s") is not None]
        weight = sum(row["end"] - row["start"] for row in inside)
        gpm = (sum(row[f"{direction}_mib_s"] * (row["end"] - row["start"]) for row in inside) / weight
               if weight > 0 else None)
        out[f"gpm_{direction}_samples"] = len(inside)
        out[f"gpm_{direction}_raw_mib_s"] = gpm
        out[f"gpm_{direction}_MB_s_if_MiB"] = None if gpm is None else gpm * MIB / MB
        out[f"gpm_{direction}_MB_s_if_MB"] = gpm
    direction = {"h2d": "rx", "d2h": "tx"}.get(phase["phase"].split("_")[0])
    if direction and not phase["phase"].endswith("_tiny"):
        true_rate = truth[direction]
        out["direction"] = direction
        out["ratio_gpm_if_MiB"] = ratio(None if out[f"gpm_{direction}_raw_mib_s"] is None
                                        else out[f"gpm_{direction}_raw_mib_s"] * MIB, true_rate)
        out["ratio_legacy_if_KB"] = ratio(None if out[f"legacy_{direction}_raw_kb_s"] is None
                                          else out[f"legacy_{direction}_raw_kb_s"] * 1000, true_rate)
        out["ratio_legacy_if_KiB"] = ratio(None if out[f"legacy_{direction}_raw_kb_s"] is None
                                           else out[f"legacy_{direction}_raw_kb_s"] * 1024, true_rate)
        other = "tx" if direction == "rx" else "rx"
        out["reverse_gpm_MB_s"] = out[f"gpm_{other}_MB_s_if_MiB"]
        out["reverse_legacy_MB_s"] = out[f"legacy_{other}_MB_s_if_KB"]
    return out


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Overall verdict from steady phases; burst bias from legacy/GPM ratio."""
    steady = [r for r in results if r["phase"] in ("h2d", "d2h")]
    bursty = [r for r in results if r["phase"] in ("h2d_bursty", "d2h_bursty")]
    tiny = [r for r in results if r["phase"] in ("h2d_tiny", "d2h_tiny")]

    def pooled(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
        values = [r[key] for r in rows if r.get(key) is not None]
        return statistics.fmean(values) if values else None

    summary = {
        "steady_ratio_gpm_if_MiB": pooled(steady, "ratio_gpm_if_MiB"),
        "steady_ratio_legacy_if_KB": pooled(steady, "ratio_legacy_if_KB"),
        "steady_ratio_legacy_if_KiB": pooled(steady, "ratio_legacy_if_KiB"),
        "bursty_ratio_gpm_if_MiB": pooled(bursty, "ratio_gpm_if_MiB"),
        "bursty_ratio_legacy_if_KB": pooled(bursty, "ratio_legacy_if_KB"),
    }
    summary["gpm_reading"] = interpret(summary["steady_ratio_gpm_if_MiB"])
    summary["legacy_reading"] = interpret(summary["steady_ratio_legacy_if_KB"])
    steady_ab = ratio(summary["steady_ratio_legacy_if_KB"], summary["steady_ratio_gpm_if_MiB"])
    bursty_ab = ratio(summary["bursty_ratio_legacy_if_KB"], summary["bursty_ratio_gpm_if_MiB"])
    summary["legacy_over_gpm_steady"] = steady_ab
    summary["legacy_over_gpm_bursty"] = bursty_ab
    # Judge each sensor against the true rate: does it keep its steady-state scale?
    deviating = []
    for sensor, key in (("GPM", "ratio_gpm_if_MiB"), ("legacy", "ratio_legacy_if_KB")):
        change = ratio(pooled(bursty, key), pooled(steady, key))
        summary[f"{sensor.lower()}_bursty_over_steady"] = change
        if change is not None and abs(change - 1.0) > 0.15:
            deviating.append(f"{sensor} ({change:.2f}x its steady scale)")
    if summary["gpm_bursty_over_steady"] is not None or summary["legacy_bursty_over_steady"] is not None:
        summary["burst_reading"] = ("for bursty traffic " + " and ".join(deviating) + " deviate(s)"
                                    if deviating else "both sensors keep their steady-state scale for bursty traffic")
    transfer_rows = [r for r in results if r["phase"] != "idle"]
    summary["gpm_dropout_intervals"] = sum(r.get("gpm_zero_intervals", 0) for r in transfer_rows)
    summary["gpm_transfer_intervals"] = sum(r.get("gpm_intervals", 0) for r in transfer_rows)
    summary["gpm_dropouts_by_phase"] = {r["phase"]: r["gpm_zero_intervals"] for r in transfer_rows
                                        if r.get("gpm_zero_intervals")}
    # Tiny copies: payload is negligible, so compare what each sensor reports
    # in total (both directions) and against each other.
    for sensor, key in (("gpm", "gpm_{}_MB_s_if_MiB"), ("legacy", "legacy_{}_MB_s_if_KB")):
        totals = [sum(r[key.format(d)] for d in ("rx", "tx")) for r in tiny
                  if all(r.get(key.format(d)) is not None for d in ("rx", "tx"))]
        summary[f"tiny_{sensor}_total_MB_s"] = statistics.fmean(totals) if totals else None
    summary["tiny_payload_MB_s"] = (statistics.fmean(r["true_h2d_MB_s"] + r["true_d2h_MB_s"] for r in tiny)
                                    if tiny else None)
    summary["legacy_over_gpm_tiny"] = ratio(summary["tiny_legacy_total_MB_s"], summary["tiny_gpm_total_MB_s"])
    tiny_ab = summary["legacy_over_gpm_tiny"]
    if tiny_ab is not None and steady_ab is not None:
        if tiny_ab > 1.25 * steady_ab:
            summary["tiny_reading"] = ("with small transactions the legacy sensor reports clearly more than "
                                       "GPM: the two count small-transaction/protocol traffic differently")
        elif tiny_ab < 0.8 * steady_ab:
            summary["tiny_reading"] = "with small transactions GPM reports clearly more than the legacy sensor"
        else:
            summary["tiny_reading"] = "both sensors treat small transactions alike (legacy/GPM as for large copies)"
    return summary


def estimate_gpm_lag(phases: List[Dict[str, Any]], gpm_rows: List[Dict[str, Any]]) -> List[float]:
    """How late GPM reports traffic, from the edges of the steady copy phases.

    GPM intervals are contiguous. Around a phase edge, the reported traffic
    (as a fraction f of the full rate) is summed over consecutive intervals:
    at a start edge from the interval containing the edge until the first full
    interval, at an end edge until the first empty one. That sum is the
    reported full-rate time R; with no lag it would place the edge exactly at
    the edge time. Works for lags longer than one interval as well.
    """
    lags: List[float] = []
    rows = sorted(gpm_rows, key=lambda r: r["start"])
    for phase in phases:
        direction = {"h2d": "rx_mib_s", "d2h": "tx_mib_s"}.get(phase["phase"])
        if direction is None:
            continue
        inner = [row[direction] for row in rows if row["start"] >= phase["start"] + 0.3
                 and row["end"] <= phase["end"] - 0.3 and row.get(direction)]
        if len(inner) < 3:
            continue
        full = statistics.median(inner)
        for edge, rising in ((phase["start"], True), (phase["end"], False)):
            index = next((i for i, row in enumerate(rows) if row["start"] <= edge < row["end"]), None)
            if index is None:
                continue
            reported, first_start = 0.0, rows[index]["start"]
            for row in rows[index:index + 6]:
                value = row.get(direction)
                if value is None:
                    break
                fraction = min(1.0, max(0.0, value / full))
                if rising and fraction >= 0.98:
                    # Traffic ran at full rate from (end - reported) onwards.
                    lags.append((row["end"] - (reported + (row["end"] - row["start"])) - edge) * 1000.0)
                    break
                if not rising and fraction <= 0.02:
                    lags.append((first_start + reported - edge) * 1000.0)
                    break
                reported += fraction * (row["end"] - row["start"])
    return lags


# --------------------------------------------------------------------------- output

def fmt(value: Optional[float], digits: int = 0, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{value:.{digits}f}{suffix}"


def print_report(results: List[Dict[str, Any]], summary: Dict[str, Any], out: Any = sys.stdout) -> None:
    print("\nPhase results (MB = 10^6 bytes; H2D = GPU receives (RX), D2H = GPU sends (TX))", file=out)
    header = (f"{'phase':<11} | {'true':>8} | {'GPM':>8} {'ratio':>6} | {'legacy':>8} {'ratio':>6} "
              f"{'KiB?':>6} | {'reverse GPM/legacy':>18} | {'samples G/L':>11}")
    print(header, file=out)
    print("-" * len(header), file=out)
    for r in results:
        direction = r.get("direction")
        if r["phase"].endswith("_tiny"):
            gpm = f"{fmt(r['gpm_rx_MB_s_if_MiB'], 1)}/{fmt(r['gpm_tx_MB_s_if_MiB'], 1)}"
            legacy = f"{fmt(r['legacy_rx_MB_s_if_KB'], 1)}/{fmt(r['legacy_tx_MB_s_if_KB'], 1)}"
            payload = r["true_h2d_MB_s"] + r["true_d2h_MB_s"]
            print(f"{r['phase']:<11} | {fmt(payload, 2):>8} | RX/TX GPM {gpm}, legacy {legacy} MB/s, "
                  f"{fmt(r.get('copies_per_s'))} copies/s", file=out)
        elif direction:
            true = r["true_h2d_MB_s"] if direction == "rx" else r["true_d2h_MB_s"]
            gpm = r[f"gpm_{direction}_MB_s_if_MiB"]
            legacy = r[f"legacy_{direction}_MB_s_if_KB"]
            reverse = f"{fmt(r['reverse_gpm_MB_s'])}/{fmt(r['reverse_legacy_MB_s'])}"
            samples = f"{r[f'gpm_{direction}_samples']}/{r[f'legacy_{direction}_samples']}"
            print(f"{r['phase']:<11} | {fmt(true):>8} | {fmt(gpm):>8} {fmt(r['ratio_gpm_if_MiB'], 3):>6} | "
                  f"{fmt(legacy):>8} {fmt(r['ratio_legacy_if_KB'], 3):>6} {fmt(r['ratio_legacy_if_KiB'], 3):>6} | "
                  f"{reverse:>18} | {samples:>11}", file=out)
        else:
            gpm = f"{fmt(r['gpm_rx_MB_s_if_MiB'], 1)}/{fmt(r['gpm_tx_MB_s_if_MiB'], 1)}"
            legacy = f"{fmt(r['legacy_rx_MB_s_if_KB'], 1)}/{fmt(r['legacy_tx_MB_s_if_KB'], 1)}"
            print(f"{r['phase']:<11} | {'0':>8} | idle RX/TX GPM {gpm}, legacy {legacy} MB/s", file=out)
    print("\nratio = sensor / true rate. GPM read as MiB/s; legacy read as KB/s (1000), 'KiB?' as KiB/s (1024).",
          file=out)
    print("\nVerdict (steady copies):", file=out)
    print(f"  GPM    ratio {fmt(summary['steady_ratio_gpm_if_MiB'], 3)} -> {summary['gpm_reading']}", file=out)
    print(f"  legacy ratio {fmt(summary['steady_ratio_legacy_if_KB'], 3)} "
          f"(as KiB: {fmt(summary['steady_ratio_legacy_if_KiB'], 3)}) -> {summary['legacy_reading']}", file=out)
    print(f"  legacy/GPM: steady {fmt(summary['legacy_over_gpm_steady'], 2)}, "
          f"bursty {fmt(summary['legacy_over_gpm_bursty'], 2)}", file=out)
    if summary.get("burst_reading"):
        print(f"  {summary['burst_reading']}", file=out)
    if summary.get("gpm_dropout_intervals"):
        by_phase = ", ".join(f"{k} {v}" for k, v in summary["gpm_dropouts_by_phase"].items())
        print(f"  GPM DROPOUT: {summary['gpm_dropout_intervals']} of {summary['gpm_transfer_intervals']} GPM "
              f"intervals read exactly 0 in both directions while copies ran ({by_phase}); excluded above",
              file=out)
    if summary.get("gpm_lag_ms") is not None:
        print(f"  GPM timing: values lag the copies by ~{summary['gpm_lag_ms']:.0f} ms "
              f"(from steady phase edges: {', '.join(f'{v:.0f}' for v in summary['gpm_lag_edges_ms'])} ms)",
              file=out)
    options = []
    if summary.get("gpm_lag_ms") is not None:
        options.append(f"--gpm-lag-ms {max(0, round(summary['gpm_lag_ms'] / 5) * 5):.0f}")
    if summary.get("legacy_over_gpm_steady") is not None:
        options.append(f"--pcie-legacy-scale {summary['legacy_over_gpm_steady']:.2f}")
    if options:
        print(f"\nSuggested ctx-cliff options for this GPU/driver: {' '.join(options)}", file=out)
    if summary.get("tiny_payload_MB_s") is not None:
        print(f"\nSmall transactions (tiny copies, payload {fmt(summary['tiny_payload_MB_s'], 2)} MB/s):", file=out)
        print(f"  reported RX+TX: GPM {fmt(summary['tiny_gpm_total_MB_s'], 1)} MB/s, "
              f"legacy {fmt(summary['tiny_legacy_total_MB_s'], 1)} MB/s, "
              f"legacy/GPM {fmt(summary['legacy_over_gpm_tiny'], 2)}", file=out)
        if summary.get("tiny_reading"):
            print(f"  {summary['tiny_reading']}", file=out)


def write_outputs(stem: str, report: Dict[str, Any], legacy_rows: List[Dict[str, Any]],
                  gpm_rows: List[Dict[str, Any]], phases: List[Dict[str, Any]]) -> None:
    def phase_of(start: float, end: float) -> str:
        for phase in phases:
            if phase["start"] <= start and end <= phase["end"]:
                return phase["phase"]
        return ""

    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        f.write("\n")
    fields = ["sensor", "phase", "start_s", "end_s", "rx", "tx", "unit"]
    origin = phases[0]["start"] if phases else 0.0
    with open(stem + ".csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in legacy_rows:
            writer.writerow({"sensor": "legacy", "phase": phase_of(row["rx_start"], row["tx_end"]),
                             "start_s": round(row["rx_start"] - origin, 4),
                             "end_s": round(row["tx_end"] - origin, 4),
                             "rx": row["rx_raw"], "tx": row["tx_raw"], "unit": "KB/s (NVML raw)"})
        for row in gpm_rows:
            writer.writerow({"sensor": "gpm", "phase": phase_of(row["start"], row["end"]),
                             "start_s": round(row["start"] - origin, 4), "end_s": round(row["end"] - origin, 4),
                             "rx": row["rx_mib_s"], "tx": row["tx_mib_s"], "unit": "MiB/s (GPM)"})


# ----------------------------------------------------------------------------- main

def calibrate(args: Any, nv: Any, copier: Any, handle: Any, out: Any = sys.stdout,
              clock: Callable[[], float] = time.perf_counter,
              sleep: Callable[[float], None] = time.sleep) -> Dict[str, Any]:
    """Run samplers and phases; return the full report (without writing files)."""
    legacy = LegacySampler(nv, handle, args.legacy_interval_ms / 1000.0, clock)
    gpm: Optional[GpmSampler] = None
    gpm_error = None
    if not args.no_gpm:
        gpm = GpmSampler(nv, handle, args.gpm_interval_ms / 1000.0, clock)
        try:
            gpm.prepare()
        except Exception as error:
            gpm_error = str(error)
            gpm = None
            print(f"WARNING: GPM unavailable ({gpm_error}); only the legacy sensor is checked", file=out)

    def link() -> Dict[str, Any]:
        return {"gen": optional_call(nv.nvmlDeviceGetCurrPcieLinkGeneration, handle),
                "width": optional_call(nv.nvmlDeviceGetCurrPcieLinkWidth, handle)}

    plan = default_plan(args.seconds, args.idle_seconds)
    phases: List[Dict[str, Any]] = []
    legacy.start()
    if gpm is not None:
        gpm.start()
    try:
        sleep(max(0.5, args.gpm_interval_ms / 1000.0 * 2))  # let both sensors settle
        for kind, seconds in plan:
            print(f"  {kind:<11} {seconds:.0f} s ...", file=out, flush=True)
            phases.append(run_phase(kind, seconds, copier, args.buffer_mib * MIB,
                                    args.burst_mib * MIB, args.burst_gap_ms / 1000.0,
                                    clock=clock, sleep=sleep, probe=link,
                                    tiny_bytes=getattr(args, "tiny_bytes", 64)))
            for sampler in (legacy, gpm):
                if sampler is not None and sampler.error:
                    raise RuntimeError(sampler.error)
    finally:
        legacy.stop_event.set()
        if gpm is not None:
            gpm.stop_event.set()
        legacy.join(5)
        if gpm is not None:
            gpm.join(5)
            gpm.free()
    results = [analyze_phase(phase, legacy.rows, gpm.rows if gpm is not None else [])
               for phase in phases]
    summary = summarize(results)
    lags = estimate_gpm_lag(phases, gpm.rows if gpm is not None else [])
    summary["gpm_lag_edges_ms"] = [round(v, 1) for v in lags]
    summary["gpm_lag_ms"] = statistics.median(lags) if lags else None
    return {"phases": phases, "results": results, "summary": summary,
            "legacy_rows": legacy.rows, "gpm_rows": gpm.rows if gpm is not None else [],
            "gpm_error": gpm_error,
            "legacy_query_ms_median": (statistics.median((r["tx_end"] - r["rx_start"]) * 1000
                                                         for r in legacy.rows) if legacy.rows else None)}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Calibrate NVML legacy and GPM PCIe throughput against known host<->GPU copies.",
        epilog="Stop llama-server and other GPU programs first; their traffic would distort the result.")
    ap.add_argument("--gpu", type=int, default=0, help="CUDA device ordinal (default: %(default)s)")
    ap.add_argument("--seconds", type=float, default=6.0, help="duration per transfer phase (default: %(default)s)")
    ap.add_argument("--idle-seconds", type=float, default=2.0, help="idle gap between phases (default: %(default)s)")
    ap.add_argument("--buffer-mib", type=int, default=256, help="copy size for steady phases (default: %(default)s)")
    ap.add_argument("--burst-mib", type=int, default=8, help="copy size for bursty phases (default: %(default)s)")
    ap.add_argument("--burst-gap-ms", type=float, default=25.0, help="pause after each burst (default: %(default)s)")
    ap.add_argument("--tiny-bytes", type=int, default=64,
                    help="copy size for the small-transaction phases (default: %(default)s)")
    ap.add_argument("--gpm-interval-ms", type=int, default=200, help="GPM interval, >100 (default: %(default)s)")
    ap.add_argument("--legacy-interval-ms", type=int, default=0,
                    help="legacy poll interval; 0 = back-to-back (default: %(default)s)")
    ap.add_argument("--no-gpm", action="store_true", help="check only the legacy sensor")
    ap.add_argument("--out-dir", default="outputs", help="report directory (default: %(default)s)")
    args = ap.parse_args(argv)
    if args.seconds < 1 or args.idle_seconds < 0.5:
        ap.error("--seconds must be >= 1 and --idle-seconds >= 0.5")
    if args.gpm_interval_ms <= 100:
        ap.error("--gpm-interval-ms must be > 100")
    if args.burst_mib < 1 or args.buffer_mib < args.burst_mib:
        ap.error("require 1 <= --burst-mib <= --buffer-mib")
    if not 1 <= args.tiny_bytes <= args.burst_mib * MIB:
        ap.error("require 1 <= --tiny-bytes <= --burst-mib")

    try:
        import pynvml as nv
        nv.nvmlInit()
    except Exception as error:
        print(f"ERROR: NVML unavailable ({error}); install with: py -m pip install -U nvidia-ml-py",
              file=sys.stderr)
        return 2
    copier = None
    try:
        try:
            copier = CudaCopier(args.gpu, args.buffer_mib * MIB)
        except Exception as error:
            print(f"ERROR: CUDA setup failed: {error}", file=sys.stderr)
            return 2
        bus_id = optional_call(copier.pci_bus_id)
        nvml_index, handle = find_nvml_handle(nv, bus_id, args.gpu)
        gpu = {"cuda_ordinal": args.gpu, "nvml_index": nvml_index, "name": optional_call(copier.name),
               "pci_bus_id": bus_id,
               "driver": nvml_text(optional_call(nv.nvmlSystemGetDriverVersion) or ""),
               "link_max_gen": optional_call(nv.nvmlDeviceGetMaxPcieLinkGeneration, handle),
               "link_max_width": optional_call(nv.nvmlDeviceGetMaxPcieLinkWidth, handle)}
        # Only compute processes (e.g. llama-server). On Windows the desktop always
        # shows up as graphics processes; its small traffic appears in the idle baseline.
        processes = optional_call(getattr(nv, "nvmlDeviceGetComputeRunningProcesses", lambda h: None),
                                  handle) or []
        others = [int(getattr(p, "pid", 0)) for p in processes if int(getattr(p, "pid", 0)) != os.getpid()]
        print(f"GPU: {gpu['name']} ({bus_id}), driver {gpu['driver']}", file=sys.stderr)
        if others:
            print(f"WARNING: other processes use this GPU (PIDs {sorted(set(others))}); "
                  "their traffic distorts the calibration", file=sys.stderr)
        total = args.idle_seconds * (len(TRANSFER_KINDS) + 1) + args.seconds * len(TRANSFER_KINDS)
        print(f"Running {total:.0f} s of calibration phases:", file=sys.stderr)
        report = calibrate(args, nv, copier, handle, out=sys.stderr)
    finally:
        if copier is not None:
            copier.close()
        optional_call(nv.nvmlShutdown)

    links = [p["link"] for p in report["phases"] if p.get("link")]
    report.update(gpu=gpu, other_gpu_pids=sorted(set(others)), link_during_copies=links,
                  arguments=vars(args), created=dt.datetime.now().astimezone().isoformat(timespec="seconds"))
    print_report(report["results"], report["summary"])
    if links:
        gen, width = links[0].get("gen"), links[0].get("width")
        print(f"\nPCIe link during copies: Gen {gen} x{width}"
              f" (NVML max reported: Gen {gpu['link_max_gen']} x{gpu['link_max_width']})")
    if report["legacy_query_ms_median"] is not None:
        print(f"Legacy RX+TX query takes {report['legacy_query_ms_median']:.0f} ms (median)")
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, "pcie-calibrate-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    raw_legacy, raw_gpm = report.pop("legacy_rows"), report.pop("gpm_rows")
    write_outputs(stem, report, raw_legacy, raw_gpm, report["phases"])
    print(f"\nsaved: {stem}.json and {stem}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
