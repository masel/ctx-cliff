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
gpm-probe.py — find out when NVML GPM drops out and whether restarting it helps.

Observed on an RTX 5060 Ti (Windows, driver 616.92): GPM sometimes reports
SM/occupancy/tensor = 0 under full load, or every GPM value = 0 while PCIe
traffic continues. The state tends to persist across several load periods and
changes at idle->load transitions. This script runs controlled load cycles

    idle (varied length)  ->  load (compute kernels, or bursty host->GPU copies)

and answers three questions:

  1. Does the legacy NVML PCIe sampler (nvmlDeviceGetPcieThroughput, used by
     ctx-cliff) cause dropouts? Cycles alternate with the sampler on and off.
  2. Does the idle length before a load period matter?
  3. Can a dropout be ended by restarting GPM sampling? When a dropout is seen
     early in a load period, the script first allocates fresh GPM sample
     buffers ("realloc"); if that does not help, it re-initialises NVML
     ("reinit") and checks whether plausible values return.

Stop llama-server and other GPU programs first. Needs nvidia-ml-py; uses the
CUDA driver (nvcuda.dll) directly and pcie-calibrate.py from the same folder.
"""

import argparse
import ctypes
import datetime as dt
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("pcie_calibrate", HERE / "pcie-calibrate.py")
pc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pc)

MIB = pc.MIB
PROBE_METRICS = (("graphics", "NVML_GPM_METRIC_GRAPHICS_UTIL"),
                 ("sm", "NVML_GPM_METRIC_SM_UTIL"),
                 ("occupancy", "NVML_GPM_METRIC_SM_OCCUPANCY"),
                 ("rx_mib_s", "NVML_GPM_METRIC_PCIE_RX_PER_SEC"),
                 ("tx_mib_s", "NVML_GPM_METRIC_PCIE_TX_PER_SEC"))

# A tiny compute kernel (dependent FMA chain), JIT-compiled by the driver.
SPIN_PTX = b"""
.version 6.0
.target sm_50
.address_size 64

.visible .entry spin(
    .param .u64 out,
    .param .u32 iters
)
{
    .reg .pred %p<2>;
    .reg .b32 %r<4>;
    .reg .f32 %f<2>;
    .reg .b64 %rd<5>;

    ld.param.u64 %rd1, [out];
    ld.param.u32 %r1, [iters];
    cvta.to.global.u64 %rd2, %rd1;
    mov.f32 %f1, 0f3F800000;
    mov.u32 %r2, 0;
$L_loop:
    fma.rn.f32 %f1, %f1, 0f3F7FFFEF, 0f33D6BF95;
    add.s32 %r2, %r2, 1;
    setp.lt.u32 %p1, %r2, %r1;
    @%p1 bra $L_loop;
    mov.u32 %r3, %tid.x;
    mul.wide.u32 %rd3, %r3, 4;
    add.s64 %rd4, %rd2, %rd3;
    st.global.f32 [%rd4], %f1;
    ret;
}
\x00"""

CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
BLOCK = 256


class ComputeLoad:
    """Launch the spin kernel on the copier's CUDA context (calling thread)."""

    def __init__(self, copier: Any, kernel_ms: float = 30.0) -> None:
        lib = copier.lib
        self.copier, self.lib = copier, lib
        lib.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        lib.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        lib.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 6 + [
            ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
        lib.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        lib.cuModuleUnload.argtypes = [ctypes.c_void_p]
        for name in ("cuModuleLoadData", "cuModuleGetFunction", "cuLaunchKernel", "cuDeviceGetAttribute",
                     "cuModuleUnload"):
            getattr(lib, name).restype = ctypes.c_int
        self.module = ctypes.c_void_p()
        self.function = ctypes.c_void_p()
        copier._check(lib.cuModuleLoadData(ctypes.byref(self.module), SPIN_PTX), "cuModuleLoadData (PTX JIT)")
        copier._check(lib.cuModuleGetFunction(ctypes.byref(self.function), self.module, b"spin"),
                      "cuModuleGetFunction")
        sms = ctypes.c_int(0)
        copier._check(lib.cuDeviceGetAttribute(ctypes.byref(sms), CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                                               copier.device), "cuDeviceGetAttribute")
        self.grid = max(1, sms.value) * 4
        # Calibrate iterations for roughly kernel_ms per launch.
        self.iters = 200_000
        elapsed = self.launch_timed()
        self.iters = int(max(1_000, min(2_000_000_000, self.iters * (kernel_ms / 1000.0) / max(elapsed, 1e-4))))
        self.kernel_ms = self.launch_timed() * 1000.0

    def launch(self) -> None:
        out = ctypes.c_uint64(self.copier.dev_ptr.value)
        iters = ctypes.c_uint32(self.iters)
        params = (ctypes.c_void_p * 2)(ctypes.cast(ctypes.pointer(out), ctypes.c_void_p),
                                       ctypes.cast(ctypes.pointer(iters), ctypes.c_void_p))
        self.copier._check(self.lib.cuLaunchKernel(self.function, self.grid, 1, 1, BLOCK, 1, 1, 0, None,
                                                   params, None), "cuLaunchKernel")

    def launch_timed(self) -> float:
        started = time.perf_counter()
        self.launch()
        self.copier.sync()
        return time.perf_counter() - started

    def close(self) -> None:
        if self.module.value:
            self.lib.cuModuleUnload(self.module)
            self.module = ctypes.c_void_p()


# ------------------------------------------------------------------ classification

def classify(rows: List[Dict[str, Any]], load: str) -> Dict[str, Any]:
    """Share of GPM intervals in a load window that look like a dropout.

    all_zero: every value exactly 0 (while the load was running).
    sm_zero : graphics >= 25 % but SM and occupancy <= 0.1 % (compute load only).
    """
    usable = [r for r in rows if r.get("graphics") is not None or r.get("rx_mib_s") is not None]
    all_zero = [r for r in usable if all((r.get(k) or 0.0) == 0.0 for k, _ in PROBE_METRICS)]
    sm_zero = [r for r in usable if load == "compute" and (r.get("graphics") or 0) >= 25
               and (r.get("sm") if r.get("sm") is not None else 1) <= 0.1
               and (r.get("occupancy") if r.get("occupancy") is not None else 1) <= 0.1]
    n = len(usable)
    return {"intervals": n, "all_zero": len(all_zero), "sm_zero": len(sm_zero),
            "dropout_share": (len(all_zero) + len(sm_zero)) / n if n else None}


def is_dropout(result: Dict[str, Any], threshold: float = 0.5) -> bool:
    return result["dropout_share"] is not None and result["dropout_share"] >= threshold


def rows_between(rows: List[Dict[str, Any]], start: float, end: float) -> List[Dict[str, Any]]:
    return [r for r in list(rows) if r["start"] >= start and r["end"] <= end]


# ------------------------------------------------------------------------ cycles

def run_load(load: str, seconds: float, copier: Any, compute: Optional[ComputeLoad],
             clock: Callable[[], float], sleep: Callable[[float], None],
             on_check: Optional[Callable[[float], None]] = None, burst_bytes: int = 8 * MIB,
             burst_gap_s: float = 0.025) -> None:
    """Keep the GPU busy for `seconds`; call on_check(elapsed) between work items."""
    started = clock()
    while clock() - started < seconds:
        if load == "compute" and compute is not None:
            compute.launch()
            copier.sync()
            copier.h2d(MIB)
        else:
            copier.h2d(burst_bytes)
            sleep(burst_gap_s)
        if on_check is not None:
            on_check(clock() - started)
    copier.sync()


def run_cycle(cycle: Dict[str, Any], copier: Any, compute: Optional[ComputeLoad], gpm: Any, legacy: Any,
              args: Any, clock: Callable[[], float] = time.perf_counter,
              sleep: Callable[[float], None] = time.sleep) -> Dict[str, Any]:
    """One idle -> load cycle, with optional remedies after an early dropout.

    Remedies run in the sampler thread while the load continues without a
    pause: an idle gap would itself be a transition that can end a dropout.
    Stages: onset -> realloc -> check -> (park legacy) -> reinit -> check -> done.
    """
    if cycle["legacy"]:
        legacy.resume()
    else:
        legacy.pause()
    sleep(cycle["idle_s"])
    result: Dict[str, Any] = dict(cycle)
    state: Dict[str, Any] = {"stage": "onset", "remedies": [], "paused_legacy": False}
    load_start = clock()

    def evaluate(since: float, now: float) -> Dict[str, Any]:
        return classify(rows_between(gpm.rows, since, now), cycle["load"])

    def on_check(elapsed: float) -> None:
        stage, now = state["stage"], clock()
        if stage == "done" or args.no_remedy and stage != "onset":
            return
        if stage == "onset":
            if elapsed < args.onset_s:
                return
            state["onset"] = evaluate(load_start + 0.3, now)
            state["stage"] = "request_realloc" if is_dropout(state["onset"]) and not args.no_remedy else "done"
        elif stage.startswith("request_"):
            kind = stage[len("request_"):]
            if kind == "reinit" and not legacy.paused.is_set():
                legacy.paused.set()  # NVML must be idle in other threads during reinit
                state["paused_legacy"] = True
            if kind == "reinit" and legacy.is_alive() and not legacy.parked.is_set():
                return  # wait (without blocking the load) until the legacy sampler is parked
            gpm.request_remedy(kind, wait=False)
            state["stage"] = f"running_{kind}"
        elif stage.startswith("running_"):
            if not gpm.remedy_done():
                return
            kind = stage[len("running_"):]
            event = gpm.events[-1] if gpm.events else {}
            if kind == "reinit":
                legacy.handle = gpm.handle
                if state["paused_legacy"]:
                    legacy.resume()
                    state["paused_legacy"] = False
            state["remedies"].append({"kind": kind, "at_s": round(now - load_start, 2),
                                      "ok": bool(event.get("ok"))})
            state["remedy_time"], state["check_at"] = now, elapsed + args.check_s
            state["stage"] = f"check_{kind}"
        elif stage.startswith("check_"):
            if elapsed < state["check_at"]:
                return
            kind = stage[len("check_"):]
            after = evaluate(state["remedy_time"] + 0.05, now)
            state["remedies"][-1]["after"] = after
            if after["intervals"] and not is_dropout(after):
                state["recovered_by"], state["stage"] = kind, "done"
            else:
                state["stage"] = "request_reinit" if kind == "realloc" else "done"

    run_load(cycle["load"], args.load_s, copier, compute, clock, sleep, on_check)
    load_end = clock()
    if state["paused_legacy"]:
        legacy.resume()
    onset = state.get("onset") or evaluate(load_start + 0.3, min(load_end, load_start + args.onset_s))
    result.update(onset=onset, onset_dropout=is_dropout(onset), remedies=state["remedies"],
                  recovered_by=state.get("recovered_by"), load_start=load_start, load_end=load_end)
    return result


def build_plan(args: Any) -> List[Dict[str, Any]]:
    loads = ["compute", "bursty_copy"] if not args.copy_only else ["bursty_copy"]
    plan = []
    for repeat in range(args.repeats):
        for idle_s in args.idle:
            for load in loads:
                # Alternate the order of legacy on/off so time drift affects both equally.
                for legacy in ((False, True) if repeat % 2 == 0 else (True, False)):
                    plan.append({"repeat": repeat + 1, "idle_s": idle_s, "load": load, "legacy": legacy})
    return plan


# ------------------------------------------------------------------------ summary

def rate(cycles: List[Dict[str, Any]]) -> str:
    if not cycles:
        return "n/a"
    hits = sum(1 for c in cycles if c["onset_dropout"])
    return f"{hits}/{len(cycles)} ({100.0 * hits / len(cycles):.0f} %)"


def summarize(cycles: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"by_legacy_and_load": {}, "by_idle": {}, "remedies": {}}
    for load in sorted({c["load"] for c in cycles}):
        for legacy in (False, True):
            group = [c for c in cycles if c["load"] == load and c["legacy"] == legacy]
            out["by_legacy_and_load"][f"{load}, legacy {'on' if legacy else 'off'}"] = rate(group)
    for idle in sorted({c["idle_s"] for c in cycles}):
        out["by_idle"][f"{idle:g} s"] = rate([c for c in cycles if c["idle_s"] == idle])
    dropped = [c for c in cycles if c["onset_dropout"] and c["remedies"]]
    out["remedies"] = {
        "cycles_with_dropout_and_remedy": len(dropped),
        "recovered_by_realloc": sum(1 for c in dropped if c.get("recovered_by") == "realloc"),
        "recovered_by_reinit": sum(1 for c in dropped if c.get("recovered_by") == "reinit"),
        "not_recovered": sum(1 for c in dropped if not c.get("recovered_by")),
    }
    on = [c for c in cycles if c["legacy"]]
    off = [c for c in cycles if not c["legacy"]]
    on_rate = sum(c["onset_dropout"] for c in on) / len(on) if on else None
    off_rate = sum(c["onset_dropout"] for c in off) / len(off) if off else None
    out["legacy_on_rate"], out["legacy_off_rate"] = on_rate, off_rate
    total = sum(c["onset_dropout"] for c in cycles)
    readings = []
    if total < 4:
        readings.append("too few dropouts for a conclusion; run longer (--repeats)")
    elif on_rate is not None and off_rate is not None:
        if on_rate >= 2 * max(off_rate, 0.02) and on_rate - off_rate >= 0.15:
            readings.append("dropouts are clearly more frequent with the legacy PCIe sampler: likely interference")
        elif off_rate >= 0.8 * on_rate:
            readings.append("dropouts also occur without the legacy PCIe sampler: it is not the cause")
        else:
            readings.append("somewhat more dropouts with the legacy sampler; not conclusive")
    rem = out["remedies"]
    if rem["cycles_with_dropout_and_remedy"]:
        fixed = rem["recovered_by_realloc"] + rem["recovered_by_reinit"]
        readings.append(f"restarting GPM ended {fixed} of {rem['cycles_with_dropout_and_remedy']} dropouts "
                        f"(realloc {rem['recovered_by_realloc']}, reinit {rem['recovered_by_reinit']})")
    out["readings"] = readings
    return out


def print_summary(summary: Dict[str, Any], out: Any = sys.stdout) -> None:
    print("\nDropout at the start of load periods (cycles with dropout / cycles):", file=out)
    for key, value in summary["by_legacy_and_load"].items():
        print(f"  {key:<32} {value}", file=out)
    print("By idle time before the load:", file=out)
    for key, value in summary["by_idle"].items():
        print(f"  idle {key:<27} {value}", file=out)
    rem = summary["remedies"]
    print("Restarting GPM during a dropout:", file=out)
    print(f"  attempts {rem['cycles_with_dropout_and_remedy']}, recovered by realloc {rem['recovered_by_realloc']}, "
          f"by NVML reinit {rem['recovered_by_reinit']}, not recovered {rem['not_recovered']}", file=out)
    print("\nReading:", file=out)
    for line in summary["readings"]:
        print(f"  - {line}", file=out)


# --------------------------------------------------------------------------- main

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Probe NVML GPM dropouts: cause and restart remedies.",
                                 epilog="Stop llama-server and other GPU programs first.")
    ap.add_argument("--gpu", type=int, default=0, help="CUDA device ordinal (default: %(default)s)")
    ap.add_argument("--repeats", type=int, default=4, help="repetitions of the full plan (default: %(default)s)")
    ap.add_argument("--idle", type=float, nargs="+", default=[0.5, 2.0, 6.0],
                    help="idle lengths before load periods, seconds (default: %(default)s)")
    ap.add_argument("--load-s", type=float, default=5.5, help="load period length (default: %(default)s)")
    ap.add_argument("--onset-s", type=float, default=1.5,
                    help="judge a dropout after this many seconds of load (default: %(default)s)")
    ap.add_argument("--check-s", type=float, default=1.2,
                    help="time to judge a remedy (default: %(default)s)")
    ap.add_argument("--gpm-interval-ms", type=int, default=200, help="GPM interval, >100 (default: %(default)s)")
    ap.add_argument("--no-remedy", action="store_true", help="only observe; never restart GPM")
    ap.add_argument("--copy-only", action="store_true", help="skip compute kernels (only bursty copies)")
    ap.add_argument("--out-dir", default="outputs", help="report directory (default: %(default)s)")
    args = ap.parse_args(argv)
    if args.gpm_interval_ms <= 100 or args.repeats < 1 or args.load_s < args.onset_s + 2 * args.check_s + 1:
        ap.error("need --gpm-interval-ms > 100, --repeats >= 1 and --load-s >= --onset-s + 2 * --check-s + 1")

    try:
        import pynvml as nv
        nv.nvmlInit()
    except Exception as error:
        print(f"ERROR: NVML unavailable ({error}); install with: py -m pip install -U nvidia-ml-py",
              file=sys.stderr)
        return 2
    copier = compute = gpm = legacy = None
    cycles: List[Dict[str, Any]] = []
    try:
        try:
            copier = pc.CudaCopier(args.gpu, 16 * MIB)
        except Exception as error:
            print(f"ERROR: CUDA setup failed: {error}", file=sys.stderr)
            return 2
        nvml_index, handle = pc.find_nvml_handle(nv, pc.optional_call(copier.pci_bus_id), args.gpu)
        if not args.copy_only:
            try:
                compute = ComputeLoad(copier)
                print(f"compute kernel: {compute.kernel_ms:.0f} ms per launch, grid {compute.grid}x{BLOCK}",
                      file=sys.stderr)
            except Exception as error:
                print(f"WARNING: compute kernel unavailable ({error}); using copy loads only", file=sys.stderr)
                args.copy_only = True
        gpm = pc.GpmSampler(nv, handle, args.gpm_interval_ms / 1000.0, metrics=PROBE_METRICS,
                            nvml_index=nvml_index)
        gpm.prepare()
        legacy = pc.LegacySampler(nv, handle, 0.0)
        plan = build_plan(args)
        duration = sum(c["idle_s"] + args.load_s for c in plan)
        print(f"GPU {pc.optional_call(copier.name)}: {len(plan)} cycles, about {duration / 60:.1f} min",
              file=sys.stderr)
        legacy.paused.set()
        legacy.start()
        gpm.start()
        for index, cycle in enumerate(plan, 1):
            result = run_cycle(cycle, copier, compute, gpm, legacy, args)
            cycles.append(result)
            remedy = (f", {result['recovered_by']} fixed it" if result.get("recovered_by")
                      else (", not fixed by restart" if result["remedies"] else ""))
            print(f"  [{index:>3}/{len(plan)}] idle {cycle['idle_s']:>4g}s {cycle['load']:<11} "
                  f"legacy {'on ' if cycle['legacy'] else 'off'} -> "
                  f"{'DROPOUT' if result['onset_dropout'] else 'ok'}{remedy}", file=sys.stderr, flush=True)
            for sampler in (gpm, legacy):
                if sampler.error:
                    raise RuntimeError(sampler.error)
    except KeyboardInterrupt:
        print("\ninterrupted; summarizing completed cycles", file=sys.stderr)
    finally:
        if legacy is not None:
            legacy.resume()
        for sampler in (legacy, gpm):
            if sampler is not None:
                sampler.stop_event.set()
                if sampler.is_alive():
                    sampler.join(5)
        if gpm is not None:
            gpm.free()
        if compute is not None:
            compute.close()
        if copier is not None:
            copier.close()
        pc.optional_call(nv.nvmlShutdown)

    if not cycles:
        return 1
    summary = summarize(cycles)
    print_summary(summary)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "gpm-probe-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S") + ".json")
    origin = cycles[0]["load_start"]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "cycles": cycles, "remedy_events": gpm.events if gpm else [],
                   "gpm_rows": [dict(r, start=r["start"] - origin, end=r["end"] - origin)
                                for r in (gpm.rows if gpm else [])],
                   "arguments": vars(args)}, f, indent=1, default=str)
    print(f"\nsaved: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
