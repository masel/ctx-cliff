"""Offline tests for pcie-calibrate.py with a simulated GPU, NVML and GPM."""
import ctypes
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

# Set CTX_CLIFF_FAST_TESTS=1 to skip the simulations that run in real time (~50 s).
SLOW = unittest.skipIf(os.environ.get("CTX_CLIFF_FAST_TESTS") == "1", "slow simulation skipped")

SCRIPT = Path(os.environ.get("PCIE_CALIBRATE_SCRIPT",
                             Path(__file__).resolve().parent.parent / "pcie-calibrate.py"))
spec = importlib.util.spec_from_file_location("pcie_calibrate_test", SCRIPT)
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)


class TrafficModel:
    """Records simulated transfers; answers 'bytes per second in [a, b]'."""

    def __init__(self):
        self.events = []
        self.lock = threading.Lock()

    def add(self, start, end, direction, nbytes, control=False):
        with self.lock:
            self.events.append((start, end, direction, nbytes, control))

    def rate(self, direction, a, b, control_weight=1.0):
        """Bytes/s in [a, b]; control (small-transaction) bytes weighted separately."""
        if b <= a:
            return 0.0
        total = 0.0
        with self.lock:
            events = list(self.events)
        for start, end, event_direction, nbytes, control in events:
            if event_direction != direction or end <= a or start >= b:
                continue
            overlap = min(end, b) - max(start, a)
            weight = control_weight if control else 1.0
            total += weight * nbytes * overlap / max(end - start, 1e-12)
        return total / (b - a)


class FakeCopier:
    """Copies at a fixed rate; every copy also costs setup time and small control
    transactions in both directions (doorbells, semaphores, completions)."""

    SETUP_S = 0.0002
    CONTROL_BYTES = 1024

    def __init__(self, model, bytes_per_s=2e9):
        self.model, self.rate = model, bytes_per_s

    def _copy(self, direction, nbytes):
        # Book the transfer up front, spread evenly over its duration, so sensors
        # sampling during the copy see it (as the real counters would).
        start = time.perf_counter()
        duration = self.SETUP_S + nbytes / self.rate
        self.model.add(start, start + duration, direction, nbytes)
        for control_direction in ("rx", "tx"):
            self.model.add(start, start + duration, control_direction, self.CONTROL_BYTES, control=True)
        time.sleep(duration)

    def h2d(self, nbytes):
        self._copy("rx", nbytes)

    def d2h(self, nbytes):
        self._copy("tx", nbytes)

    def sync(self):
        pass


class FakeMetric:
    def __init__(self):
        self.metricId, self.value, self.nvmlReturn = 0, 0.0, 0


class FakeGet:
    def __init__(self):
        self.metrics = [FakeMetric() for _ in range(2)]


class FakeNvml:
    """NVML double: legacy sensor = 20 ms window per call, scaled; GPM = exact MiB/s."""
    NVML_PCIE_UTIL_TX_BYTES = 0
    NVML_PCIE_UTIL_RX_BYTES = 1
    NVML_GPM_METRIC_PCIE_RX_PER_SEC = 20
    NVML_GPM_METRIC_PCIE_TX_PER_SEC = 21
    NVML_GPM_METRICS_GET_VERSION = 1
    NVML_SUCCESS = 0
    c_nvmlGpmMetricsGet_t = FakeGet

    def __init__(self, model, legacy_scale=1.0, legacy_unit=1000.0, gpm_scale=1.0, legacy_control_weight=1.0):
        self.model, self.legacy_scale, self.legacy_unit, self.gpm_scale = model, legacy_scale, legacy_unit, gpm_scale
        self.legacy_control_weight = legacy_control_weight

    def nvmlDeviceGetPcieThroughput(self, handle, counter):
        a = time.perf_counter()
        time.sleep(0.02)
        b = time.perf_counter()
        direction = "rx" if counter == self.NVML_PCIE_UTIL_RX_BYTES else "tx"
        rate = self.model.rate(direction, a, b, self.legacy_control_weight)
        return int(rate * self.legacy_scale / self.legacy_unit)

    def nvmlGpmQueryDeviceSupport(self, handle):
        return SimpleNamespace(isSupportedDevice=1)

    def nvmlGpmSampleAlloc(self):
        return SimpleNamespace(t=None)

    def nvmlGpmSampleGet(self, handle, sample):
        sample.t = time.perf_counter()

    def nvmlGpmMetricsGet(self, get):
        a, b = get.sample1.t, get.sample2.t
        for metric in get.metrics:
            direction = "rx" if metric.metricId == self.NVML_GPM_METRIC_PCIE_RX_PER_SEC else "tx"
            metric.value = self.model.rate(direction, a, b) * self.gpm_scale / pc.MIB

    def nvmlGpmSampleFree(self, sample):
        pass

    def nvmlDeviceGetCurrPcieLinkGeneration(self, handle):
        return 3

    def nvmlDeviceGetCurrPcieLinkWidth(self, handle):
        return 8


def run_args(**overrides):
    values = dict(seconds=0.7, idle_seconds=0.3, tiny_bytes=64, buffer_mib=32, burst_mib=2, burst_gap_ms=10.0,
                  gpm_interval_ms=120, legacy_interval_ms=0, no_gpm=False)
    values.update(overrides)
    return SimpleNamespace(**values)


def simulate(**nvml_options):
    model = TrafficModel()
    return pc.calibrate(run_args(), FakeNvml(model, **nvml_options), FakeCopier(model),
                        handle=object(), out=io.StringIO())


@SLOW
class EndToEndSimulationTests(unittest.TestCase):
    accurate = None

    @classmethod
    def setUpClass(cls):
        cls.accurate = simulate()  # shared: each simulated run takes a few seconds

    def simulate(self, **nvml_options):
        return simulate(**nvml_options)

    def test_accurate_sensors_give_ratio_near_one(self):
        report = self.accurate
        s = report["summary"]
        self.assertAlmostEqual(s["steady_ratio_gpm_if_MiB"], 1.0, delta=0.05)
        self.assertAlmostEqual(s["steady_ratio_legacy_if_KB"], 1.0, delta=0.12)
        self.assertEqual(s["gpm_reading"], "matches the transferred payload")
        steady = [r for r in report["results"] if r["phase"] == "h2d"][0]
        self.assertGreater(steady["gpm_rx_samples"], 3)
        self.assertGreater(steady["legacy_rx_samples"], 5)
        self.assertAlmostEqual(steady["true_h2d_MB_s"], 2000, delta=250)
        self.assertLess(steady["reverse_gpm_MB_s"], 1)
        self.assertEqual([p["phase"] for p in report["phases"]],
                         ["idle", "h2d", "idle", "d2h", "idle", "h2d_bursty", "idle", "d2h_bursty", "idle",
                          "h2d_tiny", "idle", "d2h_tiny", "idle"])
        self.assertIn("alike", report["summary"]["tiny_reading"])
        tiny = [r for r in report["results"] if r["phase"] == "h2d_tiny"][0]
        self.assertGreater(tiny["copies_per_s"], 500)
        self.assertNotIn("ratio_gpm_if_MiB", tiny)
        # Reported traffic far exceeds the 64-byte payloads: control transactions dominate.
        self.assertGreater(report["summary"]["tiny_gpm_total_MB_s"], 20 * report["summary"]["tiny_payload_MB_s"])
        self.assertEqual(report["phases"][1]["link"], {"gen": 3, "width": 8})

    def test_overcounting_and_small_transaction_bias_are_detected(self):
        # One run: legacy over-reports everything by 1.7 and control traffic by a further 3x.
        report = self.simulate(legacy_scale=1.7, legacy_control_weight=3.0)
        s = report["summary"]
        self.assertAlmostEqual(s["steady_ratio_legacy_if_KB"], 1.7, delta=0.2)
        self.assertEqual(s["legacy_reading"], "far off the transferred payload")
        self.assertAlmostEqual(s["legacy_over_gpm_steady"], 1.7, delta=0.2)
        self.assertAlmostEqual(s["legacy_over_gpm_tiny"], 5.1, delta=0.9)
        self.assertIn("legacy sensor reports clearly more", s["tiny_reading"])
        self.assertAlmostEqual(s["steady_ratio_gpm_if_MiB"], 1.0, delta=0.05)

    def test_kib_unit_shows_up_in_alternative_ratio(self):
        report = self.simulate(legacy_unit=1024.0)
        s = report["summary"]
        self.assertAlmostEqual(s["steady_ratio_legacy_if_KiB"], 1.0, delta=0.12)
        self.assertLess(s["steady_ratio_legacy_if_KB"], s["steady_ratio_legacy_if_KiB"])

    def test_without_gpm_only_legacy_is_checked(self):
        model = TrafficModel()
        report = pc.calibrate(run_args(no_gpm=True, seconds=1.0), FakeNvml(model), FakeCopier(model),
                              object(), out=io.StringIO())
        self.assertIsNone(report["summary"]["steady_ratio_gpm_if_MiB"])
        self.assertIsNotNone(report["summary"]["steady_ratio_legacy_if_KB"])
        self.assertEqual(report["gpm_rows"], [])


class AnalysisTests(unittest.TestCase):
    def test_only_snapshots_fully_inside_the_phase_count(self):
        phase = {"phase": "h2d", "start": 0.0, "end": 1.0, "h2d_bytes": 1_000_000_000, "d2h_bytes": 0}
        legacy = [
            {"rx_start": 0.0, "rx_end": 0.02, "tx_start": 0.02, "tx_end": 0.04, "rx_raw": 1, "tx_raw": 0},
            {"rx_start": 0.5, "rx_end": 0.52, "tx_start": 0.52, "tx_end": 0.54, "rx_raw": 1_000_000, "tx_raw": 50},
            {"rx_start": 0.97, "rx_end": 0.99, "tx_start": 0.99, "tx_end": 1.01, "rx_raw": 5, "tx_raw": 5},
        ]
        gpm = [{"start": 0.1, "end": 0.3, "rx_mib_s": 1e9 / pc.MIB, "tx_mib_s": 0.0},
               {"start": 0.3, "end": 0.9, "rx_mib_s": 1e9 / pc.MIB, "tx_mib_s": 0.0},
               {"start": 0.9, "end": 1.1, "rx_mib_s": 0.0, "tx_mib_s": 0.0}]
        result = pc.analyze_phase(phase, legacy, gpm)
        self.assertEqual(result["legacy_rx_samples"], 1)
        self.assertEqual(result["gpm_rx_samples"], 2)
        self.assertAlmostEqual(result["ratio_legacy_if_KB"], 1.0)
        self.assertAlmostEqual(result["ratio_legacy_if_KiB"], 1.024)
        self.assertAlmostEqual(result["ratio_gpm_if_MiB"], 1.0)
        self.assertAlmostEqual(result["reverse_legacy_MB_s"], 0.05)

    def test_gpm_average_is_time_weighted(self):
        phase = {"phase": "d2h", "start": 0.0, "end": 2.0, "h2d_bytes": 0, "d2h_bytes": 2 * 300 * pc.MIB}
        gpm = [{"start": 0.1, "end": 0.2, "rx_mib_s": 0, "tx_mib_s": 30.0},
               {"start": 0.2, "end": 1.9, "rx_mib_s": 0, "tx_mib_s": 300.0}]
        result = pc.analyze_phase(phase, [], gpm)
        self.assertAlmostEqual(result["gpm_tx_raw_mib_s"], (0.1 * 30 + 1.7 * 300) / 1.8)
        self.assertIsNone(result["ratio_legacy_if_KB"])

    def test_interpretation_bands(self):
        self.assertEqual(pc.interpret(None), "no data")
        self.assertEqual(pc.interpret(1.01), "matches the transferred payload")
        self.assertIn("KiB-vs-KB", pc.interpret(1.04))
        self.assertIn("protocol overhead", pc.interpret(1.2))
        self.assertEqual(pc.interpret(0.93), "slightly below payload")
        self.assertEqual(pc.interpret(1.7), "far off the transferred payload")

    def test_burst_reading_judges_each_sensor_against_truth(self):
        rows = [{"phase": "h2d", "ratio_gpm_if_MiB": 1.08, "ratio_legacy_if_KB": 1.66, "ratio_legacy_if_KiB": 1.7},
                {"phase": "h2d_bursty", "ratio_gpm_if_MiB": 1.10, "ratio_legacy_if_KB": 1.65}]
        summary = pc.summarize(rows)
        self.assertEqual(summary["burst_reading"], "both sensors keep their steady-state scale for bursty traffic")
        rows[1]["ratio_legacy_if_KB"] = 2.5
        self.assertIn("legacy (1.51x its steady scale)", pc.summarize(rows)["burst_reading"])
        self.assertNotIn("GPM (", pc.summarize(rows)["burst_reading"])

    def test_gpm_all_zero_intervals_during_copies_are_dropouts(self):
        phase = {"phase": "h2d_bursty", "start": 0.0, "end": 1.0, "h2d_bytes": 300 * pc.MB, "d2h_bytes": 0}
        gpm = [{"start": 0.1, "end": 0.3, "rx_mib_s": 300.0, "tx_mib_s": 18.0},
               {"start": 0.3, "end": 0.5, "rx_mib_s": 0.0, "tx_mib_s": 0.0},
               {"start": 0.5, "end": 0.7, "rx_mib_s": 0.0, "tx_mib_s": 0.0},
               {"start": 0.7, "end": 0.9, "rx_mib_s": 300.0, "tx_mib_s": 18.0}]
        result = pc.analyze_phase(phase, [], gpm)
        self.assertEqual((result["gpm_zero_intervals"], result["gpm_intervals"], result["gpm_rx_samples"]), (2, 4, 2))
        self.assertAlmostEqual(result["gpm_rx_raw_mib_s"], 300.0)
        idle = pc.analyze_phase(dict(phase, phase="idle", h2d_bytes=0), [], gpm)
        self.assertEqual(idle["gpm_zero_intervals"], 0)  # zeros are real in idle phases
        self.assertEqual(idle["gpm_rx_samples"], 4)
        summary = pc.summarize([result, idle])
        self.assertEqual(summary["gpm_dropout_intervals"], 2)
        self.assertEqual(summary["gpm_dropouts_by_phase"], {"h2d_bursty": 2})

    def test_gpm_lag_from_phase_edges(self):
        # Copies run 1.0-3.0 s at 1000 MiB/s; GPM intervals of 0.2 s start at 0.05 s
        # and report the traffic 60 ms late.
        lag, rows = 0.06, []
        for k in range(20):
            a, b = 0.05 + 0.2 * k, 0.25 + 0.2 * k
            covered = max(0.0, min(b, 3.0 + lag) - max(a, 1.0 + lag))
            rows.append({"start": a, "end": b, "rx_mib_s": 1000.0 * covered / 0.2, "tx_mib_s": 1.0})
        lags = pc.estimate_gpm_lag([{"phase": "h2d", "start": 1.0, "end": 3.0}], rows)
        self.assertEqual(len(lags), 2)
        for value in lags:
            self.assertAlmostEqual(value, 60.0, delta=1.0)
        no_lag = [dict(r, rx_mib_s=1000.0 * max(0.0, min(r["end"], 3.0) - max(r["start"], 1.0)) / 0.2)
                  for r in rows]
        for value in pc.estimate_gpm_lag([{"phase": "h2d", "start": 1.0, "end": 3.0}], no_lag):
            self.assertAlmostEqual(value, 0.0, delta=1.0)

    @SLOW
    def test_report_and_files(self):
        report = dict(EndToEndSimulationTests.accurate or simulate())
        out = io.StringIO()
        pc.print_report(report["results"], report["summary"], out)
        text = out.getvalue()
        for phase in ("h2d ", "d2h ", "h2d_bursty", "d2h_bursty", "h2d_tiny", "d2h_tiny", "idle"):
            self.assertIn(phase, text)
        self.assertIn("Verdict (steady copies)", text)
        # 0.7 s simulated phases are too short for a lag estimate (tested separately).
        self.assertNotIn("GPM DROPOUT", text)
        lag_text = io.StringIO()
        pc.print_report(report["results"], dict(report["summary"], gpm_lag_ms=72.0,
                                                gpm_lag_edges_ms=[63.0, 81.0]), lag_text)
        self.assertIn("GPM timing: values lag the copies by ~72 ms (from steady phase edges: 63, 81 ms)",
                      lag_text.getvalue())
        self.assertIn("Suggested ctx-cliff options for this GPU/driver: --gpm-lag-ms 70 --pcie-legacy-scale",
                      lag_text.getvalue())
        self.assertIn("Small transactions (tiny copies", text)
        with tempfile.TemporaryDirectory() as tmp:
            stem = os.path.join(tmp, "cal")
            legacy, gpm = report.pop("legacy_rows"), report.pop("gpm_rows")
            pc.write_outputs(stem, report, legacy, gpm, report["phases"])
            data = json.loads(Path(stem + ".json").read_text(encoding="utf-8"))
            self.assertIn("summary", data)
            csv_text = Path(stem + ".csv").read_text(encoding="utf-8")
            self.assertIn("legacy,h2d,", csv_text)
            self.assertIn("gpm,d2h,", csv_text)


class HelperTests(unittest.TestCase):
    def test_bus_id_normalization_and_lookup(self):
        self.assertEqual(pc.normalize_bus_id("0000:01:00.0"), pc.normalize_bus_id(b"00000000:01:00.0\x00"))
        handles = ["h0", "h1"]
        nv = SimpleNamespace(
            nvmlDeviceGetCount=lambda: 2,
            nvmlDeviceGetHandleByIndex=lambda i: handles[i],
            nvmlDeviceGetPciInfo=lambda h: SimpleNamespace(busId=b"00000000:0" + (b"1" if h == "h0" else b"2")
                                                           + b":00.0"))
        self.assertEqual(pc.find_nvml_handle(nv, "0000:02:00.0", 0), (1, "h1"))
        self.assertEqual(pc.find_nvml_handle(nv, None, 0), (0, "h0"))
        self.assertEqual(pc.find_nvml_handle(nv, "0000:09:00.0", 1), (1, "h1"))

    def test_argument_validation(self):
        for argv in (["--seconds", "0.5"], ["--gpm-interval-ms", "100"], ["--burst-mib", "0"], ["--tiny-bytes", "0"],
                     ["--buffer-mib", "4", "--burst-mib", "8"]):
            with self.subTest(argv=argv), self.assertRaises(SystemExit), \
                    unittest.mock.patch("sys.stderr", new=io.StringIO()):
                pc.main(argv)


class FakeCudaLib:
    """Python stand-in for nvcuda: writes through ctypes.byref like the driver."""

    def __init__(self, fail=None):
        self.calls, self.fail = [], fail
        for name in pc.CUDA_PROTOTYPES:
            if name != "cuDevicePrimaryCtxRelease":
                setattr(self, name, self._make(name))

    def _make(self, name):
        def call(*args):
            self.calls.append(name)
            if name == self.fail:
                return 2  # CUDA_ERROR_OUT_OF_MEMORY
            if name == "cuDeviceGet":
                args[0]._obj.value = args[1]
            elif name == "cuDevicePrimaryCtxRetain":
                args[0]._obj.value = 0x10
            elif name == "cuMemAlloc_v2":
                args[0]._obj.value = 0x1000
            elif name == "cuMemAllocHost_v2":
                args[0]._obj.value = 0x2000
            elif name == "cuGetErrorName":
                args[1]._obj.value = b"CUDA_ERROR_OUT_OF_MEMORY"
            elif name in ("cuDeviceGetName", "cuDeviceGetPCIBusId"):
                ctypes.memmove(args[0], b"GPU-X\x00" if name == "cuDeviceGetName" else b"0000:01:00.0\x00", 13)
            return 0
        return call


class CudaCopierTests(unittest.TestCase):
    def test_lifecycle_and_copies(self):
        lib = FakeCudaLib()
        copier = pc.CudaCopier(0, 1024, lib=lib)
        copier.h2d(512)
        copier.d2h(512)
        copier.sync()
        self.assertEqual(copier.pci_bus_id(), "0000:01:00.0")
        self.assertEqual(copier.name(), "GPU-X")
        copier.close()
        copier.close()  # idempotent
        self.assertEqual(lib.calls[:6], ["cuInit", "cuDeviceGet", "cuDevicePrimaryCtxRetain", "cuCtxSetCurrent",
                                         "cuMemAlloc_v2", "cuMemAllocHost_v2"])
        self.assertEqual(lib.calls[-3:], ["cuMemFree_v2", "cuMemFreeHost", "cuDevicePrimaryCtxRelease_v2"])

    def test_allocation_failure_is_named_and_cleans_up(self):
        lib = FakeCudaLib(fail="cuMemAllocHost_v2")
        with self.assertRaisesRegex(pc.CudaError, "cuMemAllocHost.*CUDA_ERROR_OUT_OF_MEMORY"):
            pc.CudaCopier(0, 1024, lib=lib)
        self.assertIn("cuMemFree_v2", lib.calls)
        self.assertEqual(lib.calls[-1], "cuDevicePrimaryCtxRelease_v2")
        self.assertNotIn("cuMemFreeHost", lib.calls)


import unittest.mock  # noqa: E402  (used in HelperTests)

if __name__ == "__main__":
    unittest.main()
