"""Direct NVML telemetry sampler replacing the nvidia-smi subprocess."""
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_csv_recording import benchmark as b


def fake_nvml(memory_ok=True):
    nv = SimpleNamespace(
        NVML_CLOCK_GRAPHICS=0, NVML_CLOCK_MEM=2, NVML_TEMPERATURE_GPU=0,
        NVML_PCIE_UTIL_TX_BYTES=0, NVML_PCIE_UTIL_RX_BYTES=1,
        nvmlInit=Mock(), nvmlShutdown=Mock(),
        nvmlDeviceGetCount=lambda: 1, nvmlDeviceGetHandleByIndex=lambda i: f"h{i}",
        nvmlDeviceGetMemoryInfo=(lambda h: SimpleNamespace(used=4 * 1048576 * 1024, total=16 * 1048576 * 1024))
        if memory_ok else Mock(side_effect=RuntimeError("not supported")),
        nvmlDeviceGetUtilizationRates=lambda h: SimpleNamespace(gpu=97, memory=41),
        nvmlDeviceGetPerformanceState=lambda h: 2,
        nvmlDeviceGetClockInfo=lambda h, kind: 2800 if kind == 0 else 14000,
        nvmlDeviceGetPowerUsage=lambda h: 165400,
        nvmlDeviceGetPowerManagementLimit=lambda h: 180000,
        nvmlDeviceGetTemperature=lambda h, sensor: 63,
        nvmlDeviceGetCurrPcieLinkGeneration=lambda h: 3,
        nvmlDeviceGetCurrPcieLinkWidth=lambda h: 8,
        nvmlDeviceGetPcieThroughput=Mock(side_effect=RuntimeError("no pcie in this fake")),
    )
    return nv


class NvmlSamplerTests(unittest.TestCase):
    def monitor(self, preference="auto"):
        monitor = b.NvidiaVramMonitor(interval_ms=100)
        monitor.backend_preference = preference
        for store in (monitor.samples, monitor.pcie_samples):
            store.archive_enabled = False
            self.addCleanup(store.close)
        return monitor

    def test_sample_has_nvidia_smi_fields_and_units(self):
        row = self.monitor()._nvml_sample(fake_nvml(), "h0")
        self.assertEqual((row["used_mib"], row["total_mib"], row["used_pct"]), (4096.0, 16384.0, 25.0))
        self.assertEqual((row["gpu_util_pct"], row["mem_util_pct"], row["pstate"]), (97.0, 41.0, "P2"))
        self.assertEqual((row["gpu_clock_mhz"], row["mem_clock_mhz"]), (2800.0, 14000.0))
        self.assertEqual((row["power_draw_w"], row["power_limit_w"], row["temperature_c"]), (165.4, 180.0, 63.0))
        self.assertEqual((row["pcie_link_gen"], row["pcie_link_width"]), (3.0, 8.0))
        self.assertTrue(set(row) <= set(b.VRAM_CSV_FIELDS))

    def test_optional_fields_may_be_missing(self):
        nv = fake_nvml()
        del nv.nvmlDeviceGetPerformanceState
        nv.nvmlDeviceGetPowerUsage = Mock(side_effect=RuntimeError("n/a"))
        row = self.monitor()._nvml_sample(nv, "h0")
        self.assertIsNone(row["pstate"])
        self.assertIsNone(row["power_draw_w"])
        self.assertIsNone(self.monitor()._nvml_sample(fake_nvml(memory_ok=False), "h0"))

    def test_auto_prefers_nvml_and_stops_cleanly(self):
        nv = fake_nvml()
        monitor = self.monitor()
        with patch.dict("sys.modules", {"pynvml": nv}), patch.object(b.subprocess, "Popen") as popen:
            self.assertTrue(monitor.start())
            time.sleep(0.25)
            monitor.stop()
        popen.assert_not_called()
        self.assertEqual(monitor.backend, "nvml")
        rows, _ = monitor.samples.window(-1e9, 1e9)
        self.assertGreaterEqual(len(rows), 2)
        self.assertEqual(rows[0]["gpu_index"], "0")
        self.assertEqual(rows[0]["phase"], "idle")
        self.assertEqual(nv.nvmlShutdown.call_count, nv.nvmlInit.call_count)
        self.assertFalse(monitor.thread.is_alive())

    def test_nvidia_smi_preference_skips_nvml(self):
        monitor = self.monitor("nvidia-smi")
        with patch.dict("sys.modules", {"pynvml": fake_nvml()}), patch.object(b.shutil, "which", return_value=None):
            self.assertFalse(monitor.start())
        self.assertEqual(monitor.error, "nvidia-smi not found")

    def test_failed_nvml_falls_back_or_errors(self):
        monitor = self.monitor("nvml")
        with patch.dict("sys.modules", {"pynvml": fake_nvml(memory_ok=False)}):
            self.assertFalse(monitor.start())
        self.assertIn("NVML sampler unavailable: NVML memory query failed", monitor.error)
        monitor = self.monitor("auto")
        with patch.dict("sys.modules", {"pynvml": fake_nvml(memory_ok=False)}), \
                patch.object(b.shutil, "which", return_value=None):
            self.assertFalse(monitor.start())
        self.assertIn("nvidia-smi not found (NVML sampler: NVML memory query failed)", monitor.error)


if __name__ == "__main__":
    unittest.main()
