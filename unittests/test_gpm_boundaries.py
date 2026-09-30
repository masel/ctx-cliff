"""Offline boundary tests for :class:`NvidiaGpmMonitor` NVML setup.

These tests use a small fake pynvml module and never require an NVIDIA driver.
"""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_csv_recording import benchmark as b


class FakeNvml:
    """Enough of pynvml to exercise NvidiaGpmMonitor.start/stop."""

    def __init__(self, count=1, supported=True):
        self.NVML_SUCCESS = 0
        self.NVML_GPM_METRICS_GET_VERSION = 1
        for index, (_, constant) in enumerate(b.NvidiaGpmMonitor.METRIC_DEFS, 10):
            setattr(self, constant, index)
        self.nvmlInit = Mock()
        self.nvmlShutdown = Mock()
        self.nvmlDeviceGetCount = Mock(return_value=count)
        self.nvmlDeviceGetHandleByIndex = Mock(side_effect=lambda i: f"handle-{i}")
        self.nvmlGpmQueryDeviceSupport = Mock(
            return_value=SimpleNamespace(isSupportedDevice=int(supported)))
        self.nvmlGpmSampleAlloc = Mock(side_effect=lambda: object())
        self.nvmlGpmSampleGet = Mock()
        self.nvmlGpmSampleFree = Mock()
        self.nvmlGpmMetricsGet = Mock()

        class MetricsGet:
            def __init__(self):
                self.metrics = [SimpleNamespace(metricId=0, nvmlReturn=0, value=0.0)
                                for _ in range(16)]

        self.c_nvmlGpmMetricsGet_t = MetricsGet


class GpmBoundaryTests(unittest.TestCase):
    def start_with(self, nvml, gpu="all"):
        monitor = b.NvidiaGpmMonitor(interval_ms=250, gpu=gpu)
        fake_thread = Mock()
        fake_thread.is_alive.return_value = False
        with patch.dict("sys.modules", {"pynvml": nvml}), \
             patch.object(b.threading, "Thread", return_value=fake_thread):
            result = monitor.start()
        return monitor, fake_thread, result

    def test_missing_pynvml_reports_install_hint(self):
        monitor = b.NvidiaGpmMonitor()
        with patch.dict("sys.modules", {"pynvml": None}):
            self.assertFalse(monitor.start())
        self.assertIn("nvidia-ml-py", monitor.error)
        self.assertFalse(monitor.supported)

    def test_missing_gpm_api_is_reported_without_nvml_init(self):
        nvml = FakeNvml()
        del nvml.nvmlGpmSampleFree
        monitor = b.NvidiaGpmMonitor()
        with patch.dict("sys.modules", {"pynvml": nvml}):
            self.assertFalse(monitor.start())
        self.assertIn("lacks GPM API", monitor.error)
        nvml.nvmlInit.assert_not_called()

    def test_no_metric_constants_initializes_then_shuts_down(self):
        nvml = FakeNvml()
        for _, constant in b.NvidiaGpmMonitor.METRIC_DEFS:
            delattr(nvml, constant)
        monitor, _, result = self.start_with(nvml)
        self.assertFalse(result)
        self.assertIn("no requested GPM metric constants", monitor.error)
        nvml.nvmlInit.assert_called_once_with()
        nvml.nvmlShutdown.assert_called_once_with()

    def test_all_gpu_handles_and_two_sample_pairs_are_allocated(self):
        nvml = FakeNvml(count=2)
        monitor, thread, result = self.start_with(nvml)
        self.assertTrue(result)
        self.assertEqual(monitor._handles,
                         [("0", "handle-0"), ("1", "handle-1")])
        self.assertEqual(len(monitor._sample_pairs), 2)
        self.assertEqual(nvml.nvmlGpmSampleAlloc.call_count, 4)
        thread.start.assert_called_once_with()
        monitor.stop()
        self.assertEqual(nvml.nvmlGpmSampleFree.call_count, 4)
        nvml.nvmlShutdown.assert_called_once_with()

    def test_explicit_indices_and_uuid_handles_are_resolved(self):
        nvml = FakeNvml(count=3)
        nvml.nvmlDeviceGetHandleByUUID = Mock(return_value="uuid-handle")
        nvml.nvmlDeviceGetIndex = Mock(return_value=2)
        monitor, _, result = self.start_with(nvml, gpu="1, GPU-abc")
        self.assertTrue(result)
        self.assertEqual(monitor._handles,
                         [("1", "handle-1"), ("2", "uuid-handle")])
        nvml.nvmlDeviceGetHandleByUUID.assert_called_once_with("GPU-abc")
        monitor.stop()

    def test_unsupported_gpu_fails_and_releases_nvml(self):
        nvml = FakeNvml(count=2, supported=False)
        monitor, _, result = self.start_with(nvml)
        self.assertFalse(result)
        self.assertIn("GPM not supported on GPU(s): 0,1", monitor.error)
        self.assertFalse(monitor._nvml_initialized)
        nvml.nvmlShutdown.assert_called_once_with()

    def test_optional_link_apis_are_allowed_to_be_absent_or_fail(self):
        nvml = SimpleNamespace()
        self.assertIsNone(b.NvidiaGpmMonitor._optional_link_int(nvml, "missing", object()))
        nvml.nvmlDeviceGetCurrPcieLinkWidth = Mock(side_effect=RuntimeError("unsupported"))
        self.assertIsNone(b.NvidiaGpmMonitor._optional_link_int(
            nvml, "nvmlDeviceGetCurrPcieLinkWidth", object()))

    def test_stop_is_idempotent_after_failed_setup(self):
        nvml = FakeNvml()
        for _, constant in b.NvidiaGpmMonitor.METRIC_DEFS:
            delattr(nvml, constant)
        monitor, _, _ = self.start_with(nvml)
        monitor.stop()
        monitor.stop()
        nvml.nvmlShutdown.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
