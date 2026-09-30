"""Offline tests for gpm-probe.py with a simulated GPU whose dropout cause is known."""
import importlib.util
import io
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest

# Set CTX_CLIFF_FAST_TESTS=1 to skip the simulations that run in real time (~50 s).
SLOW = unittest.skipIf(os.environ.get("CTX_CLIFF_FAST_TESTS") == "1", "slow simulation skipped")

SCRIPT = Path(os.environ.get("GPM_PROBE_SCRIPT", Path(__file__).resolve().parent.parent / "gpm-probe.py"))
spec = importlib.util.spec_from_file_location("gpm_probe_test", SCRIPT)
gp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gp)
pc = gp.pc


class World:
    """Simulated GPU: busy while copies run; dropout state set at load starts.

    cause='legacy'  -> a load period starts in dropout if the legacy sampler ran recently
    cause='always'  -> every load period starts in dropout
    cause='never'   -> no dropouts
    fix='realloc' | 'reinit' | None -> which GPM restart ends a dropout
    """

    def __init__(self, cause="legacy", fix="realloc"):
        self.cause, self.fix = cause, fix
        self.lock = threading.Lock()
        self.busy_until = 0.0
        self.last_busy = -10.0
        self.legacy_seen = -10.0
        self.dropout = False
        self.remedies = []
        self.handles_issued = 0

    def work(self, seconds):
        now = time.perf_counter()
        with self.lock:
            if now - self.last_busy > 0.08:  # idle -> load transition
                recent_legacy = now - self.legacy_seen < 0.05  # sampler polls every ~10 ms
                self.dropout = (self.cause == "always" or (self.cause == "legacy" and recent_legacy))
            self.busy_until = max(self.busy_until, now + seconds)
            self.last_busy = now + seconds
        time.sleep(seconds)

    def busy(self, a, b):
        with self.lock:
            return self.busy_until > a and self.last_busy - 0.5 < b

    def remedy(self, kind):
        with self.lock:
            self.remedies.append(kind)
            if self.fix == kind or (self.fix == "realloc" and kind == "reinit"):
                self.dropout = False


class FakeCopier:
    def __init__(self, world):
        self.world = world

    def h2d(self, nbytes):
        self.world.work(0.004)

    def sync(self):
        pass


class FakeMetric:
    def __init__(self):
        self.metricId, self.value, self.nvmlReturn = 0, 0.0, 0


class FakeGet:
    def __init__(self):
        self.metrics = [FakeMetric() for _ in range(8)]


class FakeNvml:
    NVML_SUCCESS = 0
    NVML_GPM_METRICS_GET_VERSION = 1
    NVML_PCIE_UTIL_TX_BYTES, NVML_PCIE_UTIL_RX_BYTES = 0, 1
    NVML_GPM_METRIC_GRAPHICS_UTIL, NVML_GPM_METRIC_SM_UTIL, NVML_GPM_METRIC_SM_OCCUPANCY = 1, 2, 3
    NVML_GPM_METRIC_PCIE_RX_PER_SEC, NVML_GPM_METRIC_PCIE_TX_PER_SEC = 20, 21
    c_nvmlGpmMetricsGet_t = FakeGet

    def __init__(self, world):
        self.world = world
        self.initialized = 1
        self.freed = 0

    # legacy sensor
    def nvmlDeviceGetPcieThroughput(self, handle, counter):
        assert self.initialized, "NVML used while shut down"
        time.sleep(0.005)
        self.world.legacy_seen = time.perf_counter()
        return 1000

    # GPM
    def nvmlGpmQueryDeviceSupport(self, handle):
        return SimpleNamespace(isSupportedDevice=1)

    def nvmlGpmSampleAlloc(self):
        return SimpleNamespace(t=None)

    def nvmlGpmSampleFree(self, sample):
        self.freed += 1

    def nvmlGpmSampleGet(self, handle, sample):
        sample.t = time.perf_counter()

    def nvmlGpmMetricsGet(self, get):
        busy = self.world.busy(get.sample1.t, get.sample2.t)
        values = {1: 90.0, 2: 60.0, 3: 30.0, 20: 300.0, 21: 20.0} if busy and not self.world.dropout else {}
        for metric in get.metrics:
            metric.value = values.get(metric.metricId, 0.0)

    # NVML lifecycle
    def nvmlShutdown(self):
        self.initialized = 0
        self.world.remedy("reinit")

    def nvmlInit(self):
        self.initialized = 1

    def nvmlDeviceGetHandleByIndex(self, index):
        self.world.handles_issued += 1
        return f"handle-{self.world.handles_issued}"


def probe_args(**overrides):
    values = dict(repeats=2, idle=[0.12], load_s=1.6, onset_s=0.6, check_s=0.3, no_remedy=False,
                  copy_only=True)
    values.update(overrides)
    return SimpleNamespace(**values)


def run_probe(world, **overrides):
    """Drive run_cycle like main() does, with fakes instead of CUDA/NVML."""
    args = probe_args(**overrides)
    nv = FakeNvml(world)
    gpm = pc.GpmSampler(nv, "handle-0", 0.05, metrics=gp.PROBE_METRICS, nvml_index=0)
    gpm.prepare()
    original_prepare_alloc = gpm._apply_remedy

    def apply(kind):
        original_prepare_alloc(kind)
        if kind == "realloc":
            world.remedy("realloc")
    gpm._apply_remedy = apply
    legacy = pc.LegacySampler(nv, "handle-0", 0.0)
    legacy.paused.set()
    legacy.start()
    gpm.start()
    copier = FakeCopier(world)
    cycles = []
    try:
        for cycle in gp.build_plan(args):
            cycles.append(gp.run_cycle(cycle, copier, None, gpm, legacy, args))
    finally:
        legacy.resume()
        for sampler in (legacy, gpm):
            sampler.stop_event.set()
            sampler.join(3)
    return cycles, gp.summarize(cycles), legacy, gpm


@SLOW
class ProbeSimulationTests(unittest.TestCase):
    def test_legacy_caused_dropouts_are_identified_and_realloc_fixes_them(self):
        world = World(cause="legacy", fix="realloc")
        cycles, summary, legacy, gpm = run_probe(world, repeats=2)
        self.assertEqual(summary["legacy_on_rate"], 1.0)
        self.assertEqual(summary["legacy_off_rate"], 0.0)
        with_legacy = [c for c in cycles if c["legacy"]]
        self.assertTrue(all(c["recovered_by"] == "realloc" for c in with_legacy))
        self.assertTrue(all(not c["remedies"] for c in cycles if not c["legacy"]))
        # Too few cycles in this short simulation for a verdict, which is reported honestly.
        self.assertIn("too few dropouts", " ".join(summary["readings"]))

    def test_verdicts_with_enough_cycles(self):
        base = {"load": "bursty_copy", "idle_s": 1.0, "remedies": [], "recovered_by": None}
        legacy_caused = ([dict(base, legacy=True, onset_dropout=True)] * 6 +
                         [dict(base, legacy=False, onset_dropout=False)] * 6)
        self.assertIn("clearly more frequent with the legacy PCIe sampler",
                      " ".join(gp.summarize(legacy_caused)["readings"]))
        independent = ([dict(base, legacy=True, onset_dropout=True)] * 4 +
                       [dict(base, legacy=False, onset_dropout=True)] * 4)
        self.assertIn("not the cause", " ".join(gp.summarize(independent)["readings"]))

    def test_reinit_is_tried_after_failed_realloc_and_updates_legacy_handle(self):
        world = World(cause="always", fix="reinit")
        cycles, summary, legacy, gpm = run_probe(world, repeats=1)
        self.assertTrue(all(c["onset_dropout"] for c in cycles))
        self.assertEqual([r["kind"] for r in cycles[0]["remedies"]], ["realloc", "reinit"])
        self.assertTrue(all(c["recovered_by"] == "reinit" for c in cycles))
        self.assertEqual(summary["remedies"]["recovered_by_reinit"], len(cycles))
        self.assertEqual(legacy.handle, gpm.handle)
        self.assertTrue(gpm.handle.startswith("handle-"))
        self.assertIsNone(legacy.error)  # never polled while NVML was shut down
        self.assertIsNone(gpm.error)

    def test_unfixable_dropout_is_reported(self):
        world = World(cause="always", fix=None)
        cycles, summary, _, _ = run_probe(world, repeats=1)
        self.assertEqual(summary["remedies"]["not_recovered"], len(cycles))
        self.assertTrue(all(len(c["remedies"]) == 2 for c in cycles))

    def test_no_remedy_only_observes(self):
        world = World(cause="always", fix="realloc")
        cycles, summary, _, _ = run_probe(world, repeats=1, no_remedy=True)
        self.assertTrue(all(c["onset_dropout"] and not c["remedies"] for c in cycles))
        self.assertEqual(world.remedies, [])

    def test_healthy_gpu_shows_no_dropouts(self):
        cycles, summary, _, _ = run_probe(World(cause="never"), repeats=1)
        self.assertFalse(any(c["onset_dropout"] for c in cycles))
        self.assertTrue(all(c["onset"]["intervals"] >= 3 for c in cycles))


class ProbeUnitTests(unittest.TestCase):
    def test_classify(self):
        busy = {"graphics": 90.0, "sm": 60.0, "occupancy": 30.0, "rx_mib_s": 300.0, "tx_mib_s": 20.0}
        zero = dict.fromkeys(busy, 0.0)
        sm_zero = dict(busy, sm=0.0, occupancy=0.05)
        self.assertEqual(gp.classify([busy, busy], "compute")["dropout_share"], 0.0)
        result = gp.classify([zero, sm_zero, busy, busy], "compute")
        self.assertEqual((result["all_zero"], result["sm_zero"], result["dropout_share"]), (1, 1, 0.5))
        self.assertEqual(gp.classify([sm_zero], "bursty_copy")["sm_zero"], 0)
        self.assertIsNone(gp.classify([], "compute")["dropout_share"])

    def test_plan_alternates_legacy_order(self):
        plan = gp.build_plan(probe_args(repeats=2, idle=[0.5, 2.0], copy_only=False))
        self.assertEqual(len(plan), 2 * 2 * 2 * 2)
        self.assertEqual([c["legacy"] for c in plan[:2]], [False, True])
        self.assertEqual([c["legacy"] for c in plan[8:10]], [True, False])

    def test_print_summary(self):
        base = {"load": "bursty_copy", "idle_s": 1.0}
        cycles = [dict(base, legacy=True, onset_dropout=True, remedies=[{"kind": "realloc"}],
                       recovered_by="realloc")] * 4 + [dict(base, legacy=False, onset_dropout=False,
                                                            remedies=[], recovered_by=None)] * 4
        out = io.StringIO()
        gp.print_summary(gp.summarize(cycles), out)
        text = out.getvalue()
        self.assertIn("bursty_copy, legacy on", text)
        self.assertIn("4/4 (100 %)", text)
        self.assertIn("restarting GPM ended 4 of 4 dropouts", text)

    def test_argument_validation(self):
        import unittest.mock
        for argv in (["--gpm-interval-ms", "100"], ["--load-s", "2"], ["--repeats", "0"]):
            with self.subTest(argv=argv), self.assertRaises(SystemExit), \
                    unittest.mock.patch("sys.stderr", new=io.StringIO()):
                gp.main(argv)


if __name__ == "__main__":
    unittest.main()
