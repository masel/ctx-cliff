"""Offline regression tests; for the current benchmark in the project root.

Run: python -m unittest discover -s unittests -v
Or set CTX_CLIFF_SCRIPT to the installed benchmark script's absolute path.
"""
import contextlib
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

DEFAULT_SCRIPT = Path(__file__).resolve().parent.parent / 'ctx-cliff.py'
SCRIPT = Path(os.environ.get('CTX_CLIFF_SCRIPT', DEFAULT_SCRIPT))
spec = importlib.util.spec_from_file_location('ctx_cliff_test', SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def read_rows(path):
    with open(path, encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.input = self.root / 'input.txt'
        self.input.write_text('text ' * 10000, encoding='utf-8')
        self.csv = self.root / 'run.csv'
        self.args = SimpleNamespace(csv=str(self.csv), file=str(self.input), server_log=None,
                                    vram_log='auto', gpm_log='auto', win_gpu_mem='auto',
                                    vram_csv=None, pcie_csv=None, gpm_csv=None, win_gpu_mem_csv=None)
        self.addCleanup(patch.stopall)
        patch('sys.stderr', new=io.StringIO()).start()
        patch('sys.stdout', new=io.StringIO()).start()

    def test_headers_and_result_visible_before_close(self):
        with benchmark.CsvRecording(self.args) as recording:
            for name, _, fields in recording.specs:
                if name in recording.paths:
                    self.assertEqual(read_rows(recording.paths[name]), (list(fields), []))
            row = dict.fromkeys(benchmark.RESULT_CSV_FIELDS, 0)
            row['target_ctx'] = 123
            recording.write_result(row)
            self.assertEqual(read_rows(self.csv)[1][0]['target_ctx'], '123')

    def test_result_writer_rejects_missing_field(self):
        with benchmark.CsvRecording(self.args) as recording:
            row = dict.fromkeys(benchmark.RESULT_CSV_FIELDS, 0)
            del row['target_ctx']
            with self.assertRaisesRegex(ValueError, 'target_ctx'):
                recording.write_result(row)
        self.assertEqual(read_rows(self.csv)[1], [])

    def test_started_monitor_error_is_reported_but_failed_auto_start_is_not(self):
        monitor = SimpleNamespace(lock=threading.Lock(), samples=[],
                                  error='GPM polling failed: synthetic')
        with benchmark.CsvRecording(self.args) as recording:
            recording.attach('gpm', monitor, 'samples')
            recording.check()  # An optional monitor may have failed during startup.
            recording.activate(monitor)
            with self.assertRaisesRegex(RuntimeError, 'GPM polling failed: synthetic'):
                recording.check()

    def test_pcie_runtime_error_is_fatal_but_unavailable_pcie_and_bus_are_not(self):
        monitor = SimpleNamespace(lock=threading.Lock(), pcie_samples=[], error=None,
                                  pcie_source='none', pcie_error='NVML PCIe unavailable',
                                  bus_error='BUS unavailable')
        with benchmark.CsvRecording(self.args) as recording:
            recording.attach('pcie', monitor, 'pcie_samples')
            recording.activate(monitor)
            recording.check()

            monitor.pcie_source = 'nvml'
            monitor.pcie_error = 'NVML PCIe polling failed: synthetic'
            with self.assertRaisesRegex(RuntimeError, 'NVML PCIe polling failed: synthetic'):
                recording.check()

            monitor.pcie_error = None
            recording.check()  # BUS diagnostics remain optional.

    def test_automatic_names_preserve_existing_trace_stems(self):
        with patch.object(benchmark.dt, 'datetime') as clock:
            clock.now.return_value.strftime.return_value = '20260913-153042'
            orphan = self.root / 'ctx-cliff-20260913-153042.gpm.csv'
            orphan.write_text('old trace', encoding='utf-8')
            first = benchmark.allocate_csv_path(str(self.root))
            second = benchmark.allocate_csv_path(str(self.root))
        self.assertEqual(Path(first).name, 'ctx-cliff-20260913-153042-1.csv')
        self.assertEqual(Path(second).name, 'ctx-cliff-20260913-153042-2.csv')
        self.assertTrue(Path(first).exists())
        self.assertTrue(Path(second).exists())
        self.assertEqual(orphan.read_text(encoding='utf-8'), 'old trace')

    def test_cli_aliases_accept_automatic_and_explicit_names(self):
        for flag in ('--csv', '--csv-export'):
            for automatic in (True, False):
                with self.subTest(flag=flag, automatic=automatic):
                    folder = self.root / f'{flag.strip("-")}-{automatic}'
                    target = self.root / f'{flag.strip("-")}-explicit.csv'
                    argv = [str(SCRIPT), '--file', str(self.input), flag]
                    if not automatic:
                        argv.append(str(target))
                    argv += ['--csv-dir', str(folder), '--vram-log', 'off', '--gpm-log', 'off', '--win-gpu-mem', 'off']
                    seen = []

                    def run(args, ap, resources, recording):
                        seen.append(args.csv)
                        self.assertEqual(read_rows(args.csv), (list(benchmark.RESULT_CSV_FIELDS), []))

                    with patch('sys.argv', argv), patch.object(benchmark, 'run_benchmark', side_effect=run):
                        benchmark.main()
                    self.assertEqual(len(seen), 1)
                    if automatic:
                        self.assertEqual(Path(seen[0]).parent, folder)
                        self.assertRegex(Path(seen[0]).name, r'^ctx-cliff-\d{8}-\d{6}(?:-\d+)?\.csv$')
                    else:
                        self.assertEqual(seen[0], str(target))
                        self.assertFalse(folder.exists())

    def test_periodic_raw_writes_and_final_drain_without_duplicates(self):
        monitor = SimpleNamespace(lock=threading.Lock(), samples=[])
        with benchmark.CsvRecording(self.args) as recording:
            recording.attach('vram', monitor, 'samples')
            with monitor.lock:
                monitor.samples.append({'timestamp': 'first', 'used_mib': 42})
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                if read_rows(recording.paths['vram'])[1]:
                    break
                time.sleep(0.02)
            self.assertEqual(len(read_rows(recording.paths['vram'])[1]), 1)
            with monitor.lock:
                monitor.samples.append({'timestamp': 'second', 'used_mib': 43})
        self.assertEqual([r['timestamp'] for r in read_rows(self.root / 'run.vram.csv')[1]], ['first', 'second'])

    def test_pcie_and_gpm_phase_enrichment_after_exception(self):
        pcie = benchmark.NvidiaVramMonitor()
        gpm = benchmark.NvidiaGpmMonitor()
        pcie.pcie_samples.append({'timestamp': 'pcie', 'gpu_index': '0', 't_mono': 11.5,
                                  'target_ctx': 100, 'repeat': 1, 'phase': 'measure',
                                  'pcie_bus_util_pct': 40})
        gpm.samples.extend([
            {'timestamp': 'boundary', 'gpu_index': '0', 'interval_start_mono': 9.5, 'interval_end_mono': 10.5},
            {'timestamp': 'contained', 'gpu_index': '0', 'interval_start_mono': 10.5, 'interval_end_mono': 11.5},
        ])
        with self.assertRaisesRegex(RuntimeError, 'request failed'):
            with benchmark.CsvRecording(self.args) as recording:
                recording.attach('pcie', pcie, 'pcie_samples', pcie.write_pcie_csv)
                recording.attach('gpm', gpm, 'samples', gpm.write_csv)
                pcie.register_pcie_phase_window(100, 1, 'decode', 10, 12)
                gpm.register_phase_window(100, 1, 'decode', 10, 12)
                raise RuntimeError('request failed')
        rows = read_rows(self.root / 'run.pcie.csv')[1]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['subphase'], 'decode')
        self.assertEqual(rows[0]['bus_subphase'], 'decode')
        rows = read_rows(self.root / 'run.gpm.csv')[1]
        self.assertEqual([r['subphase'] for r in rows], ['', 'decode'])

    def test_atomic_export_failure_preserves_live_file(self):
        destination = self.root / 'live.csv'
        destination.write_text('original\n', encoding='utf-8')
        with patch.object(benchmark.os, 'replace', side_effect=PermissionError('file open')):
            with self.assertRaises(PermissionError):
                benchmark.write_csv_atomic(str(destination), ['value'], [{'value': 1}])
        self.assertEqual(destination.read_text(encoding='utf-8'), 'original\n')
        self.assertEqual(list(self.root.glob('.ctx-cliff-*')), [])

    def test_collision_detected_before_truncation(self):
        original = self.input.read_bytes()
        self.args.csv = str(self.input)
        with self.assertRaises(ValueError):
            benchmark.CsvRecording(self.args)
        self.assertEqual(self.input.read_bytes(), original)
        self.args.csv = str(self.csv)
        self.args.gpm_csv = str(self.root / 'run.vram.csv')
        with self.assertRaises(ValueError):
            benchmark.CsvRecording(self.args)
        self.assertFalse(self.csv.exists())

    def test_background_failure_is_reported(self):
        monitor = SimpleNamespace(lock=threading.Lock(), samples=[{'timestamp': 'sample'}])
        with self.assertRaisesRegex(RuntimeError, 'disk full'):
            with benchmark.CsvRecording(self.args) as recording:
                with patch.object(recording, '_drain', side_effect=OSError('disk full')):
                    recording.attach('vram', monitor, 'samples')
                    deadline = time.monotonic() + 3
                    while recording.error is None and time.monotonic() < deadline:
                        time.sleep(0.02)
                    recording.check()

    def run_main(self, completion, warmup=0, monitors=False):
        argv = [str(SCRIPT), '--file', str(self.input), '--csv', str(self.csv),
                '--start', '100', '--end', '200', '--step', '100', '--n-predict', '8',
                '--repeat', '1', '--warmup', str(warmup), '--vram-log', 'auto' if monitors else 'off',
                '--gpm-log', 'off', '--win-gpu-mem', 'off', '--settle', '0']

        def ready(*a, **kw):
            self.assertEqual(read_rows(self.csv), (list(benchmark.RESULT_CSV_FIELDS), []))
            return True

        with patch('sys.argv', argv), patch.object(benchmark, 'server_is_ready', side_effect=ready), \
             patch.object(benchmark, 'detect_slot_n_ctx', return_value=10000), \
             patch.object(benchmark, 'tokenize', side_effect=lambda base, text, **kw: [0] * len(text)), \
             patch.object(benchmark, 'reset_slot', return_value=True), \
             patch.object(benchmark, 'fetch_server_props', return_value={'build_info': 'test'}), \
             patch.object(benchmark, 'completion', side_effect=completion):
            benchmark.main()

    def test_timeout_keeps_first_result_and_closes_files(self):
        calls = 0

        def complete(*a, **kw):
            nonlocal calls
            calls += 1
            if calls == 2:
                self.assertEqual(len(read_rows(self.csv)[1]), 1)
                raise benchmark.requests.Timeout('simulated timeout')
            return {'timings': {'predicted_n': 8, 'predicted_ms': 80, 'prompt_n': 100, 'prompt_ms': 50}}

        # A failed point no longer escapes as a traceback: completed points are
        # kept and summarized, the run exits with code 1 and records the failure.
        with contextlib.redirect_stdout(io.StringIO()) as stdout, \
                contextlib.redirect_stderr(io.StringIO()) as stderr, \
                self.assertRaises(SystemExit) as raised:
            self.run_main(complete)
        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(len(read_rows(self.csv)[1]), 1)
        self.assertIn('ERROR at target=200: Timeout: simulated timeout', stderr.getvalue())
        self.assertIn('RUN ENDED EARLY at target=200', stdout.getvalue())
        meta = json.loads((self.root / 'run.meta.json').read_text(encoding='utf-8'))
        self.assertEqual((meta['status'], meta['exit_code'], meta['completed_points']), ('failed', 1, 1))
        self.assertEqual(meta['failed_target_ctx'], 200)
        self.csv.rename(self.root / 'closed.csv')

    def test_success_does_not_duplicate_first_probe_row(self):
        self.run_main(lambda *a, **kw: {'timings': {'predicted_n': 8, 'predicted_ms': 80, 'prompt_n': 100, 'prompt_ms': 50}})
        self.assertEqual([r['target_ctx'] for r in read_rows(self.csv)[1]], ['100', '200'])

    def test_rejected_first_probe_is_not_retried(self):
        calls = []
        def reject(*a, **kw):
            calls.append(True)
            raise benchmark.CompletionRequestError(400, 'context size exceeded')
        with self.assertRaises(SystemExit) as raised:
            self.run_main(reject)
        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(read_rows(self.csv)[1], [])

    def test_warmup_interrupt_saves_raw_without_completed_results(self):
        stopped = []

        def start(monitor):
            monitor.samples.append({'timestamp': 'warmup', 'used_mib': 123})
            return True

        with patch.object(benchmark.NvidiaVramMonitor, 'start', start), \
             patch.object(benchmark.NvidiaVramMonitor, 'stop', lambda monitor: stopped.append(True)):
            with self.assertRaises(KeyboardInterrupt):
                self.run_main(lambda *a, **kw: (_ for _ in ()).throw(KeyboardInterrupt()), warmup=1, monitors=True)
        self.assertEqual(stopped, [True])
        self.assertEqual(read_rows(self.csv)[1], [])
        self.assertEqual(read_rows(self.root / 'run.vram.csv')[1][0]['timestamp'], 'warmup')

    def test_first_probe_interrupt_leaves_valid_header(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_main(lambda *a, **kw: (_ for _ in ()).throw(KeyboardInterrupt()))
        self.assertEqual(read_rows(self.csv), (list(benchmark.RESULT_CSV_FIELDS), []))

    def test_killed_process_leaves_flushed_csvs(self):
        code = '''
import importlib.util, sys, threading, time
from types import SimpleNamespace
spec = importlib.util.spec_from_file_location('bench', sys.argv[1])
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
args = SimpleNamespace(csv=sys.argv[2], file=sys.argv[3], server_log=None,
    vram_log='auto', gpm_log='off', win_gpu_mem='off', vram_csv=None,
    pcie_csv=None, gpm_csv=None, win_gpu_mem_csv=None)
with b.CsvRecording(args) as recording:
    recording.attach('vram', SimpleNamespace(lock=threading.Lock(),
        samples=[{'timestamp':'survives-kill','used_mib':321}]), 'samples')
    recording.write_result(dict.fromkeys(b.RESULT_CSV_FIELDS, 0))
    time.sleep(30)
'''
        child = subprocess.Popen([sys.executable, '-c', code, str(SCRIPT), str(self.csv), str(self.input)],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        trace = self.root / 'run.vram.csv'
        observed = False
        try:
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                if child.poll() is not None:
                    break
                if trace.exists() and read_rows(trace)[1]:
                    observed = True
                    break
                time.sleep(0.02)
        finally:
            if child.poll() is None:
                child.kill()
            _, stderr = child.communicate(timeout=5)
        self.assertTrue(observed, stderr.decode(errors='replace'))
        self.assertEqual(len(read_rows(self.csv)[1]), 1)
        self.assertEqual(read_rows(trace)[1][0]['timestamp'], 'survives-kill')


if __name__ == '__main__':
    unittest.main()
