"""Regression tests for run metadata, output fingerprints, empty-vs-zero values and
per-point failure handling (review of 2026-09-26)."""
import contextlib
import csv
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_csv_recording import SCRIPT, benchmark as b


def read_rows(path):
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def sample(**extra):
    row = dict(cache_n=10, prompt_n=100, prompt_ms=100, prefill_tps=1000,
               decode_tps=10, predicted_n=8, predicted_ms=800, draft_n=0, draft_acc=0,
               wall_s=1, truncated=False, status='OK')
    row.update(extra)
    return row


def timings(**extra):
    values = {'predicted_n': 8, 'predicted_ms': 80, 'prompt_n': 100, 'prompt_ms': 50}
    values.update(extra)
    return {'timings': values, 'content': 'def main():\n    return 42\n', 'stop_type': 'limit'}


class OutputFingerprintTests(unittest.TestCase):
    def test_loop_detection_flags_periodic_tail_only(self):
        self.assertIsNone(b.output_loop_pct(''))
        self.assertEqual(b.output_loop_pct('abc' * 20), 100.0)
        self.assertGreaterEqual(b.output_loop_pct('Intro text. ' + 'the end ' * 30), 80.0)
        normal = ('def parse(path):\n    with open(path) as f:\n        return json.load(f)\n'
                  '\n\nclass Loader:\n    """Load config files."""\n')
        self.assertLess(b.output_loop_pct(normal), 10.0)
        # Only two copies of a unit are not a loop.
        self.assertLess(b.output_loop_pct('xyz hello hello'), 50.0)

    def test_loop_detection_analyses_only_the_tail_window(self):
        text = 'unique prefix ' * 500 + 'ab' * 600
        self.assertEqual(b.output_loop_pct(text, window=1000), 100.0)

    def test_excerpt_stays_on_one_line_and_hash_is_stable(self):
        self.assertEqual(b.output_excerpt('a\nb\tc\r\\d'), 'a\\nb\\tc\\r\\\\d')
        self.assertEqual(len(b.output_excerpt('x' * 500)), b.OUTPUT_EXCERPT_CHARS)
        self.assertEqual(b.output_hash('same'), b.output_hash('same'))
        self.assertNotEqual(b.output_hash('same'), b.output_hash('same '))
        self.assertEqual(len(b.output_hash('')), 16)

    def test_missing_content_is_empty_not_hash_of_empty_string(self):
        info = b.analyze_output(None)
        self.assertEqual(set(info.values()), {None})
        info = b.analyze_output('')
        self.assertEqual(info['output_chars'], 0)
        self.assertEqual(info['output_sha256'], b.output_hash(''))

    def test_point_summary_counts_distinct_outputs(self):
        rows = [sample(output_sha256='a', output_loop_pct=5), sample(output_sha256='a', output_loop_pct=70),
                sample(output_sha256='b', output_loop_pct=None)]
        row = b.aggregate_point(rows, 100, 100, 400, SimpleNamespace(cache_mode='cold'))
        self.assertEqual((row['output_sha256'], row['output_variants'], row['output_loop_pct_max']),
                         ('a', 2, 70))
        row = b.aggregate_point([sample()], 100, 100, 400, SimpleNamespace(cache_mode='cold'))
        self.assertIsNone(row['output_sha256'])
        self.assertIsNone(row['output_variants'])


class FakeCompletionServer:
    """Minimal HTTP double: tokenize, erase, completion with generated text."""

    def __init__(self, contents):
        self.contents = list(contents)

    def post(self, url, json=None, timeout=None):
        if url.endswith('/tokenize'):
            return SimpleNamespace(ok=True, status_code=200, raise_for_status=lambda: None,
                                   json=lambda: {'tokens': list(range(10))})
        content = self.contents.pop(0)
        payload = {'timings': {'cache_n': 0, 'prompt_n': 10, 'prompt_ms': 10,
                               'predicted_n': 4, 'predicted_ms': 40},
                   'content': content, 'stop_type': 'limit'}
        return SimpleNamespace(ok=True, status_code=200, json=lambda: payload)

    def request(self, method, url, timeout=None):
        return SimpleNamespace(status_code=200)


class Recording:
    def __init__(self):
        self.rows, self.samples = [], []

    def check(self):
        return None

    def write_result(self, row):
        self.rows.append(row)

    def write_sample(self, row):
        self.samples.append(row)


class SampleOutputTests(unittest.TestCase):
    def run_point(self, contents, deterministic=True):
        args = SimpleNamespace(cache_mode='cold', repeat=len(contents), slot_id=0, settle=0,
                               n_predict=4, deterministic=deterministic, ignore_eos=True)
        builder = SimpleNamespace(build=lambda ctx: ('prompt', 10, 40), max_prompt_ctx=None,
                                  last_tokens=('prompt', list(range(10))))
        recording = Recording()
        runner = b.BenchmarkRunner(args, 'http://fake', FakeCompletionServer(contents), builder,
                                   recording, None, None, None)
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            row = runner.measure_point(10)
        return row, recording, stderr.getvalue()

    def test_samples_record_output_fields_and_prefill_mode(self):
        row, recording, _ = self.run_point(['hello world', 'hello world'])
        first = recording.samples[0]
        self.assertEqual(first['prefill_mode'], 'cold')
        self.assertEqual(first['stop_type'], 'limit')
        self.assertEqual(first['output_excerpt'], 'hello world')
        self.assertEqual(first['output_chars'], 11)
        self.assertEqual(first['output_sha256'], b.output_hash('hello world'))
        self.assertEqual((row['prefill_mode'], row['output_variants']), ('cold', 1))
        for field in b.SAMPLE_CSV_FIELDS:
            if field.startswith('output_') or field in ('stop_type', 'prefill_mode'):
                self.assertIn(field, first)

    def test_divergent_deterministic_outputs_and_loops_are_reported(self):
        row, _, stderr = self.run_point(['alpha beta', 'loop ' * 30])
        self.assertEqual(row['output_variants'], 2)
        self.assertIn('2 different outputs across 2 repeats despite --deterministic', stderr)
        self.assertIn('generated text looks degenerate', stderr)

    def test_divergence_is_not_reported_without_deterministic(self):
        _, _, stderr = self.run_point(['alpha', 'beta'], deterministic=False)
        self.assertNotIn('different outputs', stderr)


class EmptyInsteadOfZeroTests(unittest.TestCase):
    def test_no_valid_decode_or_drafts_leave_empty_values(self):
        row = b.aggregate_point([sample(status='EMPTY', decode_tps=0, predicted_n=0)] * 2,
                                100, 100, 400, SimpleNamespace(cache_mode='cold'))
        for key in ('decode_tps_median', 'decode_tps_min', 'decode_tps_max', 'draft_acc_pct'):
            self.assertIsNone(row[key], key)
        self.assertEqual(row['draft_n'], 0)
        self.assertEqual(row['valid_repeats'], 0)

    def test_zero_acceptance_with_drafts_stays_zero(self):
        row = b.aggregate_point([sample(draft_n=10, draft_acc=0)], 100, 100, 400,
                                SimpleNamespace(cache_mode='cold'))
        self.assertEqual(row['draft_acc_pct'], 0.0)

    def test_console_shows_na_for_missing_rates(self):
        row = b.aggregate_point([sample(status='EMPTY', prompt_ms=0, decode_tps=0)], 100, 100, 400,
                                SimpleNamespace(cache_mode='cold'))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            b.print_live_row(row, drafting_on=True)
        cells = [cell.strip() for cell in out.getvalue().split('|')]
        self.assertEqual(cells[1:5], ['n/a', 'n/a', 'n/a', 'n/a'])

    def test_empty_values_are_blank_in_result_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'in.txt').write_text('x', encoding='utf-8')
            args = SimpleNamespace(csv=str(root / 'r.csv'), file=str(root / 'in.txt'), server_log=None,
                                   vram_log='off', gpm_log='off', win_gpu_mem='off',
                                   vram_csv=None, pcie_csv=None, gpm_csv=None, win_gpu_mem_csv=None)
            row = b.aggregate_point([sample(status='EMPTY', decode_tps=0)], 100, 100, 400,
                                    SimpleNamespace(cache_mode='cold'))
            row.update({k: None for k in b.RESULT_CSV_FIELDS if k not in row})
            with contextlib.redirect_stderr(io.StringIO()), b.CsvRecording(args) as recording:
                recording.write_result(row)
            written = read_rows(root / 'r.csv')[0]
        self.assertEqual(written['decode_tps_median'], '')
        self.assertEqual(written['draft_acc_pct'], '')
        self.assertEqual(written['prefill_tps'], '1000')


class PrefillModeTests(unittest.TestCase):
    def test_modes(self):
        ns = SimpleNamespace
        self.assertEqual(b.prefill_mode(ns(cache_mode='cold', repeat=3)), 'cold')
        self.assertEqual(b.prefill_mode(ns(cache_mode='incremental', repeat=1)), 'incremental')
        self.assertEqual(b.prefill_mode(ns(cache_mode='incremental', repeat=3)), 'snapshot')
        self.assertEqual(b.prefill_mode(ns(cache_mode='incremental', repeat=3,
                                           prefill_repeat_enabled=False)), 'first_repeat_only')

    def test_probe_records_fallback_reason(self):
        args = SimpleNamespace(cache_mode='incremental', repeat=3, slot_id=0)
        with patch.object(b, 'slot_snapshot', side_effect=RuntimeError('no --slot-save-path')), \
                contextlib.redirect_stderr(io.StringIO()):
            b.probe_prefill_repeats(args, 'http://x', 'ctx-cliff-a.bin')
        self.assertFalse(args.prefill_repeat_enabled)
        self.assertEqual(args.prefill_repeat_error, 'no --slot-save-path')
        self.assertEqual(b.prefill_mode(args), 'first_repeat_only')


class MetadataHelperTests(unittest.TestCase):
    def test_api_key_is_redacted_in_commands(self):
        for command in ('srv --api-key secret123 --port 8080', 'srv --api-key=secret123 --port 8080',
                        'srv --api-key "secret 123" --port 8080'):
            with self.subTest(command=command):
                redacted = b.redact_command(command)
                self.assertNotIn('secret', redacted)
                self.assertIn('--port 8080', redacted)
        self.assertEqual(b.redact_command('srv --api-key-file keys.txt'), 'srv --api-key-file keys.txt')

    def test_environment_keeps_backend_variables_and_redacts_secrets(self):
        env = {'GGML_KVARN_PREFILL_HEADS': '2', 'CUDA_VISIBLE_DEVICES': '0', 'LLAMA_API_KEY': 'x',
               'PATH': '/bin', 'HOME': '/root'}
        self.assertEqual(b.relevant_environment(env), {
            'CUDA_VISIBLE_DEVICES': '0', 'GGML_KVARN_PREFILL_HEADS': '2', 'LLAMA_API_KEY': b.REDACTED})

    def test_props_drop_chat_templates_and_shorten_strings(self):
        props = {'chat_template': 'x' * 5000, 'build_info': 'b1234', 'model_path': 'm.gguf',
                 'default_generation_settings': {'n_ctx': 1000, 'api_key': 'x'}, 'long': 'y' * 5000}
        clean = b.sanitize_props(props)
        self.assertNotIn('chat_template', clean)
        self.assertEqual(clean['default_generation_settings'], {'n_ctx': 1000})
        self.assertEqual(len(clean['long']), 1003)

    def test_fetch_props_never_raises(self):
        self.assertIsNone(b.fetch_server_props('http://x', object()))
        failing = SimpleNamespace(get=lambda *a, **k: (_ for _ in ()).throw(b.requests.ConnectionError()))
        self.assertIsNone(b.fetch_server_props('http://x', failing))


class RunMetadataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.input = self.root / 'input.txt'
        self.input.write_text('abc', encoding='utf-8')
        self.path = self.root / 'run.meta.json'
        self.args = SimpleNamespace(file=str(self.input),
                                    server_command='llama-server --api-key hidden --port 8080')

    def load(self):
        return json.loads(self.path.read_text(encoding='utf-8'))

    def test_initial_content_and_redaction(self):
        with patch('sys.argv', ['ctx-cliff.py', '--server-command', 'srv --api-key hidden']), \
                patch.dict('os.environ', {'GGML_KVARN_PREFILL_HEADS': '1'}):
            b.RunMetadata(str(self.path), self.args)
        data = self.load()
        self.assertEqual(data['status'], 'running')
        self.assertEqual(data['environment']['GGML_KVARN_PREFILL_HEADS'], '1')
        self.assertEqual(len(data['input_file']['sha256']), 64)
        self.assertEqual(len(data['script']['sha256']), 64)
        self.assertNotIn('hidden', self.path.read_text(encoding='utf-8'))

    def test_exit_hook_records_outcome(self):
        cases = [((None, None), 'completed', 0),
                 ((SystemExit, SystemExit(2)), 'failed', 2),
                 ((SystemExit, SystemExit(0)), 'completed', 0),
                 ((KeyboardInterrupt, KeyboardInterrupt()), 'interrupted', 130),
                 ((ValueError, ValueError('boom')), 'failed', 1)]
        for (exc_type, exc), status, code in cases:
            with self.subTest(exc_type=exc_type), contextlib.redirect_stderr(io.StringIO()):
                meta = b.RunMetadata(str(self.path), self.args)
                self.assertFalse(meta.exit_hook(exc_type, exc, None))
                data = self.load()
                self.assertEqual((data['status'], data['exit_code']), (status, code))
                self.assertIsNotNone(data['finished_at'])
        self.assertEqual(self.load()['error'], 'ValueError: boom')

    def test_explicit_status_is_not_overwritten_on_exit(self):
        with contextlib.redirect_stderr(io.StringIO()):
            meta = b.RunMetadata(str(self.path), self.args)
            meta.update(status='completed_context_limit', exit_code=0)
            meta.exit_hook(None, None, None)
        self.assertEqual(self.load()['status'], 'completed_context_limit')

    def test_disabled_metadata_writes_nothing(self):
        meta = b.RunMetadata(None, self.args)
        meta.update(status='x')
        meta.exit_hook(None, None, None)
        self.assertEqual(list(self.root.glob('*.json')), [])

    def test_write_failure_only_warns_once(self):
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            meta = b.RunMetadata(str(self.root / 'missing' / 'run.meta.json'), self.args)
            meta.update(status='x')
        self.assertEqual(stderr.getvalue().count('could not write run metadata'), 1)

    def test_recording_reserves_and_validates_meta_path(self):
        args = SimpleNamespace(csv=str(self.root / 'run.csv'), file=str(self.input), server_log=None,
                               vram_log='off', gpm_log='off', win_gpu_mem='off',
                               vram_csv=None, pcie_csv=None, gpm_csv=None, win_gpu_mem_csv=None)
        self.assertEqual(b.CsvRecording(args).meta_path, str(self.root / 'run.meta.json'))
        args.vram_csv = str(self.root / 'run.meta.json')
        args.vram_log = 'auto'
        with self.assertRaises(ValueError):
            b.CsvRecording(args)
        args.csv, args.vram_csv = None, None
        self.assertIsNone(b.CsvRecording(args).meta_path)

    def test_allocate_csv_path_skips_stem_with_existing_metadata(self):
        with patch.object(b.dt, 'datetime') as clock:
            clock.now.return_value.strftime.return_value = '20260926-050000'
            (self.root / 'ctx-cliff-20260926-050000.meta.json').write_text('{}', encoding='utf-8')
            path = b.allocate_csv_path(str(self.root))
        self.assertEqual(Path(path).name, 'ctx-cliff-20260926-050000-1.csv')


class FakeProc:
    def __init__(self, code):
        self.code = code

    def poll(self):
        return self.code

    def wait(self, timeout=None):
        if self.code is None:
            raise subprocess.TimeoutExpired('llama-server', timeout)
        return self.code


class FailureReportTests(unittest.TestCase):
    def test_exit_code_formatting(self):
        self.assertEqual(b.format_exit_code(3221225477), '3221225477 (0xC0000005)')
        self.assertEqual(b.format_exit_code(-11), '-11 (0xFFFFFFF5)')
        self.assertEqual(b.format_exit_code(1), '1')
        self.assertEqual(b.format_exit_code(None), 'unknown')

    def test_crashed_managed_server_is_named_with_log(self):
        server = SimpleNamespace(proc=FakeProc(3221225477))
        try:
            raise b.requests.ConnectionError('connection aborted')
        except Exception as error:
            with contextlib.redirect_stderr(io.StringIO()) as stderr:
                fields = b.report_point_failure(error, 90000, server, 'logs/server.log')
        self.assertEqual(fields['server_exit_code'], 3221225477)
        self.assertEqual(fields['failed_target_ctx'], 90000)
        self.assertIn('CRASHED/EXITED with code 3221225477 (0xC0000005) during target=90000', stderr.getvalue())
        self.assertIn('server.log', stderr.getvalue())
        self.assertNotIn('Traceback', stderr.getvalue())

    def test_running_server_and_external_server(self):
        try:
            raise b.requests.Timeout('read timeout')
        except Exception as error:
            with contextlib.redirect_stderr(io.StringIO()) as stderr:
                fields = b.report_point_failure(error, 10, SimpleNamespace(proc=FakeProc(None)), None)
            self.assertNotIn('server_exit_code', fields)
            self.assertIn('still running', stderr.getvalue())
            with contextlib.redirect_stderr(io.StringIO()) as stderr:
                fields = b.report_point_failure(error, 10, None, None)
            self.assertNotIn('server_exit_code', fields)

    def test_unexpected_errors_keep_traceback(self):
        try:
            {}['missing']
        except Exception as error:
            with contextlib.redirect_stderr(io.StringIO()) as stderr:
                b.report_point_failure(error, 10, None, None)
        self.assertIn('Traceback', stderr.getvalue())


class SweepFailureTests(unittest.TestCase):
    """End-to-end through main() with a patched server, like test_csv_recording.run_main."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.input = self.root / 'input.txt'
        self.input.write_text('text ' * 10000, encoding='utf-8')
        self.csv = self.root / 'run.csv'

    def run_main(self, completion, *extra, warmup=0):
        argv = [str(SCRIPT), '--file', str(self.input), '--csv', str(self.csv),
                '--start', '100', '--end', '300', '--step', '100', '--n-predict', '8',
                '--repeat', '1', '--warmup', str(warmup), '--vram-log', 'off',
                '--gpm-log', 'off', '--win-gpu-mem', 'off', '--settle', '0', *extra]
        out, err = io.StringIO(), io.StringIO()
        code = None
        with patch('sys.argv', argv), patch.object(b, 'server_is_ready', return_value=True), \
                patch.object(b, 'detect_slot_n_ctx', return_value=10000), \
                patch.object(b, 'tokenize', side_effect=lambda base, text, **kw: [0] * len(text)), \
                patch.object(b, 'reset_slot', return_value=True), \
                patch.object(b, 'fetch_server_props', return_value={'build_info': 'b9999'}), \
                patch.object(b, 'completion', side_effect=completion), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                b.main()
            except SystemExit as exit_:
                code = exit_.code
        meta = json.loads((self.root / 'run.meta.json').read_text(encoding='utf-8'))
        return code, out.getvalue(), err.getvalue(), meta

    def test_success_writes_complete_metadata(self):
        code, out, _, meta = self.run_main(lambda *a, **k: timings(), '--nonce', 'ab-test',
                                           '--deterministic')
        self.assertIsNone(code)
        self.assertEqual((meta['status'], meta['exit_code'], meta['completed_points']), ('completed', 0, 3))
        self.assertEqual(meta['server']['props'], {'build_info': 'b9999'})
        self.assertFalse(meta['server']['managed'])
        self.assertEqual(meta['prompt']['nonce'], '[ctx-cliff run=ab-test] ')
        self.assertTrue(meta['prompt']['nonce_fixed'])
        self.assertEqual(meta['measurement']['prefill_mode'], 'incremental')
        self.assertEqual(meta['measurement']['planned_contexts'], [100, 200, 300])
        self.assertEqual(meta['arguments']['nonce'], 'ab-test')
        rows = read_rows(self.csv)
        self.assertEqual([r['prefill_mode'] for r in rows], ['incremental'] * 3)
        self.assertEqual(rows[0]['output_sha256'], b.output_hash('def main():\n    return 42\n'))
        samples = read_rows(self.root / 'run.samples.csv')
        self.assertEqual(samples[0]['output_excerpt'], 'def main():\\n    return 42\\n')
        self.assertNotIn('RUN ENDED EARLY', out)

    def test_runtime_error_mid_sweep_keeps_rows_and_prints_summary(self):
        calls = []

        def complete(*a, **k):
            calls.append(1)
            if len(calls) == 3:
                raise RuntimeError('telemetry monitor failed: synthetic')
            return timings(predicted_ms=80 if len(calls) == 1 else 160)

        code, out, err, meta = self.run_main(complete, '--cliff-min-repeats', '1')
        self.assertEqual(code, 1)
        self.assertEqual(len(read_rows(self.csv)), 2)
        self.assertIn('ERROR at target=300: RuntimeError: telemetry monitor failed: synthetic', err)
        self.assertIn('DECODE CLIFF CANDIDATE', out)
        self.assertIn('RUN ENDED EARLY at target=300', out)
        self.assertIn('2 of 3 planned points completed', out)
        self.assertEqual((meta['status'], meta['stop_reason'], meta['completed_points']), ('failed', 'failed', 2))
        self.assertEqual(meta['last_total_ctx'], int(read_rows(self.csv)[-1]['total_ctx']))

    def test_context_limit_is_a_clean_stop(self):
        calls = []

        def complete(*a, **k):
            calls.append(1)
            if len(calls) == 2:
                raise b.CompletionRequestError(400, 'context size exceeded', 'exceed_context_size_error')
            return timings()

        code, out, _, meta = self.run_main(complete)
        self.assertIsNone(code)
        self.assertEqual((meta['status'], meta['stop_reason'], meta['exit_code']),
                         ('completed_context_limit', 'context_limit', 0))
        self.assertNotIn('RUN ENDED EARLY', out)

    def test_warmup_failure_exits_cleanly_with_metadata(self):
        def complete(*a, **k):
            raise b.requests.ConnectionError('refused')

        code, _, err, meta = self.run_main(complete, warmup=1)
        self.assertEqual(code, 1)
        self.assertIn('warmup failed', err)
        self.assertNotIn('Traceback', err)
        self.assertEqual((meta['status'], meta['stop_reason'], meta['completed_points']),
                         ('failed', 'warmup_failed', 0))

    def test_invalid_nonce_is_rejected(self):
        for value in ('', 'x' * 65, 'a\nb'):
            with self.subTest(value=value), self.assertRaises(SystemExit) as raised, \
                    contextlib.redirect_stderr(io.StringIO()), \
                    patch('sys.argv', [str(SCRIPT), '--file', str(self.input), '--nonce', value]):
                b.main()
            self.assertEqual(raised.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
