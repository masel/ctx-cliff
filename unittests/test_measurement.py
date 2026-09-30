"""Offline measurement, prompt and lifecycle regression tests."""
import io
import math
import signal
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from test_csv_recording import benchmark as b


def sample(**extra):
    row = dict(cache_n=10, prompt_n=100, prompt_ms=100, prefill_tps=1000,
               decode_tps=10, predicted_n=8, predicted_ms=800, draft_n=0, draft_acc=0,
               wall_s=1, truncated=False, status='OK')
    row.update(extra)
    return row


class MeasurementTests(unittest.TestCase):
    def store(self, key='t_mono'):
        store = b.SampleStore(key)
        self.addCleanup(store.close)
        return store

    def test_finite_values_distinguish_zero_and_missing(self):
        self.assertEqual(b.finite_number(0), 0)
        for value in (None, math.nan, math.inf, 'bad'):
            self.assertIsNone(b.finite_number(value))
        self.assertEqual(b.percentile([0, 0, 100], 50), 0)
        self.assertEqual(b.percentile([0, 100], 95), 95)
        self.assertIsNone(b.percentile([], 95))
        with self.assertRaises(ValueError):
            b.percentile([1], 101)

    def test_rate_fallback_rejects_nonfinite_reported_values(self):
        for reported, expected in ((0, 0), (50, 50), (None, 20), (math.inf, 20), (math.nan, 20)):
            timings = dict(rate=reported, count=10, ms=500)
            self.assertEqual(b.rate_from_timing(timings, 'count', 'ms', 'rate'), expected)
        self.assertEqual(b.rate_from_timing(dict(count=10, ms=0), 'count', 'ms', 'rate'), 0)

    def test_stop_status_and_context_error_classification(self):
        self.assertEqual(b.sample_status(8, 8, False), 'OK')
        self.assertEqual(b.sample_status(3, 8, False), 'STOP@3')
        self.assertEqual(b.sample_status(0, 8, False), 'EMPTY')
        self.assertEqual(b.sample_status(8, 8, True), 'TRUNC')
        self.assertTrue(b.CompletionRequestError(400, 'context size exceeded').is_context_overflow)
        self.assertTrue(b.CompletionRequestError(400, '', 'exceed_context_size').is_context_overflow)
        self.assertFalse(b.CompletionRequestError(503, 'server busy').is_context_overflow)

    def test_largest_decode_drop_uses_relative_not_absolute_change(self):
        rows = [
            {'total_ctx': 1000, 'decode_tps_median': 1000},
            {'total_ctx': 2000, 'decode_tps_median': 500},
            {'total_ctx': 3000, 'decode_tps_median': 200},
        ]
        for row in rows:
            row['valid_repeats'] = 3
        drop, previous, current = b.largest_decode_drop(rows)
        self.assertEqual(drop, 60)
        self.assertEqual((previous['total_ctx'], current['total_ctx']), (2000, 3000))

    def test_largest_decode_drop_ignores_improvements_and_invalid_points(self):
        invalid = [
            {'total_ctx': 1000, 'decode_tps_median': 10},
            {'total_ctx': 2000, 'decode_tps_median': math.nan},
            {'total_ctx': 3000, 'decode_tps_median': 0},
            {'total_ctx': 4000, 'decode_tps_median': None},
        ]
        improving = [
            {'total_ctx': 1000, 'decode_tps_median': 10},
            {'total_ctx': 2000, 'decode_tps_median': 12},
        ]
        for row in invalid + improving:
            row['valid_repeats'] = 3
        self.assertIsNone(b.largest_decode_drop(invalid))
        self.assertIsNone(b.largest_decode_drop(improving))

    def test_largest_decode_drop_requires_increasing_contexts(self):
        rows = [
            {'total_ctx': 2000, 'decode_tps_median': 20},
            {'total_ctx': 1000, 'decode_tps_median': 10},
        ]
        for row in rows:
            row['valid_repeats'] = 3
        self.assertIsNone(b.largest_decode_drop(rows))

    def test_largest_decode_drop_requires_adjacent_points(self):
        rows = [
            {'total_ctx': 1000, 'decode_tps_median': 100},
            {'total_ctx': 2000, 'decode_tps_median': None},
            {'total_ctx': 3000, 'decode_tps_median': 10},
        ]
        for row in rows:
            row['valid_repeats'] = 3
        self.assertIsNone(b.largest_decode_drop(rows))

    def test_largest_prefill_drop_uses_prefill_rate(self):
        rows = [
            {'total_ctx': 1000, 'prefill_tps': 2000, 'decode_tps_median': 10},
            {'total_ctx': 2000, 'prefill_tps': 1800, 'decode_tps_median': 5},
            {'total_ctx': 3000, 'prefill_tps': 900, 'decode_tps_median': 6},
        ]
        for row in rows:
            row['prefill_valid_repeats'] = 3
        drop, previous, current = b.largest_prefill_drop(rows)
        self.assertEqual(drop, 50)
        self.assertEqual((previous['total_ctx'], current['total_ctx']), (2000, 3000))

    def test_largest_prefill_drop_rejects_invalid_adjacent_points(self):
        rows = [
            {'total_ctx': 1000, 'prefill_tps': 2000},
            {'total_ctx': 2000, 'prefill_tps': None},
            {'total_ctx': 3000, 'prefill_tps': 500},
        ]
        for row in rows:
            row['prefill_valid_repeats'] = 3
        self.assertIsNone(b.largest_prefill_drop(rows))

    def test_pcie_units_and_unknown_link(self):
        decimal = b.NvidiaVramMonitor._pcie_theoretical_mb_s(1, 1)
        binary = b.NvidiaGpmMonitor._pcie_theoretical_mib_s(1, 1)
        self.assertEqual(decimal, 250)
        self.assertAlmostEqual(binary * 1024 * 1024, decimal * 1000 * 1000)
        self.assertEqual(b.NvidiaVramMonitor._pcie_theoretical_mb_s(0, 0), 0)

    def test_repeat_quantiles_retain_real_zero(self):
        rows = [sample(gpm_decode_sm_p95_pct=value, pcie_decode_rx_p95_mb_s=value)
                for value in (0, 0, 100)]
        result = b.aggregate_point(rows, 100, 100, 400, SimpleNamespace(cache_mode='cold'))
        self.assertEqual(result['gpm_decode_sm_p95_pct'], 0)
        self.assertEqual(result['pcie_decode_rx_p95_mb_s'], 0)
        self.assertEqual(result['telemetry_quantiles'], 'median_of_repeat_quantiles')

    def test_duration_weighting_and_per_metric_validity(self):
        store = self.store('interval_end_mono')
        store.extend([
            dict(interval_start_mono=0, interval_end_mono=.2, sm_util_pct=0,
                 tensor_util_pct=None, pcie_rx_mib_s=950, pcie_tx_mib_s=0, pcie_theoretical_mib_s=1000),
            dict(interval_start_mono=.2, interval_end_mono=1, sm_util_pct=100,
                 tensor_util_pct=0, pcie_rx_mib_s=1000, pcie_tx_mib_s=0, pcie_theoretical_mib_s=2000),
        ])
        result = b.summarize_gpm(store, 0, 1)
        self.assertAlmostEqual(result['gpm_sm_avg_pct'], 80)
        self.assertAlmostEqual(result['gpm_sm_valid_ms'], 1000)
        self.assertEqual(result['gpm_tensor_avg_pct'], 0)
        self.assertEqual(result['gpm_tensor_valid_samples'], 1)
        self.assertAlmostEqual(result['gpm_tensor_valid_ms'], 800)
        self.assertAlmostEqual(result['gpm_pcie_over90_pct'], 20)
        self.assertEqual(result['gpm_pcie_ratio_samples'], 2)
        self.assertEqual(result['gpm_graphics_valid_samples'], 0)
        self.assertIsNone(result['gpm_graphics_avg_pct'])

    def test_repeat_mean_uses_metric_duration_not_total_sample_count(self):
        rows = [sample(gpm_decode_sm_avg_pct=0, gpm_decode_sm_valid_ms=100, gpm_decode_samples=500),
                sample(gpm_decode_sm_avg_pct=100, gpm_decode_sm_valid_ms=900, gpm_decode_samples=1),
                sample(gpm_decode_sm_avg_pct=None, gpm_decode_sm_valid_ms=9000, gpm_decode_samples=1000)]
        result = b.aggregate_point(rows, 100, 100, 400, SimpleNamespace(cache_mode='cold'))
        self.assertEqual(result['gpm_decode_sm_avg_pct'], 90)

    def test_coverage_merges_gpu_intervals_and_rejects_crossings(self):
        store = self.store('interval_end_mono')
        store.extend([
            dict(interval_start_mono=-.4, interval_end_mono=0, sm_util_pct=100),
            dict(interval_start_mono=-.2, interval_end_mono=.2, sm_util_pct=100),
            dict(interval_start_mono=.2, interval_end_mono=.8, gpu_index=0, sm_util_pct=20),
            dict(interval_start_mono=.2, interval_end_mono=.8, gpu_index=1, sm_util_pct=40),
            dict(interval_start_mono=.8, interval_end_mono=1.2, sm_util_pct=100),
        ])
        result = b.summarize_gpm(store, 0, 1)
        self.assertEqual(result['gpm_samples'], 2)
        self.assertAlmostEqual(result['gpm_sm_avg_pct'], 30)
        self.assertAlmostEqual(result['gpm_coverage_pct'], 60)
        self.assertEqual(result['gpm_boundary_samples'], 2)

    def test_missing_pcie_direction_is_not_assumed_idle(self):
        rows = [dict(pcie_rx_mib_s=100, pcie_tx_mib_s=None, pcie_theoretical_mib_s=100)]
        result = b.summarize_pcie(rows, 'gpm_pcie_', 'mib_s', lambda r: r['pcie_theoretical_mib_s'])
        self.assertEqual(result['gpm_pcie_rx_valid_samples'], 1)
        self.assertEqual(result['gpm_pcie_tx_valid_samples'], 0)
        self.assertIsNone(result['gpm_pcie_tx_p95_mib_s'])
        self.assertIsNone(result['gpm_pcie_over90_pct'])

    def test_fallback_age_and_zero_memory_survive(self):
        store = self.store()
        store.append(dict(t_mono=9.5, dedicated_mib=0, shared_mib=0, adapter_instances=1))
        result = b.summarize_windows(store, 10, 10.1)
        self.assertEqual(result['win_fallback_samples'], 1)
        self.assertAlmostEqual(result['win_fallback_max_age_ms'], 600)
        self.assertEqual(result['win_shared_peak_mib'], 0)
        no_match = b.summarize_windows(store, 12, 12.1)
        self.assertEqual(no_match['win_gpu_samples'], 0)
        self.assertIsNone(no_match['win_shared_peak_mib'])

    def test_pcie_survives_missing_slow_vram_samples(self):
        slow, fast = self.store(), self.store()
        fast.append(dict(t_mono=10, pcie_rx_mb_s=0, pcie_tx_mb_s=0, pcie_link_gen=4, pcie_link_width=16))
        summary = b.summarize_vram(slow, fast, 9.9, 10.1, 250)
        self.assertEqual(summary['vram_samples'], 0)
        self.assertEqual(summary['pcie_samples'], 1)
        self.assertEqual(summary['pcie_rx_p95_mb_s'], 0)
        result = b.aggregate_point([sample(**summary)], 100, 100, 400, SimpleNamespace(cache_mode='cold'))
        self.assertEqual(result['pcie_samples'], 1)
        self.assertIsNone(result['vram_free_min_mib'])

    def test_invalid_decode_repeats_are_not_used_for_telemetry(self):
        rows = [sample(status='STOP', gpm_decode_sm_avg_pct=90, gpm_decode_sm_valid_ms=100)]
        result = b.aggregate_point(rows, 100, 100, 400, SimpleNamespace(cache_mode='cold'))
        self.assertEqual(result['valid_repeats'], 0)
        self.assertIsNone(result['gpm_decode_sm_avg_pct'])

    def test_incremental_prefill_without_repeat_uses_only_first_repeat(self):
        rows = [sample(prompt_n=100, gpm_prefill_sm_avg_pct=0, gpm_prefill_sm_valid_ms=100),
                sample(prompt_n=0, gpm_prefill_sm_avg_pct=100, gpm_prefill_sm_valid_ms=100)]
        result = b.aggregate_point(rows, 100, 100, 400,
                                   SimpleNamespace(cache_mode='incremental', prefill_repeat_enabled=False))
        self.assertEqual(result['prompt_n'], 100)
        self.assertEqual(result['gpm_prefill_sm_avg_pct'], 0)
        self.assertEqual(set(result), set(b.RESULT_CSV_FIELDS))

    def test_runner_integrates_all_monitors_and_releases_request_buffers(self):
        vram = b.NvidiaVramMonitor()
        gpm = b.NvidiaGpmMonitor()
        windows = b.WindowsGpuMemoryMonitor()
        for store in (vram.samples, vram.pcie_samples, gpm.samples, windows.samples):
            self.addCleanup(store.close)
        args = SimpleNamespace(cache_mode='cold', slot_id=0, settle=0, n_predict=8,
                               deterministic=True, ignore_eos=True, repeat=1)
        recording = Mock()
        runner = b.BenchmarkRunner(args, 'http://test', object(),
            b.PromptBuilder('x' * 1000, '', lambda text: [0] * len(text), 1, 0), recording, vram, gpm, windows)

        def complete(*a, **kw):
            start = b.time.perf_counter()
            end = b.time.perf_counter()
            vram.samples.append(dict(t_mono=end, gpu_index='0', used_mib=100, total_mib=1000,
                used_pct=10, gpu_util_pct=0, mem_util_pct=0, power_draw_w=0))
            vram.pcie_samples.append(dict(t_mono=end, gpu_index='0', pcie_rx_mb_s=0,
                pcie_tx_mb_s=0, pcie_link_gen=4, pcie_link_width=16))
            gpm.samples.append(dict(interval_start_mono=start, interval_end_mono=end,
                sm_util_pct=0, pcie_rx_mib_s=0, pcie_tx_mib_s=0, pcie_theoretical_mib_s=1000))
            windows.samples.append(dict(t_mono=end, dedicated_mib=100, shared_mib=0, adapter_instances=1))
            return {'timings': {'predicted_n': 8, 'predicted_ms': 1000, 'prompt_n': 100}}

        with patch.object(b, 'completion', side_effect=complete), patch.object(b, 'reset_slot'), \
             patch.object(b, 'tokenize', return_value=list(range(100))):
            row = runner.measure_point(100)
        self.assertEqual(set(row), set(b.RESULT_CSV_FIELDS))
        self.assertEqual(row['gpm_decode_sm_avg_pct'], 0)
        self.assertEqual(row['vram_free_min_mib'], 900)
        self.assertEqual(row['win_shared_peak_mib'], 0)
        self.assertEqual(row['power_avg_w'], 0)
        recording.write_result.assert_called_once_with(row)
        self.assertIsNone(gpm.samples.pin)

    def test_phase_reconstruction_exposes_residual_and_clipping(self):
        windows, quality = b.reconstruct_phases(10, 12, 500, 1000)
        self.assertEqual(windows, (('prefill', 10.5, 11), ('decode', 11, 12)))
        self.assertEqual(quality['phase_wall_residual_ms'], 500)
        windows, quality = b.reconstruct_phases(10, 11, 800, 800)
        self.assertEqual(windows[0][1], 10)
        self.assertEqual(quality['phase_timing_excess_ms'], 600)

    def test_console_displays_missing_metrics_separately(self):
        row = b.aggregate_point([sample(gpm_decode_sm_avg_pct=0, gpm_decode_sm_valid_ms=100)],
                                100, 100, 400, SimpleNamespace(cache_mode='cold'))
        with patch('sys.stdout', new=io.StringIO()) as stream:
            b.print_live_row(row, False)
        self.assertIn('0/n/a/n/a/n/a', stream.getvalue())

    def test_display_group_compacts_only_wholly_missing_values(self):
        self.assertEqual(b.display_group([None, None, None, None]), 'n/a')
        self.assertEqual(b.display_group([None, math.nan]), 'n/a')
        self.assertEqual(b.display_group([0, 0, 0, 0]), '0/0/0/0')
        self.assertEqual(b.display_group([0, None, 12, 0]), '0/n/a/12/0')

    def test_first_measurement_bypasses_warmup_cache_then_reuses_prefix(self):
        args = SimpleNamespace(cache_mode='incremental', slot_id=0, settle=0,
                               n_predict=8, deterministic=True, ignore_eos=True)
        runner = b.BenchmarkRunner(args, 'http://test', object(), None, Mock(), None, None, None)
        previous = ''
        flags = []
        def complete(base, prompt, *args, **kwargs):
            nonlocal previous
            reuse = kwargs['cache_prompt']
            flags.append(reuse)
            cached = len(previous) if reuse and prompt.startswith(previous) else 0
            previous = prompt
            return {'timings': {'cache_n': cached, 'prompt_n': len(prompt) - cached,
                                'prompt_ms': 10, 'predicted_n': 8, 'predicted_ms': 100}}
        with patch.object(b, 'completion', side_effect=complete):
            runner.take_sample('x' * 1000, 1000, 0, phase='warmup')
            runner.take_sample('x' * 1000, 1000, 1, phase='warmup')
            first = runner.take_sample('x' * 1000, 1000, 0)
            repeat = runner.take_sample('x' * 1000, 1000, 1)
            next_point = runner.take_sample('x' * 2000, 2000, 0)
        self.assertEqual(flags, [True, True, False, True, True])
        self.assertEqual(first['prompt_n'], 1000)
        self.assertEqual(first['cache_n'], 0)
        self.assertEqual(repeat['cache_n'], 1000)
        self.assertEqual(next_point['prompt_n'], 1000)

    def test_no_warmup_and_failed_first_request_keep_full_prefill(self):
        args = SimpleNamespace(cache_mode='incremental', slot_id=0, settle=0,
                               n_predict=8, deterministic=True, ignore_eos=True)
        runner = b.BenchmarkRunner(args, 'http://test', object(), None, Mock(), None, None, None)
        with patch.object(b, 'completion', side_effect=b.requests.Timeout('timeout')) as request:
            with self.assertRaises(b.requests.Timeout):
                runner.take_sample('prompt', 100, 0)
            self.assertFalse(request.call_args.kwargs['cache_prompt'])
        self.assertFalse(runner.measurement_started)


def char_tokens(text):
    return list(range(len(text)))


class PromptTests(unittest.TestCase):
    def builder(self, content='word ' * 10000, nonce='nonce ', tokenize=char_tokens, cap=None, density=4):
        return b.PromptBuilder(content, nonce, tokenize, density, len(tokenize(nonce)), cap)

    def test_exact_token_count_across_density_changes(self):
        tokenize = lambda text: list(range(len(text[:500]) // 5 + len(text[500:])))
        builder = self.builder(tokenize=tokenize)
        for target in (100, 500, 1000, 5000):
            window, tokens, chars = builder.build(target)
            self.assertEqual(tokens, target)
            self.assertEqual(builder.last_tokens, (window, tokenize(window)[:target]))
            self.assertLessEqual(chars, len(window) - len(builder.nonce))

    def test_hard_cap(self):
        builder = self.builder(cap=97)
        self.assertEqual(builder.build(1000)[1], 97)
        self.assertEqual(len(builder.last_tokens[1]), 97)

    def test_short_input_returns_whole_file(self):
        window, tokens, chars = self.builder(content='tiny').build(100)
        self.assertEqual((window, tokens, chars), ('nonce tiny', 10, 4))

    def test_counting_tokenizer_is_rejected(self):
        with self.assertRaises(b.PromptBuildError):
            b.PromptBuilder('word ' * 100, 'nonce ', len, 1, 6).build(100)

    def test_impossible_budget_fails_without_returning_oversized_prompt(self):
        builder = self.builder(nonce='x' * 200)
        with self.assertRaises(b.PromptBuildError):
            builder.build(100)
        with self.assertRaises(b.PromptBuildError):
            self.builder(content='')
        with self.assertRaises(b.PromptBuildError):
            self.builder(cap=0).build(100)

    def test_reset_restores_initial_estimate(self):
        builder = self.builder()
        builder.build(100)
        self.assertNotEqual(builder.density, builder.initial_density)
        builder.reset()
        self.assertEqual(builder.density, builder.initial_density)


class StorageAndLifecycleTests(unittest.TestCase):
    def test_archive_failure_is_reported_and_store_cannot_reopen_after_close(self):
        store = b.SampleStore()
        with patch.object(b.tempfile, 'TemporaryFile', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                store.append(dict(t_mono=1))
        with self.assertRaisesRegex(RuntimeError, 'disk full'):
            store.read_since(0)
        self.assertEqual(len(store), 0)
        store.close()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            store.append(dict(t_mono=2))

    def test_no_csv_does_not_require_a_disk_archive(self):
        store = b.SampleStore()
        self.addCleanup(store.close)
        store.archive_enabled = False
        with patch.object(b.tempfile, 'TemporaryFile', side_effect=AssertionError('unnecessary disk archive')):
            store.append(dict(t_mono=1))
        self.assertEqual(len(store.window(0, 2)[0]), 1)

    def test_driver_resources_are_not_freed_while_worker_is_running(self):
        monitor = b.NvidiaGpmMonitor()
        self.addCleanup(monitor.samples.close)
        monitor.thread = Mock()
        monitor.thread.is_alive.return_value = True
        monitor._pynvml = Mock()
        monitor._sample_pairs = {'0': ('a', 'b')}
        with self.assertRaisesRegex(RuntimeError, 'still in use'):
            monitor.stop()
        monitor._pynvml.nvmlGpmSampleFree.assert_not_called()

    def test_archive_supports_concurrent_append_and_csv_consumption(self):
        store = b.SampleStore()
        self.addCleanup(store.close)
        done = threading.Event()
        failures = []
        def produce():
            try:
                for index in range(5000):
                    store.append(dict(t_mono=index, value=index))
            except Exception as error:
                failures.append(error)
            finally:
                done.set()
        thread = threading.Thread(target=produce)
        thread.start()
        consumed = []
        while not done.wait(.001):
            consumed.extend(store.read_since(len(consumed)))
        thread.join(timeout=2)
        while len(consumed) < len(store):
            consumed.extend(store.read_since(len(consumed)))
        self.assertEqual(failures, [])
        self.assertEqual([row['value'] for row in consumed], list(range(5000)))

    def test_idle_history_is_bounded_but_archive_is_complete(self):
        store = b.SampleStore()
        self.addCleanup(store.close)
        for index in range(12000):
            store.append(dict(t_mono=float(index), value=index))
        self.assertLess(len(store.rows), 1100)
        rows, _ = store.window(11995, 11999)
        self.assertEqual([r['value'] for r in rows], list(range(11995, 12000)))
        exported = list(store.iter_all())
        self.assertEqual(len(exported), 12000)
        self.assertEqual(exported[0]['value'], 0)
        drained = []
        while len(drained) < len(store):
            drained.extend(store.read_since(len(drained)))
        self.assertEqual(drained, exported)

    def test_active_request_is_retained_until_summarized(self):
        store = b.SampleStore()
        self.addCleanup(store.close)
        store.begin(0)
        for index in range(3000):
            store.append(dict(t_mono=float(index)))
        self.assertEqual(len(store.window(0, 2999)[0]), 3000)
        store.end()
        store.append(dict(t_mono=3001))
        self.assertLess(len(store.rows), 1100)
        self.assertEqual(len(list(store.iter_all())), 3001)

    def test_out_of_order_append_does_not_corrupt_archive(self):
        store = b.SampleStore()
        self.addCleanup(store.close)
        store.append(dict(t_mono=2))
        with self.assertRaises(ValueError):
            store.append(dict(t_mono=1))
        self.assertEqual(len(list(store.iter_all())), 1)

    def test_phase_lookup_uses_time_even_with_repeated_target_keys(self):
        index = b.PhaseIndex([(100, 1, 'prefill', 0, 1), (100, 1, 'decode', 1, 2),
                             (100, 1, 'prefill', 3, 4)])
        self.assertEqual(index.find(3.2, 3.8)[2], 'prefill')
        self.assertIsNone(index.find(.8, 1.2))
        self.assertEqual(index.find(1, 1)[2], 'decode')

    def test_polling_stop_interrupts_long_interval(self):
        stop = threading.Event()
        done = threading.Event()
        thread = threading.Thread(target=lambda: (list(b.polling_ticks(stop, 30)), done.set()))
        thread.start()
        stop.set()
        self.assertTrue(done.wait(1))
        thread.join()

    def test_session_is_used_for_completion_without_retry(self):
        http = Mock()
        http.post.return_value.ok = True
        http.post.return_value.json.return_value = {'timings': {}}
        self.assertEqual(b.completion('http://test', 'prompt', 8, True, False, 0, True, http=http), {'timings': {}})
        http.post.assert_called_once()
        self.assertEqual(http.post.call_args.kwargs['json']['prompt'], 'prompt')

    def test_sigterm_handler_restores_previous_handler_after_cleanup(self):
        previous = signal.getsignal(signal.SIGTERM)
        with self.assertRaises(KeyboardInterrupt):
            with b.termination_handler():
                handler = signal.getsignal(signal.SIGTERM)
                handler(signal.SIGTERM, None)
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)


if __name__ == '__main__':
    unittest.main()
