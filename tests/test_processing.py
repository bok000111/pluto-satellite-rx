import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import freqz, resample_poly, upfirdn

from analyze_live_doppler import analyze as analyze_doppler
from analyze_rs44 import BEACON, analyze as analyze_rs44, extract_audio, search_track
from channel_filter import channel_taps
from decode_meteor import frames, recover
from oqpsk_carrier import estimate_paired


plot_figures = []


def capture_figure(fig):
    plot_figures.append(fig)


class ProcessingTests(unittest.TestCase):
    def test_narrowband_doppler_spectrum_has_finite_peak_metrics(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            live = root / 'live-doppler'
            live.mkdir()
            rate, count = 48000, 32768 + 12000
            samples = np.arange(count)
            for name, frequency in [('uncorrected', 6000), ('corrected', 3000)]:
                iq = np.exp(2j * np.pi * frequency * samples / rate).astype('<c8')
                iq.tofile(live / f'{name}.cf32')
            (root / 'iq.cs16').write_bytes(bytes(count * 4))
            metadata = dict(target='Synthetic tone', sample_count=count, bytes_written=count * 4,
                            live_doppler=dict(output_sample_rate_hz=rate, output_samples_per_branch=count,
                                              processed_samples=count, rejected_samples=0,
                                              status='completed', errors=[]))
            (root / 'recording.json').write_text(json.dumps(metadata))
            plot_figures.clear()
            with plt.rc_context({'figure.hooks': ['test_processing:capture_figure']}):
                report = analyze_doppler(root)
            self.assertTrue(report['raw_size_matches'])
            for name, frequency in [('uncorrected', 6000), ('corrected', 3000)]:
                branch = report['branches'][name]
                self.assertTrue(np.isfinite(branch['median_peak_excess_db']))
                self.assertEqual(branch['peak_candidate_snapshots'], 1)
                with np.load(live / f'{name}-spectra.npz') as spectrum:
                    peak = spectrum['offset_hz'][spectrum['psd_dbfs_hz'][0].argmax()]
                    np.testing.assert_array_equal(spectrum['elapsed_seconds'], [0])
                self.assertAlmostEqual(peak, frequency, delta=rate / 8192)
            waterfalls = [image for fig in plot_figures for ax in fig.axes for image in ax.images]
            self.assertEqual(len(waterfalls), 2)
            for image in waterfalls:
                np.testing.assert_allclose(image.get_extent()[2:], 16384 / rate + np.array([-.5, .5]))
            plot_figures.clear()
            metadata['live_doppler']['output_sample_rate_hz'] = True
            (root / 'recording.json').write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'Invalid Doppler output rate'):
                analyze_doppler(root)

    def test_rs44_rejects_unsorted_or_incomplete_prediction(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rate, count = 262144, 524288
            metadata = dict(sample_rate_hz=rate, sample_count=count, center_frequency_hz=435605000,
                            actual_start_utc='1970-01-01T00:00:00+00:00')
            (root / 'recording.json').write_text(json.dumps(metadata))
            (root / 'iq.cs16').write_bytes(bytes(count * 4))
            prediction_file = root / 'prediction.json'
            for times, error in [([0, 2, 1], 'increasing'), ([1, 2, 3], 'outside')]:
                with self.subTest(times=times):
                    prediction = dict(reference_frequency_hz=435605000, time_utc_unix=times,
                                      doppler_hz=[0, 0, 0])
                    prediction_file.write_text(json.dumps(prediction))
                    with self.assertRaisesRegex(ValueError, error):
                        analyze_rs44(root, prediction_file)
            self.assertFalse((root / 'cw-candidate.wav').exists())

    def test_rs44_fits_a_moving_peak_and_rejects_an_unrecorded_band(self):
        frequency = 435605000 + np.arange(-8000, 8001, 25)
        expected = 435605000 + np.arange(-1500, 1501, 50)
        power = np.ones((len(expected), len(frequency)))
        power[np.arange(len(expected)), (expected + 650 - frequency[0]) // 25] = 200
        report, peaks, _, _ = search_track(power, frequency, expected)
        self.assertTrue(report['doppler_consistent_candidate'])
        self.assertEqual(report['estimated_frequency_offset_hz'], 650)
        np.testing.assert_array_equal(peaks, expected + 650)
        with self.assertRaisesRegex(ValueError, 'recorded beacon band'):
            search_track(power, frequency, expected + 30000)

    def test_rs44_prefers_strong_peak_between_grid_points(self):
        frequency = BEACON + np.arange(-8000, 8001, 2)
        expected = BEACON + np.arange(-1500, 1501, 50)
        power = np.ones((len(expected), len(frequency)))
        power[np.arange(len(expected)), (expected + 624 - frequency[0]) // 2] = 200
        power[np.arange(len(expected)), (expected - 1500 - frequency[0]) // 2] = 20
        report, peaks, _, fitted = search_track(power, frequency, expected)
        self.assertTrue(report['doppler_consistent_candidate'])
        self.assertEqual(report['estimated_frequency_offset_hz'], 624)
        np.testing.assert_array_equal(peaks, expected + 624)
        np.testing.assert_array_equal(fitted, peaks)
        with self.assertRaisesRegex(ValueError, 'no snapshots'):
            search_track(power[:0], frequency, expected[:0])

    def test_cw_chunk_boundaries_match_continuous_resampling(self):
        rate, begin = 262144, 137
        end = begin + 2 * rate + 123
        prediction = dict(time_utc_unix=[0, 3], doppler_hz=[-300, 120])
        t = np.arange(end + 200) / rate
        increments = 2 * np.pi * (1850 + np.interp(t, [0, 3], [-300, 120])) / rate
        phase = np.r_[0., np.cumsum(increments[:-1])]
        raw = np.round(1500 * np.column_stack((np.cos(phase), np.sin(phase)))).astype('<i2')
        audio, envelope = extract_audio(raw, rate, BEACON - 1226, begin, end, 0, prediction, 624)
        iq = raw[begin:end, 0] + 1j * raw[begin:end, 1]
        baseband = resample_poly(iq * np.exp(-1j * (phase[begin:end] - phase[begin])), 6000, rate)
        expected = np.real(baseband * np.exp(2j * np.pi * 650 * np.arange(baseband.size) / 6000))
        expected = np.round(expected / np.max(abs(expected)) * .9 * 32767).astype(np.int16)
        np.testing.assert_allclose(audio, expected, atol=1)
        np.testing.assert_allclose(envelope, abs(baseband[:12000].reshape(-1, 60)).mean(axis=1), atol=1e-6)

    def test_rs44_waterfall_rows_follow_snapshot_centers(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rate, count = 262144, 3 * 262144 + 100
            metadata = dict(sample_rate_hz=rate, sample_count=count, center_frequency_hz=BEACON,
                            actual_start_utc='1970-01-01T00:16:40+00:00')
            (root / 'recording.json').write_text(json.dumps(metadata))
            (root / 'iq.cs16').write_bytes(bytes(count * 4))
            prediction = root / 'prediction.json'
            prediction.write_text(json.dumps(dict(reference_frequency_hz=BEACON,
                                                   time_utc_unix=[1000, 1004], doppler_hz=[0, 0])))
            plot_figures.clear()
            with plt.rc_context({'figure.hooks': ['test_processing:capture_figure']}):
                analyze_rs44(root, prediction)
            waterfall = plot_figures[0].axes[0]
            plot_figures.clear()
            with np.load(root / 'cw-spectrum.npz') as spectrum:
                centers = spectrum['time_utc_unix'] - 1000
                np.testing.assert_array_equal(centers, [1., 2.])
            np.testing.assert_array_equal(waterfall.images[0].get_extent()[2:], [.5, 2.5])
            np.testing.assert_array_equal(waterfall.lines[0].get_ydata(), centers)

    def test_lrpt_filter_and_paired_carrier_against_known_frequencies(self):
        taps = channel_taps(288000, 65000, 75000)
        _, response = freqz(taps, worN=[55000, 90000], fs=288000)
        self.assertAlmostEqual(abs(response[0]), 1, delta=1e-3)
        self.assertLess(abs(response[1]), 1e-3)
        rng = np.random.default_rng(2)
        pulse = [.1, .4, .9, 1, .9, .4, .1]
        for clock, offset in [(72000, -873), (80000, 631)]:
            with self.subTest(clock=clock, offset=offset):
                rate = 4 * clock
                i = upfirdn(pulse, rng.choice([-1, 1], clock + 2), up=4)[8:rate + 8]
                q = upfirdn(pulse, rng.choice([-1, 1], clock + 2), up=4)[6:rate + 6]
                seconds = np.arange(rate) / rate
                iq = (i + 1j * q) * np.exp(2j * np.pi * offset * seconds)
                result = estimate_paired(iq, rate, 18000)
                self.assertTrue(result['accepted'])
                self.assertAlmostEqual(result['frequency_hz'], offset, delta=.5)
                self.assertAlmostEqual(result['symbol_clock_hz'], clock, delta=1)

    def test_lrpt_rejects_two_unmodulated_cw_lines(self):
        rate = 288000
        t = np.arange(rate) / rate
        iq = np.exp(2j * np.pi * -36000 * t) + np.exp(2j * np.pi * 36000 * t)
        self.assertEqual(estimate_paired(iq, rate, 18000),
                         {'accepted': False, 'reason': 'dominant_unmodulated_tone'})

    def test_cadu_deduplication_and_payload_conflict(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'frames.cadu'
            original = bytes.fromhex('1acffc1d') + bytes(1020)
            path.write_bytes(original + original)
            self.assertEqual(list(frames(path).values()), [original])
            conflict = bytearray(original)
            conflict[895] = 1
            path.write_bytes(original + conflict)
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                frames(path)

    def test_lrpt_fallback_product_arguments_timeout_and_wrap(self):
        def cadu(header, counter):
            return bytes.fromhex('1acffc1d') + header.to_bytes(2, 'big') + counter.to_bytes(3, 'big') + bytes(1015)

        before, at, after = [cadu(1, counter) for counter in (0xffffff, 0, 1)]
        other_before, other_after = [cadu(2, counter) for counter in (0xffffff, 0)]
        for outcome in ('72k', '80k-product-timeout', 'zero'):
            with self.subTest(outcome=outcome), TemporaryDirectory() as directory:
                root = Path(directory)
                live = root / 'live-doppler'
                live.mkdir()
                (live / 'corrected.cf32').write_bytes(bytes(16))
                (root / 'recording.json').write_text(json.dumps(dict(
                    status='completed', settings_restored=True, compare_constant_cfo=True,
                    live_doppler=dict(status='completed', covers_all_recorded_samples=True,
                                      file_sizes_match=True, output_sample_rate_hz=288000,
                                      output_samples_per_branch=2, fir_passband_hz=100000))))
                calls = []

                def run(command, **kwargs):
                    calls.append(command)
                    _, pipeline, mode, _, output, *_ = command
                    if mode == 'cadu':
                        if outcome == '80k-product-timeout':
                            raise subprocess.TimeoutExpired(command, kwargs['timeout'])
                    else:
                        swap = command[command.index('--iq_swap') + 1] == 'true'
                        accepted_pipeline = {'72k': 'meteor_m2-x_lrpt',
                                             '80k-product-timeout': 'meteor_m2-x_lrpt_80k'}.get(outcome)
                        if pipeline == accepted_pipeline:
                            folder = Path(output)
                            folder.mkdir()
                            (folder / (pipeline + '.cadu')).write_bytes(
                                before + at + other_after if swap else at + other_before + after)
                    return subprocess.CompletedProcess(command, 0)

                with patch('decode_meteor.subprocess.run', side_effect=run):
                    report = recover(root)
                    with self.assertRaisesRegex(FileExistsError, 'fresh recording copy'):
                        recover(root)
                expected_rates = [72000, 72000] + ([80000, 80000] if outcome != '72k' else [])
                baseband = [call for call in calls if call[2] == 'baseband']
                self.assertEqual([(call[1], call[call.index('--symbolrate') + 1],
                                   call[call.index('--iq_swap') + 1]) for call in baseband],
                                 [('meteor_m2-x_lrpt_80k' if rate == 80000 else 'meteor_m2-x_lrpt',
                                   str(rate), str(bool(iq % 2)).lower()) for iq, rate in enumerate(expected_rates)])
                self.assertIsNone(report['estimated_symbol_rate'])
                self.assertIsNone(report['target'])
                self.assertEqual(report['failed_runs'], [])
                merged = Path(report['merged_cadu']).read_bytes()
                if outcome == 'zero':
                    self.assertEqual(len(calls), len(baseband))
                    self.assertIsNone(report['symbol_rate'])
                    self.assertEqual(report['unique_valid_cadu'], 0)
                    self.assertEqual(merged, b'')
                else:
                    self.assertEqual(calls[-1][1:4], [baseband[-1][1], 'cadu', report['merged_cadu']])
                    self.assertEqual(report['symbol_rate'], expected_rates[-1])
                    self.assertEqual(report['unique_valid_cadu'], 5)
                    self.assertEqual(merged, before + at + after + other_before + other_after)
                timeout = outcome == '80k-product-timeout'
                self.assertEqual(report['product_timeout'], timeout)
                self.assertEqual(report['status'], 'partial' if timeout else 'completed')
                self.assertEqual(json.loads((root / 'meteor-recovery/comparison.json').read_text()), report)
