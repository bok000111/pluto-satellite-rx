import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample_poly

from analyze_live_doppler import analyze as analyze_doppler
from analyze_rs44 import BEACON, analyze as analyze_rs44, extract_audio, search_track


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
