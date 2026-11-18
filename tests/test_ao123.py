import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from scipy.io import wavfile

from analyze_ao123 import match_candidates
from demod_nfm import demod


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


class Ao123Tests(unittest.TestCase):
    def test_candidate_gate_and_noise_subtracted_power(self):
        frequency = np.arange(-16_000, 16_001, 10)
        doppler = np.full(7, 2000.)
        before = np.full((len(doppler), len(frequency)), -100.)
        after = before.copy()
        for i, (peak, residual, height) in enumerate([
                (210, 250, 20), (210, 260, 20), (140, 140, 20),
                (380, 380, 20), (210, 210, 11), (210, 210, 12), (210, 210, 20)]):
            after[i, np.searchsorted(frequency, peak)] += height
            before[i, np.searchsorted(frequency, doppler[i] + residual)] += 20
        before[0, np.searchsorted(frequency, 2360)] = -20  # Exactly 150 Hz from prediction + peak.
        doppler[-1] = 15_000
        matched, before_residual, after_residual, snr, power = match_candidates(frequency, before, after, doppler)
        np.testing.assert_array_equal(matched, [True, False, False, False, False, True, False])
        np.testing.assert_array_equal(before_residual[:2], [250, 260])
        np.testing.assert_array_equal(after_residual[:2], [210, 210])
        self.assertAlmostEqual(snr[0], 10 * np.log10(99 / 5))
        self.assertAlmostEqual(power[0], 10 * np.log10(99 * 1e-10 * 10))
        self.assertTrue(np.isnan(snr[~matched]).all())

    def test_nfm_discriminator_and_audio_recover_known_tone(self):
        t = np.arange(200_000) / 100_000
        iq = np.exp(2j * np.pi * 208 * t + 2.5j * np.sin(2 * np.pi * 1000 * t))
        channel, raw, deemphasis, audio = demod(iq, 100_000, 208)
        self.assertEqual((len(channel), len(raw), len(deemphasis), len(audio)), (50_000, 49_999, 49_999, 25_000))
        center = raw[6250:-6250]
        self.assertAlmostEqual(center.mean(), 0, delta=.2)
        self.assertAlmostEqual(center.std() * np.sqrt(2), 2500, delta=20)
        steady_audio = audio[3125:-3125]
        frequencies = np.fft.rfftfreq(len(steady_audio), 1 / 12_500)
        power = abs(np.fft.rfft(steady_audio))
        self.assertAlmostEqual(frequencies[power.argmax()], 1000)
        self.assertGreater(power.max(), 0)

    def test_cli_smoke_for_all_figures_with_no_candidates(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            recording = root / 'recording'
            live = recording / 'live-doppler'
            live.mkdir(parents=True)
            output = root / 'output'
            t = np.arange(400_000) / 100_000
            iq = np.exp(2j * np.pi * (208 * t + 300 * np.maximum(t - 3.25, 0))
                        + 2.5j * np.sin(2 * np.pi * 1000 * t)).astype('<c8')
            meta = {'carrier_hz': 435_400_000, 'input_sample_rate_hz': 1_000_000,
                    'output_sample_rate_hz': 100_000, 'output_samples_per_branch': len(iq),
                    'fir_delay_input_samples': 200}
            (live / 'result.json').write_text(json.dumps(meta))
            (recording / 'recording.json').write_text(json.dumps({
                'live_doppler': meta, 'actual_start_utc': '2000-01-01T00:00:00+00:00',
                'actual_end_utc': '2000-01-01T00:00:04+00:00',
            }))
            epoch = 946684800
            controls = [{'input_sample_offset': i * 1_000_000, 'model_start_utc_unix': epoch + i}
                        for i in range(4)]
            (live / 'control.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in controls))
            frequency = np.fft.fftshift(np.fft.fftfreq(8192, 1 / 100_000))
            prediction = root / 'prediction.json'
            prediction.write_text(json.dumps({
                'time_utc_unix': [epoch, epoch + 4], 'doppler_hz': [1000., 1000.],
                'reference_frequency_hz': 217_700_000,
                'passes': [{'aos_utc': '1999-12-31T23:59:59+00:00',
                            'peak_utc': '2000-01-01T00:00:02+00:00',
                            'los_utc': '2000-01-01T00:00:05+00:00'}],
            }))
            np.savez(recording / 'spectrum-history.npz', frequency_hz=meta['carrier_hz'] + np.arange(-16_000, 16_001, 1000),
                     time_utc_unix=[epoch + 1], psd_dbfs_hz=np.full((1, 33), -100.))
            windows = root / 'windows.json'
            windows.write_text(json.dumps([{
                'start': '2000-01-01T00:00:01+00:00', 'end': '2000-01-01T00:00:03+00:00',
                'strong_start': '2000-01-01T00:00:01.500+00:00', 'strong_end': '2000-01-01T00:00:02.500+00:00',
                'example_start': '2000-01-01T09:00:01.250+09:00',
            }, {
                'start': '2000-01-01T00:00:03.250+00:00', 'end': '2000-01-01T00:00:03.450+00:00',
                'strong_start': '2000-01-01T00:00:03.320+00:00', 'strong_end': '2000-01-01T00:00:03.370+00:00',
            }]))

            nfm = [sys.executable, str(SCRIPTS / 'demod_nfm.py'), '--recording', str(recording),
                   '--output', str(output), '--offset-hz', '208']
            for branch in ('uncorrected', 'corrected'):
                iq.tofile(live / f'{branch}.cf32')
                np.savez(live / f'{branch}-spectra.npz', elapsed_seconds=[0.],
                         offset_hz=frequency, psd_dbfs_hz=np.full((1, len(frequency)), -100.))
                subprocess.run([*nfm, '--branch', branch], check=True, capture_output=True, text=True)
                rate, pcm = wavfile.read(output / f'{branch}-nfm.wav')
                self.assertEqual((rate, pcm.shape, pcm.dtype.kind, pcm.dtype.itemsize), (12_500, (50_000,), 'i', 2))

            subprocess.run([*nfm, '--windows', str(windows)], check=True, capture_output=True, text=True)
            nfm_report = json.loads((output / 'nfm-report.json').read_text())
            self.assertEqual(nfm_report['example_start_utc_unix'], epoch + 1.25)
            self.assertEqual(nfm_report['example_duration_seconds'], 1)
            self.assertEqual(nfm_report['clips'][0]['samples'], 25_000)
            self.assertEqual(wavfile.read(output / 'nfm-01.wav')[1].shape, (25_000,))
            self.assertEqual(nfm_report['clips'][1]['samples'], 2_500)
            self.assertAlmostEqual(nfm_report['clips'][1]['rf']['peak_offset_hz'], 300, delta=20)

            ao = [sys.executable, str(SCRIPTS / 'analyze_ao123.py'), '--recording', str(recording),
                  '--prediction', str(prediction), '--output', str(output),
                  '--before-wav', str(output / 'uncorrected-nfm.wav'), '--after-wav', str(output / 'corrected-nfm.wav')]
            subprocess.run(ao, check=True, capture_output=True, text=True)
            for index, name in enumerate(('pass-waterfall', 'doppler-comparison', 'signal-metrics', 'fm-spectrogram',
                                          'nfm-clips-spectrogram', 'nfm-processing-stages'), 1):
                image = output / f'{index:02d}-{name}.png'
                self.assertGreater(image.stat().st_size, 1024)
            report = json.loads((output / 'candidate-report.json').read_text())
            self.assertEqual((report['snapshot_count'], report['matched_snapshots'], report['candidates']), (1, 0, []))

            before_db, after_db = (np.full((1, len(frequency)), -100.) for _ in range(2))
            before_db[0, np.abs(frequency - 2208).argmin()] += 20
            after_db[0, np.abs(frequency - 208).argmin()] += 20
            for branch, values in [('uncorrected', before_db), ('corrected', after_db)]:
                np.savez(live / f'{branch}-spectra.npz', elapsed_seconds=[0.],
                         offset_hz=frequency, psd_dbfs_hz=values)
            legacy_prediction = json.loads(prediction.read_text())
            del legacy_prediction['passes']
            prediction.write_text(json.dumps(legacy_prediction))
            subprocess.run(ao[:-4], check=True, capture_output=True, text=True)
            report = json.loads((output / 'candidate-report.json').read_text())
            self.assertEqual(report['matched_snapshots'], 1)
            candidate = report['candidates'][0]
            self.assertEqual(candidate['doppler_hz'], 2000)
            self.assertAlmostEqual(candidate['time_utc_unix'] - epoch, 2 * 8192 / 100_000 - .0002, places=6)


if __name__ == '__main__':
    unittest.main()
