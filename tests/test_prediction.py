import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sgp4.api import Satrec

from predict_pass import geometry


TLE = (
    '1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753',
    '2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667',
)
SITE = (0., 0., 0.)
FREQUENCY = 137_500_000


class PredictionTest(unittest.TestCase):
    def test_passes_and_doppler(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'sat.tle').write_text('VANGUARD 1\n' + '\n'.join(TLE) + '\n')
            (root / 'config.toml').write_text('[observer]\nlatitude_deg = 0\nlongitude_deg = 0\naltitude_m = 0\n')
            start = datetime(2000, 6, 27, tzinfo=timezone.utc).timestamp()
            end = start + 86401
            command = [sys.executable, 'scripts/predict_pass.py', '--tle', str(root / 'sat.tle'),
                       '--start', datetime.fromtimestamp(start, timezone.utc).isoformat(),
                       '--end', datetime.fromtimestamp(end, timezone.utc).isoformat(),
                       '--frequency', str(FREQUENCY), '--output', str(root / 'prediction.json'),
                       '--config', str(root / 'config.toml')]
            subprocess.run(command, check=True, capture_output=True, text=True)
            result = json.loads((root / 'prediction.json').read_text())
            times = np.array(result['time_utc_unix'])
            doppler = np.array(result['doppler_hz'])
            self.assertEqual(times[-1], end)
            np.testing.assert_array_equal(np.diff(times)[:-1], 2)
            self.assertEqual(result['reference_frequency_hz'], FREQUENCY)
            self.assertTrue(result['passes'])
            self.assertNotIn('observer', result)
            sat = Satrec.twoline2rv(*TLE)
            for entry in result['passes']:
                aos, peak, los = (datetime.fromisoformat(entry[key]).timestamp()
                                  for key in ('aos_utc', 'peak_utc', 'los_utc'))
                self.assertLessEqual(aos, peak)
                self.assertLessEqual(peak, los)
                if aos > start and los < end:
                    elevations = geometry(sat, [aos, (aos + peak) / 2, peak,
                                                (peak + los) / 2, los], *SITE)[1]
                    self.assertAlmostEqual(elevations[0], 10, delta=.02)
                    self.assertAlmostEqual(elevations[-1], 10, delta=.02)
                    self.assertGreater(elevations[1], elevations[0])
                    self.assertGreaterEqual(elevations[2], elevations[1])
                    self.assertGreaterEqual(elevations[2], elevations[3])
                    self.assertGreater(elevations[3], elevations[4])
                    self.assertAlmostEqual(elevations[2], entry['peak_elevation_deg'], places=5)
            entry = result['passes'][1]
            aos, los = (datetime.fromisoformat(entry[key]).timestamp() for key in ('aos_utc', 'los_utc'))
            grid_peak = geometry(sat, times[(times >= aos) & (times <= los)], *SITE)[1].max()
            self.assertGreater(entry['peak_elevation_deg'], grid_peak + 1e-6)
            samples = times[::100]
            difference = (geometry(sat, samples + .5, *SITE)[0] -
                          geometry(sat, samples - .5, *SITE)[0])
            expected = -difference / 299792.458 * FREQUENCY
            np.testing.assert_allclose(doppler[::100], expected, atol=1, rtol=0)
            self.assertGreater(doppler.max(), 0)
            self.assertLess(doppler.min(), 0)
            first_pass = result['passes'][0]
            clipped_start = datetime.fromisoformat(first_pass['peak_utc']).timestamp()
            clipped_end = min(datetime.fromisoformat(first_pass['los_utc']).timestamp(), clipped_start + 10)
            command[command.index('--start') + 1] = datetime.fromtimestamp(clipped_start, timezone.utc).isoformat()
            command[command.index('--end') + 1] = datetime.fromtimestamp(clipped_end, timezone.utc).isoformat()
            subprocess.run(command, check=True, capture_output=True, text=True)
            clipped = json.loads((root / 'prediction.json').read_text())['passes'][0]
            self.assertAlmostEqual(datetime.fromisoformat(clipped['aos_utc']).timestamp(), clipped_start, places=5)
            self.assertAlmostEqual(datetime.fromisoformat(clipped['los_utc']).timestamp(), clipped_end, places=5)

    def test_bad_checksum_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'sat.tle').write_text(TLE[0][:-1] + '0\n' + TLE[1] + '\n')
            result = subprocess.run([sys.executable, 'scripts/predict_pass.py', '--tle', str(root / 'sat.tle'),
                                     '--start', '2000-06-27T00:00:00+00:00', '--end', '2000-06-28T00:00:00+00:00',
                                     '--frequency', str(FREQUENCY), '--output', str(root / 'out.json')],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('checksum', result.stderr)


if __name__ == '__main__':
    unittest.main()
