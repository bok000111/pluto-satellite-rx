from datetime import datetime, timedelta, timezone
import errno
import json
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from live_doppler import DopplerCurve, PairedChannelizer
from record_meteor import record
from record_pass import run


class ReceiveTests(unittest.TestCase):
    def test_paired_chunking_preserves_filter_phase_and_decimation(self):
        rate, output_rate, offset = 240_000, 40_000, 17_000
        utc = 1_700_000_000.
        curve = DopplerCurve([utc, utc + .5, utc + 1.], [800., 1100., 900.])
        start = .375
        samples = (np.arange(40_013) / rate) + start
        phase = offset * (samples - start) + curve.integral(samples) - curve.integral(start)
        iq = np.asarray(np.exp(2j * np.pi * phase), dtype=np.complex64)
        whole = PairedChannelizer(rate, output_rate, offset, curve, 11_000).process(iq, start)

        stream = PairedChannelizer(rate, output_rate, offset, curve, 11_000)
        cuts = (0, 1, 991, 4096, 16_777, 32_768, iq.size)
        chunks = [stream.process(iq[a:b], start + a / rate) for a, b in zip(cuts, cuts[1:])]
        np.testing.assert_allclose(np.concatenate(chunks, axis=1), whole, rtol=3e-6, atol=1e-6)
        self.assertEqual(whole.shape, (2, (iq.size + 5) // 6))

    def test_linear_prediction_removes_positive_doppler_without_phase_jumps(self):
        rate, output_rate = 240_000, 40_000
        curve = DopplerCurve([0., 1.], [900., 1900.])
        t = np.arange(rate // 2) / rate
        iq = np.exp(2j * np.pi * (900. * t + 500. * t**2)).astype(np.complex64)
        channelizer = PairedChannelizer(rate, output_rate, 0., curve, 11_000)
        before, after = channelizer.process(iq, 0.)
        valid = slice(200, None)
        np.testing.assert_allclose(after[valid], 1, atol=2e-6)
        measured = np.angle(before[201:] * before[200:-1].conj()) * output_rate / (2 * np.pi)
        delayed_t = (np.arange(200, before.size - 1) + .5) / output_rate
        delayed_t -= channelizer.filter_info["fir_delay_input_samples"] / rate
        np.testing.assert_allclose(measured, 900. + 1000. * delayed_t, atol=.02)

    def test_rate_readback_and_sigterm_restore(self):
        for case in ("raw", "paired", "sigterm", "rejected_rate", "raw_error"):
            with self.subTest(case=case), TemporaryDirectory() as directory:
                self.check_capture(Path(directory) / "capture", case)

    def check_capture(self, output, case):
        now = datetime(2030, 1, 1, tzinfo=timezone.utc)
        requested_rate = 2_800_000 if case == "rejected_rate" else 2_500_000

        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return now

        class Rate:
            def __init__(self):
                self.writes = []

            @property
            def value(self):
                return str(radio.sample_rate)

            @value.setter
            def value(self, value):
                rate = int(value)
                self.writes.append(rate)
                if rate == 2_800_000:
                    raise OSError(errno.EINVAL, "Rate rejected by IIO")
                radio.rate_hz = 2_499_999 if rate == 2_500_000 else rate

        class Fir:
            writes = 0

            @property
            def value(self):
                return "1"

            @value.setter
            def value(self, value):
                self.writes += 1

        class Receiver:
            rate_hz = 3_000_000
            rx_rf_bandwidth = 1_500_000
            rx_buffer_size = 1024
            gain_control_mode_chan0 = "manual"
            rx_hardwaregain_chan0 = 42.
            calls = 0
            destroyed = False

            @property
            def sample_rate(self):
                return self.rate_hz

            @property
            def rx_lo(self):
                return self.lo

            @rx_lo.setter
            def rx_lo(self, value):
                self.lo = value + 5 if value == 137_500_000 else value

            def rx(self):
                nonlocal now
                self.calls += 1
                if case == "sigterm" and self.calls == 3:
                    signal.raise_signal(signal.SIGTERM)
                now += timedelta(seconds=.05 if case == "sigterm" else .10)
                return np.full(65_536, self.calls - 2j, dtype=np.complex64)

            def rx_destroy_buffer(self):
                self.destroyed = True

        radio = Receiver()
        radio.lo = 100_000_000
        rate, fir = Rate(), Fir()
        channel = SimpleNamespace(attrs={"sampling_frequency": rate, "filter_fir_en": fir})
        phy = SimpleNamespace(find_channel=lambda name, output: channel)
        radio.ctx = SimpleNamespace(find_device=lambda name: phy, set_timeout=lambda value: None)
        previous_handler = signal.getsignal(signal.SIGTERM)
        original_open = Path.open

        def open_with_partial_write(path, *args, **kwargs):
            file = original_open(path, *args, **kwargs)
            if case != "raw_error" or path.name != "iq.cs16":
                return file

            class PartialWrite:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    file.close()

                def write(self, data):
                    file.write(data[:4])
                    raise OSError(errno.ENOSPC, "Disk full")

            return PartialWrite()

        with patch("record_meteor.datetime", Clock), patch("record_meteor.connect", return_value=radio), \
                patch.object(Path, "open", open_with_partial_write):
            result = record(output, start=now, end=now + timedelta(seconds=.18), uri="test", expected_serial=None,
                            target="Synthetic", center_hz=137_500_000, sample_rate_hz=requested_rate,
                            bandwidth_hz=2_000_000, gain_control_mode="slow_attack",
                            sample_sink=(lambda *args: self.fail("Paired capture accepted a rounded rate"))
                            if case == "paired" else None)
        metadata = json.loads((output / "recording.json").read_text())
        self.assertEqual(rate.writes, [requested_rate, 3_000_000])
        self.assertEqual(fir.writes, 0)
        self.assertTrue(radio.destroyed)
        self.assertEqual(metadata["after"], metadata["before"])
        self.assertTrue(metadata["settings_restored"])
        self.assertIs(signal.getsignal(signal.SIGTERM), previous_handler)
        if case == "raw_error":
            self.assertEqual((result, metadata["status"], metadata["bytes_written"]), (2, "partial", 4))
            self.assertEqual(metadata["error"], "record IQ: OSError")
            self.assertEqual(metadata["error_errno"], errno.ENOSPC)
            return
        if case == "rejected_rate":
            self.assertEqual((result, radio.calls, metadata["status"]), (2, 0, "failed"))
            self.assertEqual(metadata["error_errno"], errno.EINVAL)
            return
        self.assertTrue(metadata["gain_probe"]["restored"])
        self.assertEqual(metadata["receiver"]["rx_frequency_hz"], 137_500_005)
        self.assertEqual(metadata["receiver"]["sample_rate_hz"], 2_499_999)
        if case == "paired":
            self.assertEqual((result, radio.calls, metadata["status"]), (2, 0, "failed"))
            self.assertFalse((output / "iq.cs16").exists())
        else:
            self.assertEqual((result, metadata["status"]), (2, "partial") if case == "sigterm" else (0, "completed"))
            self.assertEqual(metadata["sample_rate_hz"], 2_499_999)
            self.assertEqual(metadata["sample_count"], 65_536)
            self.assertEqual(metadata["bytes_written"], 262_144)
            iq = np.fromfile(output / "iq.cs16", dtype="<i2").reshape(-1, 2)
            np.testing.assert_array_equal(iq, np.tile([2, -2], (65_536, 1)))
            self.assertEqual(metadata["quality_summary"]["measured_samples"], iq.shape[0])
            self.assertTrue(all(row["gain_readback_index"] is None for row in metadata["quality_history"]))
            self.assertEqual([row["phase"] for row in metadata["gain_history"]["readings"]],
                             ["before_rx", "after_rx"])
            with np.load(output / "spectrum-history.npz") as spectrum:
                self.assertEqual(spectrum["frequency_hz"][2048], 137_500_005)
                self.assertEqual(spectrum["psd_dbfs_hz"].shape, (1, 4096))

    def test_excursion_uses_signed_values_within_capture_window(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            start = datetime.now(timezone.utc) + timedelta(seconds=5)
            end = start + timedelta(seconds=1)
            plan = {"target": "Synthetic", "output": "capture", "center_hz": 137_500_000,
                    "sample_rate_hz": 240_000, "rf_bandwidth_hz": 180_000,
                    "record_start": start.isoformat(), "record_end": end.isoformat(),
                    "prediction": "prediction.json", "doppler_channels": [
                        {"key": "A", "carrier_hz": 137_570_000,
                         "sample_rate_hz": 40_000, "cutoff_hz": 11_000}]}
            (root / "plan.json").write_text(json.dumps(plan))
            (root / "test.toml").write_text('[pluto]\nuri = "example"\n')
            prediction = {"reference_frequency_hz": 137_570_000,
                          "time_utc_unix": [start.timestamp() - 10, start.timestamp(),
                                            start.timestamp() + 1, end.timestamp() + 2,
                                            end.timestamp() + 10],
                          "doppler_hz": [100_000, -20_000, -20_000, -20_000, 100_000]}
            (root / "prediction.json").write_text(json.dumps(prediction))
            with patch("record_pass.record", side_effect=RuntimeError("capture reached")), \
                    patch("record_pass.LiveDopplerTap"):
                with self.assertRaisesRegex(RuntimeError, "capture reached"):
                    run(root / "plan.json", root / "test.toml")
            prediction["doppler_hz"][2] = 20_000
            (root / "prediction.json").write_text(json.dumps(prediction))
            with self.assertRaisesRegex(ValueError, "exceeds the raw RF passband"):
                run(root / "plan.json", root / "test.toml")

    def test_multichannel_views_survive_raw_error(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            start = datetime.now(timezone.utc) + timedelta(seconds=5)
            end = start + timedelta(seconds=1)
            (root / "plan.json").write_text(json.dumps({
                "target": "Synthetic", "output": "captures/new", "center_hz": 137_500_000,
                "sample_rate_hz": 240_000, "rf_bandwidth_hz": 180_000,
                "record_start": start.isoformat(), "record_end": end.isoformat(), "prediction": "prediction.json",
                "doppler_channels": [
                    {"key": key, "carrier_hz": carrier, "sample_rate_hz": 40_000, "cutoff_hz": 11_000}
                    for key, carrier in (("A", 137_515_000), ("B", 137_490_000))
                ],
            }))
            (root / "prediction.json").write_text(json.dumps({
                "reference_frequency_hz": 137_500_000, "doppler_hz": [-100, 100],
                "time_utc_unix": [start.timestamp() - 1, end.timestamp() + 2],
            }))
            (root / "test.toml").write_text('[pluto]\nuri = "example"\n')

            def failed_capture(output, *, sample_sink, **kwargs):
                output.mkdir(parents=True)
                iq = np.full(4096, 1 + 1j, dtype=np.complex64)
                (output / "iq.cs16").write_bytes(np.ones((iq.size, 2), dtype="<i2").tobytes())
                np.savez(output / "spectrum-history.npz")
                metadata = {"center_frequency_hz": 137_500_005, "actual_start_utc": start.isoformat(),
                            "sample_count": iq.size, "bytes_written": iq.size * 4,
                            "status": "partial", "settings_restored": True, "target": "Synthetic"}
                sample_sink(iq, 0, start.timestamp() + iq.size / 240_000, metadata)
                (output / "recording.json").write_text(json.dumps(metadata))
                raise RuntimeError("Raw file I/O failed")

            with patch("record_pass.record", side_effect=failed_capture):
                with self.assertRaisesRegex(RuntimeError, "Raw file I/O failed"):
                    run(root / "plan.json", root / "test.toml")
            output = root / "captures/new"
            metadata = json.loads((output / "recording.json").read_text())
            self.assertEqual([entry["view"] for entry in metadata["live_doppler_channels"]],
                             ["targets/A", "targets/B"])
            for key in ("A", "B"):
                view = output / "targets" / key
                for filename in ("iq.cs16", "spectrum-history.npz"):
                    self.assertEqual((view / filename).readlink(), Path("../..") / filename)
                live = json.loads((view / "recording.json").read_text())["live_doppler"]
                self.assertTrue(live["covers_all_recorded_samples"])
                self.assertTrue(live["file_sizes_match"])
                self.assertEqual(live["actual_center_hz"], 137_500_005)
                self.assertNotIn(str(root), json.dumps(live))
                self.assertEqual(live["residual_correction"]["reason"], "recording_or_restoration_not_completed")
