"""Continuous-phase receive-side Doppler correction with a bounded raw-IQ tap."""

from datetime import datetime
import json
from pathlib import Path
import queue
import threading
import time

import numpy as np
from scipy.signal import fftconvolve, firwin, kaiserord


class DopplerCurve:
    def __init__(self, utc, frequency_hz):
        utc, frequency_hz = np.asarray(utc, dtype=float), np.asarray(frequency_hz, dtype=float)
        if (utc.ndim != 1 or utc.size < 2 or frequency_hz.shape != utc.shape
                or not np.isfinite(utc).all() or not np.isfinite(frequency_hz).all()
                or np.any(np.diff(utc) <= 0)):
            raise ValueError("Doppler prediction must have finite, increasing UTC samples")
        self.origin = float(utc[0])
        self.times = utc - self.origin
        self.hz = frequency_hz
        self.slopes = np.diff(frequency_hz) / np.diff(self.times)
        self.areas = np.r_[0., np.cumsum(np.diff(self.times) * (frequency_hz[:-1] + frequency_hz[1:]) / 2)]

    @classmethod
    def from_file(cls, filename, carrier_hz):
        data = json.loads(Path(filename).read_text())
        reference = data['reference_frequency_hz']
        if type(reference) not in (int, float) or not np.isfinite(reference) or reference <= 0:
            raise ValueError("Doppler prediction requires a positive reference frequency")
        scale = float(carrier_hz) / reference
        return cls(data['time_utc_unix'], np.asarray(data['doppler_hz']) * scale)

    def integral(self, seconds):
        seconds = np.asarray(seconds, dtype=float)
        if np.any(seconds < 0) or np.any(seconds > self.times[-1]):
            raise ValueError("IQ time is outside the Doppler prediction; extrapolation refused")
        index = np.clip(np.searchsorted(self.times, seconds, side='right') - 1, 0, self.times.size - 2)
        dt = seconds - self.times[index]
        return self.areas[index] + self.hz[index] * dt + .5 * self.slopes[index] * dt**2


class ContinuousMixer:
    def __init__(self, sample_rate, fixed_offset_hz, curve):
        self.sample_rate = float(sample_rate)
        self.fixed_offset = float(fixed_offset_hz)
        self.curve = curve
        self.phase_cycles = 0.

    def process(self, iq, model_start_seconds):
        offsets = np.arange(iq.size, dtype=float) / self.sample_rate
        base = self.curve.integral(model_start_seconds)
        cycles = (self.phase_cycles + self.fixed_offset * offsets
                  + self.curve.integral(model_start_seconds + offsets) - base)
        duration = iq.size / self.sample_rate
        advance = (self.fixed_offset * duration
                   + float(self.curve.integral(model_start_seconds + duration)) - float(base))
        self.phase_cycles = (self.phase_cycles + advance) % 1.
        return np.asarray(iq * np.exp(-2j * np.pi * np.remainder(cycles, 1.)), dtype=np.complex64)


class PairedChannelizer:
    def __init__(self, sample_rate, output_rate, fixed_offset_hz, curve, cutoff_hz=40_000):
        if sample_rate <= 0 or output_rate <= 0 or sample_rate % output_rate:
            raise ValueError("Output rate must be an integer divisor of the input rate")
        if not 0 < cutoff_hz < output_rate / 2:
            raise ValueError("Channel filter must fit inside the output Nyquist frequency")
        self.decimation = int(sample_rate // output_rate)
        stopband_hz = output_rate / 2
        filter_cutoff_hz = (cutoff_hz + stopband_hz) / 2
        count, beta = kaiserord(90., (stopband_hz - cutoff_hz) / (sample_rate / 2))
        self.taps = firwin(count | 1, filter_cutoff_hz, fs=sample_rate,
                           window=('kaiser', beta)).astype(np.float32)
        self.filter_info = {
            'fir_passband_hz': cutoff_hz, 'fir_stopband_hz': stopband_hz,
            'fir_cutoff_hz': filter_cutoff_hz, 'fir_design_attenuation_db': 90.,
            'fir_taps': int(self.taps.size),
            'fir_delay_input_samples': (self.taps.size - 1) // 2,
        }
        self.history = np.zeros((2, self.taps.size - 1), dtype=np.complex64)
        zero = DopplerCurve(curve.origin + curve.times, np.zeros_like(curve.hz))
        self.mixers = [ContinuousMixer(sample_rate, fixed_offset_hz, zero),
                       ContinuousMixer(sample_rate, fixed_offset_hz, curve)]
        self.input_samples = 0

    def process(self, iq, model_start_seconds):
        mixed = np.stack([m.process(iq, model_start_seconds) for m in self.mixers])
        joined = np.concatenate((self.history, mixed), axis=1)
        filtered = fftconvolve(joined, self.taps[None, :], mode='valid', axes=1)
        self.history = joined[:, -(self.taps.size - 1):].copy()
        first = (-self.input_samples) % self.decimation
        result = filtered[:, first::self.decimation].astype('<c8')
        self.input_samples += iq.size
        return result


class LiveDopplerTap:
    def __init__(self, output, *, sample_rate, output_rate, carrier_hz, prediction,
                 cutoff_hz=40_000, cfo_hz=0., queue_blocks=16):
        if queue_blocks < 1:
            raise ValueError('The DSP queue must be bounded and have positive capacity')
        if sample_rate <= 0 or output_rate <= 0 or sample_rate % output_rate:
            raise ValueError('Invalid input/output sample rates')
        if not 0 < cutoff_hz < output_rate / 2:
            raise ValueError('Invalid channel cutoff')
        self.output = Path(output) / 'live-doppler'
        self.rate = int(sample_rate)
        self.output_rate = int(output_rate)
        self.carrier = float(carrier_hz)
        self.cfo = float(cfo_hz)
        self.cutoff = float(cutoff_hz)
        self.curve = (prediction if isinstance(prediction, DopplerCurve)
                      else DopplerCurve.from_file(prediction, carrier_hz))
        self.pending = queue.Queue(maxsize=queue_blocks)
        self.disabled = False
        self.closed = False
        self.created_output = False
        self.stats = {
            'status': 'waiting',
            'carrier_hz': carrier_hz, 'constant_cfo_hz': cfo_hz,
            'input_sample_rate_hz': self.rate, 'output_sample_rate_hz': self.output_rate,
            'output_format': 'cf32_le', 'output_scaling': 'raw ADC counts / 2048',
            'fir_passband_hz': cutoff_hz, 'fir_taps': None,
            'fir_delay_input_samples': None, 'queue_capacity_blocks': queue_blocks,
            'submitted_samples': 0, 'processed_samples': 0, 'rejected_samples': 0,
            'output_samples_per_branch': 0, 'file_sizes_match': False, 'maximum_queue_depth': 0,
            'maximum_processing_seconds': 0., 'maximum_processing_lag_seconds': 0.,
            'sample_time_note': 'UTC is estimated from host buffer delivery, not hardware timestamps. NCO phase is continuous across updates.',
            'sample_sink_contract': 'Accepted buffers are contiguous. On overflow or error the DSP branch stops accepting data; raw recording continues.',
            'errors': [], 'identity_confirmed': False,
        }
        self.thread = threading.Thread(target=self._run, name='live-doppler', daemon=True)
        self.thread.start()

    def submit(self, iq, sample_offset, received_utc, metadata):
        if self.closed:
            raise RuntimeError("Doppler tap has already closed")
        if self.disabled:
            self.stats['rejected_samples'] += int(iq.size)
            return
        item = (np.asarray(iq, dtype=np.complex64).copy(), int(sample_offset), float(received_utc),
                int(metadata['center_frequency_hz']), metadata['actual_start_utc'])
        try:
            self.pending.put_nowait(item)
        except queue.Full:
            self.stats['errors'].append('DSP queue overflow; paired channel files are partial')
            self.stats['rejected_samples'] += int(iq.size)
            self.disabled = True
            return
        self.stats['submitted_samples'] += int(iq.size)
        self.stats['maximum_queue_depth'] = max(self.stats['maximum_queue_depth'], self.pending.qsize())

    def _save(self):
        if self.created_output:
            temporary = self.output / 'progress.tmp'
            temporary.write_text(json.dumps(self.stats, indent=2) + '\n')
            temporary.replace(self.output / 'progress.json')

    def _run(self):
        processor = None
        files = []
        controls = None
        next_log = 0.
        try:
            while not (self.closed and self.pending.empty()):
                item = self.pending.get()
                if item is None:
                    break
                iq, offset, received, actual_center, actual_start = item
                if processor is None:
                    self.output.mkdir(exist_ok=False)
                    self.created_output = True
                    processor = PairedChannelizer(self.rate, self.output_rate,
                        self.carrier + self.cfo - actual_center, self.curve, self.cutoff)
                    self.stats.update(processor.filter_info)
                    for name in ('uncorrected.cf32', 'corrected.cf32'):
                        files.append((self.output / name).open('xb'))
                    controls = (self.output / 'control.jsonl').open('x')
                    self.stats.update(status='processing', actual_center_hz=actual_center,
                                      input_start_utc=actual_start)
                if offset != processor.input_samples:
                    raise RuntimeError(f'Noncontiguous IQ tap: expected {processor.input_samples}, received {offset}')
                before = time.monotonic()
                model_start_utc = received - iq.size / self.rate
                model_start = model_start_utc - self.curve.origin
                outputs = processor.process(iq / 2048., model_start)
                for target, values in zip(files, outputs):
                    target.write(values.tobytes())
                duration = time.monotonic() - before
                self.stats['processed_samples'] += int(iq.size)
                self.stats['output_samples_per_branch'] += int(outputs.shape[1])
                self.stats['maximum_processing_seconds'] = max(self.stats['maximum_processing_seconds'], duration)
                self.stats['maximum_processing_lag_seconds'] = max(self.stats['maximum_processing_lag_seconds'], time.time() - received)
                if received >= next_log:
                    elapsed = (received - datetime.fromisoformat(actual_start).timestamp())
                    row = {'input_sample_offset': offset, 'buffer_received_utc_unix': received,
                           'model_start_utc_unix': model_start_utc,
                           'predicted_doppler_hz': float(np.interp(model_start, self.curve.times, self.curve.hz)),
                           'iq_to_wall_ratio': (offset + iq.size) / self.rate / max(elapsed, .000001),
                           'processing_seconds': duration, 'queue_depth': self.pending.qsize()}
                    controls.write(json.dumps(row) + '\n')
                    controls.flush()
                    for target in files:
                        target.flush()
                    self._save()
                    next_log = received + 1.
        except OSError as error:
            self.stats['errors'].append(f'DSP file I/O: {type(error).__name__}')
            self.disabled = True
        except Exception as error:
            self.stats['errors'].append(f'{type(error).__name__}: {error}')
            self.disabled = True
        finally:
            for target in files + ([controls] if controls is not None else []):
                try:
                    target.close()
                except OSError as error:
                    self.stats['errors'].append(f'Closing DSP output: {type(error).__name__}')
            self.stats['status'] = 'partial' if self.stats['errors'] else ('completed' if processor else 'no_samples')
            try:
                self._save()
            except OSError as error:
                self.stats['errors'].append(f'Saving DSP progress: {type(error).__name__}')
                self.stats['status'] = 'partial'

    def close(self):
        if self.closed:
            return self.stats
        self.closed = True
        if self.thread.is_alive():
            try:
                self.pending.put(None, timeout=5.)
            except queue.Full:
                self.stats['errors'].append('DSP queue did not drain after RX stopped')
            self.thread.join(timeout=30.)
        if self.thread.is_alive():
            self.stats['errors'].append('DSP worker did not stop within 30 seconds')
            self.stats['status'] = 'partial'
        elif self.created_output:
            try:
                for name in ('uncorrected.cf32', 'corrected.cf32'):
                    path = self.output / name
                    self.stats[name + '_bytes'] = path.stat().st_size if path.exists() else None
                self.stats['file_sizes_match'] = all(
                    self.stats[name + '_bytes'] == self.stats['output_samples_per_branch'] * 8
                    for name in ('uncorrected.cf32', 'corrected.cf32'))
                self._save()
                (self.output / 'result.json').write_text(json.dumps(self.stats, indent=2) + '\n')
            except OSError as error:
                self.stats['errors'].append(f'Finalizing DSP output: {type(error).__name__}')
                self.stats['status'] = 'partial'
        return self.stats
