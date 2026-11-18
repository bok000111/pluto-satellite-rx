"""Confidence-gated, capture-local carrier correction after RX has stopped."""

import json
from pathlib import Path
import shutil
import time

import numpy as np

from oqpsk_carrier import estimate_paired, fit_paired, paired_cycles


def _peak(power, frequencies, limit, order):
    indices = np.flatnonzero(abs(frequencies) < limit * order)
    if indices.size < 16:
        return None
    index = int(indices[np.argmax(power[indices])])
    others = indices[abs(indices - index) > 4]
    floor = max(float(np.median(power[indices])), 1e-30)
    prominence = 10 * np.log10(max(float(power[index]), 1e-30) / floor)
    dominance = 10 * np.log10(max(float(power[index]), 1e-30)
                             / max(float(np.max(power[others])), 1e-30))
    if prominence < 12 or dominance < 6 or index in (indices[0], indices[-1]):
        return None
    left, middle, right = np.log(np.maximum(power[index - 1:index + 2], 1e-30))
    denominator = left - 2 * middle + right
    fraction = .5 * (left - right) / denominator if denominator < 0 else 0.
    resolution = float(frequencies[1] - frequencies[0]) / order
    return {'frequency_hz': float(frequencies[index] / order + np.clip(fraction, -.5, .5) * resolution),
            'order': order, 'peak_above_floor_db': float(prominence),
            'peak_dominance_db': float(dominance), 'resolution_hz': resolution}


def estimate_carrier(iq, sample_rate, max_cfo_hz):
    iq = np.asarray(iq, dtype=np.complex128)
    if iq.size < 2048 or not np.isfinite(iq).all():
        return {'accepted': False, 'reason': 'insufficient_or_invalid_samples'}
    rms = np.sqrt(np.mean(abs(iq)**2))
    if rms <= 1e-15:
        return {'accepted': False, 'reason': 'zero_power'}
    nfft = min(65536, 2**int(np.floor(np.log2(iq.size // 4))))
    blocks = (iq[:iq.size // nfft * nfft] / rms).reshape(-1, nfft)
    window = np.hanning(nfft)
    frequencies = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / sample_rate))
    raw = np.mean(abs(np.fft.fftshift(np.fft.fft(blocks * window, axis=1), axes=1))**2, axis=0)
    peak = int(np.argmax(raw))
    tone_fraction = float(raw[max(0, peak - 2):peak + 3].sum() / raw.sum())
    if tone_fraction > .15:
        return {'accepted': False, 'reason': 'dominant_unmodulated_tone', 'tone_fraction': tone_fraction}
    excess = np.maximum(raw - np.median(raw), 0.)
    cumulative = np.cumsum(excess) / max(float(excess.sum()), 1e-30)
    lower, upper = np.interp([.25, .75], cumulative, frequencies)
    candidates = []
    for order in (2, 4):
        powered = blocks**order
        spectra = abs(np.fft.fftshift(np.fft.fft(powered * window, axis=1), axes=1))**2
        mean = spectra.mean(axis=0)
        if order == 2:
            inside = np.flatnonzero((abs(frequencies) < max_cfo_hz * order)
                                   & (frequencies / order >= lower) & (frequencies / order <= upper))
            if inside.size > 16:
                strongest = int(inside[np.argmax(mean[inside])])
                others = inside[abs(inside - strongest) > 4]
                if others.size:
                    competitor = float(np.max(mean[others]))
                    floor = max(float(np.median(mean[inside])), 1e-30)
                    if competitor > floor * 16 and mean[strongest] < competitor * 4:
                        return {'accepted': False, 'reason': 'multiple_modulation_carriers'}
        estimate = _peak(mean, frequencies, max_cfo_hz, order)
        if estimate is None:
            continue
        if not lower <= estimate['frequency_hz'] <= upper:
            continue
        halves = [_peak(part.mean(axis=0), frequencies, max_cfo_hz, order)
                  for part in np.array_split(spectra, 2)]
        if any(part is None or abs(part['frequency_hz'] - estimate['frequency_hz'])
               > estimate['resolution_hz'] for part in halves):
            continue
        candidates.append(estimate)
    if not candidates:
        return {'accepted': False, 'reason': 'no_consistent_modulation_carrier'}
    if len(candidates) == 2 and abs(candidates[0]['frequency_hz'] - candidates[1]['frequency_hz']) > max(
            item['resolution_hz'] for item in candidates):
        return {'accepted': False, 'reason': 'ambiguous_carrier_hypotheses'}
    return {'accepted': True, **candidates[0]}


def fit_segments(measurements):
    groups, group = [], []
    for row in measurements:
        if not row['accepted']:
            if group:
                groups.append(group)
                group = []
            continue
        if group:
            gap = row['midpoint_seconds'] - group[-1]['midpoint_seconds']
            jump = abs(row['frequency_hz'] - group[-1]['frequency_hz'])
            if (row['order'] != group[-1]['order'] or gap > 3. or jump > 5. * gap + 2.
                    or row['end_seconds'] - group[0]['start_seconds'] > 30.):
                groups.append(group)
                group = []
        group.append(row)
    if group:
        groups.append(group)
    segments = []
    for group in groups:
        if len(group) < 6:
            continue
        times = np.array([r['midpoint_seconds'] for r in group])
        origin = float(times.mean())
        relative = times - origin
        frequency = np.array([r['frequency_hz'] for r in group])
        limit = max(1., max(r['resolution_hz'] for r in group) * .5)
        slope, intercept = np.polyfit(relative[::2], frequency[::2], 1)
        held_out = frequency[1::2] - (intercept + slope * relative[1::2])
        if np.max(abs(held_out)) > limit or abs(slope) > 5.:
            continue
        slope, intercept = np.polyfit(relative, frequency, 1)
        error = frequency - (intercept + slope * relative)
        if np.max(abs(error)) > limit or abs(slope) > 5.:
            continue
        segments.append({
            'start_seconds': group[0]['start_seconds'], 'end_seconds': group[-1]['end_seconds'],
            'reference_seconds': origin, 'frequency_hz': float(intercept),
            'drift_hz_per_second': float(slope), 'order': group[0]['order'],
            'measurement_count': len(group), 'fit_rms_hz': float(np.sqrt(np.mean(error**2))),
            'held_out_max_error_hz': float(np.max(abs(held_out))),
        })
    return segments


def correction_cycles(seconds, segments):
    seconds = np.asarray(seconds, dtype=float)
    phase = np.zeros_like(seconds)
    for segment in segments:
        if segment.get('estimator') == 'oqpsk_paired':
            phase += paired_cycles(seconds, segment)
            continue
        elapsed = np.clip(seconds - segment['start_seconds'], 0.,
                          segment['end_seconds'] - segment['start_seconds'])
        initial = (segment['frequency_hz'] + segment['drift_hz_per_second']
                   * (segment['start_seconds'] - segment['reference_seconds']))
        phase += initial * elapsed + .5 * segment['drift_hz_per_second'] * elapsed**2
    return phase


def analyze_carrier(source, sample_rate, cutoff_hz, *, deadline=float('inf')):
    count = source.size
    window_samples = int(sample_rate)
    limit = min(sample_rate / 16, cutoff_hz / 4)
    measurements = []
    for offset in range(0, count - window_samples + 1, 2 * window_samples):
        if time.monotonic() >= deadline:
            raise TimeoutError('Residual carrier analysis exceeded its time budget')
        row = estimate_carrier(source[offset:offset + window_samples], sample_rate, limit)
        row.update(start_seconds=offset / sample_rate, end_seconds=(offset + window_samples) / sample_rate,
                   midpoint_seconds=(offset + window_samples / 2) / sample_rate)
        measurements.append(row)
    segments = fit_segments(measurements)
    result = {'measurements': measurements, 'segments': segments,
              'maximum_cfo_hz': limit, 'analysis_window_seconds': 1., 'analysis_step_seconds': 2.}
    if not segments:
        paired = []
        for offset in range(0, count - window_samples + 1, 2 * window_samples):
            if time.monotonic() >= deadline:
                raise TimeoutError('Paired carrier analysis exceeded its time budget')
            previous = measurements[len(paired)]
            if previous.get('reason') in ('dominant_unmodulated_tone', 'multiple_modulation_carriers'):
                row = {'accepted': False, 'reason': previous['reason']}
            else:
                row = estimate_paired(source[offset:offset + window_samples], sample_rate, limit)
            row.update(start_seconds=offset / sample_rate, end_seconds=(offset + window_samples) / sample_rate,
                       midpoint_seconds=(offset + window_samples / 2) / sample_rate)
            paired.append(row)
        result.update(paired_measurements=paired, segments=fit_paired(paired))
    return result


def correct_residual(output, live, metadata, *, budget_seconds=30., source_path=None, work_dir=None):
    standalone = source_path is None and work_dir is None
    live_root = Path(output) / 'live-doppler'
    root = Path(work_dir) if work_dir is not None else live_root
    result = {'status': 'skipped', 'source': 'corrected.cf32', 'output': None,
              'method': 'Second/fourth-power estimates, with paired cyclic-line fallback and held-out drift checks.',
              'note': 'Frequency correction is limited to validated segments. This does not identify modulation, spacecraft, valid frames, or RF SNR.'}
    if not metadata.get('settings_restored') or metadata.get('status') != 'completed':
        return {**result, 'reason': 'recording_or_restoration_not_completed'}
    if live.get('status') != 'completed' or not live.get('covers_all_recorded_samples') or not live.get('file_sizes_match'):
        return {**result, 'reason': 'incomplete_doppler_source'}
    source_path = Path(source_path) if source_path is not None else live_root / 'corrected.cf32'
    report_path = root / 'residual-correction.json'
    destination = root / 'residual-corrected.cf32'
    partial = root / 'residual-corrected.partial.cf32'
    deadline = time.monotonic() + budget_seconds
    try:
        stat = source_path.stat()
        if stat.st_size != live['output_samples_per_branch'] * 8:
            raise ValueError('Residual source size does not match the Doppler metadata')
        if standalone and report_path.exists():
            saved = json.loads(report_path.read_text())
            if saved.get('source_bytes') != stat.st_size or saved.get('source_mtime_ns') != stat.st_mtime_ns:
                raise ValueError('Existing residual report belongs to different input samples')
            if saved['status'] == 'completed' and (not destination.exists() or destination.stat().st_size != stat.st_size):
                raise ValueError('Existing residual output is missing or incomplete')
            return saved
        result['source_bytes'] = stat.st_size
        if standalone:
            result['source_mtime_ns'] = stat.st_mtime_ns
        if stat.st_size == 0:
            result['reason'] = 'empty_source'
        else:
            iq = np.memmap(source_path, dtype='<c8', mode='r')
            rate = live['output_sample_rate_hz']
            result.update(analyze_carrier(iq, rate, live['fir_passband_hz'], deadline=deadline))
            if not result['segments']:
                result['reason'] = 'insufficient_consistent_carrier_measurements'
            elif shutil.disk_usage(root).free < stat.st_size + 2_000_000_000:
                result['reason'] = 'insufficient_space_for_additional_branch_and_2GB_reserve'
            elif destination.exists() or partial.exists():
                raise FileExistsError('A residual output already exists without a matching report')
            else:
                with partial.open('xb') as target:
                    for offset in range(0, iq.size, 262144):
                        if time.monotonic() >= deadline:
                            raise TimeoutError('Residual correction exceeded its time budget')
                        values = np.asarray(iq[offset:offset + 262144])
                        seconds = (offset + np.arange(values.size)) / rate
                        phase = correction_cycles(seconds, result['segments'])
                        corrected = values * np.exp(-2j * np.pi * np.remainder(phase, 1.))
                        target.write(corrected.astype('<c8').tobytes())
                if partial.stat().st_size != stat.st_size:
                    raise RuntimeError('Residual output length differs from its source')
                partial.replace(destination)
                result.update(status='completed', output=str(destination), output_bytes=stat.st_size,
                              corrected_seconds=sum(s['end_seconds'] - s['start_seconds'] for s in result['segments']))
    except Exception as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        if partial.exists():
            result['partial_output'] = str(partial)
    if not report_path.exists():
        try:
            report_path.write_text(json.dumps(result, indent=2) + '\n')
        except OSError as error:
            result['report_error'] = str(error)
    return result
