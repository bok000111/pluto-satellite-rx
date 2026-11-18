"""Paired cyclic-line carrier tracking with capture-local validation."""

import numpy as np
from scipy.fft import fft, fftshift, next_fast_len
from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks


def estimate_paired(iq, sample_rate, max_cfo_hz):
    iq = np.asarray(iq, dtype=np.complex64)
    if iq.size < 2048 or not np.isfinite(iq).all():
        return {'accepted': False, 'reason': 'insufficient_or_invalid_samples'}
    nfft = next_fast_len(iq.size * 2)
    window = np.hanning(iq.size).astype('f4')
    raw_power = abs(fft(iq * window))**2
    peak = int(np.argmax(raw_power))
    if raw_power.take(np.arange(peak - 2, peak + 3), mode='wrap').sum() > .15 * raw_power.sum():
        return {'accepted': False, 'reason': 'dominant_unmodulated_tone'}
    frequencies = fftshift(np.fft.fftfreq(nfft, 1 / sample_rate))
    power = abs(fftshift(fft(iq**2 * window, n=nfft)))**2
    search = (abs(frequencies) > 2 * max_cfo_hz) & (abs(frequencies) < sample_rate * .48)
    floor = max(float(np.median(power[search])), 1e-30)
    peaks, _ = find_peaks(power, height=floor * 10**1.2, distance=5)
    low = peaks[search[peaks] & (frequencies[peaks] < 0)]
    high = peaks[search[peaks] & (frequencies[peaks] > 0)]
    low = low[np.argsort(power[low])[-12:]]
    high = high[np.argsort(power[high])[-12:]]
    candidates = []
    for left in low:
        for right in high:
            carrier = (frequencies[left] + frequencies[right]) / 4
            clock = (frequencies[right] - frequencies[left]) / 2
            if abs(carrier) >= max_cfo_hz:
                continue
            contrast = 10 * np.log10(float(min(power[left], power[right])) / floor)
            candidates.append({'frequency_hz': float(carrier), 'symbol_clock_hz': float(clock),
                               'peak_above_floor_db': float(contrast), 'order': 2,
                               'estimator': 'oqpsk_paired', 'resolution_hz': sample_rate / iq.size / 4})
    candidates.sort(key=lambda item: item['peak_above_floor_db'], reverse=True)
    if not candidates:
        return {'accepted': False, 'reason': 'no_paired_cyclic_lines'}
    if len(candidates) > 1 and candidates[0]['peak_above_floor_db'] - candidates[1]['peak_above_floor_db'] < 3:
        return {'accepted': False, 'reason': 'ambiguous_paired_cyclic_lines'}
    return {'accepted': True, **candidates[0]}


def fit_paired(measurements):
    groups, group = [], []
    for row in measurements:
        if not row['accepted']:
            continue
        if group:
            gap = row['midpoint_seconds'] - group[-1]['midpoint_seconds']
            clock_limit = max(5., group[-1]['symbol_clock_hz'] * 50e-6)
            if (gap > 32 or abs(row['frequency_hz'] - group[-1]['frequency_hz']) > 5 * gap + 2
                    or abs(row['symbol_clock_hz'] - group[-1]['symbol_clock_hz']) > clock_limit):
                groups.append(group)
                group = []
        group.append(row)
    if group:
        groups.append(group)
    segments = []
    for group in groups:
        if len(group) < 12:
            continue
        times = np.array([row['midpoint_seconds'] for row in group])
        frequencies = np.array([row['frequency_hz'] for row in group])
        buckets = np.floor((times - times[0]) / 10).astype(int)

        def knots(selection):
            pairs = [(np.median(times[selection & (buckets == bucket)]),
                      np.median(frequencies[selection & (buckets == bucket)]))
                     for bucket in np.unique(buckets[selection])]
            return np.array(pairs).T

        training = np.arange(len(times)) % 2 == 0
        kt, kf = knots(training)
        if len(kt) < 3:
            continue
        validation = (~training) & (times >= kt[0]) & (times <= kt[-1])
        if validation.sum() < 4:
            continue
        errors = frequencies[validation] - PchipInterpolator(kt, kf)(times[validation])
        if np.max(abs(errors)) > 5 or np.sqrt(np.mean(errors**2)) > 2:
            continue
        kt, kf = knots(np.ones(len(times), dtype=bool))
        if np.max(abs(np.diff(kf) / np.diff(kt))) > 5:
            continue
        segments.append({'start_seconds': float(times[0] - .5), 'end_seconds': float(times[-1] + .5),
                         'knots_seconds': kt.tolist(), 'cfo_hz': kf.tolist(),
                         'estimator': 'oqpsk_paired', 'order': 2,
                         'symbol_clock_hz': float(np.median([r['symbol_clock_hz'] for r in group])),
                         'measurement_count': len(group), 'held_out_count': int(validation.sum()),
                         'held_out_rms_hz': float(np.sqrt(np.mean(errors**2))),
                         'held_out_max_error_hz': float(np.max(abs(errors)))})
    return segments


def paired_cycles(seconds, segment):
    knots = np.asarray(segment['knots_seconds'])
    frequency = np.asarray(segment['cfo_hz'])
    bounded = np.clip(seconds, segment['start_seconds'], segment['end_seconds'])
    integral = PchipInterpolator(knots, frequency).antiderivative()
    return (integral(np.clip(bounded, knots[0], knots[-1])) - integral(knots[0])
            + np.minimum(bounded - knots[0], 0) * frequency[0]
            + np.maximum(bounded - knots[-1], 0) * frequency[-1]
            - (segment['start_seconds'] - knots[0]) * frequency[0])
