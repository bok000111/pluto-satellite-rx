"""Capture-local OQPSK tracking and independently decodable LRPT intervals."""

import json
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.signal import fftconvolve

from channel_filter import channel_taps, filter_channel
from oqpsk_carrier import estimate_paired, fit_paired, paired_cycles


def analyze(source, rate, cutoff, *, deadline=float('inf')):
    passband, stopband = 65000, 75000
    if stopband >= min(rate / 2, cutoff):
        raise ValueError('LRPT analysis filter must fit inside the recorded passband')
    taps = channel_taps(rate, passband, stopband)
    half = len(taps) // 2
    window = int(rate)
    limit = min(rate / 16, cutoff / 4)
    measurements = []
    for first in range(0, source.size - window + 1, window):
        if time.monotonic() >= deadline:
            raise TimeoutError('Segmented carrier analysis exceeded its time budget')
        last = first + window
        values = np.asarray(source[max(0, first - half):min(source.size, last + half)])
        values = np.pad(values, (max(0, half - first), max(0, last + half - source.size)))
        filtered = fftconvolve(values, taps, mode='valid')
        row = estimate_paired(filtered, rate, limit)
        if row.get('accepted'):
            clock = row['symbol_clock_hz']
            nominal = min((72000, 80000), key=lambda value: abs(value - clock))
            if abs(nominal - clock) > nominal * 100e-6:
                row.update(accepted=False, reason='unsupported_lrpt_symbol_clock')
        row.update(start_seconds=first / rate, end_seconds=last / rate,
                   midpoint_seconds=(first + window / 2) / rate)
        measurements.append(row)
    return {'measurements': measurements, 'segments': fit_paired(measurements),
            'analysis_window_seconds': 1., 'analysis_step_seconds': 1.,
            'analysis_filter': {'passband_hz': passband, 'stopband_hz': stopband,
                                'fir_taps': len(taps), 'sample_delay': 0},
            'maximum_cfo_hz': limit}


def prepare(source, folder, rate, cutoff, bandwidths_khz, *, budget_seconds=120.):
    source, folder = Path(source), Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stat = source.stat()
    widths = list(dict.fromkeys(bandwidths_khz))
    for width in widths:
        if not 0 < width * 500 < width * 500 + 10000 < min(rate / 2, cutoff):
            raise ValueError('Segment filter must fit inside the recorded passband')
    identity = {'source': str(source.resolve()), 'source_bytes': stat.st_size,
                'sample_rate_hz': rate,
                'source_passband_hz': cutoff, 'bandwidths_khz': widths, 'analysis_version': 1}
    report_path = folder / 'segments.json'
    result = {**identity, 'status': 'skipped', 'branches': [], 'segments': [],
              'additional_doppler_applied': False}
    deadline = time.monotonic() + budget_seconds
    try:
        if not stat.st_size or stat.st_size % 8:
            raise ValueError('Empty or unaligned CF32 source')
        iq = np.memmap(source, dtype='<c8', mode='r')
        result.update(analyze(iq, rate, cutoff, deadline=deadline))
        intervals = [(int(np.ceil(s['start_seconds'] * rate)), int(np.floor(s['end_seconds'] * rate)))
                     for s in result['segments']]
        required = sum((b - a) * 8 * (1 + len(widths)) for a, b in intervals)
        result['required_additional_iq_bytes'] = required
        if not intervals:
            result['reason'] = 'insufficient_consistent_carrier_measurements'
        elif shutil.disk_usage(folder).free < required + 2_000_000_000:
            result.update(status='failed', reason='insufficient_space_for_segments_and_2GB_reserve')
        else:
            for index, (segment, (a, b)) in enumerate(zip(result['segments'], intervals)):
                destination = folder / f'segment-{index:03d}-wide.cf32'
                partial = destination.with_suffix('.partial.cf32')
                if destination.exists() or partial.exists():
                    raise FileExistsError('Segment output already exists')
                with partial.open('xb') as sink:
                    for first in range(a, b, 262144):
                        if time.monotonic() >= deadline:
                            raise TimeoutError('Segment correction exceeded its time budget')
                        z = iq[first:min(first + 262144, b)]
                        seconds = (first + np.arange(z.size)) / rate
                        phase = paired_cycles(seconds, segment)
                        corrected = z * np.exp(-2j * np.pi * np.remainder(phase, 1.))
                        corrected.astype('<c8').tofile(sink)
                if partial.stat().st_size != (b - a) * 8:
                    raise RuntimeError('Segment output length differs from its interval')
                partial.replace(destination)
                outputs = [('wide', destination)]
                for width in widths:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Segment filtering exceeded its time budget')
                    target = folder / f'segment-{index:03d}-{width:g}k.cf32'
                    filter_channel(destination, target, rate, width * 500, width * 500 + 10000)
                    outputs.append((f'{width:g}k', target))
                nominal = min((72000, 80000), key=lambda value: abs(value - segment['symbol_clock_hz']))
                for label, path in outputs:
                    result['branches'].append({'label': f'segment-{index:03d}-{label}', 'path': str(path),
                                               'bytes': path.stat().st_size,
                                               'sample_start': a, 'sample_end': b, 'symbol_rate': nominal,
                                               'measured_symbol_clock_hz': segment['symbol_clock_hz']})
            result.update(status='completed', corrected_seconds=sum((b - a) / rate for a, b in intervals))
    except Exception as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
    report_path.write_text(json.dumps(result, indent=2) + '\n')
    return result
