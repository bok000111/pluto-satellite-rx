"""Offline channel filtering with explicit passband, stopband and sample alignment."""

from pathlib import Path
import shutil

import numpy as np
from scipy.signal import fftconvolve, firwin, kaiserord


def channel_taps(sample_rate, passband_hz, stopband_hz):
    if not 0 < passband_hz < stopband_hz < sample_rate / 2:
        raise ValueError('Channel edges must be positive, ordered and below Nyquist')
    count, beta = kaiserord(90., (stopband_hz - passband_hz) / (sample_rate / 2))
    return firwin(count | 1, (passband_hz + stopband_hz) / 2,
                  fs=sample_rate, window=('kaiser', beta)).astype('f4')


def filter_channel(source, destination, sample_rate, passband_hz, stopband_hz, *, chunk_samples=262144):
    source, destination = Path(source), Path(destination)
    if destination.exists() or destination.with_suffix('.partial').exists():
        raise FileExistsError(destination)
    if chunk_samples < 1 or source.stat().st_size % 8:
        raise ValueError('Invalid chunk size or unaligned CF32 source')
    taps = channel_taps(sample_rate, passband_hz, stopband_hz)
    if shutil.disk_usage(destination.parent).free < source.stat().st_size + 2_000_000_000:
        raise RuntimeError('Insufficient space for filtered branch and 2GB reserve')
    iq = np.memmap(source, dtype='<c8', mode='r')
    half = len(taps) // 2
    partial = destination.with_suffix('.partial')
    with partial.open('xb') as sink:
        for start in range(0, iq.size, chunk_samples):
            end = min(start + chunk_samples, iq.size)
            values = np.asarray(iq[max(0, start - half):min(iq.size, end + half)])
            values = np.pad(values, (max(0, half - start), max(0, end + half - iq.size)))
            filtered = fftconvolve(values, taps, mode='valid')
            filtered.astype('<c8').tofile(sink)
    if partial.stat().st_size != source.stat().st_size:
        raise RuntimeError('Filtered channel length changed')
    partial.replace(destination)
    return {'source': str(source), 'output': str(destination), 'sample_rate_hz': sample_rate,
            'passband_hz': passband_hz, 'stopband_hz': stopband_hz, 'fir_taps': len(taps),
            'output_bytes': destination.stat().st_size, 'sample_delay': 0,
            'edge_transient_samples': half, 'design_stopband_attenuation_db': 90}
