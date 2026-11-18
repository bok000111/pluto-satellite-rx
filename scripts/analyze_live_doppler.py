#!/usr/bin/env python3
"""Audit paired live DSP files and plot spectra without asserting satellite identity."""

import argparse
import json
from pathlib import Path

import numpy as np


def analyze(output):
    output = Path(output)
    metadata = json.loads((output / 'recording.json').read_text())
    live = metadata['live_doppler']
    root = output / 'live-doppler'
    rate = live['output_sample_rate_hz']
    count = live['output_samples_per_branch']
    if type(rate) not in (int, float) or not np.isfinite(rate) or rate <= 0 or type(count) is not int or count < 0:
        raise ValueError('Invalid Doppler output rate or sample count')
    raw_bytes = (output / 'iq.cs16').stat().st_size
    report = {
        'raw_size_matches': raw_bytes == metadata['sample_count'] * 4 == metadata['bytes_written'],
        'raw_bytes': raw_bytes, 'raw_iq_seconds': metadata.get('recorded_iq_seconds'),
        'settings_restored': metadata.get('settings_restored'),
        'gain_restored': metadata.get('gain_probe', {}).get('restored'),
        'dsp_status': live['status'], 'dsp_errors': live['errors'],
        'covers_all_raw_samples': live['processed_samples'] == metadata['sample_count'],
        'rejected_samples': live['rejected_samples'], 'branches': {},
        'satellite_identity_confirmed': False, 'valid_frame_count': None,
        'measurement_note': 'Peak tracks and adjacent-band excess are signal candidates, not calibrated SNR or verified frames.',
        'quality_summary': metadata.get('quality_summary'),
        'residual_correction': live.get('residual_correction'),
    }
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fft_size = 8192
    hop = max(int(rate), fft_size * 4)
    starts = np.arange(0, max(0, count - fft_size * 4 + 1), hop, dtype=int)
    window = np.hanning(fft_size)
    frequency = np.fft.fftshift(np.fft.fftfreq(fft_size, 1 / rate))
    band_edge = min(25_000, rate * 25 / 70)
    inside = abs(frequency) < band_edge
    adjacent = (abs(frequency) >= band_edge) & (abs(frequency) < min(35_000, rate / 2))
    branches = ['uncorrected', 'corrected']
    if live.get('residual_correction', {}).get('status') == 'completed':
        branches.append('residual-corrected')
    for name in branches:
        path = root / (name + '.cf32')
        size = path.stat().st_size
        branch = {'bytes': size, 'size_matches': size == count * 8}
        report['branches'][name] = branch
        if size != count * 8 or not starts.size:
            branch['analysis_status'] = 'insufficient_or_inconsistent_samples'
            continue
        iq = np.memmap(path, dtype='<c8', mode='r')
        spectra = []
        for start in starts:
            samples = np.asarray(iq[start:start + fft_size * 4]).reshape(4, fft_size)
            power = np.mean(abs(np.fft.fftshift(np.fft.fft(samples * window, axis=1), axes=1))**2, axis=0)
            spectra.append(power / (rate * np.sum(window**2)))
        power = np.asarray(spectra)
        db = 10 * np.log10(np.maximum(power, 1e-30))
        indices = np.flatnonzero(inside)[np.argmax(power[:, inside], axis=1)]
        peak_hz = frequency[indices]
        peak_excess = db[np.arange(len(indices)), indices] - np.median(db[:, adjacent], axis=1)
        selected = peak_excess >= 10
        branch.update(analysis_status='completed', snapshot_count=len(starts),
                      peak_candidate_snapshots=int(selected.sum()),
                      median_peak_excess_db=float(np.median(peak_excess)),
                      peak_track_span_hz=float(np.ptp(peak_hz[selected])) if selected.any() else None)
        np.savez_compressed(root / (name + '-spectra.npz'), elapsed_seconds=starts / rate,
                            offset_hz=frequency, psd_dbfs_hz=db, peak_hz=peak_hz,
                            peak_excess_db=peak_excess, candidate_mask=selected)
        fig, ax = plt.subplots(figsize=(10, 3.8), layout='constrained')
        ax.plot(frequency / 1000, 10 * np.log10(np.maximum(power.mean(axis=0), 1e-30)), linewidth=.7)
        ax.set(xlabel='Frequency offset (kHz)', ylabel='PSD (dBFS/Hz, uncalibrated)',
               title=metadata['target'] + ' / ' + name)
        fig.savefig(root / (name + '-spectrum.png'), dpi=150)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
        centers = (starts + fft_size * 2) / rate
        half_hop = hop / (2 * rate)
        image = ax.imshow(db, origin='lower', aspect='auto',
                          extent=(frequency[0] / 1000, frequency[-1] / 1000,
                                  centers[0] - half_hop, centers[-1] + half_hop),
                          vmin=float(np.percentile(db, 10)), vmax=float(np.percentile(db, 99.5)))
        ax.set(xlabel='Frequency offset (kHz)', ylabel='IQ elapsed time (s)',
               title=metadata['target'] + ' / ' + name)
        fig.colorbar(image, ax=ax, label='dBFS/Hz, uncalibrated')
        fig.savefig(root / (name + '-waterfall.png'), dpi=150)
        plt.close(fig)
    (root / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    analyze(parser.parse_args().output)
