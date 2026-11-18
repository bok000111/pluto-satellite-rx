#!/usr/bin/env python3
"""Compare AO-123 signal candidates in saved live Doppler spectra."""

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann


def match_candidates(frequency, before_db, after_db, doppler_hz):
    """Return the match mask, before/after residuals in Hz, SNR in dB and line power in dBFS."""
    rows = np.arange(len(after_db))
    df = frequency[1] - frequency[0]
    search = (frequency >= 100) & (frequency <= 450)
    noise = (abs(frequency) >= 500) & (abs(frequency) <= 1500)
    if not search.any() or not noise.any():
        raise ValueError('Spectra do not resolve the AO-123 search and noise bands')
    peak = np.flatnonzero(search)[after_db[:, search].argmax(axis=1)]
    after = frequency[peak]
    excess = after_db[rows, peak] - np.median(after_db[:, noise], axis=1)
    near = abs(frequency) <= 12_000
    before_frequency = frequency[near]
    search_before = abs(before_frequency - (doppler_hz + after)[:, None]) < 150
    before_peak = np.where(search_before, before_db[:, near], -np.inf).argmax(axis=1)
    before = before_frequency[before_peak] - doppler_hz
    matched = (search_before.any(axis=1) & (excess >= 12) & (abs(before - after) <= 45)
               & (after >= 145) & (after <= 375))
    if np.any(peak[matched] < 2) or np.any(peak[matched] + 2 >= len(frequency)):
        raise ValueError('Spectra do not contain five bins around the matched line')
    band = after_db[rows[matched, None], peak[matched, None] + np.arange(-2, 3)]
    local_noise = np.mean(10 ** (after_db[matched][:, noise] * 0.1), axis=1)
    line = (np.sum(10 ** (band * 0.1), axis=1) - 5 * local_noise) * df
    snr = np.full(len(rows), np.nan)
    power = np.full(len(rows), np.nan)
    positive = line > 0
    measured = rows[matched][positive]
    snr[measured] = 10 * np.log10(line[positive] / (5 * local_noise[positive] * df))
    power[measured] = 10 * np.log10(line[positive])
    return matched, before, after, snr, power


def edges(values, step):
    if len(values) == 1:
        return values[0] + np.array([-0.5, 0.5]) * step
    mids = (values[:-1] + values[1:]) / 2
    return np.r_[2 * values[0] - mids[0], mids, 2 * values[-1] - mids[-1]]


def analyze(recording, prediction, output, before_wav=None, after_wav=None):
    if (before_wav is None) != (after_wav is None):
        raise ValueError('--before-wav and --after-wav must be supplied together')
    metadata = json.loads((recording / 'recording.json').read_text())
    live = metadata['live_doppler']
    root = recording / 'live-doppler'
    prediction = json.loads(prediction.read_text())
    carrier = live['carrier_hz']
    input_rate = live['input_sample_rate_hz']
    output_rate = live['output_sample_rate_hz']
    controls = [json.loads(line) for line in (root / 'control.jsonl').read_text().splitlines()]
    control_sample = np.array([row['input_sample_offset'] for row in controls]) / input_rate
    control_utc = np.array([row['model_start_utc_unix'] for row in controls])
    predicted_times = prediction['time_utc_unix']
    predicted_hz = np.asarray(prediction['doppler_hz']) * (carrier / prediction['reference_frequency_hz'])

    with np.load(root / 'uncorrected-spectra.npz') as before, np.load(root / 'corrected-spectra.npz') as after:
        elapsed, frequency = after['elapsed_seconds'], after['offset_hz']
        if not np.array_equal(elapsed, before['elapsed_seconds']) or not np.array_equal(frequency, before['offset_hz']):
            raise ValueError('Before and after spectra must use the same time and frequency grids')
        u, c = before['psd_dbfs_hz'], after['psd_dbfs_hz']
    if not len(elapsed) or u.shape != c.shape or c.shape != (len(elapsed), len(frequency)):
        raise ValueError('Expected nonempty paired spectra on the saved grids')
    if not len(controls):
        raise ValueError('control.jsonl contains no timing records')
    elapsed = elapsed + 2 * 8192 / output_rate - live['fir_delay_input_samples'] / input_rate
    times = np.interp(elapsed, control_sample, control_utc) + np.maximum(elapsed - control_sample[-1], 0)
    doppler = np.interp(times, predicted_times, predicted_hz)
    matched, before, after, snr, power = match_candidates(frequency, u, c, doppler)

    with np.load(recording / 'spectrum-history.npz') as raw:
        raw_f = raw['frequency_hz'] - carrier
        raw_t = raw['time_utc_unix']
        narrow = abs(raw_f) <= 16_000
        raw_psd = raw['psd_dbfs_hz'][:, narrow]
    if not raw_psd.size:
        raise ValueError('Receiver spectrum history has no bins within 16 kHz of the carrier')
    start = datetime.fromisoformat(metadata['actual_start_utc']).timestamp()
    end = datetime.fromisoformat(metadata['actual_end_utc']).timestamp()
    markers = [datetime.fromisoformat(pass_[key]).timestamp()
               for pass_ in prediction.get('passes', ()) for key in ('aos_utc', 'peak_utc', 'los_utc')
               if pass_[key] is not None]
    kst = timezone(timedelta(hours=9))
    formatter = mdates.DateFormatter('%H:%M:%S' if end - start < 120 else '%H:%M', tz=kst)

    output.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.savefig(output / name, dpi=155)
        plt.close(fig)

    def time_axis(ax, left=start, right=end):
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(tz=kst))
        ax.xaxis.set_major_formatter(formatter)
        ax.set(xlim=(left / 86400, right / 86400), xlabel='Time (KST)')

    fig, ax = plt.subplots(figsize=(12, 4.4), layout='constrained')
    image = ax.pcolormesh(edges(raw_t, 1) / 86400, edges(raw_f[narrow] / 1000, 1), raw_psd.T,
                          cmap='magma', vmin=np.percentile(raw_psd, 12), vmax=np.percentile(raw_psd, 99.3),
                          shading='flat', rasterized=True)
    ax.plot(raw_t / 86400, np.interp(raw_t, predicted_times, predicted_hz) / 1000,
            '--', color='#6be7ef', label='Predicted Doppler')
    for marker in markers:
        if start <= marker <= end:
            ax.axvline(marker / 86400, lw=.8, ls=':', color='#b8c7d6')
    ax.legend()
    ax.set(ylabel=f'Offset from {carrier / 1e6:.3f} MHz (kHz)', ylim=(-16, 16))
    time_axis(ax)
    fig.suptitle('AO-123 pass | Receiver spectrum history')
    fig.colorbar(image, ax=ax, label='PSD (dBFS/Hz)')
    save(fig, '01-pass-waterfall.png')

    visible = abs(frequency) <= 16_100
    scale = np.percentile(np.concatenate([u[:, visible], c[:, visible]]), [12, 99.8])
    fig, axs = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, layout='constrained')
    for ax, psd, track, observed, label in zip(
            axs, (u, c), (doppler, np.zeros_like(doppler)), (before + doppler, after),
            ('Before correction', 'After correction')):
        image = ax.imshow(psd[:, visible].T, origin='lower', aspect='auto', cmap='magma',
                          extent=[*edges(times, max(1, 32768 / output_rate))[[0, -1]] / 86400,
                                  *edges(frequency[visible] / 1000, 1)[[0, -1]]],
                          vmin=scale[0], vmax=scale[1], interpolation='nearest')
        ax.plot(times / 86400, track / 1000, '--', color='#6be7ef', label='Predicted signal')
        ax.scatter(times[matched] / 86400, observed[matched] / 1000, s=8, color='#d7ff88',
                   linewidth=0, label='Matched candidate')
        for marker in markers:
            if start <= marker <= end:
                ax.axvline(marker / 86400, lw=.8, ls=':', color='#b8c7d6')
        ax.set(title=label, ylim=(-16, 16), ylabel='Offset (kHz)')
    axs[0].legend(loc='upper right', ncols=2)
    time_axis(axs[1])
    fig.suptitle(f'AO-123 pass | Doppler correction at {carrier / 1e6:.3f} MHz')
    fig.colorbar(image, ax=axs, label='PSD (dBFS/Hz)', shrink=.9)
    save(fig, '02-doppler-comparison.png')

    width = 5 * (frequency[1] - frequency[0])
    fig, axs = plt.subplots(3, 1, figsize=(12, 7.1), sharex=True, layout='constrained')
    axs[0].scatter(times[matched] / 86400, power[matched], s=18, color='#17728c')
    axs[0].set_ylabel('Line power (dBFS)')
    axs[1].scatter(times[matched] / 86400, snr[matched], s=18, color='#17728c')
    axs[1].set_ylabel(f'SNR in {width:.1f} Hz (dB)')
    axs[2].scatter(times[matched] / 86400, before[matched], s=21, color='#8c4b14', label='Before minus prediction')
    axs[2].scatter(times[matched] / 86400, after[matched], s=19, facecolor='none', edgecolor='#17728c',
                   label='After correction')
    axs[2].set_ylabel('Residual (Hz)')
    axs[2].legend(loc='upper right')
    for ax in axs:
        ax.grid(axis='y', alpha=.23)
    if matched.any():
        time_axis(axs[2], times[matched][0] - 25, times[matched][-1] + 25)
    else:
        time_axis(axs[2])
    fig.suptitle('AO-123 pass | Intermittent signal candidate')
    save(fig, '03-signal-metrics.png')

    if before_wav is not None:
        audio = []
        rate = 12_500
        nfft, hop, avg = 512, 256, 12
        transform = ShortTimeFFT(hann(nfft, sym=False), hop=hop, fs=rate, scale_to='psd')
        for path in (before_wav, after_wav):
            wav_rate, pcm = wavfile.read(path, mmap=True)
            frames = ((len(pcm) - nfft) // hop + 1) // avg * avg
            if wav_rate != rate or pcm.ndim != 1 or pcm.dtype.kind != 'i' or pcm.dtype.itemsize != 2 or frames < avg:
                raise ValueError('Expected 12.5 kHz mono PCM16 WAV with at least 12 complete STFT frames')
            zz = transform.spectrogram(pcm.astype(np.float32) / 32768, p0=1, p1=frames + 1)
            power_avg = zz.reshape(len(transform.f), -1, avg).mean(axis=-1)
            audio.append(10 * np.log10(np.maximum(power_avg, 1e-16)))
        af = transform.f
        selected = (af >= 100) & (af <= 5000)
        vmin, vmax = np.percentile(np.concatenate([z[selected] for z in audio]), [10, 99.5])
        fig, axs = plt.subplots(2, 1, figsize=(12, 7.1), sharex=True, layout='constrained')
        for ax, psd, label in zip(axs, audio, ('FM audio before correction', 'FM audio after correction')):
            elapsed = (1 + np.arange(psd.shape[1]) * avg + (avg - 1) / 2) * hop / rate
            audio_t = np.interp(elapsed, control_sample, control_utc) + np.maximum(elapsed - control_sample[-1], 0)
            image = ax.imshow(psd[selected], origin='lower', aspect='auto', cmap='magma',
                              extent=[*edges(audio_t, avg * hop / rate)[[0, -1]] / 86400,
                                      *edges(af[selected] / 1000, 1)[[0, -1]]],
                              vmin=vmin, vmax=vmax, interpolation='nearest')
            ax.scatter(times[matched] / 86400, np.full(matched.sum(), 4.91), s=5, color='#6be7ef',
                       label='RF candidate snapshots')
            ax.set(title=label, ylabel='Audio frequency (kHz)', ylim=(.1, 5))
        axs[0].legend(loc='upper right')
        time_axis(axs[1])
        fig.suptitle('AO-123 pass | Narrowband FM audio')
        fig.colorbar(image, ax=axs, label='Audio PSD (dBFS/Hz)', shrink=.9)
        save(fig, '04-fm-spectrogram.png')

    indices = np.flatnonzero(matched)
    report = {
        'carrier_hz': carrier, 'satellite_identity_confirmed': False, 'line_bandwidth_hz': width,
        'snapshot_count': len(times), 'matched_snapshots': len(indices),
        'before_track_span_hz': float(np.ptp((before + doppler)[matched])) if len(indices) else None,
        'after_track_span_hz': float(np.ptp(after[matched])) if len(indices) else None,
        'before_residual_median_hz': float(np.median(before[matched])) if len(indices) else None,
        'after_residual_median_hz': float(np.median(after[matched])) if len(indices) else None,
        'candidates': [
            {'time_utc_unix': times[i], 'doppler_hz': doppler[i],
             'before_residual_hz': before[i], 'after_residual_hz': after[i],
             'snr_db': snr[i] if np.isfinite(snr[i]) else None,
             'line_power_dbfs': power[i] if np.isfinite(power[i]) else None}
            for i in indices
        ],
    }
    (output / 'candidate-report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', required=True, type=Path)
    parser.add_argument('--prediction', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--before-wav', type=Path)
    parser.add_argument('--after-wav', type=Path)
    args = parser.parse_args()
    analyze(args.recording, args.prediction, args.output, args.before_wav, args.after_wav)
