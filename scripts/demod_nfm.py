#!/usr/bin/env python3
"""Demodulate 100 kS/s paired live Doppler IQ to 12.5 kS/s mono NFM audio."""

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile


IQ_RATE = 100_000
RATE = 25_000
AUDIO_RATE = 12_500
CTCSS = np.array([
    62.5, 67, 69.3, 71.9, 74.4, 77, 79.7, 82.5, 85.4, 88.5, 91.5, 94.8, 97.4, 100,
    103.5, 107.2, 110.9, 114.8, 118.8, 123, 127.3, 131.8, 136.5, 141.3, 146.2, 151.4,
    156.7, 159.8, 162.2, 165.5, 167.9, 171.3, 173.8, 177.3, 179.9, 183.5, 186.2,
    189.9, 192.8, 196.6, 199.5, 203.5, 206.5, 210.7, 218.1, 225.7, 229.1, 233.6, 241.8, 250.3, 254.1,
])


def demod(iq, sample_rate, offset_hz):
    """Return channel IQ, raw discriminator, de-emphasized discriminator and audio."""
    if sample_rate != IQ_RATE or len(iq) < 5 or not np.isfinite(offset_hz):
        raise ValueError('NFM requires 100 kS/s IQ with at least five samples and a finite offset')
    fir = signal.firwin(401, 6300, fs=sample_rate, window=('kaiser', 8.6))
    tuned = iq * np.exp(-2j * np.pi * offset_hz / sample_rate * np.arange(len(iq)))
    channel = signal.resample_poly(tuned, 1, 4, window=fir)
    discriminator = np.angle(channel[1:] * channel[:-1].conj()) * RATE / (2 * np.pi)
    alpha = -np.expm1(-1 / (RATE * 750e-6))
    deemphasis = signal.lfilter([alpha], [1, alpha - 1], discriminator)
    voice_sos = signal.butter(4, (300, 3000), 'bandpass', fs=RATE, output='sos')
    audio = signal.resample_poly(signal.sosfilt(voice_sos, deemphasis), 1, 2)
    return channel, discriminator, deemphasis, audio


def db(power):
    return 10 * np.log10(np.maximum(power, np.finfo(float).tiny))


def voice_power(values):
    f, p = signal.welch(values, fs=RATE, nperseg=min(2048, len(values)), detrend='constant')
    band = (f >= 300) & (f <= 3000)
    return db(p[band].sum() * (f[1] - f[0]))


def tone_powers(low):
    if len(low) < 1000:
        return np.empty((0, len(CTCSS))), np.empty((0, len(CTCSS)))
    frames = np.lib.stride_tricks.sliding_window_view(low, 1000)[::500]
    power = abs(np.fft.rfft((frames - frames.mean(1, keepdims=True)) * np.hanning(1000), n=4000, axis=1)) ** 2
    freq = np.fft.rfftfreq(4000, 1 / 1000)
    best = np.empty((len(power), len(CTCSS)))
    found = np.empty_like(best)
    for j, nominal in enumerate(CTCSS):
        bins = np.flatnonzero(abs(freq - nominal) <= .75)
        adjacent = (abs(freq - nominal) >= 2) & (abs(freq - nominal) <= 8)
        peak = bins[power[:, bins].argmax(1)]
        best[:, j] = db(power[np.arange(len(power)), peak]) - db(np.median(power[:, adjacent], axis=1))
        found[:, j] = freq[peak]
    return best, found


def rf_metrics(values):
    f, p = signal.welch(values, fs=RATE, nperseg=min(8192, len(values)), detrend='constant')
    near = abs(f) <= 500
    i = np.flatnonzero(near)[p[near].argmax()]
    distance = abs(f - f[i])
    local = (distance > 400) & (distance < 850)
    excess = db(p[i]) - db(np.median(p[local]))
    wide = db(p[distance < 5000].sum())
    return {'peak_offset_hz': round(float(f[i]), 1), 'peak_excess_db': float(excess),
            'relative_power_db': {str(b): round(float(db(p[distance <= b].sum()) - wide), 1)
                                  for b in (30, 100, 300)}}


def render(recording, output, offset_hz, windows, branch):
    root = recording / 'live-doppler'
    live = json.loads((root / 'result.json').read_text())
    path = root / (branch + '.cf32')
    fs = live['output_sample_rate_hz']
    if fs != IQ_RATE or path.stat().st_size != live['output_samples_per_branch'] * 8:
        raise ValueError('Expected a 100 kS/s cf32 branch matching the recorded sample count')
    iq = np.memmap(path, mode='r', dtype='<c8')
    output.mkdir(parents=True, exist_ok=True)
    if windows is None:
        *_, audio = demod(iq, fs, offset_hz)
        gain = .9 / max(np.max(abs(audio)), .9 * 2500)
        wavfile.write(output / f'{branch}-nfm.wav', AUDIO_RATE, np.rint(audio * gain * 32767).astype('<i2'))
        return

    controls = [json.loads(line) for line in (root / 'control.jsonl').read_text().splitlines()]
    if not controls:
        raise ValueError('control.jsonl contains no timing records')
    sample = np.array([row['input_sample_offset'] for row in controls]) / live['input_sample_rate_hz']
    ts = np.array([row['model_start_utc_unix'] for row in controls])
    delay = live['fir_delay_input_samples'] / live['input_sample_rate_hz']
    metadata = json.loads((recording / 'recording.json').read_text())
    origin = datetime.fromisoformat(metadata['actual_start_utc']).timestamp()
    windows = json.loads(windows.read_text())
    if not isinstance(windows, list) or not windows:
        raise ValueError('--windows must contain a nonempty JSON array')

    clips = []
    example = None
    explicit_example = False
    for row in windows:
        clip_start, clip_end, strong_start, strong_end = (
            datetime.fromisoformat(row[key]) for key in ('start', 'end', 'strong_start', 'strong_end'))
        if any(bound.utcoffset() is None for bound in (clip_start, clip_end, strong_start, strong_end)):
            raise ValueError('Window timestamps require a time zone')
        start = clip_start.timestamp()
        length, j, k = ((bound - clip_start).total_seconds() for bound in (clip_end, strong_start, strong_end))
        strong_duration = strong_end - strong_start
        if not 0 <= j < k <= length or strong_duration < timedelta(milliseconds=50):
            raise ValueError('A window must contain its strong interval with at least 0.05 s for Welch analysis')
        example_start = strong_start
        duration = min(1., strong_duration.total_seconds())
        has_example = 'example_start' in row
        if has_example:
            if explicit_example:
                raise ValueError('Only one window may specify example_start')
            example_start = datetime.fromisoformat(row['example_start'])
            if example_start.utcoffset() is None:
                raise ValueError('example_start requires a time zone')
            duration = 1.
            if not clip_start <= example_start <= clip_end - timedelta(seconds=1):
                raise ValueError('The one-second example must fit inside its window')
            explicit_example = True
        first = round((np.interp(start, ts, sample) + max(0., start - ts[-1]) + delay) * fs)
        count = round(length * fs)
        pad = fs // 2
        if start < ts[0] or first < pad or first + count + pad > len(iq):
            raise ValueError('Window must fit in the branch with 0.5 seconds of IQ padding on both ends')
        channel, raw, deemphasis, audio = demod(iq[first - pad:first + count + pad], fs, offset_hz)
        channel, raw, deemphasis = (values[RATE // 2:RATE // 2 + round(length * RATE)]
                                     for values in (channel, raw, deemphasis))
        audio = audio[AUDIO_RATE // 2:AUDIO_RATE // 2 + round(length * AUDIO_RATE)]
        strong = slice(round(j * RATE), round(k * RATE))
        low = signal.resample_poly(raw, 1, 25)
        contrasts, freqs = tone_powers(low)
        valid = contrasts >= 12
        ii, jj = np.where(valid[1:] & valid[:-1] & (abs(freqs[1:] - freqs[:-1]) <= .5))
        active = np.flatnonzero((np.arange(len(contrasts)) * .5 >= j) & (np.arange(len(contrasts)) * .5 + 1 <= k))
        top = None
        if len(active):
            frame, tone = np.unravel_index(contrasts[active].argmax(), contrasts[active].shape)
            frame = active[frame]
            top = {'nominal_hz': CTCSS[tone], 'peak_hz': freqs[frame, tone], 'contrast_db': contrasts[frame, tone]}
        report = {
            'window': {'start_seconds': start - origin, 'end_seconds': start - origin + length,
                       'strong_start_seconds': start - origin + j, 'strong_end_seconds': start - origin + k},
            'rf': rf_metrics(channel[strong]),
            'voice_delta_db': float(voice_power(raw[strong]) - voice_power(raw[:2 * RATE])),
            'max_67_hz_contrast_db': float(contrasts[:, 1].max()) if len(contrasts) else None,
            'strong_top_tone': top,
            'tone_matches': [{'start_seconds': int(i) * .5, 'nominal_hz': CTCSS[l],
                              'peak_hz': freqs[i, l], 'contrasts_db': contrasts[i:i + 2, l].tolist()}
                             for i, l in zip(ii, jj)],
        }
        contrast = report['rf']['peak_excess_db']
        if has_example or (not explicit_example and (example is None or contrast > example[0])):
            position = (example_start - clip_start).total_seconds()
            section = slice(round(position * RATE), round((position + duration) * RATE))
            example = (contrast, example_start.timestamp(), duration,
                       iq[first + round(position * fs):first + round((position + duration) * fs)],
                       channel[section], raw[section], deemphasis[section],
                       audio[round(position * AUDIO_RATE):round((position + duration) * AUDIO_RATE)])
        clips.append((start, length, j, k, audio, low, report))

    peak = max(np.max(abs(clip[4])) for clip in clips)
    gain = .9 / max(peak, .9 * 2500)
    voice_specs, sub_specs = [], []
    for index, (start, length, j, k, audio, low, report) in enumerate(clips, 1):
        name = f'nfm-{index:02d}.wav'
        wavfile.write(output / name, AUDIO_RATE, np.rint(audio * gain * 32767).astype('<i2'))
        report.update(wav=name, samples=len(audio))
        for bank, values, rate, nfft, hop, band in (
                (voice_specs, audio * gain, AUDIO_RATE, 512, 64, (300, 3000)),
                (sub_specs, low, 1000, 1000, 125, (40, 260))):
            nfft = min(nfft, len(values))
            f, t, p = signal.spectrogram(values, fs=rate, window='hann', nperseg=nfft,
                                         noverlap=max(0, nfft - hop), mode='psd')
            selected = (f >= band[0]) & (f <= band[1])
            bank.append((f[selected], t, 10 * np.log10(np.maximum(p[selected], 1e-20))))

    fig, axs = plt.subplots(len(clips), 2, figsize=(13, max(3.5, 2 * len(clips))),
                            squeeze=False, layout='constrained')
    for col, bank in enumerate((voice_specs, sub_specs)):
        vmin, vmax = np.percentile(np.concatenate([p.ravel() for _, _, p in bank]), [10, 99])
        for ax, (start, length, j, k, *_), (f, t, p) in zip(axs[:, col], clips, bank):
            elapsed = start - origin
            image = ax.pcolormesh(elapsed + t, f, p, shading='nearest', cmap='magma', vmin=vmin, vmax=vmax, rasterized=True)
            ax.axvspan(elapsed + j, elapsed + k, fill=False, edgecolor='#64e5e6', lw=.85)
            ax.set(xlim=(elapsed, elapsed + length), ylabel='Hz')
            if col == 1:
                ax.axhline(67, lw=.6, color='#c5def0', ls='--', alpha=.8)
            else:
                ax.text(.01, .96, f'{start - origin:.1f} s after recording start',
                        transform=ax.transAxes, va='top', color='white')
        axs[-1, col].set_xlabel('Seconds since recording start')
        axs[0, col].set_title(('Voice-band WAV, 300–3000 Hz', 'Raw FM discriminator, 40–260 Hz')[col])
        fig.colorbar(image, ax=axs[:, col], label=('Audio PSD (dBFS/Hz)', 'Deviation PSD (dB re 1 Hz²/Hz)')[col])
    fig.suptitle('NFM clips and subaudible tone check')
    fig.savefig(output / '05-nfm-clips-spectrogram.png', dpi=150)
    plt.close(fig)

    contrast, start, duration, pre, post, raw, deemphasis, audio = example
    fig, axs = plt.subplots(3, 1, figsize=(11, 8.5), layout='constrained')
    for values, rate, nfft, offset, label in (
            (pre, fs, 16384, offset_hz, f'{branch.capitalize()} IQ'),
            (post, RATE, 4096, 0, 'After retune and channel filter')):
        f, p = signal.welch(values, fs=rate, nperseg=min(nfft, len(values)), detrend='constant')
        axs[0].plot((np.fft.fftshift(f) - offset) / 1000, db(np.fft.fftshift(p)), lw=.8, label=label)
    axs[0].axvspan(-5.5, 5.5, color='#ccdef0', alpha=.3)
    axs[0].set(xlim=(-12, 12), xlabel=f'Offset after {offset_hz:g} Hz retune (kHz)', ylabel='RF PSD (dBFS/Hz)')
    for values, rate, nfft, label in (
            (raw, RATE, 4096, 'FM discriminator'), (deemphasis, RATE, 4096, 'After de-emphasis'),
            (audio, AUDIO_RATE, 2048, 'After voice bandpass')):
        f, p = signal.welch(values, fs=rate, nperseg=min(nfft, len(values)), detrend='constant')
        axs[1].plot(f / 1000, db(p), lw=.8, label=label)
    axs[1].axvspan(.3, 3, color='#ccdef0', alpha=.3)
    axs[1].set(xlim=(0, 5), xlabel='Audio frequency (kHz)', ylabel='PSD (dB re 1 Hz²/Hz)')
    elapsed = start - origin
    axs[2].plot(elapsed + np.arange(len(raw)) / RATE, raw, lw=.7, label='FM discriminator')
    axs[2].plot(elapsed + np.arange(len(audio)) / AUDIO_RATE, audio, lw=.9, label='Output audio, before PCM scaling')
    axs[2].axhline(2500, lw=.8, ls=':', color='gray')
    axs[2].axhline(-2500, lw=.8, ls=':', color='gray')
    trace_start = elapsed + min(.25, duration - .04)
    axs[2].set(xlim=(trace_start, trace_start + .04), xlabel='Seconds since recording start', ylabel='Deviation (Hz)')
    for ax in axs:
        ax.legend()
    fig.suptitle(f'NFM demodulation stages at {start - origin:.1f} s after recording start')
    fig.savefig(output / '06-nfm-processing-stages.png', dpi=155)
    plt.close(fig)

    report = {'branch': branch, 'offset_hz': offset_hz, 'pcm_gain_per_hz': float(gain),
              'unscaled_peak_hz': float(peak), 'example_elapsed_seconds': start - origin,
              'example_duration_seconds': duration, 'example_rf_peak_excess_db': contrast,
              'clips': [clip[-1] for clip in clips]}
    (output / 'nfm-report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--offset-hz', required=True, type=float)
    parser.add_argument('--windows', type=Path)
    parser.add_argument('--branch', choices=('corrected', 'uncorrected'), default='corrected')
    args = parser.parse_args()
    render(args.recording, args.output, args.offset_hz, args.windows, args.branch)
