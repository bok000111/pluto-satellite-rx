#!/usr/bin/env python3
"""Inspect an RS-44 recording and extract Doppler-corrected CW audio."""

import argparse
from datetime import datetime
import json
from math import gcd
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

BEACON = 435_605_000


def search_track(psd, frequency, expected):
    if not expected.size:
        raise ValueError("Predicted track has no snapshots")
    offsets = np.arange(-5000, 5001, 50)
    if frequency.size:
        lower = max(-5000, np.ceil(frequency[0] + 400 - expected.min()))
        upper = min(5000, np.floor(frequency[-1] - 400 - expected.max()))
        offsets = offsets[(offsets >= lower) & (offsets <= upper)]
    if not frequency.size or not offsets.size:
        raise ValueError("Predicted track does not fit inside the recorded beacon band")
    baseline = np.maximum(np.median(psd, axis=1), 1e-30)
    relative = psd / baseline[:, None]
    scores = []
    for offset in offsets:
        first = np.searchsorted(frequency, expected + offset - 25)
        last = np.searchsorted(frequency, expected + offset + 25, side="right")
        values = [row[a:b].max(initial=0.) for row, a, b in zip(relative, first, last)]
        scores.append(float(np.mean(np.log1p(values))))
    best = int(np.argmax(scores))
    offset = int(offsets[best])
    peak_indices = []
    for row, target in zip(psd, expected + offset):
        indices = np.flatnonzero(abs(frequency - target) < 400)
        peak_indices.append(int(indices[np.argmax(row[indices])]))
    peak_indices = np.array(peak_indices)
    excess = 10 * np.log10(np.maximum(relative[np.arange(len(psd)), peak_indices], 1e-30))
    observed = frequency[peak_indices]
    usable = excess >= 10
    if usable.any():
        offset = int(np.clip(round(float(np.median(observed[usable] - expected[usable]))), lower, upper))
    expected = expected + offset
    residual = observed - expected
    enough = int(np.count_nonzero(usable)) >= 20
    variation = float(np.ptp(expected[usable])) if np.any(usable) else 0.
    moving_rms = float(np.sqrt(np.mean(residual[usable]**2))) if enough else None
    fixed_rms = float(np.std(observed[usable])) if enough else None
    consistent = bool(enough and variation > 1000 and moving_rms < 150 and fixed_rms > 3 * moving_rms)
    return {
        "estimated_frequency_offset_hz": offset,
        "doppler_consistent_candidate": consistent,
        "above_10db_snapshot_count": int(np.count_nonzero(usable)),
        "predicted_doppler_span_hz": variation,
        "moving_track_residual_rms_hz": moving_rms,
        "fixed_frequency_residual_rms_hz": fixed_rms,
        "note": "Heuristic track search; relative power is not calibrated SNR. CW identity requires separate verification.",
    }, observed, excess, expected


def extract_audio(raw, sample_rate, center, start_sample, stop_sample, start_utc,
                  prediction, frequency_offset):
    chunks = []
    phase = 0.
    step = sample_rate // gcd(sample_rate, 6000)
    # Align padding to the output grid and cover the default FIR half length.
    margin = ((max(sample_rate, 6000) + 599) // 600 + step - 1) // step * step
    for begin in range(start_sample, stop_sample, sample_rate):
        end = min(begin + sample_rate, stop_sample)
        a, b = max(start_sample, begin - margin), min(stop_sample, end + margin)
        packed = raw[a:b]
        iq = packed[:, 0].astype(np.float64) + 1j * packed[:, 1]
        t = start_utc + np.arange(a, b) / sample_rate
        doppler = np.interp(t, prediction["time_utc_unix"], prediction["doppler_hz"])
        increments = 2 * np.pi * (BEACON - center + doppler + frequency_offset) / sample_rate
        angles = np.r_[0., np.cumsum(increments)]
        angles += phase - angles[begin - a]
        phase = float(angles[end - a] % (2 * np.pi))
        resampled = resample_poly(iq * np.exp(-1j * angles[:-1]), 6000, sample_rate)
        first = (begin - a) * 6000 // sample_rate
        count = ((end - begin) * 6000 + sample_rate - 1) // sample_rate
        chunks.append(resampled[first:first + count])
    baseband = np.concatenate(chunks)
    t = np.arange(baseband.size) / 6000
    audio = np.real(baseband * np.exp(2j * np.pi * 650 * t))
    scale = max(float(np.max(abs(audio))), 1.)
    audio = np.round(np.clip(audio / scale * .9, -1, 1) * 32767).astype(np.int16)
    envelope = abs(baseband[:baseband.size // 60 * 60].reshape(-1, 60)).mean(axis=1)
    return audio, envelope


def analyze(folder, prediction_file):
    metadata = json.loads((folder / "recording.json").read_text())
    prediction = json.loads(prediction_file.read_text())
    if prediction["reference_frequency_hz"] != BEACON:
        raise ValueError("Prediction is not for the RS-44 beacon")
    sample_rate = metadata["sample_rate_hz"]
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("Sample rate must be a positive integer in Hz")
    center = int(metadata["center_frequency_hz"])
    start = datetime.fromisoformat(metadata["actual_start_utc"])
    if start.utcoffset() is None:
        raise ValueError("Recording start time must include a UTC offset")
    start_utc = start.timestamp()
    knots = np.asarray(prediction["time_utc_unix"], dtype=float)
    doppler = np.asarray(prediction["doppler_hz"], dtype=float)
    if (knots.ndim != 1 or knots.size < 2 or knots.shape != doppler.shape
            or not np.isfinite(knots).all() or not np.isfinite(doppler).all()
            or not np.all(np.diff(knots) > 0)):
        raise ValueError("Prediction needs paired, finite samples in strictly increasing UTC order")
    raw = np.memmap(folder / "iq.cs16", dtype="<i2", mode="r").reshape(-1, 2)
    if raw.shape[0] != metadata["sample_count"]:
        raise ValueError("IQ size differs from recording metadata")
    nfft, averages = 131072, 4
    frame = nfft * averages
    if len(raw) < frame:
        raise ValueError("Insufficient IQ for CW analysis")
    if knots[0] > start_utc or knots[-1] < start_utc + (len(raw) - 1) / sample_rate:
        raise ValueError("Recording times are outside the prediction window")
    times, spectra = [], []
    frequency = center + np.fft.fftshift(np.fft.fftfreq(nfft, 1 / sample_rate))
    selected = abs(frequency - BEACON) < 18000
    window = np.hanning(nfft)
    for offset in range(0, len(raw) - frame + 1, sample_rate):
        packed = raw[offset:offset + frame]
        iq = (packed[:, 0].astype(float) + 1j * packed[:, 1]).reshape(averages, nfft)
        fft = np.fft.fftshift(np.fft.fft(iq * window, axis=1), axes=1)
        power = np.mean(abs(fft)**2, axis=0) / (sample_rate * np.sum(window**2) * 2048**2)
        spectra.append(power[selected])
        times.append(start_utc + (offset + frame / 2) / sample_rate)
    frequency = frequency[selected]
    times, psd = np.array(times), np.array(spectra)
    expected = BEACON + np.interp(times, knots, doppler)
    result, observed, excess, fitted = search_track(psd, frequency, expected)
    smooth = np.convolve(excess, np.ones(min(60, len(excess))) / min(60, len(excess)), mode="same")
    audio_center = times[int(np.argmax(smooth))] - start_utc
    audio_start = max(0, min(audio_center - 30, len(raw) / sample_rate - 60))
    begin = int(audio_start * sample_rate)
    end = min(len(raw), begin + 60 * sample_rate)
    audio, envelope = extract_audio(raw, sample_rate, center, begin, end, start_utc,
                                   prediction, result["estimated_frequency_offset_hz"])
    wavfile.write(folder / "cw-candidate.wav", 6000, audio)
    result.update(rs44_identity_confirmed=False, cw_identifier="unverified",
                  audio_start_seconds=audio_start, audio_duration_seconds=len(audio) / 6000,
                  fft_resolution_hz=sample_rate / nfft,
                  timing_note="Offline sample times assume uniform delivery from actual_start_utc; inspect recorded-IQ vs wall duration.")
    np.savez_compressed(folder / "cw-spectrum.npz", frequency_hz=frequency,
                        time_utc_unix=times, psd_dbfs_hz=10*np.log10(np.maximum(psd, 1e-30)),
                        fitted_frequency_hz=fitted, peak_frequency_hz=observed,
                        relative_peak_power_db=excess, audio_envelope=envelope)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (waterfall, pulses) = plt.subplots(2, 1, figsize=(11, 8), layout="constrained")
    db = 10 * np.log10(np.maximum(psd, 1e-30))
    elapsed = times - start_utc
    image = waterfall.imshow(db, origin="lower", aspect="auto",
        extent=((frequency[0]-BEACON)/1000, (frequency[-1]-BEACON)/1000, elapsed[0] - .5, elapsed[-1] + .5),
        vmin=float(np.percentile(db, 15)), vmax=float(np.percentile(db, 99)))
    waterfall.plot((fitted-BEACON)/1000, elapsed, color="cyan", linewidth=.8, label="Predicted Doppler + fitted offset")
    waterfall.set(xlabel="Offset from 435.605 MHz (kHz)", ylabel="Seconds since recording start", title="RS-44 CW candidate; satellite identity unverified")
    waterfall.legend(loc="upper right", fontsize=8)
    fig.colorbar(image, ax=waterfall, label="PSD (dBFS/Hz; uncalibrated)")
    pulses.plot(audio_start + np.arange(len(envelope)) / 100, envelope, linewidth=.6)
    pulses.set(xlabel="Seconds since recording start", ylabel="Envelope (ADC counts)", title="CW audio envelope")
    fig.savefig(folder / "cw-analysis.png", dpi=160)
    plt.close(fig)
    (folder / "cw-analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def render_saved(folder, prediction_file, output, wide_spectrum, cw_audio, audio_start):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram

    metadata = json.loads((folder / 'recording.json').read_text())
    origin = datetime.fromisoformat(metadata['actual_start_utc']).timestamp()
    prediction = json.loads(prediction_file.read_text())
    knots = prediction['time_utc_unix']
    doppler = prediction['doppler_hz']
    with np.load(folder / 'cw-spectrum.npz') as recorded:
        frequency = recorded['frequency_hz']
        times = recorded['time_utc_unix']
        psd = recorded['psd_dbfs_hz']
    elapsed = times - origin
    expected = BEACON + np.interp(times, knots, doppler)
    result, peak, excess, fit = search_track(10 ** (psd / 10), frequency, expected)
    good = excess >= 10

    output.mkdir(parents=True)

    def save(fig, name):
        fig.savefig(output / name, dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5), layout='constrained')
    image = ax.pcolormesh(elapsed, (frequency - BEACON) / 1000, psd.T, shading='nearest',
                         cmap='magma', vmin=np.percentile(psd, 15), vmax=np.percentile(psd, 99.5))
    ax.plot(elapsed, (expected - BEACON) / 1000, ':', color='white', lw=.8, label='Predicted Doppler')
    ax.plot(elapsed, (fit - BEACON) / 1000, color='cyan', lw=.8, label='Fitted Doppler')
    ax.set(xlabel='Seconds since recording start', ylabel='Offset from 435.605 MHz (kHz)')
    ax.legend()
    fig.colorbar(image, ax=ax, label='PSD (dBFS/Hz)')
    save(fig, '01-pass-waterfall.png')

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 5.5), layout='constrained')
    axes[0].plot(elapsed, (fit - BEACON) / 1000, lw=.8)
    axes[0].scatter(elapsed[good], (peak[good] - BEACON) / 1000, s=10, color='#a05228')
    axes[0].set_ylabel('Offset from 435.605 MHz (kHz)')
    axes[1].scatter(elapsed[good], peak[good] - fit[good], s=12, color='#a05228')
    axes[1].axhline(0, color='.4', lw=.8)
    axes[1].set(xlabel='Seconds since recording start', ylabel='Residual (Hz)')
    save(fig, '02-doppler-fit.png')

    fs, audio = wavfile.read(cw_audio)
    hz, second, power = spectrogram(audio.astype(float) / 32768, fs=fs, window='hann',
                                    nperseg=1024, noverlap=896)
    band = (hz >= 550) & (hz <= 750)
    cw_band = (hz >= 600) & (hz <= 700)
    db = 10 * np.log10(np.maximum(power[band], 1e-20))
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 5.5), layout='constrained')
    image = axes[0].pcolormesh(second + audio_start, hz[band], db, shading='nearest', cmap='magma',
                              vmin=np.percentile(db, 15), vmax=np.percentile(db, 99))
    axes[0].set_ylabel('Audio frequency (Hz)')
    fig.colorbar(image, ax=axes[0], label='PSD (dBFS/Hz)')
    envelope = np.sqrt(np.sum(power[cw_band], axis=0) * (hz[1] - hz[0]))
    axes[1].plot(second + audio_start, envelope, lw=.65)
    axes[1].set(xlabel='Seconds since recording start', ylabel='600–700 Hz envelope')
    save(fig, '03-cw-morse.png')

    offsets = np.arange(-400, 401, 12.5)
    detail = np.vstack([np.interp(center + offsets, frequency, row) for center, row in zip(fit, psd)])
    fig, ax = plt.subplots(figsize=(12, 4.5), layout='constrained')
    image = ax.pcolormesh(elapsed, offsets, detail.T, shading='nearest', cmap='magma',
                         vmin=np.percentile(detail, 15), vmax=np.percentile(detail, 99))
    ax.scatter(elapsed[good], peak[good] - fit[good], s=8, color='cyan')
    ax.set(xlabel='Seconds since recording start', ylabel='Offset from fitted Doppler (Hz)')
    fig.colorbar(image, ax=ax, label='PSD (dBFS/Hz)')
    save(fig, '04-beacon-detail.png')

    with np.load(wide_spectrum) as spectrum:
        wide_t, wide_hz, wide_psd = spectrum['time'], spectrum['freq'], spectrum['psd']
    fig, ax = plt.subplots(figsize=(12, 5.5), layout='constrained')
    image = ax.pcolormesh(wide_t - origin, wide_hz / 1e6, wide_psd.T, shading='nearest', cmap='magma',
                         vmin=np.percentile(wide_psd, 15), vmax=np.percentile(wide_psd, 99.7))
    correction = np.interp(wide_t, knots, doppler) + result['estimated_frequency_offset_hz']
    ax.plot(wide_t - origin, (BEACON + correction) / 1e6, color='cyan', lw=.8, label='CW beacon')
    ax.fill_between(wide_t - origin, (435610000 + correction) / 1e6, (435670000 + correction) / 1e6,
                    color='cyan', alpha=.2, label='Transponder downlink')
    ax.set(xlabel='Seconds since recording start', ylabel='Frequency (MHz)')
    ax.legend()
    fig.colorbar(image, ax=ax, label='PSD (dBFS/Hz)')
    save(fig, '05-transponder-waterfall.png')

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Render observation figures from saved spectra")
    parser.add_argument("--wide-spectrum", type=Path)
    parser.add_argument("--cw-audio", type=Path)
    parser.add_argument("--audio-start", type=float, help="CW clip start in seconds after recording start")
    args = parser.parse_args()
    if args.output is not None:
        if args.wide_spectrum is None or args.cw_audio is None or args.audio_start is None:
            parser.error("--output requires --wide-spectrum, --cw-audio and --audio-start")
        render_saved(args.folder, args.prediction, args.output, args.wide_spectrum, args.cw_audio, args.audio_start)
    else:
        analyze(args.folder, args.prediction)
