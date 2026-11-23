#!/usr/bin/env python3
"""Analyze DJIBOUTI-1B GMSK/AX.25 from the original CS16 recording."""

import argparse
from binascii import crc_hqx
from csv import writer
from datetime import datetime
import json
from pathlib import Path
import re
import tomllib

import numpy as np
from scipy import fft, signal

from live_doppler import DopplerCurve, PairedChannelizer
from predict_pass import Satrec, geometry
from record_meteor import write_metadata


CARRIER_HZ = 401_800_000
SYMBOL_RATE = 2400
CHANNEL_RATE = 100_000
NFFT = 32768
FFTS_PER_SLICE = 16
BIT_REVERSE = bytes(int(f'{value:08b}'[::-1], 2) for value in range(256))


def db(power):
    return 10 * np.log10(np.maximum(power, 1e-30))


def raw_spectra(raw, rate, center_hz, full_scale):
    offsets = fft.fftfreq(NFFT, 1 / rate) + center_hz - CARRIER_HZ
    band = np.flatnonzero(abs(offsets) <= 42_000)
    if len(band) < 2:
        raise ValueError('The recording does not cover the 401.800 MHz analysis band')
    band = band[np.argsort(offsets[band])]
    window = np.hanning(NFFT).astype(np.float32)
    scale = rate * np.sum(window**2) * full_scale**2
    stride = NFFT * FFTS_PER_SLICE
    spectra, seconds = [], []
    for start in range(0, len(raw), stride * 8):
        count = min(stride * 8, len(raw) - start) // NFFT
        if not count:
            break
        iq = raw[start:start + count * NFFT].astype(np.float32).view(np.complex64).reshape(count, NFFT)
        bins = fft.fft(iq * window, axis=1, workers=4)[:, band]
        psd = (bins.real**2 + bins.imag**2) / scale
        for index in range(0, count, FFTS_PER_SLICE):
            end = min(index + FFTS_PER_SLICE, count)
            spectra.append(psd[index:end].mean(axis=0))
            seconds.append((start + (index + end) * NFFT / 2) / rate)
    return np.array(seconds), offsets[band], np.stack(spectra)


def ax25_frames(levels):
    nrzi = np.r_[True, levels[1:] == levels[:-1]]
    bits = np.packbits(nrzi)
    stream = ''.join(f'{byte:08b}' for byte in bits)[:len(nrzi)]
    flags = [flag.start() for flag in re.finditer('(?=01111110)', stream)]
    for left, right in zip(flags, flags[1:]):
        stuffed = stream[left + 8:right]
        if not 144 <= len(stuffed) <= 3200 or '111111' in stuffed:
            continue
        data = stuffed.replace('111110', '11111')
        if len(data) % 8:
            continue
        frame = np.packbits(np.frombuffer(data.encode(), dtype=np.uint8) - 48, bitorder='little').tobytes()
        crc = crc_hqx(frame[:-2].translate(BIT_REVERSE), 0xffff)
        expected = int.from_bytes(crc.to_bytes(2, 'big').translate(BIT_REVERSE), 'little') ^ 0xffff
        if len(frame) >= 18 and expected == int.from_bytes(frame[-2:], 'little'):
            yield left, frame


def decode_burst(raw, rate, metadata, curve, start_utc, start_s, end_s, peak_s):
    decimation = rate // CHANNEL_RATE
    start = int(max(0, start_s - .7) * CHANNEL_RATE) * decimation
    end = min(len(raw), int((end_s + .8) * CHANNEL_RATE) * decimation)
    channelizer = PairedChannelizer(rate, CHANNEL_RATE, CARRIER_HZ - metadata['center_frequency_hz'],
                                   curve, cutoff_hz=12_000)
    chunks = []
    for first in range(start, end, rate // 2):
        block = raw[first:min(first + rate // 2, end)].astype(np.float32).view(np.complex64).ravel()
        _, corrected = channelizer.process(block / metadata['adc_full_scale'],
                                           start_utc - curve.origin + first / rate)
        chunks.append(corrected)
    samples_per_symbol = 8
    rate_out = SYMBOL_RATE * samples_per_symbol
    iq = signal.resample_poly(np.concatenate(chunks), rate_out, CHANNEL_RATE)
    iq = signal.sosfiltfilt(signal.butter(5, 1200, fs=rate_out, output='sos'), iq)
    fm = np.angle(iq[1:] * iq[:-1].conj()) * rate_out / (2 * np.pi)
    peak = int((peak_s - start / rate) * rate_out)
    half_window = int(.30 * rate_out)
    window = fm[max(0, peak - half_window):peak + half_window]
    estimate = float(np.mean(np.percentile(window, [10, 90])))
    frames = {}
    for phase in np.arange(0, samples_per_symbol, .25):
        indices = np.arange(phase, len(fm) - 1, samples_per_symbol)
        samples = np.interp(indices, np.arange(len(fm)), fm)
        for threshold in (0, estimate, estimate - 160, estimate + 160):
            for first, frame in ax25_frames(samples > threshold):
                frames.setdefault(frame, start / rate + indices[first] / rate_out)
    return [{'elapsed_seconds': elapsed, 'bytes': len(frame), 'frame_hex': frame.hex()}
            for frame, elapsed in frames.items()]


def render(output, times, rf, raw, residual, corrected, doppler, floor, contrast,
           detected, burst_times, burst_centers):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout='constrained')
    for ax, frequency, psd, limit, label in zip(
            axes, (rf, residual), (raw, corrected), (14_000, 3800),
            ('RF offset from 401.800 MHz (kHz)', 'Doppler residual (kHz)')):
        band = abs(frequency) <= limit
        im = ax.pcolormesh(times, frequency[band] / 1000, db(psd[:, band] / floor[:, None]).T,
                           vmin=-2.5, vmax=14, cmap='viridis', shading='nearest', rasterized=True)
        ax.set_ylabel(label)
    axes[0].plot(times, doppler / 1000, '--', color='white', alpha=.65, linewidth=.75)
    axes[0].set_title('DJIBOUTI-1B 401.800 MHz / Raw IQ and predicted Doppler')
    axes[1].axhline(0, color='white', alpha=.45, linestyle='--', linewidth=.75)
    axes[1].scatter(burst_times, np.asarray(burst_centers) / 1000,
                    s=16, facecolors='none', edgecolors='#ffcc77', linewidths=1.05)
    axes[1].set_xlabel('Seconds since recording start')
    fig.colorbar(im, ax=axes, label='PSD above local floor (dB)')
    fig.savefig(output / '01-pass-waterfall.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True, layout='constrained')
    axes[0].set_title('DJIBOUTI-1B / 2.4 kHz channel around predicted Doppler')
    axes[0].plot(times, db(floor), color='#416b50', linewidth=.7, alpha=.7)
    axes[0].set_ylabel('Noise (dBFS/Hz)')
    axes[1].plot(times, db(contrast), color='#8296a3', linewidth=.6, label='Channel / noise')
    axes[1].scatter(times[detected], db(contrast[detected] - 1), s=6, c='#a63e21', label='Burst SNR')
    axes[1].set(ylabel='Ratio (dB)', xlabel='Seconds since recording start')
    axes[1].legend(loc='upper right')
    fig.savefig(output / '02-snr.png', dpi=150)
    plt.close(fig)


def analyze(folder, prediction, tle, config, output):
    metadata = json.loads((folder / 'recording.json').read_text())
    rate = metadata['sample_rate_hz']
    count = metadata['sample_count']
    raw_path = folder / 'iq.cs16'
    if (metadata['sample_format'] != 'cs16_le' or type(rate) is not int or rate < CHANNEL_RATE
            or rate % CHANNEL_RATE or type(count) is not int or count < NFFT
            or raw_path.stat().st_size != count * 4 or metadata['adc_full_scale'] <= 0):
        raise ValueError('Expected a complete CS16 sample array and an input rate divisible by 100 kS/s')
    start = datetime.fromisoformat(metadata['actual_start_utc'])
    if start.utcoffset() is None:
        raise ValueError('Recording timestamp requires a time zone')
    origin = start.timestamp()
    curve = DopplerCurve.from_file(prediction, CARRIER_HZ)
    if not 0 <= origin - curve.origin <= origin - curve.origin + count / rate <= curve.times[-1]:
        raise ValueError('Prediction does not cover the complete recording')

    lines = tle.read_text().splitlines()
    satellite = Satrec.twoline2rv(*lines[-2:])
    if satellite.satnum != 62402:
        raise ValueError('Expected the DJIBOUTI-1B TLE')
    observer = tomllib.loads(config.read_text())['observer']
    site = [float(observer[k]) for k in ('latitude_deg', 'longitude_deg', 'altitude_m')]
    _, _, radial = geometry(satellite, curve.origin + curve.times, *site)
    if not np.allclose(-radial * CARRIER_HZ / 299_792.458, curve.hz, atol=.1, rtol=0):
        raise ValueError('Prediction does not match the supplied TLE and observer configuration')
    raw = np.memmap(raw_path, dtype='<i2', mode='r', shape=(count, 2))
    seconds, rf, original = raw_spectra(raw, rate, metadata['center_frequency_hz'], metadata['adc_full_scale'])
    doppler = np.interp(origin - curve.origin + seconds, curve.times, curve.hz)
    _, elevation, _ = geometry(satellite, origin + seconds, *site)
    resolution = rate / NFFT
    residual = np.arange(-30_000, 30_000 + resolution / 2, resolution)
    if rf[0] > residual[0] + min(doppler) or rf[-1] < residual[-1] + max(doppler):
        raise ValueError('The selected spectrum does not cover the Doppler analysis band')
    corrected = np.vstack([np.interp(residual + shift, rf, row)
                           for shift, row in zip(doppler, original)]).astype(np.float32)
    distance = abs(residual)
    floor = np.median(corrected[:, (distance >= 15_000) & (distance <= 25_000)], axis=1)
    if np.any(floor <= 0):
        raise ValueError('Noise estimate is zero in at least one PSD slice')
    channel = distance <= 1200
    contrast = corrected[:, channel].mean(axis=1) / floor
    detected = db(contrast) >= 3
    edges = np.flatnonzero(np.diff(np.r_[False, detected, False]))
    bursts, peaks, centers = [], [], []
    labels = np.zeros(len(seconds), dtype=int)
    for first, end in zip(edges[::2], edges[1::2]):
        peak = first + int(np.argmax(contrast[first:end]))
        near = abs(rf - doppler[peak]) < 1200
        weights = np.maximum(original[peak, near] - floor[peak], 0)
        offset = float(np.average(rf[near], weights=weights)) - float(doppler[peak])
        peaks.append(peak)
        centers.append(offset)
        labels[first:end] = len(bursts) + 1
        frames = decode_burst(raw, rate, metadata, curve, origin, seconds[first], seconds[end - 1], seconds[peak])
        bursts.append({'elapsed_seconds': seconds[peak], 'peak_snr_db': float(db(contrast[peak] - 1)),
                       'doppler_residual_hz': offset, 'frames': frames})

    output.mkdir(parents=True)
    visible = db(floor[elevation >= 10])
    with (output / 'snr-history.csv').open('w', newline='') as stream:
        rows = writer(stream)
        rows.writerow(('elapsed_seconds', 'predicted_doppler_hz',
                       'noise_dbfs_hz', 'channel_to_noise_db', 'burst', 'burst_snr_db'))
        for i, second in enumerate(seconds):
            rows.writerow((f'{second:.6f}',
                           f'{doppler[i]:.2f}', f'{db(floor[i]):.3f}', f'{db(contrast[i]):.3f}',
                           labels[i] or '', f'{db(contrast[i] - 1):.3f}' if detected[i] else ''))
    summary = {'sample_count': count, 'fft_slices': len(seconds), 'fft_bin_spacing_hz': resolution,
               'time_slice_seconds': NFFT * FFTS_PER_SLICE / rate, 'signal_bandwidth_hz': channel.sum().item() * resolution,
               'visible_noise_dbfs_hz': np.quantile(visible, [.1, .5, .9]).tolist() if visible.size else None,
               'bursts': bursts}
    write_metadata(output / 'analysis.json', summary)
    render(output, seconds, rf, original, residual, corrected, doppler, floor, contrast,
           detected, seconds[peaks], centers)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    parser.add_argument('--prediction', required=True, type=Path)
    parser.add_argument('--tle', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.recording, args.prediction, args.tle, args.config, args.output)
    print(f"{len(result['bursts'])} bursts, {sum(len(b['frames']) for b in result['bursts'])} valid AX.25 frames")
