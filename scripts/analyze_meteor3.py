#!/usr/bin/env python3
"""Reproduce the METEOR-M2-3 LRPT images and SNR figures from a saved pass."""

import argparse
from csv import writer
import hashlib
import json
import re
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import tomllib

import numpy as np
from scipy import fft
from scipy.signal import fftconvolve

from channel_filter import channel_taps, filter_channel
from decode_meteor import frames
from live_doppler import ContinuousMixer, DopplerCurve
from predict_pass import Satrec, geometry
from record_meteor import write_metadata


CARRIER_HZ = 137_900_000
SNR_CENTER_HZ = -2600
SNR_BANDWIDTH_HZ = 130_000
DECODE_RATE = 250_000
DECODE_SECONDS = 563.835045
PIPELINE = 'meteor_m2-x_lrpt'


def db(power):
    return 10 * np.log10(np.maximum(power, 1e-30))


def spectra(source, rate):
    iq = np.memmap(source, dtype='<c8', mode='r')
    size = 4096
    per_slice = round(rate / size)
    window = np.hanning(size).astype(np.float32)
    scale = rate * np.sum(window**2)
    frequency = fft.fftshift(fft.fftfreq(size, 1 / rate))
    stride = size * per_slice
    rows, elapsed = [], []
    for start in range(0, len(iq), stride):
        count = min(per_slice, (len(iq) - start) // size)
        if not count:
            break
        bins = fft.fft(iq[start:start + count * size].reshape(count, size) * window, axis=1, workers=2)
        rows.append(fft.fftshift(np.mean(bins.real**2 + bins.imag**2, axis=0)) / scale)
        elapsed.append((start + count * size / 2) / rate)
    return frequency, np.stack(rows), np.array(elapsed)


def pass_geometry(folder, raw_rate, elapsed, prediction, tle, config):
    controls = [json.loads(row) for row in (folder / 'live-doppler/control.jsonl').read_text().splitlines()]
    offsets = np.array([row['input_sample_offset'] for row in controls]) / raw_rate
    model_start = np.array([row['model_start_utc_unix'] for row in controls])
    if (offsets.size < 2 or np.any(np.diff(offsets) <= 0) or np.any(np.diff(model_start) <= 0)
            or elapsed[0] < offsets[0] or elapsed[-1] > offsets[-1]):
        raise ValueError('Timing records do not cover the spectrum windows')
    utc = np.interp(elapsed, offsets, model_start)
    tle_lines = tle.read_text().splitlines()
    satellite = Satrec.twoline2rv(*tle_lines[-2:])
    if satellite.satnum != 57166:
        raise ValueError('Expected the METEOR-M2-3 TLE')
    observer = tomllib.loads(config.read_text())['observer']
    site = [float(observer[key]) for key in ('latitude_deg', 'longitude_deg', 'altitude_m')]
    curve = DopplerCurve.from_file(prediction, CARRIER_HZ)
    _, _, radial = geometry(satellite, curve.origin + curve.times, *site)
    if not np.allclose(curve.hz, -radial * CARRIER_HZ / 299_792.458, atol=.1, rtol=0):
        raise ValueError('Prediction does not match the TLE and observer configuration')
    _, elevation, _ = geometry(satellite, utc, *site)
    return elevation


def render(output, frequency, psd, elapsed, side_noise, inband_noise, snr):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4.5), layout='constrained')
    band = abs(frequency) <= 100_000
    image = ax.pcolormesh(elapsed, frequency[band] / 1000, db(psd[:, band] / side_noise[:, None]).T,
                         shading='nearest', cmap='viridis', vmin=-8, vmax=9, rasterized=True)
    for edge in (SNR_CENTER_HZ - SNR_BANDWIDTH_HZ / 2, SNR_CENTER_HZ + SNR_BANDWIDTH_HZ / 2):
        ax.axhline(edge / 1000, ls=':', color='white', alpha=.7, lw=.7)
    ax.set(title='METEOR-M2-3 / Doppler-corrected 137.900 MHz', xlabel='Seconds since recording start',
           ylabel='Frequency offset (kHz)')
    fig.colorbar(image, ax=ax, label='PSD relative to sideband floor (dB)')
    fig.savefig(output / '01-lrpt-waterfall.png', dpi=155)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 5.5), layout='constrained')
    axes[0].set_title('METEOR-M2-3 / 130 kHz LRPT SNR and noise')
    axes[0].plot(elapsed, snr, lw=.85, color='#a05228')
    if np.any(np.isfinite(snr)):
        strongest = int(np.nanargmax(snr))
        axes[0].scatter(elapsed[strongest], snr[strongest], s=23, c='#73290f')
    axes[0].set(ylabel='Estimated SNR (dB)', ylim=(-14, 6))
    axes[1].plot(elapsed, db(inband_noise), lw=.7, color='#456b53')
    axes[1].set(ylabel='Noise (dBFS/Hz)', xlabel='Seconds since recording start')
    fig.savefig(output / '02-snr.png', dpi=150)
    plt.close(fig)


def measure(folder, metadata, prediction, tle, config, output):
    source = folder / 'live-doppler/corrected.cf32'
    rate = metadata['live_doppler']['output_sample_rate_hz']
    freq, psd, elapsed = spectra(source, rate)
    elevation = pass_geometry(folder, metadata['sample_rate_hz'], elapsed, prediction, tle, config)

    distance = abs(freq)
    side_noise = np.median(psd[:, (distance >= 80_000) & (distance <= 95_000)], axis=1)
    channel = abs(freq - SNR_CENTER_HZ) <= SNR_BANDWIDTH_HZ / 2
    power = psd[:, channel].mean(axis=1)
    visible = elevation >= 10
    if not np.any(visible) or np.all(visible) or np.any(side_noise <= 0):
        raise ValueError('Need signal-band and sideband measurements above and below 10 degrees')
    background = float(np.median(power[~visible] / side_noise[~visible]))
    if not np.isfinite(background) or background <= 0:
        raise ValueError('Invalid off-pass noise estimate')
    noise = background * side_noise
    excess = power / noise - 1
    snr = np.where(excess > 0, db(excess), np.nan)
    peak = int(np.nanargmax(snr)) if np.any(np.isfinite(snr)) else None

    with (output / 'snr-history.csv').open('w', newline='') as stream:
        csv = writer(stream)
        csv.writerow(('elapsed_seconds', 'channel_to_noise_db', 'estimated_snr_db',
                      'equivalent_inband_noise_dbfs_hz', 'sideband_noise_dbfs_hz'))
        for i, second in enumerate(elapsed):
            csv.writerow((f'{second:.6f}',
                          f'{db(power[i] / noise[i]):.4f}', f'{snr[i]:.4f}' if excess[i] > 0 else '',
                          f'{db(noise[i]):.4f}', f'{db(side_noise[i]):.4f}'))

    def sample(index):
        return {'elapsed_seconds': float(elapsed[index]),
                'snr_db': float(snr[index]) if excess[index] > 0 else None}

    result = {'fft_slices': len(elapsed), 'center_offset_hz': SNR_CENTER_HZ,
              'bandwidth_hz': float(channel.sum() * (freq[1] - freq[0])),
              'noise_to_sideband_db': float(db(background)),
              'visible_noise_dbfs_hz': np.quantile(db(noise[visible]), [.1, .5, .9]).tolist(),
              'visible_sideband_noise_dbfs_hz': np.quantile(db(side_noise[visible]), [.1, .5, .9]).tolist(),
              'peak_snr': sample(peak) if peak is not None else None}
    write_metadata(output / 'snr-analysis.json', result)
    render(output, freq, psd, elapsed, side_noise, noise, snr)
    return result


def channelize(folder, metadata, prediction, destination):
    rate = metadata['sample_rate_hz']
    if rate % DECODE_RATE:
        raise ValueError('Original IQ sample rate must be divisible by 250 kS/s')
    decimation = rate // DECODE_RATE
    count = min(metadata['sample_count'], round(rate * DECODE_SECONDS))
    raw = np.memmap(folder / 'iq.cs16', mode='r', dtype='<i2', shape=(metadata['sample_count'], 2))
    controls = [json.loads(row) for row in (folder / 'live-doppler/control.jsonl').read_text().splitlines()]
    offsets = np.array([row['input_sample_offset'] for row in controls])
    times = np.array([row['model_start_utc_unix'] for row in controls])
    if not len(offsets) or offsets[0] != 0 or offsets[-1] < count:
        raise ValueError('Timing records do not cover the LRPT decode interval')
    curve = DopplerCurve.from_file(prediction, CARRIER_HZ)
    mixer = ContinuousMixer(rate, CARRIER_HZ - metadata['center_frequency_hz'] + SNR_CENTER_HZ, curve)
    taps = channel_taps(rate, 95_000, 122_000)
    history = np.zeros(len(taps) - 1, dtype='c8')
    block = 393216
    with destination.open('xb') as sink:
        for start in range(0, count, block):
            raw_iq = raw[start:min(start + block, count)].astype('f4').view('c8').ravel()
            mixed = mixer.process(raw_iq / metadata['adc_full_scale'],
                                  float(np.interp(start, offsets, times)) - curve.origin)
            joined = np.concatenate((history, mixed))
            filtered = fftconvolve(joined, taps, mode='valid')
            history = joined[-len(history):].copy()
            filtered[(-start) % decimation::decimation].astype('<c8').tofile(sink)


def satdump_json(path):
    text = re.sub(r'("(?:\\.|[^"\\])*")|/\*.*?\*/|//[^\n]*',
                  lambda match: match[1] or '', path.read_text(), flags=re.S)
    return json.loads(text)


def decode(folder, metadata, prediction, tle, output, binary, resources):
    bytes_per_branch = int(DECODE_SECONDS * DECODE_RATE + 1) * 8
    if shutil.disk_usage(output).free < 30 * 2**30 + 2 * bytes_per_branch + 2**28:
        raise RuntimeError('Insufficient free space for LRPT analysis and 30 GiB reserve')
    with TemporaryDirectory(prefix='lrpt-work-', dir=output) as work:
        work = Path(work)
        wide, narrow = work / 'wide.cf32', work / '110k.cf32'
        channelize(folder, metadata, prediction, wide)
        filter_channel(wide, narrow, DECODE_RATE, 55_000, 69_000)
        wide.unlink()

        pipelines = work / 'pipelines'
        pipelines.mkdir()
        pipeline = satdump_json(resources / 'pipelines/Meteor-M.json')[PIPELINE]
        pipeline['name'] = 'METEOR-M2-3 LRPT CADU'
        del pipeline['work']['products']
        write_metadata(pipelines / 'decode.json', {'lrpt72_cadu': pipeline})
        (work / 'plugins').symlink_to(resources / 'plugins', target_is_directory=True)
        (work / 'resources').symlink_to(resources / 'resources', target_is_directory=True)
        settings = satdump_json(resources / 'satdump_cfg.json')
        settings['satdump_general']['tle_update_interval']['value'] = 'Never'
        write_metadata(work / 'satdump_cfg.json', settings)
        with (output / 'decoder.log').open('w') as log:
            subprocess.run([binary, 'lrpt72_cadu', 'baseband', narrow, work / 'decoded',
                            '--samplerate', str(DECODE_RATE), '--baseband_format', 'cf32',
                            '--iq_swap', 'true', '--dc_block', 'true', '--enable_doppler', 'false',
                            '--rs_usecheck', 'true', '--freq_shift', '0', '--pll_bw', '0.0006',
                            '--rrc_alpha', '0.45', '--clock_alpha', '0.015',
                            '--clock_omega_relative_limit', '0.0001'],
                           cwd=work, stdout=log, stderr=subprocess.STDOUT, timeout=300, check=True)
        narrow.unlink()
        accepted = frames(work / 'decoded/lrpt72_cadu.cadu')
        if not accepted:
            raise RuntimeError('SatDump returned no valid CADUs')
        data = b''.join(accepted.values())
        cadu = output / 'lrpt.cadu'
        cadu.write_bytes(data)
        render_products(work, cadu, tle, resources, binary, output, settings)

    result = {'unique_valid_cadu': len(accepted), 'sha256': hashlib.sha256(data).hexdigest()}
    write_metadata(output / 'decode-summary.json', result)
    return result


def render_products(work, cadu, tle, resources, binary, output, settings):
    stage = work / 'render'
    stage.mkdir()
    for name in ('pipelines', 'plugins'):
        (stage / name).symlink_to(resources / name, target_is_directory=True)
    assets = stage / 'resources'
    assets.mkdir()
    for source in (resources / 'resources').iterdir():
        if source.name != 'projections_settings':
            (assets / source.name).symlink_to(source)
    projections = assets / 'projections_settings'
    projections.mkdir()
    for source in (resources / 'resources/projections_settings').iterdir():
        if source.name == 'meteor_m2-3_msumr_lrpt.json':
            sensor = satdump_json(source)
            sensor['timestamp_offset'] += 7
            write_metadata(projections / source.name, sensor)
        else:
            (projections / source.name).symlink_to(source)

    instrument = settings['viewer']['instruments']['msu_mr']
    rgb = instrument['rgb_composites']['AVHRR 221 False Color']
    grid = {'type': 'equirec', 'offset_x': 106, 'offset_y': 48, 'scalar_x': .02,
            'scalar_y': -.02, 'width': 2350, 'height': 1150}
    overlays = {'draw_qth_overlay': False, 'draw_cities_overlay': False, 'draw_latlon_overlay': False}
    instrument['rgb_composites'] = {
        'RGB 221': rgb,
        'Gray 1': {'equation': 'ch1', 'geo_correct': True},
        'Gray 2': {'equation': 'ch2', 'geo_correct': True},
        'Base 221': dict(rgb, project=dict(overlays, config=grid, draw_map_overlay=False,
                                          draw_shores_overlay=False)),
        'Map 221': dict(rgb, project=dict(overlays, config=grid, draw_map_overlay=True,
                                         draw_shores_overlay=True, borders_color=[.95, .77, .36, 1.0],
                                         shores_color=[.99, .99, .94, 1.0])),
    }
    instrument.pop('project_channels', None)
    write_metadata(stage / 'satdump_cfg.json', settings)
    tle_lines = tle.read_text().splitlines()[-2:]
    tle_override = stage / 'meteor.tle'
    tle_override.write_text('METEOR-M2-3\n' + '\n'.join(tle_lines) + '\n')
    with (output / 'products.log').open('w') as log:
        subprocess.run([binary, PIPELINE, 'cadu', cadu, stage / 'products', '--tle_override',
                        tle_override, '--satellite_number', 'Auto', '--rs_usecheck', 'true',
                        '--fill_missing', 'false', '--enable_doppler', 'false'],
                       cwd=stage, stdout=log, stderr=subprocess.STDOUT, timeout=300, check=True)
    products = stage / 'products/MSU-MR'
    for native, public in (
        ('msu_mr_rgb_RGB_221_corrected.png', '03-msu-mr-221.png'),
        ('msu_mr_Gray_1_corrected.png', '04-msu-mr-ch1.png'),
        ('msu_mr_Gray_2_corrected.png', '05-msu-mr-ch2.png'),
        ('rgb_msu_mr_rgb_Map_221_projected.png', '06-msu-mr-map.png'),
        ('rgb_msu_mr_rgb_Base_221_projected.png', 'msu-mr-map-base.png'),
    ):
        shutil.copyfile(products / native, output / public)


def analyze(folder, prediction, tle, config, output, binary, resources):
    metadata = json.loads((folder / 'recording.json').read_text())
    live = metadata['live_doppler']
    if (metadata['status'] != 'completed' or not metadata['settings_restored']
            or live['status'] != 'completed' or not live['covers_all_recorded_samples']
            or not live['file_sizes_match'] or live['fir_passband_hz'] < 95_000):
        raise ValueError('Analysis requires a complete recording and 95 kHz corrected passband')
    if (folder / 'iq.cs16').stat().st_size != metadata['sample_count'] * 4:
        raise ValueError('Original IQ size does not match the metadata')
    rate, count = live['output_sample_rate_hz'], live['output_samples_per_branch']
    if type(rate) is not int or rate <= 190_000 or type(count) is not int or count < 4096:
        raise ValueError('Invalid CF32 rate or sample count')
    if (folder / 'live-doppler/corrected.cf32').stat().st_size != count * 8:
        raise ValueError('Corrected CF32 size does not match the metadata')
    output = output.resolve()
    output.mkdir(parents=True)
    snr = measure(folder, metadata, prediction, tle, config, output)
    lrpt = decode(folder.resolve(), metadata, prediction, tle, output, binary, resources)
    print(json.dumps({'peak_snr': snr['peak_snr'], 'unique_valid_cadu': lrpt['unique_valid_cadu']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    parser.add_argument('--prediction', required=True, type=Path)
    parser.add_argument('--tle', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--satdump', default='satdump')
    parser.add_argument('--satdump-resources', type=Path)
    args = parser.parse_args()
    binary = shutil.which(args.satdump)
    if binary is None:
        parser.error('SatDump CLI was not found. Install it or specify --satdump.')
    binary = Path(binary).resolve()
    resources = args.satdump_resources or binary.parent.parent / 'Resources'
    analyze(args.recording, args.prediction, args.tle, args.config, args.output, binary, resources.resolve())
