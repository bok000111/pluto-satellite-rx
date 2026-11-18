#!/usr/bin/env python3
"""Recover Meteor LRPT using measured carrier tracking and compared channel filters."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from channel_filter import filter_channel
from residual_carrier import correct_residual
from meteor_segments import prepare as prepare_segments


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def frames(path):
    data = path.read_bytes()
    if len(data) % 1024:
        raise ValueError('Unaligned CADU output')
    result = {}
    for first in range(0, len(data), 1024):
        frame = data[first:first + 1024]
        if frame[:4] != bytes.fromhex('1acffc1d'):
            raise ValueError('Invalid CADU synchronization marker')
        key = frame[4:9].hex()
        if key in result and result[key][4:896] != frame[4:896]:
            raise ValueError('Conflicting corrected CADU payloads')
        result[key] = frame
    return result


def decode_branch(source, folder, rate, symbol_rate, iq_swap, binary, *, fine_tracking=True):
    folder.mkdir()
    parameters = {'samplerate': rate, 'baseband_format': 'cf32', 'freq_shift': 0,
                  'dc_block': True, 'rs_usecheck': True, 'satellite_number': 'Auto',
                  'fill_missing': False, 'enable_doppler': False}
    if fine_tracking:
        parameters.update(pll_bw=.0005, rrc_alpha=.6, clock_alpha=.002, clock_omega_relative_limit=.00005)
    parameters.update(symbolrate=symbol_rate, iq_swap=iq_swap)
    pipeline = 'meteor_m2-x_lrpt_80k' if symbol_rate == 80000 else 'meteor_m2-x_lrpt'
    command = [binary, pipeline, 'baseband', str(source), str(folder / 'products')]
    for key, value in parameters.items():
        command.extend(['--' + key, str(value).lower() if isinstance(value, bool) else str(value)])
    path = folder / 'result.json'
    result = {'command': command, 'source_bytes': source.stat().st_size, 'parameters': parameters,
              'exit_code': None, 'timeout': False}
    with (folder / 'decoder.log').open('w') as log:
        try:
            run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=300)
            result['exit_code'] = run.returncode
        except subprocess.TimeoutExpired:
            result['timeout'] = True
    path_cadu = folder / 'products' / (pipeline + '.cadu')
    accepted = {}
    if result['exit_code'] == 0:
        if path_cadu.exists():
            accepted = frames(path_cadu)
            result.update(cadu_path=str(path_cadu),
                          validation='Native four-interleave Reed-Solomon checks enabled; aligned CADU and payload consistency checked.')
        result['unique_valid_cadu'] = len(accepted)
    else:
        result.update(unique_valid_cadu=None, error='Native decoder failed')
    save(path, result)
    print(json.dumps({'branch': folder.name, 'frames': result['unique_valid_cadu'],
                      'exit_code': result['exit_code'], 'timeout': result['timeout']}), flush=True)
    return result, accepted


def recover(output, *, bandwidths_khz=(140, 130), binary='satdump'):
    output = Path(output).resolve()
    metadata = json.loads((output / 'recording.json').read_text())
    live = metadata['live_doppler']
    if (metadata.get('status') != 'completed' or not metadata.get('settings_restored')
            or live.get('status') != 'completed' or not live.get('covers_all_recorded_samples')
            or not live.get('file_sizes_match')):
        raise ValueError('Recovery requires a completed, restored recording and complete DSP input')
    rate = live['output_sample_rate_hz']
    count = live['output_samples_per_branch']
    if (type(rate) is not int or type(count) is not int or count <= 0
            or not 0 < live['fir_passband_hz'] <= rate / 2):
        raise ValueError('Invalid LRPT sample rate, count or preserved passband')
    root = output / 'meteor-recovery'
    original = output / 'live-doppler/corrected.cf32'
    if original.stat().st_size != count * 8:
        raise ValueError('Doppler source length differs from the recorded sample count')
    try:
        root.mkdir()
    except FileExistsError:
        raise FileExistsError('meteor-recovery already exists. Use a fresh recording copy.') from None
    work = root / 'carrier/live-doppler'
    work.mkdir(parents=True)
    residual = correct_residual(output, live, metadata, source_path=original, work_dir=work)
    bandwidths_khz = list(dict.fromkeys(bandwidths_khz))
    branches = [('doppler-default', original)]
    if residual.get('status') == 'completed':
        corrected = work / 'residual-corrected.cf32'
        branches.append(('carrier-wide', corrected))
        for width in bandwidths_khz:
            passband = width * 500
            stopband = passband + 10000
            if stopband >= min(rate / 2, live['fir_passband_hz']):
                raise ValueError('Narrow channel must fit inside the preserved source passband')
            destination = root / f'carrier-{width:g}k.cf32'
            report = filter_channel(corrected, destination, rate, passband, stopband)
            save(destination.with_suffix('.json'), report)
            branches.append((f'carrier-{width:g}k', destination))
    segmented = prepare_segments(original, root / 'segments', rate, live['fir_passband_hz'], bandwidths_khz)
    clocks = [segment['symbol_clock_hz'] for analysis in (residual, segmented)
              for segment in analysis.get('segments', []) if 'symbol_clock_hz' in segment]
    supported = [nominal for nominal in (72000, 80000)
                 if any(abs(clock - nominal) <= nominal * 100e-6 for clock in clocks)]
    estimated_rate = supported[0] if len(supported) == 1 else None
    runs, merged = {}, {}
    for wave, symbol_rate in enumerate((estimated_rate,) if estimated_rate else (72000, 80000)):
        suffix = '-80k' if wave else ''
        wave_branches = branches + [(branch['label'], Path(branch['path'])) for branch in segmented['branches']
                                    if branch['symbol_rate'] == symbol_rate]
        for label, source in wave_branches:
            for swap in (False, True):
                name = label + suffix + f'-iq{int(swap)}'
                result, accepted = decode_branch(source, root / name, rate, symbol_rate, swap, binary,
                                                 fine_tracking=label != 'doppler-default')
                runs[name] = result
                for key, frame in accepted.items():
                    if key in merged and merged[key][4:896] != frame[4:896]:
                        raise ValueError('Decoder branches contain conflicting corrected payloads')
                    merged[key] = frame
        if merged:
            break
    anchors = {}
    for key in merged:
        anchors.setdefault(key[:4], int(key[4:], 16))
    # Order each stream by signed 24-bit distance from its first observed counter.
    ordered = sorted(merged, key=lambda key: (key[:4],
        (int(key[4:], 16) - anchors[key[:4]] + 0x800000) % 0x1000000 - 0x800000))
    payload = b''.join(merged[key] for key in ordered)
    destination = root / 'merged-rs-accepted.cadu'
    destination.write_bytes(payload)
    pipeline = 'meteor_m2-x_lrpt_80k' if symbol_rate == 80000 else 'meteor_m2-x_lrpt'
    command = [binary, pipeline, 'cadu', str(destination), str(root / 'merged-products'),
               '--satellite_number', 'Auto', '--rs_usecheck', 'true', '--fill_missing', 'false']
    products = None
    product_timeout = False
    if merged:
        with (root / 'products.log').open('w') as log:
            try:
                products = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=300).returncode
            except subprocess.TimeoutExpired:
                product_timeout = True
    failed = [label for label, run in runs.items() if run.get('unique_valid_cadu') is None]
    report = {'status': 'completed' if not failed and not product_timeout and products in (0, None)
              and residual.get('status') != 'failed' and segmented['status'] != 'failed' else 'partial',
              'target': metadata.get('target'), 'source': str(original),
              'estimated_symbol_rate': estimated_rate, 'symbol_rate': symbol_rate if merged else estimated_rate,
              'residual_correction': residual, 'runs': runs, 'failed_runs': failed,
              'segmented_recovery': segmented,
              'unique_valid_cadu': len(merged), 'merged_cadu': str(destination),
              'merged_sha256': hashlib.sha256(payload).hexdigest(),
              'product_exit_code': products, 'product_timeout': product_timeout,
              'baseline_preserved_in_union': all(
                  runs[f'doppler-default{suffix}-iq{iq}']['unique_valid_cadu'] is not None for iq in (0, 1)),
              'additional_doppler_applied': False}
    save(root / 'comparison.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--bandwidths-khz', nargs='*', type=float, default=[140, 130])
    parser.add_argument('--satdump', default='satdump', help='Path to the SatDump CLI (default: satdump from PATH)')
    args = parser.parse_args()
    binary = shutil.which(args.satdump)
    if binary is None:
        parser.error('SatDump CLI was not found. Install it or specify --satdump.')
    report = recover(args.output, bandwidths_khz=args.bandwidths_khz, binary=str(Path(binary).resolve()))
    print(json.dumps({key: report[key] for key in ('status', 'unique_valid_cadu', 'failed_runs')}))
    raise SystemExit(0 if report['status'] == 'completed' else 2)
