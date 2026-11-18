"""Predict satellite passes and Doppler from a local TLE."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tomllib

import numpy as np
from scipy.optimize import minimize_scalar
from sgp4.api import Satrec, jday


def geometry(sat, times, latitude_deg, longitude_deg, altitude_m):
    times = np.atleast_1d(times).astype(float)
    lat, lon = np.deg2rad([latitude_deg, longitude_deg])
    e2 = 6.69437999014e-3
    normal = 6378.137 / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    height = altitude_m / 1000
    observer = np.array([(normal + height) * np.cos(lat) * np.cos(lon),
                         (normal + height) * np.cos(lat) * np.sin(lon),
                         (normal * (1 - e2) + height) * np.sin(lat)])
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    d = datetime.fromtimestamp(float(times[0]), timezone.utc)
    jd0, fr0 = jday(d.year, d.month, d.day, d.hour, d.minute, d.second + d.microsecond / 1e6)
    jd = np.full(times.shape, jd0)
    fr = fr0 + (times - times[0]) / 86400
    err, pos, vel = sat.sgp4_array(jd, fr)
    if np.any(err):
        raise RuntimeError(f'SGP4 error {np.unique(err)}')
    days = jd + fr
    t = (days - 2451545) / 36525
    theta = np.deg2rad((280.46061837 + 360.98564736629 * (days - 2451545)
                        + .000387933 * t * t - t * t * t / 38710000) % 360)
    c, s = np.cos(theta), np.sin(theta)
    fixed = np.column_stack((c * pos[:, 0] + s * pos[:, 1],
                             -s * pos[:, 0] + c * pos[:, 1], pos[:, 2]))
    velocity = np.column_stack((c * vel[:, 0] + s * vel[:, 1],
                                -s * vel[:, 0] + c * vel[:, 1], vel[:, 2]))
    velocity -= np.cross(np.array([0., 0., 7.29211514670698e-5]), fixed)
    rel = fixed - observer
    ranges = np.linalg.norm(rel, axis=1)
    elev = np.rad2deg(np.arcsin((rel @ up) / ranges))
    radial = np.sum(rel * velocity, axis=1) / ranges
    return ranges, elev, radial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tle', type=Path, required=True)
    parser.add_argument('--start', required=True)
    parser.add_argument('--end', required=True)
    parser.add_argument('--frequency', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--min-elevation', type=float, default=10)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parent.parent / 'config/local.toml')
    args = parser.parse_args()
    try:
        lines = args.tle.read_text().splitlines()
        if len(lines) == 3:
            lines = lines[1:]
        if (len(lines) != 2 or not lines[0].startswith('1 ') or not lines[1].startswith('2 ')
                or any(len(line) != 69 or not line[68].isdigit()
                       or sum(int(c) if c.isdigit() else int(c == '-') for c in line[:68]) % 10 != int(line[68])
                       for line in lines) or lines[0][2:7] != lines[1][2:7]):
            parser.error('invalid TLE lines or checksum')
        sat = Satrec.twoline2rv(*lines)
        start = datetime.fromisoformat(args.start)
        end = datetime.fromisoformat(args.end)
        if start.utcoffset() is None or end.utcoffset() is None:
            raise ValueError('start and end must include a timezone')
        first, last = start.timestamp(), end.timestamp()
        if not first < last:
            raise ValueError('end must be after start')
        if not np.isfinite(args.frequency) or args.frequency <= 0:
            raise ValueError('frequency must be positive and finite')
        if not np.isfinite(args.min_elevation) or not -90 <= args.min_elevation <= 90:
            raise ValueError('min-elevation must be between -90 and 90 degrees')
        with args.config.open('rb') as f:
            observer = tomllib.load(f)['observer']
        site = tuple(float(observer[key]) for key in ('latitude_deg', 'longitude_deg', 'altitude_m'))
        if not all(np.isfinite(site)) or not -90 <= site[0] <= 90 or not -180 <= site[1] <= 180:
            raise ValueError('invalid observer coordinates')
    except (ValueError, KeyError, OSError, TypeError) as exc:
        parser.error(str(exc))
    grid = np.arange(first, last, 2.)
    grid = np.append(grid, last)
    _, elev, radial = geometry(sat, grid, *site)
    mask = elev >= args.min_elevation
    edges = np.diff(np.r_[False, mask, False].astype(int))
    passes = []
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        aos = first if a == 0 else float(np.interp(args.min_elevation, elev[a-1:a+1], grid[a-1:a+1]))
        los = last if b == len(grid) else float(np.interp(args.min_elevation, elev[b-1:b+1][::-1], grid[b-1:b+1][::-1]))
        k = a + int(np.argmax(elev[a:b]))
        left, right = max(aos, grid[max(0, k-1)]), min(los, grid[min(len(grid)-1, k+1)])
        peak = minimize_scalar(lambda dt: -geometry(sat, [left + dt], *site)[1][0],
                               bounds=(0., right - left), method='bounded') if left < right else None
        candidates = [aos, los, float(grid[k])]
        if peak is not None and peak.success:
            candidates.append(float(left + peak.x))
        heights = geometry(sat, candidates, *site)[1]
        peak_index = int(np.argmax(heights))
        iso = lambda t: datetime.fromtimestamp(float(t), timezone.utc).isoformat()
        passes.append({'aos_utc': iso(aos), 'peak_utc': iso(candidates[peak_index]),
                       'los_utc': iso(los), 'peak_elevation_deg': float(heights[peak_index])})
    result = {'reference_frequency_hz': args.frequency, 'time_utc_unix': grid.tolist(),
              'doppler_hz': (-radial / 299792.458 * args.frequency).tolist(), 'passes': passes}
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
