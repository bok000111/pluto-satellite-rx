"""Record one planned raw or Doppler-corrected receive window."""

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import tomllib

import numpy as np

from live_doppler import DopplerCurve, LiveDopplerTap
from record_meteor import check_space, record, validate_window, write_metadata
from residual_carrier import correct_residual


def run(plan_path, config_path):
    plan = json.loads(plan_path.read_text())
    with config_path.open("rb") as file:
        pluto = tomllib.load(file)["pluto"]
    uri, serial = pluto["uri"], pluto.get("serial")
    if not isinstance(uri, str) or not uri.strip():
        raise ValueError("[pluto].uri must be a nonempty string")
    if serial is not None and (not isinstance(serial, str) or not serial.strip()):
        raise ValueError("[pluto].serial must be a nonempty string")

    output = plan_path.parent / plan["output"]
    if output.exists() or output.is_symlink():
        raise FileExistsError("Recording output already exists")
    target = plan["target"]
    center, rate, bandwidth = (plan[key] for key in ("center_hz", "sample_rate_hz", "rf_bandwidth_hz"))
    if not isinstance(target, str) or not target.strip():
        raise ValueError("The plan target must be a nonempty string")
    if any(type(value) is not int or value <= 0 for value in (center, rate, bandwidth)):
        raise ValueError("Center frequency, sample rate and RF bandwidth must be positive integers")
    mode = plan.get("gain_control_mode")
    if mode is not None and mode not in ("manual", "slow_attack", "fast_attack", "hybrid"):
        raise ValueError("Unsupported RX gain mode")

    channels = plan.get("doppler_channels", [])
    if not isinstance(channels, list):
        raise ValueError("doppler_channels must be a list")
    now = datetime.now(timezone.utc)
    if "seconds" in plan:
        if channels or "record_start" in plan or "record_end" in plan:
            raise ValueError("seconds selects immediate raw-only recording")
        seconds = plan["seconds"]
        if type(seconds) not in (int, float) or not 0 < seconds <= 900:
            raise ValueError("Recording must be between 0 and 900 seconds")
        start, end = now, now + timedelta(seconds=seconds)
    else:
        start, end = (datetime.fromisoformat(plan[key]) for key in ("record_start", "record_end"))
    start, end = validate_window(start, end, now)

    views = []
    keys = set()
    for channel in channels:
        key, output_rate, cutoff, carrier = (channel[name] for name in (
            "key", "sample_rate_hz", "cutoff_hz", "carrier_hz"))
        cfo = channel.get("cfo_hz", 0)
        if (not isinstance(key, str) or not key or key in (".", "..") or Path(key).name != key or "\\" in key):
            raise ValueError("Doppler channel keys must be safe directory names")
        folded = key.casefold()
        if folded in keys:
            raise ValueError("Doppler channel keys must be unique directory names")
        keys.add(folded)
        if (type(output_rate) is not int or output_rate <= 0 or rate % output_rate
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in (cutoff, carrier, cfo))
                or not 0 < cutoff < output_rate / 2 or carrier <= 0):
            raise ValueError("Invalid Doppler carrier, CFO, cutoff or decimation")
        curve = DopplerCurve.from_file(plan_path.parent / plan["prediction"], carrier)
        window = np.array([start.timestamp(), end.timestamp() + 2]) - curve.origin
        curve.integral(window)
        doppler = np.r_[np.interp(window, curve.times, curve.hz),
                        curve.hz[(curve.times > window[0]) & (curve.times < window[1])]]
        excursion = np.max(abs(carrier + cfo - center + doppler)) + cutoff + 5
        if excursion >= min(rate, bandwidth) / 2:
            raise ValueError("Doppler channel exceeds the raw RF passband")
        view = output / "targets" / key if len(channels) > 1 else output
        views.append((key, view, output_rate, cutoff, carrier, cfo, curve))
    check_space(output, (end - max(start, now)).total_seconds(), rate, (view[2] for view in views))

    taps = []
    result = None
    metadata_path = output / "recording.json"

    def submit(iq, offset, received, metadata):
        for _, view, tap in taps:
            if offset == 0 and view != output:
                view.mkdir(parents=True)
            tap.submit(iq, offset, received, metadata)

    try:
        for key, view, output_rate, cutoff, carrier, cfo, curve in views:
            tap = LiveDopplerTap(view, sample_rate=rate, output_rate=output_rate, carrier_hz=carrier,
                                 prediction=curve, cutoff_hz=cutoff, cfo_hz=cfo)
            taps.append((key, view, tap))
        result = record(output, start=start, end=end, uri=uri, expected_serial=serial, target=target,
                        center_hz=center, sample_rate_hz=rate, bandwidth_hz=bandwidth,
                        gain_control_mode=mode, sample_sink=submit if views else None)
    finally:
        results = [(key, view, tap, tap.close()) for key, view, tap in taps]
        if results and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            summaries = []
            for key, view, tap, live in results:
                live["covers_all_recorded_samples"] = live["processed_samples"] == metadata["sample_count"]
                if live["status"] == "completed" and (
                        not live["covers_all_recorded_samples"] or not live.get("file_sizes_match")):
                    live["status"] = "partial"
                live["residual_correction"] = correct_residual(view, live, metadata)
                if (tap.output / "result.json").exists():
                    write_metadata(tap.output / "result.json", live)
                if view == output:
                    metadata["live_doppler"] = live
                    metadata["nominal_downlink_frequencies_hz"] = [live["carrier_hz"]]
                else:
                    view.mkdir(parents=True, exist_ok=True)
                    for filename in ("iq.cs16", "spectrum-history.npz"):
                        if (output / filename).is_file():
                            (view / filename).symlink_to(Path("../..") / filename)
                    write_metadata(view / "recording.json", {
                        **metadata, "target": f"{target} / {key}", "live_doppler": live,
                        "nominal_downlink_frequencies_hz": [live["carrier_hz"]],
                    })
                summaries.append({"key": key, "view": str(view.relative_to(output)), **live})
            if len(views) > 1:
                metadata["live_doppler_channels"] = summaries
            write_metadata(metadata_path, metadata)

    if any(live["status"] != "completed" or not live["file_sizes_match"] for *_, live in results):
        return 2
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "local.toml")
    args = parser.parse_args()
    try:
        return run(args.plan, args.config)
    except OSError as error:
        parser.exit(1, f"record_pass: {type(error).__name__}: {error.strerror or 'recording output already exists'}\n")
    except (ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"record_pass: {type(error).__name__}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
