"""Record one bounded receive window as streaming cs16 IQ."""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import signal
import time

import numpy as np

from pluto import capture_rate_attributes, capture_rate_matches, connect, radio_info
from rx_quality import ReceiveQuality, summarize_quality

SAMPLES = 65_536
FULL_SCALE = 2048


def write_metadata(path, metadata):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def validate_window(start, end, now):
    if start.utcoffset() is None or end.utcoffset() is None:
        raise ValueError("Start and end must include a time zone")
    start, end = start.astimezone(timezone.utc), end.astimezone(timezone.utc)
    if not 0 < (end - start).total_seconds() <= 900:
        raise ValueError("Recording must be between 0 and 900 seconds")
    if end <= now:
        raise ValueError("The capture window has already ended")
    if (start - now).total_seconds() > 900:
        raise ValueError("Run this recorder within 15 minutes of the capture window")
    return start, end


def check_space(output, seconds, raw_rate, channel_rates=()):
    needed = math.ceil(seconds * (4 * raw_rate + 16 * sum(channel_rates))) + 512 * 1024**2
    parent = next(parent for parent in output.absolute().parents if parent.exists())
    if shutil.disk_usage(parent).free < needed:
        raise RuntimeError(f"At least {needed} free bytes are needed for this recording")


def record_gain_readback(radio, metadata, phase):
    started = time.monotonic()
    reading = {"time_utc_unix": time.time(), "sample_offset": metadata["sample_count"], "phase": phase}
    try:
        reading["gain_mode"] = radio.gain_control_mode_chan0
        gain = float(radio.rx_hardwaregain_chan0)
        if not math.isfinite(gain):
            raise ValueError("RX gain readback is not finite")
        reading["actual_gain_db"] = gain
    except Exception as error:
        reading["error"] = type(error).__name__
        metadata["gain_history"]["disabled"] = True
    reading["read_duration_seconds"] = time.monotonic() - started
    metadata["gain_history"]["readings"].append(reading)


def record(output, *, start, end, uri, expected_serial, target, center_hz, sample_rate_hz,
           bandwidth_hz, gain_control_mode=None, sample_sink=None):
    now = datetime.now(timezone.utc)
    start, end = validate_window(start, end, now)
    if any(type(value) is not int or value <= 0 for value in (center_hz, sample_rate_hz, bandwidth_hz)):
        raise ValueError("Center frequency, sample rate and RF bandwidth must be positive integers")
    output = Path(output)
    check_space(output, (end - max(start, now)).total_seconds(), sample_rate_hz)
    metadata_path = output / "recording.json"
    raw_path = output / "iq.cs16"
    metadata = {
        "status": "preparing", "planned_start": start.isoformat(), "planned_end": end.isoformat(),
        "target": target, "target_identity_confirmed": False, "lrpt_identity_confirmed": False,
        "center_frequency_hz": center_hz, "requested_center_frequency_hz": center_hz,
        "sample_rate_hz": sample_rate_hz, "requested_sample_rate_hz": sample_rate_hz,
        "rx_bandwidth_hz": bandwidth_hz, "requested_rx_bandwidth_hz": bandwidth_hz,
        "sample_format": "cs16_le",
        "sample_layout": "I then Q, signed little-endian int16; unscaled 12-bit ADC counts",
        "adc_full_scale": FULL_SCALE, "transmit_used": False, "sample_count": 0, "bytes_written": 0,
        "quality_history": [],
        "gain_history": {
            "interval_seconds": None, "policy": "capture_boundaries", "disabled": False,
            "sample_offset_reference": "Number of IQ samples written before host readback; not a hardware timestamp or per-sample gain.",
            "readings": [],
        },
    }
    before = None
    restore_rate = False
    first_count = None
    snapshots, timestamps = [], []
    powers, clipped = 0., 0
    quality = ReceiveQuality()
    window = np.hanning(4096)
    finalizing = interrupted = False

    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True
        if not finalizing:
            raise KeyboardInterrupt(f"Stopped by signal {signum}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    previous_handler = signal.signal(signal.SIGTERM, stop)
    try:
        phase = "write initial metadata"
        write_metadata(metadata_path, metadata)
        phase = "connect and verify TX disable"
        radio = connect(uri, expected_serial=expected_serial)
        metadata["connected_at_utc"] = datetime.now(timezone.utc).isoformat()
        radio.ctx.set_timeout(2000)
        phase = "read original RX settings"
        settings = radio_info(radio)
        rate_attribute, fir = capture_rate_attributes(radio)
        settings["rx_fir_enabled"] = int(fir.value)
        before = metadata["before"] = settings

        phase = "set RX sample rate, RF bandwidth, LO and buffer size"
        restore_rate = before["sample_rate_hz"] != sample_rate_hz
        if restore_rate:
            rate_attribute.value = str(sample_rate_hz)
        radio.rx_rf_bandwidth = bandwidth_hz
        radio.rx_lo = center_hz
        radio.rx_buffer_size = SAMPLES
        if gain_control_mode is not None:
            phase = "set RX gain mode"
            metadata["gain_probe"] = {
                "requested_mode": gain_control_mode,
                "original_mode": before["rx_gain_mode"],
                "original_gain_db": before["rx_gain_db"],
                "restored": False,
            }
            radio.gain_control_mode_chan0 = gain_control_mode

        phase = "read active RX settings"
        receiver = metadata["receiver"] = radio_info(radio)
        receiver["rx_fir_enabled"] = int(fir.value)
        actual_rate = receiver["sample_rate_hz"]
        actual_center = receiver["rx_frequency_hz"]
        metadata.update(sample_rate_hz=actual_rate, center_frequency_hz=actual_center,
                        rx_bandwidth_hz=receiver["rx_bandwidth_hz"])
        rate_matches = (capture_rate_matches(sample_rate_hz, actual_rate)
                        if sample_sink is None else actual_rate == sample_rate_hz)
        if not rate_matches:
            phase = "verify RX sample rate"
            raise RuntimeError("RX sample rate differs from requested")
        if (abs(actual_center - center_hz) > 5 or receiver["rx_bandwidth_hz"] != bandwidth_hz
                or receiver["rx_buffer_size"] != SAMPLES
                or receiver["rx_fir_enabled"] != before["rx_fir_enabled"]
                or (gain_control_mode is not None and receiver["rx_gain_mode"] != gain_control_mode)):
            phase = "verify active RX LO, bandwidth, buffer size, FIR and gain mode"
            raise RuntimeError("Active RX settings differ from requested")
        normalization = actual_rate * np.sum(window**2) * FULL_SCALE**2
        phase = "wait for capture window"
        metadata["receiver_ready_at_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["status"] = "waiting"
        write_metadata(metadata_path, metadata)
        while (remaining := (start - datetime.now(timezone.utc)).total_seconds()) > 0:
            time.sleep(min(1., remaining))
        if datetime.now(timezone.utc) >= end:
            raise RuntimeError("The capture window ended during receiver preparation")
        phase = "discard startup buffer"
        record_gain_readback(radio, metadata, "before_rx")
        radio.rx()
        metadata["actual_start_utc"] = datetime.now(timezone.utc).isoformat()
        phase = "record IQ"
        metadata["status"] = "recording"
        write_metadata(metadata_path, metadata)
        next_spectrum = next_metadata = next_quality = 0.
        with raw_path.open("xb", buffering=4 * 1024**2) as raw:
            while datetime.now(timezone.utc) < end:
                iq = np.asarray(radio.rx())
                received_utc = time.time()
                if not np.isfinite(iq).all() or not np.any(iq):
                    phase = "validate IQ (nonfinite or entirely zero)"
                    raise RuntimeError("Invalid IQ")
                offset = metadata["sample_count"]
                packed = np.empty((iq.size, 2), dtype="<i2")
                packed[:, 0], packed[:, 1] = iq.real, iq.imag
                raw.write(packed.tobytes())
                if first_count is None:
                    first_count = iq.size
                    metadata["first_recorded_buffer_received_utc_unix"] = received_utc
                    metadata["host_receive_span_note"] = "Host receive timing only; not a device timestamp or proof of USB loss."
                metadata["last_recorded_buffer_received_utc_unix"] = received_utc
                metadata["sample_count"] += iq.size
                metadata["bytes_written"] += packed.nbytes
                if sample_sink is not None:
                    try:
                        sample_sink(iq, offset, received_utc, metadata)
                    except Exception as error:
                        metadata["sample_sink_error"] = type(error).__name__
                        sample_sink = None
                block_power, block_clipped = quality.add(iq, offset, received_utc)
                powers += block_power
                clipped += block_clipped
                tick = time.monotonic()
                if tick >= next_quality:
                    quality.flush(metadata)
                    next_quality = tick + 1.
                if tick >= next_spectrum and iq.size >= SAMPLES:
                    fft = np.fft.fft(iq[:SAMPLES].reshape(16, 4096) * window, axis=1)
                    snapshots.append(np.fft.fftshift(np.mean(abs(fft)**2, axis=0) / normalization))
                    timestamps.append(received_utc)
                    next_spectrum = tick + 1.
                if tick >= next_metadata:
                    raw.flush()
                    metadata["last_samples_utc"] = datetime.now(timezone.utc).isoformat()
                    write_metadata(metadata_path, metadata)
                    next_metadata = tick + 10.
        if not metadata["sample_count"]:
            phase = "verify capture contains IQ samples"
            raise RuntimeError("No IQ samples were recorded within the capture window")
        metadata["status"] = "completed"
    except (Exception, KeyboardInterrupt) as error:
        metadata["status"] = "partial" if metadata["sample_count"] else "failed"
        metadata["error"] = f"{phase}: {type(error).__name__}"
        if isinstance(error, OSError) and error.errno is not None:
            metadata["error_errno"] = error.errno
    finally:
        finalizing = True
        metadata["actual_end_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            restoration_errors = []
            if before is not None:
                try:
                    radio.rx_destroy_buffer()
                except Exception as error:
                    restoration_errors.append(f"destroy RX buffer: {type(error).__name__}")
                if "actual_start_utc" in metadata:
                    record_gain_readback(radio, metadata, "after_rx")
                if restore_rate:
                    try:
                        rate_attribute.value = str(before["sample_rate_hz"])
                    except Exception as error:
                        restoration_errors.append(f"restore sample rate: {type(error).__name__}")
                restore = {
                    "rx_rf_bandwidth": "rx_bandwidth_hz", "rx_lo": "rx_frequency_hz",
                    "rx_buffer_size": "rx_buffer_size",
                }
                if "gain_probe" in metadata:
                    restore["gain_control_mode_chan0"] = "rx_gain_mode"
                    if before["rx_gain_mode"] == "manual":
                        restore["rx_hardwaregain_chan0"] = "rx_gain_db"
                for attribute, key in restore.items():
                    try:
                        setattr(radio, attribute, before[key])
                    except Exception as error:
                        restoration_errors.append(f"{key}: {type(error).__name__}")
                try:
                    after = metadata["after"] = radio_info(radio)
                    after["rx_fir_enabled"] = int(fir.value)
                    gain_keys = {"rx_gain_mode"}
                    if before["rx_gain_mode"] == "manual":
                        gain_keys.add("rx_gain_db")
                    for key in before:
                        if (key != "rx_gain_db" or key in gain_keys) and after.get(key) != before[key]:
                            restoration_errors.append(f"{key} was not restored")
                    if "gain_probe" in metadata:
                        metadata["gain_probe"]["restored"] = all(after[key] == before[key] for key in gain_keys)
                except Exception as error:
                    restoration_errors.append(f"restore readback: {type(error).__name__}")
            metadata["settings_restored"] = before is not None and not restoration_errors
            metadata["restoration_errors"] = restoration_errors
            quality.flush(metadata)
            summarize_quality(metadata)
            if metadata["sample_count"]:
                metadata["recorded_iq_seconds"] = metadata["sample_count"] / metadata["sample_rate_hz"]
                metadata["host_receive_span_minus_iq_seconds"] = (
                    metadata["last_recorded_buffer_received_utc_unix"]
                    - metadata["first_recorded_buffer_received_utc_unix"]
                    - (metadata["sample_count"] - first_count) / metadata["sample_rate_hz"])
                metadata["rms_adc_counts"] = math.sqrt(powers / metadata["sample_count"])
                metadata["clipped_sample_fraction"] = clipped / metadata["sample_count"]
            try:
                if raw_path.exists():
                    size = raw_path.stat().st_size
                    if size != metadata["bytes_written"]:
                        metadata["bytes_written"] = size
                        metadata.setdefault("error", "Raw file size differs from the recorded IQ count")
                        metadata["status"] = "partial"
                if snapshots:
                    frequency = np.fft.fftshift(np.fft.fftfreq(4096, 1 / actual_rate)) + actual_center
                    np.savez_compressed(output / "spectrum-history.npz", frequency_hz=frequency,
                        time_utc_unix=timestamps, psd_dbfs_hz=10 * np.log10(np.maximum(np.stack(snapshots), 1e-30)))
            except Exception as error:
                metadata["analysis_error"] = type(error).__name__
            if interrupted:
                metadata.setdefault("error", "Interrupted by SIGTERM")
            if metadata["status"] == "completed" and (
                    not metadata["settings_restored"] or "sample_sink_error" in metadata
                    or "analysis_error" in metadata or interrupted):
                metadata["status"] = "partial"
            write_metadata(metadata_path, metadata)
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
    return 0 if metadata["status"] == "completed" else 2
