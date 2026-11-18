"""Pluto receive connection and sample-rate control."""


def connect(uri, *, expected_serial=None):
    if not isinstance(uri, str) or not uri.strip():
        raise ValueError("An explicit Pluto URI is required")
    if expected_serial is not None and (not isinstance(expected_serial, str) or not expected_serial.strip()):
        raise ValueError("Expected Pluto serial must be a nonempty string")

    import adi

    try:
        radio = adi.Pluto(uri=uri)
        radio.ctx.set_timeout(5000)
        if expected_serial is not None and radio.ctx.attrs.get("hw_serial") != expected_serial:
            raise RuntimeError("Connected SDR serial does not match the reception plan")
        disable_tx(radio)
    except OSError as error:
        raise RuntimeError(f"Pluto connection or TX disable failed ({type(error).__name__})") from None
    return radio


def disable_tx(radio):
    phy = radio.ctx.find_device("ad9361-phy")
    dds = radio.ctx.find_device("cf-ad9361-dds-core-lpc")
    if phy is None or dds is None:
        raise RuntimeError("Cannot disable TX: Pluto TX devices not found")
    lo = phy.find_channel("altvoltage1", True)
    gain = phy.find_channel("voltage0", True)
    settings = [(lo.attrs["powerdown"], 1), (gain.attrs["hardwaregain"], -89.75)]
    channels = [channel for channel in dds.channels
                if channel.output and channel.id.startswith("altvoltage")]
    if not channels:
        raise RuntimeError("Cannot disable TX: DDS channels not found")
    settings.extend((channel.attrs[name], 0)
                    for name in ("raw", "scale") for channel in channels)
    for attribute, value in settings:
        attribute.value = str(value)
    for attribute, value in settings:
        if float(attribute.value.split()[0]) != value:
            raise RuntimeError(f"TX disable readback failed: {attribute.name}")


def radio_info(radio):
    return {
        "sample_rate_hz": int(radio.sample_rate),
        "rx_frequency_hz": int(radio.rx_lo),
        "rx_bandwidth_hz": int(radio.rx_rf_bandwidth),
        "rx_buffer_size": int(radio.rx_buffer_size),
        "rx_gain_mode": radio.gain_control_mode_chan0,
        "rx_gain_db": float(radio.rx_hardwaregain_chan0),
    }


def capture_rate_attributes(radio):
    phy = radio.ctx.find_device("ad9361-phy")
    attrs = phy.find_channel("voltage0", False).attrs
    return attrs["sampling_frequency"], attrs["filter_fir_en"]


def capture_rate_matches(requested, actual):
    return actual == requested or (requested, actual) == (2_500_000, 2_499_999)
