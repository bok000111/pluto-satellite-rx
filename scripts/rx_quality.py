"""Sample-bounded RX quality measurements aligned with gain readbacks."""

import math

import numpy as np


class ReceiveQuality:
    def __init__(self):
        self.count = 0
        self.power = 0.
        self.clipped = 0
        self.peak_component = 0.
        self.peak_magnitude = 0.

    def add(self, iq, sample_offset, received_utc):
        if not self.count:
            self.start = int(sample_offset)
            self.first_utc = received_utc
        power = iq.real.astype(np.float64)**2 + iq.imag.astype(np.float64)**2
        component = max(float(np.max(abs(iq.real))), float(np.max(abs(iq.imag))))
        clipped = int(np.count_nonzero((abs(iq.real) >= 2047) | (abs(iq.imag) >= 2047)))
        total = float(np.sum(power))
        self.power += total
        self.clipped += clipped
        self.count += iq.size
        self.last_utc = received_utc
        self.peak_component = max(self.peak_component, component)
        self.peak_magnitude = max(self.peak_magnitude, math.sqrt(float(np.max(power))))
        return total, clipped

    def flush(self, metadata):
        if not self.count:
            return
        metadata['quality_history'].append({
            'sample_start': self.start, 'sample_end': self.start + self.count,
            'sample_count': self.count,
            'first_buffer_received_utc_unix': self.first_utc,
            'last_buffer_received_utc_unix': self.last_utc,
            'rms_adc_counts': math.sqrt(self.power / self.count),
            'peak_component_adc_counts': self.peak_component,
            'peak_magnitude_adc_counts': self.peak_magnitude,
            'clipped_sample_count': self.clipped,
            'clipped_sample_fraction': self.clipped / self.count,
            'gain_readback_index': None,
        })
        self.__init__()


def summarize_quality(metadata):
    rows = metadata['quality_history']
    if not rows:
        return
    gains = metadata['gain_history']['readings']
    observed = [row['actual_gain_db'] for row in gains if 'actual_gain_db' in row]
    worst = max(range(len(rows)), key=lambda i: rows[i]['clipped_sample_fraction'])
    metadata['quality_summary'] = {
        'measured_samples': sum(row['sample_count'] for row in rows),
        'covers_recording': sum(row['sample_count'] for row in rows) == metadata['sample_count'],
        'interval_count': len(rows),
        'intervals_with_clipping': sum(row['clipped_sample_count'] > 0 for row in rows),
        'worst_clipping_interval_index': worst,
        'minimum_rms_adc_counts': min(row['rms_adc_counts'] for row in rows),
        'maximum_rms_adc_counts': max(row['rms_adc_counts'] for row in rows),
        'minimum_observed_gain_db': min(observed) if observed else None,
        'maximum_observed_gain_db': max(observed) if observed else None,
        'gain_readback_changes': sum(a != b for a, b in zip(observed, observed[1:])),
    }
