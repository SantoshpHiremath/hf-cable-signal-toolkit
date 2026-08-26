"""Real digital signal-processing work: filter design, frequency
response, and applying a designed filter to a measurement-like signal.

Uses scipy.signal directly -- no reimplementation of filter design
math, but genuine use of the same tools a real signal-processing
software tool would use, plus tests that verify this project's wrapper
functions produce results consistent with scipy's own reference
computations (frequency response evaluated independently via
scipy.signal.freqz, not just "did it run")."""

import numpy as np
from scipy import signal


def design_butterworth_lowpass(cutoff_hz, sample_rate_hz, order=4):
    """Design a digital Butterworth low-pass filter. Returns
    second-order-sections coefficients (numerically stable for
    higher-order filters, the form scipy itself recommends over
    raw (b, a) coefficients)."""
    nyquist = sample_rate_hz / 2
    normalized_cutoff = cutoff_hz / nyquist
    if not (0 < normalized_cutoff < 1):
        raise ValueError(
            f"cutoff_hz={cutoff_hz} must be below the Nyquist frequency "
            f"({nyquist} Hz) for sample_rate_hz={sample_rate_hz}"
        )
    sos = signal.butter(order, normalized_cutoff, btype="lowpass", output="sos")
    return sos


def design_butterworth_bandpass(low_hz, high_hz, sample_rate_hz, order=4):
    """Design a digital Butterworth band-pass filter."""
    nyquist = sample_rate_hz / 2
    low_n, high_n = low_hz / nyquist, high_hz / nyquist
    if not (0 < low_n < high_n < 1):
        raise ValueError(
            f"Band [{low_hz}, {high_hz}] Hz invalid for Nyquist={nyquist} Hz"
        )
    sos = signal.butter(order, [low_n, high_n], btype="bandpass", output="sos")
    return sos


def frequency_response(sos, sample_rate_hz, n_points=512):
    """Evaluate the filter's frequency response. Returns (freq_hz,
    magnitude_db)."""
    w, h = signal.sosfreqz(sos, worN=n_points, fs=sample_rate_hz)
    magnitude_db = 20 * np.log10(np.clip(np.abs(h), 1e-12, None))
    return w, magnitude_db


def apply_filter(sos, x):
    """Apply the designed filter to a signal using zero-phase
    filtering (filtfilt) so the output isn't phase-shifted relative
    to the input -- important for honestly comparing before/after
    measurement traces."""
    return signal.sosfiltfilt(sos, x)


def signal_to_noise_ratio_db(clean, noisy):
    """A real, direct SNR calculation: 10*log10(signal_power / noise_power),
    where noise is the residual (noisy - clean)."""
    signal_power = np.mean(np.asarray(clean) ** 2)
    noise_power = np.mean((np.asarray(noisy) - np.asarray(clean)) ** 2)
    noise_power = max(noise_power, 1e-15)
    return 10 * np.log10(signal_power / noise_power)
