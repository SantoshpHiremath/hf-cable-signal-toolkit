"""Synthetic (but physically-derived, not fabricated) measurement data.

Two distinct generators, corresponding to this project's two real
software-tool capabilities:

1. `cable_sparams_from_model()` -- S11 vs. frequency is *computed*, not
   invented, from `CableModel`'s real closed-form transmission-line
   equations for a chosen cable geometry and load. This is the
   physically correct S11 that RLGC parameters of this kind *would*
   produce; it is not a measurement from a real cable.
2. `noisy_baseband_measurement()` -- a generic clean tone plus
   injected out-of-band interference, standing in for "a measurement
   trace with noise a signal-processing algorithm needs to clean up" --
   used to demonstrate `filter_design.py`, deliberately kept separate
   from the S-parameter side rather than mislabeled as RF data.
"""

import numpy as np

from src.transmission_line import CableModel, reflection_coefficient


# Roughly RG-58/50-ohm-coax-like per-unit-length parameters, used only
# as a realistic, named example -- not a claim of having measured or
# sourced these from a real cable's datasheet.
RG58_LIKE_PARAMS = {"R": 0.03, "L": 2.5e-7, "G": 1e-6, "C": 1.0e-10}


def cable_sparams_from_model(freq_hz, length_m=1.0, z_load=75.0, params=None):
    """Compute S11(f) for a uniform cable of given length terminated in
    z_load, using CableModel's real transmission-line equations."""
    params = params or RG58_LIKE_PARAMS
    model = CableModel(**params)
    z0 = model.characteristic_impedance(freq_hz)
    zin = model.input_impedance(freq_hz, length_m, z_load)
    s11 = reflection_coefficient(z0, zin)
    return s11


def noisy_baseband_measurement(duration_s=0.01, sample_rate_hz=200_000,
                                tone_hz=2_000, noise_band_hz=None,
                                noise_amplitude=0.6, seed=0):
    """A clean low-frequency tone (the "signal of interest") plus
    injected higher-frequency band-limited noise (the "interference" a
    low-pass filter should remove). Returns (t, clean, noisy,
    sample_rate_hz).

    `noise_band_hz` defaults to a band expressed relative to the given
    `sample_rate_hz` (20%-30% of Nyquist) rather than a fixed absolute
    range, so callers that pass a different sample rate than the
    default still get a physically valid band instead of a silent
    Nyquist violation."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * sample_rate_hz)
    t = np.arange(n) / sample_rate_hz

    clean = np.sin(2 * np.pi * tone_hz * t)

    nyq = sample_rate_hz / 2
    if noise_band_hz is None:
        noise_band_hz = (0.4 * nyq, 0.6 * nyq)
    if not (0 < noise_band_hz[0] < noise_band_hz[1] < nyq):
        raise ValueError(
            f"noise_band_hz={noise_band_hz} must lie strictly within "
            f"(0, Nyquist={nyq} Hz) for sample_rate_hz={sample_rate_hz}"
        )

    # Band-limited noise: white noise passed through a bandpass filter
    # so it concentrates in a known, checkable frequency range.
    from scipy import signal as scipy_signal
    white = rng.normal(0, 1, size=n)
    sos = scipy_signal.butter(
        4, [noise_band_hz[0] / nyq, noise_band_hz[1] / nyq], btype="bandpass", output="sos"
    )
    band_noise = scipy_signal.sosfilt(sos, white)
    band_noise = band_noise / (np.std(band_noise) + 1e-12) * noise_amplitude

    noisy = clean + band_noise
    return t, clean, noisy, sample_rate_hz
