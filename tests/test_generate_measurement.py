import numpy as np

from src.generate_measurement import cable_sparams_from_model, noisy_baseband_measurement
from src.transmission_line import CableModel


def test_cable_sparams_matched_load_gives_low_reflection():
    freq = np.linspace(1e6, 6e9, 50)
    model_params = {"R": 0.03, "L": 2.5e-7, "G": 1e-6, "C": 1.0e-10}
    z0_dc_estimate = np.sqrt(model_params["L"] / model_params["C"])  # ~50 ohm-ish for these params
    s11 = cable_sparams_from_model(freq, length_m=1.0, z_load=z0_dc_estimate, params=model_params)
    # Matching the load close to the line's own characteristic impedance
    # should give small reflection across the band.
    assert np.all(np.abs(s11) < 0.05)


def test_cable_sparams_mismatched_load_gives_higher_reflection():
    freq = np.linspace(1e6, 6e9, 50)
    matched = cable_sparams_from_model(freq, length_m=1.0, z_load=50.19)
    mismatched = cable_sparams_from_model(freq, length_m=1.0, z_load=200.0)
    assert np.mean(np.abs(mismatched)) > np.mean(np.abs(matched))


def test_cable_sparams_shape_matches_frequency_array():
    freq = np.linspace(1e6, 1e9, 17)
    s11 = cable_sparams_from_model(freq)
    assert s11.shape == freq.shape


def test_noisy_measurement_clean_signal_is_correct_frequency():
    t, clean, noisy, fs = noisy_baseband_measurement(tone_hz=1000, sample_rate_hz=50000, duration_s=0.02)
    spectrum = np.abs(np.fft.rfft(clean))
    freqs = np.fft.rfftfreq(len(clean), d=1 / fs)
    peak_freq = freqs[np.argmax(spectrum)]
    assert abs(peak_freq - 1000) < 60  # FFT bin resolution tolerance


def test_noisy_measurement_noise_is_concentrated_in_requested_band():
    t, clean, noisy, fs = noisy_baseband_measurement(
        tone_hz=500, noise_band_hz=(10000, 12000), sample_rate_hz=100000, duration_s=0.02
    )
    residual = noisy - clean
    spectrum = np.abs(np.fft.rfft(residual))
    freqs = np.fft.rfftfreq(len(residual), d=1 / fs)

    in_band_power = np.sum(spectrum[(freqs >= 9000) & (freqs <= 13000)] ** 2)
    total_power = np.sum(spectrum ** 2)
    assert in_band_power / total_power > 0.6  # most noise energy is where we put it


def test_same_seed_is_reproducible():
    _, _, noisy_a, _ = noisy_baseband_measurement(seed=42)
    _, _, noisy_b, _ = noisy_baseband_measurement(seed=42)
    assert np.array_equal(noisy_a, noisy_b)
