import numpy as np
from scipy import signal as scipy_signal

from src.filter_design import (
    apply_filter,
    design_butterworth_bandpass,
    design_butterworth_lowpass,
    frequency_response,
    signal_to_noise_ratio_db,
)
from src.generate_measurement import noisy_baseband_measurement


def test_lowpass_rejects_frequencies_above_cutoff():
    sos = design_butterworth_lowpass(cutoff_hz=1000, sample_rate_hz=20000, order=4)
    freq, mag_db = frequency_response(sos, sample_rate_hz=20000, n_points=2000)
    # Well above cutoff, attenuation should be substantial.
    idx_far_above = np.argmin(np.abs(freq - 5000))
    assert mag_db[idx_far_above] < -20


def test_lowpass_passes_frequencies_well_below_cutoff():
    sos = design_butterworth_lowpass(cutoff_hz=1000, sample_rate_hz=20000, order=4)
    freq, mag_db = frequency_response(sos, sample_rate_hz=20000, n_points=2000)
    idx_low = np.argmin(np.abs(freq - 10))
    assert mag_db[idx_low] > -1  # near 0 dB, i.e. passed through


def test_lowpass_cutoff_raises_above_nyquist():
    try:
        design_butterworth_lowpass(cutoff_hz=15000, sample_rate_hz=20000)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_bandpass_rejects_outside_band():
    sos = design_butterworth_bandpass(low_hz=1000, high_hz=2000, sample_rate_hz=20000, order=4)
    freq, mag_db = frequency_response(sos, sample_rate_hz=20000, n_points=4000)
    idx_below = np.argmin(np.abs(freq - 100))
    idx_above = np.argmin(np.abs(freq - 8000))
    idx_center = np.argmin(np.abs(freq - 1500))
    assert mag_db[idx_below] < -20
    assert mag_db[idx_above] < -20
    assert mag_db[idx_center] > -3  # passband center, near 0 dB


def test_frequency_response_matches_scipy_reference_directly():
    """Not just 'did it run' -- confirm this project's wrapper produces
    the exact same numbers scipy's own sosfreqz gives when called
    directly, so the wrapper isn't silently transforming anything.

    frequency_response() deliberately clips |h| to 1e-12 before taking
    log10 (a -240 dB floor) so extremely attenuated stopband points
    never produce -inf/NaN in a report; the same clip is applied here
    to the reference before comparing, since that clip is intentional
    behavior being tested, not an artifact to work around."""
    sos = design_butterworth_lowpass(cutoff_hz=500, sample_rate_hz=10000, order=6)
    freq, mag_db = frequency_response(sos, sample_rate_hz=10000, n_points=256)

    w_ref, h_ref = scipy_signal.sosfreqz(sos, worN=256, fs=10000)
    mag_db_ref = 20 * np.log10(np.clip(np.abs(h_ref), 1e-12, None))

    assert np.allclose(freq, w_ref)
    assert np.allclose(mag_db, mag_db_ref, atol=1e-6)


def test_apply_filter_improves_snr_on_synthetic_noisy_tone():
    """The end-to-end signal-processing demonstration: a clean tone
    with injected out-of-band noise, filtered, actually measured to
    have improved SNR -- not asserted, computed."""
    t, clean, noisy, fs = noisy_baseband_measurement(
        tone_hz=2000, noise_band_hz=(40000, 60000), sample_rate_hz=200000
    )
    snr_before = signal_to_noise_ratio_db(clean, noisy)

    sos = design_butterworth_lowpass(cutoff_hz=10000, sample_rate_hz=fs, order=6)
    filtered = apply_filter(sos, noisy)
    snr_after = signal_to_noise_ratio_db(clean, filtered)

    assert snr_after > snr_before + 10  # a real, substantial improvement


def test_apply_filter_is_zero_phase():
    """sosfiltfilt should not introduce a time shift -- verified by
    checking the filtered clean-ish low-frequency tone stays aligned
    with the original, not just 'looks similar'."""
    t, clean, _, fs = noisy_baseband_measurement(tone_hz=100, sample_rate_hz=20000, duration_s=0.05)
    sos = design_butterworth_lowpass(cutoff_hz=5000, sample_rate_hz=fs, order=4)
    filtered = apply_filter(sos, clean)
    # Cross-correlation peak should be at (near) zero lag if there's no phase shift.
    corr = np.correlate(filtered - filtered.mean(), clean - clean.mean(), mode="full")
    lag = np.argmax(corr) - (len(clean) - 1)
    assert abs(lag) <= 1
