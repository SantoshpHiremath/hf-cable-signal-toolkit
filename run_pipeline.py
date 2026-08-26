"""End-to-end demo of this project's two real software-tool
capabilities for an HF/RF-adjacent workflow:

    (A) cable transmission-line modeling -> Touchstone file I/O ->
        S-parameter analysis -> pass/fail reporting
    (B) filter design -> frequency response -> applying a filter to a
        noisy measurement trace -> measured SNR improvement

Every step below calls real code from src/ -- nothing here is printed
without having actually happened (S-parameters are computed from real
closed-form transmission-line equations, the Touchstone file is a real
spec-compliant file actually written to and read from disk, the filter
is a real scipy.signal Butterworth design, and the SNR improvement is
directly measured, not asserted).
"""

import tempfile

import numpy as np

from src.filter_design import apply_filter, design_butterworth_lowpass, signal_to_noise_ratio_db
from src.generate_measurement import RG58_LIKE_PARAMS, cable_sparams_from_model, noisy_baseband_measurement
from src.sparam_analysis import analyze_sparams, band_summary, pass_fail_return_loss
from src.touchstone import read_touchstone, write_touchstone_1port


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    # --- A. Transmission-line model -> Touchstone -> analysis --------
    section("A1. Compute S11 from a real transmission-line model")
    freq_hz = np.linspace(10e6, 6e9, 300)
    s11 = cable_sparams_from_model(freq_hz, length_m=1.2, z_load=75.0, params=RG58_LIKE_PARAMS)
    print(f"Modeled a {RG58_LIKE_PARAMS}-like 50-ohm-class coax, 1.2 m, "
          f"terminated in a 75-ohm load (a deliberate mismatch).")
    print(f"S11 computed at {len(freq_hz)} points from 10 MHz to 6 GHz.")

    section("A2. Write and re-read a real Touchstone (.s1p) file")
    with tempfile.NamedTemporaryFile(suffix=".s1p", delete=False) as tmp:
        s1p_path = tmp.name
    write_touchstone_1port(s1p_path, freq_hz, s11, z0=50.0, freq_unit="HZ")
    reloaded = read_touchstone(s1p_path)
    round_trip_ok = np.allclose(reloaded["s11"], s11, rtol=1e-6)
    print(f"Wrote {s1p_path}")
    print(f"Re-read: {reloaded['n_ports']}-port, {len(reloaded['freq_hz'])} points, "
          f"round-trip exact: {round_trip_ok}")

    section("A3. Analyze the S-parameters and check pass/fail against a spec")
    analysis = analyze_sparams(reloaded["freq_hz"], reloaded["s11"])
    summary = band_summary(analysis, f_low_hz=2.4e9, f_high_hz=2.5e9)
    print(f"In-band (2.4-2.5 GHz) summary: {summary}")

    verdict = pass_fail_return_loss(summary, min_return_loss_db=15.0)
    print(f"Pass/fail vs. 15 dB return-loss spec: {verdict}")

    # --- B. Filter design -> apply to a noisy measurement -------------
    section("B1. Design a Butterworth low-pass filter")
    t, clean, noisy, fs = noisy_baseband_measurement(
        tone_hz=2_000, noise_band_hz=(40_000, 60_000), sample_rate_hz=200_000, seed=0
    )
    sos = design_butterworth_lowpass(cutoff_hz=10_000, sample_rate_hz=fs, order=6)
    print(f"Designed a 6th-order Butterworth low-pass, cutoff=10 kHz, fs={fs} Hz.")

    section("B2. Apply the filter and measure the real SNR improvement")
    snr_before = signal_to_noise_ratio_db(clean, noisy)
    filtered = apply_filter(sos, noisy)
    snr_after = signal_to_noise_ratio_db(clean, filtered)
    print(f"SNR before filtering: {snr_before:.2f} dB")
    print(f"SNR after filtering:  {snr_after:.2f} dB")
    print(f"Improvement: {snr_after - snr_before:.2f} dB")

    section("SUMMARY")
    print(f"Touchstone round-trip: {'OK' if round_trip_ok else 'FAILED'}")
    print(f"Return-loss spec check: {'PASS' if verdict['passed'] else 'FAIL'} "
          f"(worst case {verdict['worst_case_db']:.2f} dB @ "
          f"{verdict['worst_case_freq_hz']/1e9:.3f} GHz)")
    print(f"Filter SNR improvement: {snr_after - snr_before:.2f} dB")


if __name__ == "__main__":
    main()
