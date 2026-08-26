import numpy as np

from src.transmission_line import (
    CableModel,
    insertion_loss_db,
    reflection_coefficient,
    return_loss_db,
    transform_reflection_to_input,
    vswr,
)

LOSSLESS = {"R": 0.0, "L": 2.5e-7, "G": 0.0, "C": 1.0e-10}
LOSSY = {"R": 0.02, "L": 2.5e-7, "G": 5e-7, "C": 1.0e-10}


def test_lossless_line_characteristic_impedance_is_real_sqrt_L_over_C():
    model = CableModel(**LOSSLESS)
    freqs = np.array([1e6, 100e6, 1e9, 5e9])
    z0 = model.characteristic_impedance(freqs)
    expected = np.sqrt(LOSSLESS["L"] / LOSSLESS["C"])
    assert np.allclose(z0.real, expected, rtol=1e-10)
    assert np.allclose(z0.imag, 0.0, atol=1e-8)


def test_lossless_line_propagation_constant_is_purely_imaginary():
    model = CableModel(**LOSSLESS)
    freqs = np.array([1e6, 1e9])
    gamma = model.propagation_constant(freqs)
    assert np.allclose(gamma.real, 0.0, atol=1e-6)  # alpha = 0, no attenuation
    expected_beta = 2 * np.pi * freqs * np.sqrt(LOSSLESS["L"] * LOSSLESS["C"])
    assert np.allclose(gamma.imag, expected_beta, rtol=1e-8)


def test_lossy_line_has_nonzero_attenuation():
    model = CableModel(**LOSSY)
    gamma = model.propagation_constant(np.array([1e9]))
    assert gamma.real[0] > 0  # real lines attenuate


def test_matched_load_gives_zero_reflection_and_vswr_one():
    model = CableModel(**LOSSLESS)
    freq = np.array([1e9])
    z0 = model.characteristic_impedance(freq)
    gamma = reflection_coefficient(z0, z0)  # ZL == Z0
    assert np.allclose(np.abs(gamma), 0.0, atol=1e-10)
    assert np.allclose(vswr(gamma), 1.0, atol=1e-8)


def test_short_circuit_reflection_coefficient_is_minus_one():
    z0 = np.array([50.0 + 0j])
    gamma = reflection_coefficient(z0, np.array([0.0 + 0j]))
    assert np.allclose(gamma, -1.0)


def test_open_circuit_reflection_coefficient_approaches_plus_one():
    z0 = np.array([50.0 + 0j])
    gamma = reflection_coefficient(z0, np.array([1e9 + 0j]))
    assert np.allclose(np.abs(gamma), 1.0, atol=1e-6)
    assert gamma.real[0] > 0.999


def test_input_impedance_equals_z0_for_matched_load_at_any_length():
    """A classic transmission-line property: a matched load reflects
    nothing, so the line's input impedance equals Z0 regardless of
    line length."""
    model = CableModel(**LOSSLESS)
    freq = np.array([2.4e9])
    z0 = model.characteristic_impedance(freq)[0]
    for length in [0.1, 0.5, 1.3, 2.77]:
        zin = model.input_impedance(freq, length, z0)[0]
        assert np.isclose(zin, z0, rtol=1e-9)


def test_quarter_wave_transformer_property():
    """Textbook result: for a lossless line, at a length equal to a
    quarter wavelength, Zin = Z0^2 / ZL."""
    model = CableModel(**LOSSLESS)
    freq = 1e9
    z0 = model.characteristic_impedance(np.array([freq]))[0]
    v_phase = 1.0 / np.sqrt(LOSSLESS["L"] * LOSSLESS["C"])
    wavelength = v_phase / freq
    quarter_wave_length = wavelength / 4.0

    z_load = 25.0 + 0j
    zin = model.input_impedance(np.array([freq]), quarter_wave_length, z_load)[0]
    expected = z0 ** 2 / z_load
    assert np.isclose(zin, expected, rtol=1e-6)


def test_vswr_matches_hand_computed_value():
    # Gamma = 0.2 -> VSWR = (1+0.2)/(1-0.2) = 1.5
    gamma = np.array([0.2 + 0j])
    assert np.isclose(vswr(gamma)[0], 1.5)


def test_return_loss_db_matches_hand_computed_value():
    # |Gamma| = 0.1 -> RL = -20*log10(0.1) = 20 dB
    gamma = np.array([0.1 + 0j])
    assert np.isclose(return_loss_db(gamma)[0], 20.0)


def test_return_loss_increases_as_match_improves():
    gammas = np.array([0.5, 0.1, 0.01])
    rl = return_loss_db(gammas)
    assert rl[0] < rl[1] < rl[2]


def test_insertion_loss_zero_for_unity_transmission():
    s21 = np.array([1.0 + 0j])
    assert np.isclose(insertion_loss_db(s21)[0], 0.0, atol=1e-9)


def test_insertion_loss_matches_hand_computed_value():
    # |S21| = 0.5 -> IL = -20*log10(0.5) ~= 6.02 dB
    s21 = np.array([0.5 + 0j])
    assert np.isclose(insertion_loss_db(s21)[0], 6.0206, atol=1e-3)


def test_transform_reflection_to_input_matches_direct_computation():
    model = CableModel(**LOSSY)
    freq = np.array([1e9])
    z0 = model.characteristic_impedance(freq)
    gamma_prop = model.propagation_constant(freq)
    length = 0.75
    z_load = 30.0 + 5j

    gamma_load = reflection_coefficient(z0, np.array([z_load]))
    gamma_in_via_transform = transform_reflection_to_input(gamma_load, gamma_prop, length)

    zin = model.input_impedance(freq, length, np.array([z_load]))
    gamma_in_direct = reflection_coefficient(z0, zin)

    assert np.allclose(gamma_in_via_transform, gamma_in_direct, rtol=1e-6)
