import numpy as np

from src.sparam_analysis import analyze_sparams, band_summary, pass_fail_return_loss


def test_analyze_sparams_return_loss_and_vswr():
    freq = np.array([1e9, 2e9, 3e9])
    s11 = np.array([0.1 + 0j, 0.05 + 0j, 0.3 + 0j])
    result = analyze_sparams(freq, s11)
    assert np.isclose(result["return_loss_db"][0], 20.0)
    assert np.isclose(result["vswr"][0], 1.2222222, atol=1e-5)


def test_analyze_sparams_includes_insertion_loss_when_s21_given():
    freq = np.array([1e9])
    s11 = np.array([0.1 + 0j])
    s21 = np.array([0.9 + 0j])
    result = analyze_sparams(freq, s11, s21)
    assert "insertion_loss_db" in result
    assert result["insertion_loss_db"][0] > 0


def test_analyze_sparams_omits_insertion_loss_without_s21():
    result = analyze_sparams(np.array([1e9]), np.array([0.1 + 0j]))
    assert "insertion_loss_db" not in result


def test_band_summary_finds_worst_case_correctly():
    freq = np.array([1e9, 2e9, 3e9, 4e9])
    s11 = np.array([0.05, 0.3, 0.02, 0.1]) + 0j  # worst match at 2 GHz
    analysis = analyze_sparams(freq, s11)
    summary = band_summary(analysis, 1e9, 4e9)
    assert summary["n_points_in_band"] == 4
    assert np.isclose(summary["worst_case_freq_hz"], 2e9)
    assert np.isclose(summary["return_loss_min_db"], analysis["return_loss_db"][1])


def test_band_summary_filters_to_the_requested_band():
    freq = np.array([1e9, 2e9, 3e9, 4e9, 5e9])
    s11 = np.full(5, 0.1 + 0j)
    analysis = analyze_sparams(freq, s11)
    summary = band_summary(analysis, 2e9, 4e9)
    assert summary["n_points_in_band"] == 3


def test_band_summary_raises_when_band_is_empty():
    freq = np.array([1e9, 2e9])
    s11 = np.array([0.1, 0.1]) + 0j
    analysis = analyze_sparams(freq, s11)
    try:
        band_summary(analysis, 10e9, 20e9)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_pass_fail_return_loss_passes_when_above_threshold():
    summary = {"return_loss_min_db": 18.0, "worst_case_freq_hz": 2.4e9}
    result = pass_fail_return_loss(summary, min_return_loss_db=15.0)
    assert result["passed"] is True
    assert np.isclose(result["margin_db"], 3.0)


def test_pass_fail_return_loss_fails_when_below_threshold():
    summary = {"return_loss_min_db": 8.0, "worst_case_freq_hz": 2.4e9}
    result = pass_fail_return_loss(summary, min_return_loss_db=15.0)
    assert result["passed"] is False
    assert result["margin_db"] < 0
