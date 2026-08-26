import numpy as np

from src.touchstone import read_touchstone, write_touchstone_1port, write_touchstone_2port


def test_1port_round_trip_exact(tmp_path):
    freq_hz = np.array([1e6, 100e6, 1e9, 2.4e9, 5.8e9])
    rng = np.random.default_rng(0)
    s11 = rng.uniform(-0.9, 0.9, size=5) + 1j * rng.uniform(-0.9, 0.9, size=5)

    path = tmp_path / "test.s1p"
    write_touchstone_1port(path, freq_hz, s11, z0=50.0)
    data = read_touchstone(path)

    assert data["n_ports"] == 1
    assert data["z0"] == 50.0
    assert np.allclose(data["freq_hz"], freq_hz, rtol=1e-9)
    assert np.allclose(data["s11"], s11, rtol=1e-6)


def test_2port_round_trip_exact(tmp_path):
    freq_hz = np.linspace(1e9, 6e9, 10)
    rng = np.random.default_rng(1)
    s11 = rng.uniform(-0.5, 0.5, 10) + 1j * rng.uniform(-0.5, 0.5, 10)
    s21 = rng.uniform(0.1, 0.9, 10) + 1j * rng.uniform(-0.2, 0.2, 10)
    s12 = s21 * 0.98  # nearly reciprocal, as a real passive cable would be
    s22 = rng.uniform(-0.5, 0.5, 10) + 1j * rng.uniform(-0.5, 0.5, 10)

    path = tmp_path / "test.s2p"
    write_touchstone_2port(path, freq_hz, s11, s21, s12, s22, z0=50.0)
    data = read_touchstone(path)

    assert data["n_ports"] == 2
    assert np.allclose(data["freq_hz"], freq_hz, rtol=1e-9)
    assert np.allclose(data["s11"], s11, rtol=1e-6)
    assert np.allclose(data["s21"], s21, rtol=1e-6)
    assert np.allclose(data["s12"], s12, rtol=1e-6)
    assert np.allclose(data["s22"], s22, rtol=1e-6)


def test_frequency_unit_conversion_ghz(tmp_path):
    freq_hz = np.array([1e9, 2e9, 3e9])
    s11 = np.array([0.1 + 0j, 0.2 + 0j, 0.3 + 0j])

    path = tmp_path / "test_ghz.s1p"
    write_touchstone_1port(path, freq_hz, s11, freq_unit="GHZ")

    with open(path) as f:
        content = f.read()
    assert "# GHZ S RI R 50" in content
    assert "1 0.1 0" in content  # 1 GHz written as "1", not "1000000000"

    data = read_touchstone(path)
    assert np.allclose(data["freq_hz"], freq_hz)


def test_comment_lines_are_ignored(tmp_path):
    path = tmp_path / "manual.s1p"
    path.write_text(
        "! this is a hand-written comment\n"
        "! another comment\n"
        "# HZ S RI R 50\n"
        "! comment after the option line too\n"
        "1000000 0.1 0.05\n"
        "2000000 0.2 -0.05\n"
    )
    data = read_touchstone(path)
    assert data["n_ports"] == 1
    assert len(data["freq_hz"]) == 2
    assert np.isclose(data["s11"][0], 0.1 + 0.05j)


def test_missing_option_line_raises(tmp_path):
    path = tmp_path / "bad.s1p"
    path.write_text("1000000 0.1 0.05\n")
    try:
        read_touchstone(path)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_non_ri_format_raises_not_implemented(tmp_path):
    path = tmp_path / "ma.s1p"
    path.write_text("# HZ S MA R 50\n1000000 0.5 30\n")
    try:
        read_touchstone(path)
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass
