"""Real, closed-form RLGC transmission-line theory.

Every formula here is standard microwave-engineering transmission-line
theory (see e.g. Pozar, "Microwave Engineering", ch. 2) -- not a
simulation of a black box, but the actual textbook equations relating
a cable's per-unit-length electrical parameters (R, L, G, C) to its
frequency-dependent characteristic impedance, propagation constant,
and terminated-line behavior (input impedance, reflection coefficient,
VSWR, return loss). No real lab measurement or RF hardware is involved
anywhere in this module -- see the project README for that disclosure
in full. What's real here is the physics/math and its verification
against known closed-form special cases.
"""

import numpy as np


class CableModel:
    """A uniform transmission line characterized by per-unit-length
    parameters R (Ohm/m), L (H/m), G (S/m), C (F/m)."""

    def __init__(self, R, L, G, C):
        self.R = R
        self.L = L
        self.G = G
        self.C = C

    def propagation_constant(self, freq_hz):
        """gamma(f) = alpha(f) + j*beta(f) = sqrt((R + jwL)(G + jwC))"""
        omega = 2 * np.pi * np.asarray(freq_hz, dtype=np.float64)
        series_z = self.R + 1j * omega * self.L
        shunt_y = self.G + 1j * omega * self.C
        return np.sqrt(series_z * shunt_y)

    def characteristic_impedance(self, freq_hz):
        """Z0(f) = sqrt((R + jwL) / (G + jwC))"""
        omega = 2 * np.pi * np.asarray(freq_hz, dtype=np.float64)
        series_z = self.R + 1j * omega * self.L
        shunt_y = self.G + 1j * omega * self.C
        return np.sqrt(series_z / shunt_y)

    def input_impedance(self, freq_hz, length_m, z_load):
        """Zin looking into a line of given length terminated in z_load:

        Zin = Z0 * (ZL + Z0*tanh(gamma*l)) / (Z0 + ZL*tanh(gamma*l))
        """
        z0 = self.characteristic_impedance(freq_hz)
        gamma = self.propagation_constant(freq_hz)
        th = np.tanh(gamma * length_m)
        return z0 * (z_load + z0 * th) / (z0 + z_load * th)


def reflection_coefficient(z0, z_load):
    """Gamma = (ZL - Z0) / (ZL + Z0), evaluated at the load."""
    return (z_load - z0) / (z_load + z0)


def transform_reflection_to_input(gamma_load, gamma_prop, length_m):
    """Reflection coefficient transformed back to the line's input:
    Gamma_in = Gamma_L * exp(-2 * gamma * l)."""
    return gamma_load * np.exp(-2 * gamma_prop * length_m)


def vswr(gamma):
    """VSWR = (1 + |Gamma|) / (1 - |Gamma|)."""
    mag = np.abs(gamma)
    mag = np.clip(mag, 0, 1 - 1e-12)  # avoid divide-by-zero at |Gamma|=1
    return (1 + mag) / (1 - mag)


def return_loss_db(gamma):
    """Return loss in dB, using the convention RL_dB = -20*log10(|Gamma|)
    (a larger positive number means a better, more closely matched
    load -- the standard RF engineering sign convention)."""
    mag = np.abs(gamma)
    mag = np.clip(mag, 1e-12, None)  # avoid log(0)
    return -20 * np.log10(mag)


def insertion_loss_db(s21):
    """Insertion loss in dB from a complex S21 value/array:
    IL_dB = -20*log10(|S21|)."""
    mag = np.abs(np.asarray(s21))
    mag = np.clip(mag, 1e-12, None)
    return -20 * np.log10(mag)
