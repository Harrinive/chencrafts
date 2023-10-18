import numpy as np
from scipy.constants import (
    h, hbar, pi, e, 
)


def capacitance_2_EC(C):
    """
    Give capacitance in fF, return charging energy in GHz.

    Charging energy EC = e^2 / (2C)
    """
    return e**2 / (2 * C * 1e-15) / h / 1e9

def EC_2_capacitance(EC):
    """
    Give charging energy in GHz, return capacitance in fF

    Charging energy EC = e^2 / (2C)
    """
    return e**2 / (2 * h * EC * 1e9) / 1e-15

def EL_2_inductance(EL):
    """
    Give EL in GHz, return inductance in pH

    Inductive energy, coefficient of (phi - phi_ext)^2, 
    EL = 1 / (2L) * Phi_0^2 / (2 pi)^2. Flux quantum Phi_0 = h / (2e)
    """
    Phi_0 = h / (2 * e)
    return Phi_0**2 / (2 * pi)**2 / (2 * h * EL * 1e9) / 1e-12

def inductance_2_EL(L):
    """
    Give inductance in pH, return EL in GHz

    Inductive energy, coefficient of (phi - phi_ext)^2, 
    EL = 1 / (2L) * Phi_0^2 / (2 pi)^2. Flux quantum Phi_0 = h / (2e)
    """
    Phi_0 = h / (2 * e)
    return Phi_0**2 / (2 * pi)**2 / (2 * L * 1e-12) / h / 1e9

def EC_EL_2_omega_Z(EC, EL):
    """
    Give EC and EL in GHz, return oscillation frequency in GHz and
    impedence in ohms

    freq = 1 / sqrt(LC) / (2 pi)
    Z = sqrt(L / C)
    """
    C = EC_2_capacitance(EC) * 1e-15
    L = EL_2_inductance(EL) * 1e-12
    
    freq = 1 / np.sqrt(L * C) / 2 / pi / 1e9
    Z = np.sqrt(L / C)

    return freq, Z

def omega_Z_2_EC_EL(freq, Z):
    """
    Give oscillation frequency in GHz and impedence in ohms, return
    EC and EL in GHz

    L = Z / (freq * 2 pi)
    C = L / Z^2
    """
    L = Z / (freq * 2 * pi * 1e9)
    C = L / Z**2

    EC = capacitance_2_EC(C)
    EL = inductance_2_EL(L)

    return EC, EL



