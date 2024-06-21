import numpy as np
from scipy.constants import (
    h, hbar, pi, e, 
)
from IPython.display import display, Math
from sympy import latex


# unit conversion ======================================================
def EC_by_C(C):
    """
    Give capacitance in fF, return charging energy in GHz.

    Charging energy EC = e^2 / (2C)
    """
    return e**2 / (2 * C * 1e-15) / h / 1e9

def C_by_EC(EC):
    """
    Give charging energy in GHz, return capacitance in fF

    Charging energy EC = e^2 / (2C)
    """
    return e**2 / (2 * h * EC * 1e9) / 1e-15

def L_by_EL(EL):
    """
    Give EL in GHz, return inductance in uH

    Inductive energy, coefficient of 1/2 * (phi - phi_ext)^2, 
    EL = 1 / L * Phi_0^2 / (2 pi)^2. Flux quantum Phi_0 = h / (2e)
    """
    Phi_0 = h / (2 * e)
    return Phi_0**2 / (2 * pi)**2 / (h * EL * 1e9) / 1e-6

def EL_by_L(L):
    """
    Give inductance in uH, return EL in GHz

    Inductive energy, coefficient of 1/2 * (phi - phi_ext)^2, 
    EL = 1 / L * Phi_0^2 / (2 pi)^2. Flux quantum Phi_0 = h / (2e)
    """
    Phi_0 = h / (2 * e)
    return Phi_0**2 / (2 * pi)**2 / (L * 1e-6) / h / 1e9

def omega_Z_by_EC_EL(EC, EL):
    """
    Give EC and EL in GHz, return oscillation frequency in GHz and
    impedence in ohms, where 
    EC is the charging energy, defined as e^2 / (2C), and 
    EL is the inductive energy, defined as a coeefficient of 1/2 * (phi - phi_ext)^2,

    We make use of the fact that the oscillation frequency is given by
    freq = 1 / sqrt(LC) / (2 pi), and the impedence is given by
    Z = sqrt(L / C)
    """
    C = C_by_EC(EC) * 1e-15
    L = L_by_EL(EL) * 1e-6
    
    freq = 1 / np.sqrt(L * C) / np.pi / 2 / 1e9
    Z = np.sqrt(L / C)

    return freq, Z

def EC_EL_by_omega_Z(freq, Z):
    """
    Give oscillation frequency in GHz and impedence in ohms, return
    EC and EL in GHz, where
    EC is the charging energy, defined as e^2 / (2C), and 
    EL is the inductive energy, defined as a coeefficient of 1/2 * (phi - phi_ext)^2,

    L = Z / (freq * 2 pi)
    C = L / Z^2
    """
    L = Z / (freq * 2 * np.pi * 1e9)
    C = L / Z**2

    EC = EC_by_C(C * 1e15)
    EL = EL_by_L(L * 1e6)

    return EC, EL

def phi_zpf_by_Z(Z):
    """
    For a resonator, give impedence in ohms, return zero point fluctuation of 
    flux in the unit of Phi_0 / 2pi. 
    To convert it to oscillator length, multiply by sqrt(2).
    """
    Phi_zpf = np.sqrt(hbar * Z / 2)
    Phi_0 = h / (2 * e)
    return Phi_zpf / Phi_0 * 2 * np.pi

def Z_by_phi_zpf(phi_zpf):
    """
    For a resonator, give zero point fluctuation of flux in the unit of Phi_0 / 2pi,
    return impedence in ohms.
    When you have a oscillator length, divide by sqrt(2) first.
    """
    Phi_0 = h / (2 * e)
    Phi_zpf = phi_zpf * Phi_0 / 2 / np.pi
    return 2 * Phi_zpf**2 / hbar

def n_zpf_by_Z(Z):
    """
    For a resonator, give impedence in ohms, return zero point fluctuation of 
    charge in the unit of 2e. 
    The relationship between n_zpf and oscillator length is n_zpf = 1 / (sqrt(2) l_zpf).
    """
    Q_zpf = np.sqrt(hbar / 2 / Z)
    return Q_zpf / 2 / e

def Z_by_n_zpf(n_zpf):
    """
    For a resonator, give zero point fluctuation of charge in the unit of 2e,
    return impedence in ohms.
    The relationship between n_zpf and oscillator length is n_zpf = 1 / (sqrt(2) l_zpf).
    """
    return hbar / (n_zpf * 2 * e)**2 / 2

# Josephson junctions ==================================================
def I_crit_by_EJ(EJ):
    """
    Convert EJ in GHz to critical current in uA.
    
    The relationship between EJ and critical current is given by
    EJ = hbar * I_crit / 2e
    """
    I_crit = (EJ * 1e9) * (2 * pi) * (2 * e)
    return I_crit * 1e6

def EJ_by_I_crit(I_crit):
    """
    Convert critical current in uA to EJ in GHz.
    
    The relationship between EJ and critical current is given by
    EJ = hbar * I_crit / 2e
    """
    EJ = (I_crit * 1e-6) / (2 * pi) / (2 * e)
    return EJ * 1e9

# display ==============================================================
def display_expr(expr):
    """
    Display sympy expression in LaTeX format in a Jupyter notebook.
    """
    display(Math(latex(expr)))
    
# math =================================================================
def mod_c(a, b):
    """
    Modulo operation that always return a number in the range of [-b/2, b/2).
    """
    return (a + b/2) % b - b/2