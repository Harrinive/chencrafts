import numpy as np
import sympy as sp
from scipy.constants import (
    h, hbar, pi, e, 
)
from IPython.display import display, Math

from typing import List


# unit conversion ======================================================
Phi_0 = h / (2 * e)

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
    return Phi_0**2 / (2 * pi)**2 / (h * EL * 1e9) / 1e-6

def EL_by_L(L):
    """
    Give inductance in uH, return EL in GHz

    Inductive energy, coefficient of 1/2 * (phi - phi_ext)^2, 
    EL = 1 / L * Phi_0^2 / (2 pi)^2. Flux quantum Phi_0 = h / (2e)
    """
    return Phi_0**2 / (2 * pi)**2 / (L * 1e-6) / h / 1e9

# oscillator ===========================================================
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
    return Phi_zpf / Phi_0 * 2 * np.pi

def Z_by_phi_zpf(phi_zpf):
    """
    For a resonator, give zero point fluctuation of flux in the unit of Phi_0 / 2pi,
    return impedence in ohms.
    When you have a oscillator length, divide by sqrt(2) first.
    """
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
    EJ_by_h = (I_crit * 1e-6) / (2 * pi) / (2 * e)
    return EJ_by_h * 1e-9

def CJ_by_area(area, J_crit):
    """ 
    For a junction, calculate the capacitance in fF from its area in um^2
    and critical current density in uA/um^2.
    
    Typically, a junction has a width of 0.2 um.
    """
    if J_crit == 0.2:
        C_density = 48 # fF/um^2
    elif J_crit == 1.0: 
        C_density = 53 # fF/um^2
    else:
        print(f'For, J_crit = {J_crit} uA/um^2, junction capacitance is '
              'not known')
        print('Attempt to approximate linearly (not a good one)')
        C_density = (
            (53 - 48)
            / (1.0 - 0.2)
            * (J_crit - 0.2)
            + 48 #fF/um^2
        )
    return area * C_density

def LJ_by_area(area, J_crit, Phi_ext=0):
    """
    For a junction, calculate the effective inductance in uH from its area 
    in um^2, critical current density in uA/um^2, external flux in Phi_0 / 2PI. 
    The effective inductance is external flux dependent.
    
    To get EJ, use function EJ_by_I_crit.
    
    Typically, a junction has a width of 0.2 um.
    """
    return (
        Phi_0 
        / ((2 * np.pi * J_crit * 1e-6 * area) * np.cos(Phi_ext)) 
        * 1e6 # in uH
    )

# display ==============================================================
def display_expr(expr: sp.Expr):
    """
    Display sympy expression in LaTeX format in a Jupyter notebook.
    """
    display(Math(sp.latex(expr)))
    
# math =================================================================
def mod_c(
    a, 
    b = np.pi * 2, 
    center = 0.0,
):
    """
    Modulo operation that always return a number in the range of 
    [-b/2 + center, b/2 + center).
    """
    return ((a - center) + b/2) % b - b/2 + center

def perturbative_inverse(
    M0: sp.Matrix | np.ndarray | np.matrix, 
    M1: sp.Matrix | np.ndarray | np.matrix, 
    order: int = 2
) -> List[sp.Matrix | np.ndarray | np.matrix]:
    """
    Perturbative inverse of a matrix M0 + M1.
    
    Parameters
    ----------
    M0: sp.Matrix
        The matrix to be inverted.
    M1: sp.Matrix
        The perturbation matrix.
    order: int
        The order of the perturbative expansion.

    Returns
    -------
    List[sp.Matrix]
        The perturbative inverse of M0 + M1, order by order.
    """
    if isinstance(M0, np.ndarray | np.matrix):
        M0_inv = np.linalg.inv(M0)
    elif isinstance(M0, sp.Matrix):
        M0_inv = M0.inv()
    else:
        raise ValueError(f"Unsupported matrix type: {type(M0)}")
    result = [M0_inv]
    
    for i in range(1, order + 1):
        if isinstance(M0, np.ndarray | np.matrix):
            term = (-1)**i * (M0_inv @ M1)**i @ M0_inv
        elif isinstance(M0, sp.Matrix):
            term = (-1)**i * (M0_inv @ M1)**i @ M0_inv
        else:
            raise ValueError(f"Unsupported matrix type: {type(M0)}")
        
        result.append(term)
    
    return result