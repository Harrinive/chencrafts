import numpy as np

from scipy.special import erfc
from scipy.constants import h, k

from typing import Tuple

def n_th(freq, temp, n_th_base: float | np.ndarray = 0.0):
    """freq is in the unit of GHz, temp is in the unit of K"""
    return 1 / (np.exp(freq * h * 1e9 / temp / k) - 1) + n_th_base

def readout_error(disp, relax_rate, int_time) -> float | np.ndarray:
    SNR = 2 * np.abs(disp) * np.sqrt(relax_rate * int_time)
    return 0.5 * erfc(SNR / 2)

def qubit_addi_energy_relax_w_res(
    qubit_relax_rate: float | np.ndarray, 
    qubit_deph_rate: float | np.ndarray,
    g_over_delta: float | np.ndarray, 
    readout_photon_num: float | np.ndarray, 
    n_crit: float | np.ndarray, 
    res_relax_rate: float | np.ndarray, 
) -> Tuple[float | np.ndarray, float | np.ndarray]:
    """
    Following Boissonneault et al. (2009), equation (5.7) and (5.8).

    The returned value is a tuple of relaxation rate and excitation rate when the qubit 
    is coupled with a resonator with some photons in it. 
    
    Qubit natural relaxation rate is NOT included in the returned value.
    """
    # in the Equation (5.7), the "0" should be "1". The change here is to make the expression
    # exclude the qubit natural relaxation rate. 
    k_down_ro = (
        qubit_relax_rate * (0 - (readout_photon_num + 0.5) / 2 / n_crit)
        + g_over_delta**2 * res_relax_rate
        + 2 * g_over_delta**2 * qubit_deph_rate * (readout_photon_num + 1)
    )
    k_up_ro = 2 * g_over_delta**2 * qubit_deph_rate * readout_photon_num
    return k_down_ro, k_up_ro

def qubit_shot_noise_dephasing_w_res(
        res_relax_rate, chi, n_th_r,
        drive_strength = 0.0, drive_detuning = 0.0,
    ) -> float | np.ndarray:
    """
    Follow Clerk and Utami (2007), Equation (43), (44), (66) and (69).
    """
    # Equation (44) depahsing rate without drive
    Gamma_phi_th = res_relax_rate / 2 * (np.sqrt(
        (1 + 2j * chi / res_relax_rate)**2 + 8j * chi * n_th_r / res_relax_rate
    ) - 1).real

    if drive_strength != 0.0:
        # Equation (43) qubit frequency shift
        Delta_th = res_relax_rate / 2 * (np.sqrt(
            (1 + 2j * chi / res_relax_rate)**2 + 8j * chi * n_th_r / res_relax_rate
        )).imag
        # Equation (69) depahsing rate with drive
        gamma_th = res_relax_rate + 2 * Gamma_phi_th
        Gamma_phi_dr = (
            drive_strength**2 / 2 * chi * Delta_th * gamma_th 
            / ((drive_detuning + Delta_th)**2 + (gamma_th / 2)**2)
            / ((drive_detuning - Delta_th)**2 + (gamma_th / 2)**2)
        )

        Gamma_phi = Gamma_phi_th + Gamma_phi_dr
    else:
        Gamma_phi = Gamma_phi_th

    return Gamma_phi

