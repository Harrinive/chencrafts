import numpy as np
import qutip as qt
from scqubits.core.hilbert_space import HilbertSpace
from scipy.special import erfc
from scipy.constants import h, k

from typing import Tuple, Callable, List
from chencrafts.cqed.mode_assignment import single_mode_dressed_esys


def n_th(
    freq: float | np.ndarray, 
    temp: float | np.ndarray, 
    n_th_base: float | np.ndarray = 0.0
) -> float | np.ndarray:
    """
    Calculate the thermal occupation number of a mode at a given temperature.
    
    Parameters
    ----------
    freq: float | np.ndarray
        The frequency of the interested transition, in GHz
    temp: float | np.ndarray
        The temperature of the environment, in Kelvin
    n_th_base: float | np.ndarray
        Sometimes adding a constant to the thermal occupation number is necessary
        to fit the experimental data. This parameter is used to do so.

    Returns
    -------
    float | np.ndarray
        Thermal occupation number
    
    """
    return 1 / (np.exp(freq * h * 1e9 / temp / k) - 1) + n_th_base

def readout_error(
    n: float | np.ndarray, 
    relax_rate: float | np.ndarray,
    int_time: float | np.ndarray,
) -> float | np.ndarray:
    """
    Calculate the readout error probability of a qubit with a given dispersive shift,
    relaxation rate and integration time.

    Parameters
    ----------
    n: float | np.ndarray
        The average resonator photon number during readout
    relax_rate: float | np.ndarray
        The relaxation rate of the resonator, in GHz
    int_time: float | np.ndarray
        The integration time of the readout, in ns

    Returns
    -------
    float | np.ndarray
        Readout error probability
    """

    SNR = 2 * np.abs(n) * np.sqrt(relax_rate * int_time)
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

    Parameters
    ----------
    qubit_relax_rate: float | np.ndarray
        Qubit relaxation rate, in GHz
    qubit_deph_rate: float | np.ndarray
        Qubit dephasing rate, in GHz
    g_over_delta: float | np.ndarray
        Coupling strength devided by detuning
    readout_photon_num: float | np.ndarray
        Number of photons in the resonator
    n_crit: float | np.ndarray
        Critical photon number of the resonator
    res_relax_rate: float | np.ndarray
        Resonator relaxation rate, in GHz

    Returns
    -------
    Tuple[float | np.ndarray, float | np.ndarray]
        Readout introduced relaxation rate and excitation rate
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

    The returned value is the dephasing rate of a qubit coupled with a resonator
    when the resonator is excited by a the environment and a drive.

    Parameters
    ----------
    res_relax_rate: float | np.ndarray
        Resonator relaxation rate, in GHz
    chi: float | np.ndarray
        Dispersice shift of the qubit-resonator system, in GHz
    n_th_r: float | np.ndarray  
        Thermal occupation number of the resonator when not driven
    drive_strength: float | np.ndarray
        Drive strength of the resonator, in GHz
    drive_detuning: float | np.ndarray
        Drive detuning of the resonator, in GHz

    Returns
    -------
    float | np.ndarray
        Dephasing rate of the qubit

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

def purcell_factor(
    hilbertspace: HilbertSpace,
    res_mode_idx: int, qubit_mode_idx: int, 
    res_state_func: Callable | int = 0, qubit_state_index: int = 0,
    collapse_op_list: List[qt.Qobj] = [],
    dressed_indices: np.ndarray | None = None, eigensys = None,
    **kwargs
) -> List[float]:
    """
    It returns some factors between two mode: osc and qubit, in order to 
    calculate the decay rate of the state's occupation probability. The returned numbers, 
    say, n_osc and n_qubit, can be used in this way:  
     - state's decay rate = n_osc * osc_decay_rate + n_qubit * qubit_decay_rate

    Parameters
    ----------
    osc_mode_idx, qubit_mode_idx:
        The index of the two modes in the hilberspace's subsystem_list
    osc_state_func, qubit_state_index:
        The purcell decay rate of a joint system when the joint state can be described by 
        some bare product state of osc and B. Those two arguments can be an integer
        (default, 0), indicating a bare fock state. Additionally, A_state_func can
        also be set to a function with signature `osc_state_func(<some basis of osc mode>, 
        **kwargs)`. Such a fuction should check the validation of the basis, and raise a
        RuntimeError if invalid.
    dressed_indeces: 
        An array mapping the FLATTENED bare state label to the eigenstate label. Usually
        given by `ParameterSweep["dressed_indices"]`.
    eigensys: 
        The eigensystem for the hilbertspace.
    collapse_op_list:
        If empty, the purcell factors will be evaluated assuming the collapse operators
        are osc mode's annilation operator and qubit mode's sigma_minus operator. Otherwise,
        will calculate the factors using operators specified by the user by:
         - factor = qutip.expect(operator, state) for all operators
    kwargs:
        kwyword arguments for osc_state_func

    Returns
    -------
    Purcell factors
    """

    # obtain collapse operators
    if collapse_op_list == []:
        collapse_op_list = [
            hilbertspace.annihilate(hilbertspace.subsystem_list[res_mode_idx]),
            hilbertspace.hubbard_operator(0, 1, hilbertspace.subsystem_list[qubit_mode_idx])
        ]

    # Construct the desired state
    state_label = np.zeros_like(hilbertspace.subsystem_dims, dtype=int)
    state_label[qubit_mode_idx] = qubit_state_index

    _, osc_evecs = single_mode_dressed_esys(
        hilbertspace,
        res_mode_idx,
        tuple(state_label),
        dressed_indices,
        eigensys,
    )
    try: 
        if callable(res_state_func):
            state = res_state_func(osc_evecs, **kwargs)
        else:
            state = osc_evecs[res_state_func]
    except (RuntimeError, IndexError):
        # if the state is invalid
        return [np.nan] * len(collapse_op_list)
        
    # calculate expectation value of collapse operators
    factor_list = []
    for op in collapse_op_list:
        factor_list.append(
            qt.expect(op.dag() * op, state)
        )

    return factor_list
