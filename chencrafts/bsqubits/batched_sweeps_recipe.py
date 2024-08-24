import numpy as np
import qutip as qt
import scipy as sp
import scqubits as scq
from scqubits.core.param_sweep import ParameterSweep
from scqubits.core.namedslots_array import NamedSlotsNdarray

from chencrafts.bsqubits.batched_custom_sweeps import (
    batched_sweep_general, batched_sweep_pulse
)
from chencrafts.cqed.decoherence import thermal_ratio, thermal_factor
from chencrafts.bsqubits import cat_ideal as cat_ideal
from chencrafts.cqed.qt_helper import oprt_in_basis
from chencrafts.cqed.mode_assignment import two_mode_dressed_esys
from chencrafts.bsqubits.cat_recipe import get_jump_ops

from typing import List, Tuple, Any, Dict, Callable

# When considering dephasing, we only consider the zero-frequency transition
# because the noise spectral density is 1/f.
TPHI_HI_FREQ_CUTOFF = 1e-8

# When considering depolarization, we limit the lowest frequency, or the 
# thermal factor blows up.
T1_LO_FREQ_CUTOFF = 1e-2

# Step1: Transform the system-bath coupling operators ##################
# ######################################################################
# Requirements: 
# 1. Two modes: cavity and qubit. 
# 2. Dimension of the cavity: smaller than the original sweep by at least 1,
#    Dimension of the qubit: 2, 

def sweep_organized_evecs(
    sweep: ParameterSweep, idx, 
    res_mode_idx = 0, qubit_mode_idx = 1,
    res_trunc_dim = 5,
    qubit_trunc_dim = 2,
):
    """
    Obtain a list of eigenvectors ordered by bare labels.
    """
    hilbertspace = sweep.hilbertspace

    evals, evecs = two_mode_dressed_esys(
        hilbertspace=hilbertspace,
        res_mode_idx=res_mode_idx, qubit_mode_idx=qubit_mode_idx,
        state_label=(-1, -1),
        res_truncated_dim=res_trunc_dim, 
        qubit_truncated_dim=qubit_trunc_dim,
        dressed_indices=sweep["dressed_indices"][idx],
        eigensys=(sweep["evals"][idx], sweep["evecs"][idx]),
        adjust_phase=True,
    )
    
    if np.prod(evecs.shape) < res_trunc_dim * qubit_trunc_dim:
        print(evecs.shape)
        raise ValueError(
            "The truncated dimension is too large, please make it smaller."
        )
    
    return evecs.ravel()

def sweep_dressed_qubit_op(
    sweep: ParameterSweep, idx,
    res_mode_idx = 0,
    qubit_mode_idx = 1, 
    res_trunc_dim = 5,
    qubit_trunc_dim = 2,
    operator_name = "n_operator"
):
    """
    Transform a qubit operator from the bare basis to the dressed basis.
    operator_name can be an attribute of the qubit (like "n_operator"), 
    or a projector (like "proj_11")
    
    Sweep must contains org_evecs key, from sweep_organized_evecs function.    
    """
    hilbertspace = sweep.hilbertspace
    qubit = hilbertspace.subsystem_list[qubit_mode_idx]
    
    # obtain and transform the operator
    if operator_name.startswith("proj_"):
        op = hilbertspace.hubbard_operator(
            j = int(operator_name[5]),
            k = int(operator_name[6]),
            subsystem = qubit,
        )
    else:
        op = scq.identity_wrap(
            operator = operator_name,
        subsystem = qubit,
        subsys_list = hilbertspace.subsystem_list,
    )
    op_dressed = oprt_in_basis(op, sweep["org_evecs"][idx])
    
    # change it to the desired dimension
    dim = hilbertspace.subsystem_dims
    dim[qubit_mode_idx] = qubit_trunc_dim
    dim[res_mode_idx] = res_trunc_dim

    return qt.Qobj(op_dressed, dims=[dim, dim])

def sweep_dressed_res_op(
    sweep: ParameterSweep, idx, 
    res_mode_idx = 0, 
    qubit_mode_idx = 1,
    res_trunc_dim = 5,
    qubit_trunc_dim = 2,
    operator_name = "a_m_adag",
):
    """
    Transform a resonator operator from the bare basis to the dressed basis.
    operator_name can be "a_m_adag" or "adag_a".
    
    Sweep must contains org_evecs key, from sweep_organized_evecs function.
    """
    hilbertspace = sweep.hilbertspace
    res = hilbertspace.subsystem_list[res_mode_idx]
    
    # obtain and transform the operator
    a_op = hilbertspace.annihilate(res)
    
    if operator_name == "a_m_adag":
        op = 1j * a_op - 1j * a_op.dag()
    elif operator_name == "adag_a":
        op = a_op.dag() * a_op
    else:
        raise ValueError(f"Invalid resonator operator name: {operator_name}")

    op_dressed = oprt_in_basis(op, sweep["org_evecs"][idx])

    # change it to the desired dimension
    dim = hilbertspace.subsystem_dims
    dim[res_mode_idx] = res_trunc_dim
    dim[qubit_mode_idx] = qubit_trunc_dim

    return qt.Qobj(
        op_dressed, 
        dims=[dim, dim]
    )

def batched_sweep_dressed_op(
    sweep: ParameterSweep, 
    res_mode_idx = 0, qubit_mode_idx = 1,
    res_trunc_dim = 5,
    qubit_trunc_dim = 2,
    qubit_op_names: List[str] = [],
    res_op_names: List[str] = [],
    **kwargs
):
    """
    A batch of add_sweeps for calculating dressed operators.
    
    Accepted qubit operators (keys of the qubit_spectral_density):
    - <some>_operator
    - proj_{i}{j}
    
    Accepted resonator operators (keys of the res_spectral_density):
    - a_m_adag
    - adag_a
    
    It will add the following sweeps:
    - org_evecs: the organized eigenvectors of the system.
    - qubit_{op_name}: the dressed qubit operators.
    - res_amadag: the dressed (a - adag) operator.
    """    
    
    sweep.add_sweep(
        sweep_organized_evecs, 
        sweep_name="org_evecs",
        res_mode_idx=res_mode_idx, qubit_mode_idx=qubit_mode_idx,
        res_trunc_dim=res_trunc_dim,
        qubit_trunc_dim=qubit_trunc_dim,
    )
    for op_name in qubit_op_names:
        sweep.add_sweep(
            sweep_dressed_qubit_op, 
            sweep_name=f"qubit_{op_name}",
            qubit_mode_idx=qubit_mode_idx,
            res_mode_idx=res_mode_idx,
            res_trunc_dim=res_trunc_dim,
            qubit_trunc_dim=qubit_trunc_dim,
            operator_name=op_name,
        )
    for op_name in res_op_names:
        sweep.add_sweep(
            sweep_dressed_res_op, 
            sweep_name=f"res_{op_name}",
            res_mode_idx=res_mode_idx,
            qubit_mode_idx=qubit_mode_idx,
            res_trunc_dim=res_trunc_dim,
            qubit_trunc_dim=qubit_trunc_dim,
            operator_name=op_name,
        )

# Step2: Extract the jump operators from the system-bath coupling operators
# and compute the associated rates
# #########################################################################
def ohmic_spectral_density(omega, T, Q = 10e6):
    """
    Return the ohmic spectral density that is linearly dependent on frequency.
    
    Parameters
    ----------
    omega: float | np.ndarray
        The frequency of the noise, GHz.
    T: float | np.ndarray
        The temperature of the noise, K.
    Q: float | np.ndarray
        The quality factor.

    Returns
    -------
    float | np.ndarray
        The ohmic spectral density.
    """
    # add a cutoff to avoid zero division
    omega = np.where(np.abs(omega) < T1_LO_FREQ_CUTOFF, T1_LO_FREQ_CUTOFF, omega)
    
    therm_factor = thermal_factor(omega, T)
    return np.pi * 2 * np.abs(omega) / Q * therm_factor

def q_cap_fun(omega, T, Q_cap = 1e6):
    """
    See Smith et al (2020). Return the capacitive noise's spectral density.
    
    Parameters
    ----------
    omega: float
        The frequency of the noise, GHz.
    T: float
        The temperature of the noise, K.
    Q_cap: float
        The quality factor of the capacitor.

    Returns
    -------
    float
        The capacitive noise's spectral density.
    """
    return (
        Q_cap
        * (6 / np.abs(omega)) ** 0.7
    )

def cap_spectral_density(omega, T, EC, Q_cap = 1e6):
    """
    Return the capacitive noise's spectral density.
    
    Parameters
    ----------
    omega: float
        The frequency of the noise, GHz.
    T: float
        The temperature of the noise, K.
    EC: float
        The charging energy of the qubit.
    Q_cap: float
        The quality factor of the capacitor.

    Returns
    -------
    float
        The capacitive noise's spectral density.
    """
    # add a cutoff to avoid zero division
    omega = np.where(np.abs(omega) < T1_LO_FREQ_CUTOFF, T1_LO_FREQ_CUTOFF, omega)
    
    therm_factor = thermal_factor(omega, T)
    s = (
        2
        * 8
        * EC
        / q_cap_fun(omega, T, Q_cap)
        * therm_factor
    )
    s *= (
        2 * np.pi
    )  # We assume that system energies are given in units of frequency
    return s

def q_ind_fun(omega, T, Q_ind = 500e6):
    """
    Return the inductor's quality factor.
    
    Parameters
    ----------
    omega: float
        The frequency of the noise, GHz.
    T: float
        The temperature of the noise, K.
    Q_ind: float
        The quality factor of the inductor.

    Returns
    -------
    float
        The inductor's quality factor.
    """
    therm_ratio = abs(thermal_ratio(omega, T))
    therm_ratio_500MHz = thermal_ratio(0.5, T)
    return (
        Q_ind
        * (
            sp.special.kv(0, 1 / 2 * therm_ratio_500MHz)
            * np.sinh(1 / 2 * therm_ratio_500MHz)
        )
        / (
            sp.special.kv(0, 1 / 2 * therm_ratio)
            * np.sinh(1 / 2 * therm_ratio)
        )
    )

def ind_spectral_density(omega, T, EL, Q_ind = 500e6):
    """
    Return the inductive noise's spectral density.
    
    Parameters
    ----------
    omega: float
        The frequency of the noise, GHz.
    T: float
        The temperature of the noise, K.
    EL: float
        The inductive energy of the qubit, GHz.
    Q_ind: float
        The quality factor of the inductor.

    Returns
    -------
    float
        The inductive noise's spectral density.
    """
    # add a cutoff to avoid zero division
    omega = np.where(np.abs(omega) < T1_LO_FREQ_CUTOFF, T1_LO_FREQ_CUTOFF, omega)
    
    therm_factor = thermal_factor(omega, T)
    s = (
        2
        * EL
        / q_ind_fun(omega, T, Q_ind)
        * therm_factor
    )
    s *= (
        2 * np.pi
    )  # We assume that system energies are given in units of frequency
    return s

def delta_spectral_density(omega, peak_value, peak_loc = 0, peak_width = 1e-10):
    """
    Obtain a delta function spectral density. It's used to model the dephasing
    noise, which have a 1/f spectrum. We assume it dies down very quickly so
    any non-zero frequency will have a zero spectral density.
    
    Parameters
    ----------
    omega: float
        The frequency of the noise, GHz.
    peak_value: float
        The value of the delta function.
    peak_loc: float
        The location of the delta function.
    peak_width: float
        The width of the delta function, for numrical purposes.

    Returns
    -------
    float
        The delta function spectral density.
    """
    if np.allclose(omega, peak_loc, atol=peak_width):
        return peak_value
    else:
        return 0

def sweep_spectral_density(
    sweep: ParameterSweep,
    idx,
    qubit_mode_idx = 1,
    qubit_op_names: List[str] = [],
    res_op_names: List[str] = [],
    **kwargs
):
    """
    Add the spectral density functions to the sweep.
    
    Required keys in either sweep.parameters or kwargs
    - qubit:
        - temp_a        (a stands for ancilla, unit: K)
        - Q_ind
        - Q_cap
        - kappa_a_phi   (dephasing rate, unit: ns-1)
    - resonator:
        - temp_s        (s stands for system)
        - Q_s
        - kappa_s_phi   (dephasing rate, unit: ns-1)
    """
    params = {
        key: val[idx] for key, val in sweep.parameters.meshgrid_by_name().items()
    } | kwargs
    qubit = sweep.hilbertspace.subsystem_list[qubit_mode_idx]
    
    spec_dens = {}
    for op in qubit_op_names:
        if op == "n_operator":
            EC = qubit.EC
            func = lambda freq: cap_spectral_density(
                freq, T = params["temp_a"], EC = EC, Q_cap = params["Q_cap"]
            )
        elif op == "phi_operator":
            EL = qubit.EL
            func = lambda freq: ind_spectral_density(
                freq, T = params["temp_a"], EL = EL, Q_ind = params["Q_ind"]
            )
        elif op == "proj_11":
            func = lambda freq: delta_spectral_density(
                freq, peak_value = 2 * params["kappa_a_phi"], peak_loc = 0, peak_width = TPHI_HI_FREQ_CUTOFF
            )
        else:
            raise ValueError(f"Invalid qubit operator name: {op}")
        spec_dens[f"qubit_{op}"] = func
    
    for op in res_op_names:
        if op == "a_m_adag":
            func = lambda freq: ohmic_spectral_density(
                freq, T = params["temp_s"], Q = params["Q_s"]
            )
        elif op == "adag_a":
            func = lambda freq: delta_spectral_density(
                freq, peak_value = params["kappa_s_phi"], peak_loc = 0, peak_width = TPHI_HI_FREQ_CUTOFF
            )
        else:
            raise ValueError(f"Invalid resonator operator name: {op}")
        spec_dens[f"res_{op}"] = func
        
    return spec_dens


# Just a note here, for a qubit, coefficient_of_proj_11 = 2 --> dephasing rate = 1
# to see this, run the following code:

# import qutip as qt
# import chencrafts.toolbox as tb
# liouv = qt.liouvillian(
#     c_ops = [qt.projection(2, 1, 1) * np.sqrt(1)]
# )
# prop = lambda t: (liouv * t).expm()
# init_state = qt.ket2dm((qt.basis(2, 1) + qt.basis(2, 0)).unit())
# t_array = np.linspace(0, 10, 100)
# pop1 = []
# for t in t_array:
#     pop1.append(qt.expect(init_state.dag() * init_state, cqed.superop_evolve(prop(t), init_state)))
# print(tb.decay_rate(t_array, pop1, extract_envelope=False))
    

def ravel_res_qubit_index(
    res_mode_idx = 0, qubit_mode_idx = 1,
    res_trunc_dim = 5, qubit_trunc_dim = 2,
    res_state = 0, qubit_state = 0,
) -> int:
    """
    Ravel the resonator and qubit indices into a single index. 
    """
    if res_mode_idx == 0 and qubit_mode_idx == 1:
        return np.ravel_multi_index(
            (res_state, qubit_state), 
            (res_trunc_dim, qubit_trunc_dim)
        )
    elif res_mode_idx == 1 and qubit_mode_idx == 0:
        return np.ravel_multi_index(
            (qubit_state, res_state), 
            (qubit_trunc_dim, res_trunc_dim)
        )
    raise ValueError(f"Invalid mode indices: {res_mode_idx}, {qubit_mode_idx}")

def calc_mat_elem(
    op: qt.Qobj,
    res_mode_idx = 0, qubit_mode_idx = 1,
    res_trunc_dim = 5, qubit_trunc_dim = 2,
    res_state_1 = 0, qubit_state_1 = 0,
    res_state_2 = 1, qubit_state_2 = 0,
    **kwargs
) -> float:
    """
    Calculate matrix element <state1|op|state2>, which is associated with
    the jump operator |state1><state2|.
    """
    idx1 = ravel_res_qubit_index(
        res_mode_idx, qubit_mode_idx, res_trunc_dim, qubit_trunc_dim, 
        res_state_1, qubit_state_1
    )
    idx2 = ravel_res_qubit_index(
        res_mode_idx, qubit_mode_idx, res_trunc_dim, qubit_trunc_dim, 
        res_state_2, qubit_state_2
    )
    
    # matrix element
    mat_elem = op[idx1, idx2]
    
    return mat_elem
    
def calc_freq_diff(
    sweep: ParameterSweep, idx,
    res_mode_idx = 0, qubit_mode_idx = 1,
    res_state_1 = 0, qubit_state_1 = 0,
    res_state_2 = 1, qubit_state_2 = 0,
    **kwargs
) -> float:
    """
    Calculate the frequency difference between two dressed states using 
    the bare indices.
    """
    res = sweep.hilbertspace.subsystem_list[res_mode_idx]
    qubit = sweep.hilbertspace.subsystem_list[qubit_mode_idx]
    
    # note that we don't use "res_trunc_dim" and "qubit_trunc_dim" here
    # because evals are still organized before organization & truncation
    res_dim = res.truncated_dim
    qubit_dim = qubit.truncated_dim
    
    idx1 = ravel_res_qubit_index(
        res_mode_idx, qubit_mode_idx, 
        res_dim, qubit_dim, 
        res_state_1, qubit_state_1
    )
    idx2 = ravel_res_qubit_index(
        res_mode_idx, qubit_mode_idx, 
        res_dim, qubit_dim, 
        res_state_2, qubit_state_2
    )
    
    # frequency difference
    drs_idx1 = sweep["dressed_indices"][idx][idx1]
    drs_idx2 = sweep["dressed_indices"][idx][idx2]
    omega_1 = sweep["evals"][idx][drs_idx1]
    omega_2 = sweep["evals"][idx][drs_idx2]
    omega_diff = omega_2 - omega_1
    
    return omega_diff

def compute_rate_by_a_m_adag(
    sweep: ParameterSweep,
    res_mode_idx = 0,
    qubit_mode_idx = 1,
    res_trunc_dim = 5,
    qubit_trunc_dim = 2,
    **kwargs
):
    """
    From a transformed a-adag operator, calculate the rates for jump operators
    appears in second-order perturbative expansion of the system-bath interaction:
    - a
    - adag
    - qubit |i><j|
    
    It performs loop over parameters, not supposed to work with add_sweep.
    
    Up to the second order of the dispersive approximation, the jump operators
    are a, adag and qubit ij.
    """

    for idx in np.ndindex(sweep.parameters.counts):
        try:
            spectral_density = sweep["spectral_density"][idx]
            spec_dens = spectral_density["res_a_m_adag"]
        except KeyError:
            return
        sys_bath_inter_op = sweep["res_a_m_adag"][idx]
        # set <00|op|00> to 0
        sys_bath_inter_op = (
            sys_bath_inter_op 
            - qt.qeye_like(sys_bath_inter_op) * sys_bath_inter_op[0, 0]
        )

        # a 
        state_kwargs = dict(
            res_mode_idx = res_mode_idx, qubit_mode_idx = qubit_mode_idx,
            res_trunc_dim = res_trunc_dim, qubit_trunc_dim = qubit_trunc_dim,
            res_state_1 = 0, qubit_state_1 = 0,
            res_state_2 = 1, qubit_state_2 = 0,
        )
        a_mat_elem = calc_mat_elem(
            sys_bath_inter_op, **state_kwargs
        )
        a_freq_diff = calc_freq_diff(
            sweep, idx, **state_kwargs
        )
        jump_a = spec_dens(a_freq_diff) * np.abs(a_mat_elem)**2
        sweep[f"jump_a"][idx] = sweep[f"jump_a"][idx] + jump_a

        # adag
        state_kwargs = dict(
            res_mode_idx = res_mode_idx, qubit_mode_idx = qubit_mode_idx,
            res_trunc_dim = res_trunc_dim, qubit_trunc_dim = qubit_trunc_dim,
            res_state_1 = 1, qubit_state_1 = 0,
            res_state_2 = 0, qubit_state_2 = 0,
        )
        adag_mat_elem = calc_mat_elem(
            sys_bath_inter_op, **state_kwargs
        )
        adag_freq_diff = calc_freq_diff(
            sweep, idx, **state_kwargs
        )
        jump_adag = spec_dens(adag_freq_diff) * np.abs(adag_mat_elem)**2
        sweep[f"jump_adag"][idx] = sweep[f"jump_adag"][idx] + np.real(jump_adag)
        
        # qubit |i><j|
        jump_q_matelems = np.zeros(
            (qubit_trunc_dim, qubit_trunc_dim),
            dtype=complex
        )
        for i, j in np.ndindex(qubit_trunc_dim, qubit_trunc_dim):
            state_kwargs = dict(
                res_mode_idx = res_mode_idx, qubit_mode_idx = qubit_mode_idx,
                res_trunc_dim = res_trunc_dim, qubit_trunc_dim = qubit_trunc_dim,
                res_state_1 = 0, qubit_state_1 = i,
                res_state_2 = 0, qubit_state_2 = j,
            )
            q_mat_elem = calc_mat_elem(
                sys_bath_inter_op, **state_kwargs
            )
            q_freq_diff = calc_freq_diff(
                sweep, idx, **state_kwargs
            )
            jump_q_matelems[i, j] = spec_dens(q_freq_diff) * np.abs(q_mat_elem)**2
            
        # <00|op|00> should already be 0, so the following is not needed
        # q_matelems = q_matelems - np.eye(qubit_trunc_dim) * q_matelems[0, 0]
        sweep[f"jump_ij"][idx] = sweep[f"jump_ij"][idx] + np.real(jump_q_matelems)
    
def compute_rate_by_adag_a(
    sweep: ParameterSweep,
    res_mode_idx = 0,
    qubit_mode_idx = 1,
    res_trunc_dim = 5,
    qubit_trunc_dim = 2,
    **kwargs
):
    """
    From a transformed adag_a operator, calculate the rates for jump operators
    appears in second-order perturbative expansion of the system-bath interaction:
    - adag_a
    - a / adag
    - a / adag * |i><j|
    
    It performs looping over parameters, not supposed to work with add_sweep.
    
    Up to the second order of the dispersive approximation, the jump operators
    are adag_a and a/adag * qubit ij.
    """

    for idx in np.ndindex(sweep.parameters.counts):
        try:
            spectral_density = sweep["spectral_density"][idx]
            spec_dens = spectral_density["res_adag_a"]
        except KeyError:
            return
        
        sys_bath_inter_op = sweep["res_adag_a"][idx].copy()
        # set <00|op|00> to 0
        sys_bath_inter_op = (
            sys_bath_inter_op 
            - qt.qeye_like(sys_bath_inter_op) * sys_bath_inter_op[0, 0]
        )
        
        # adag_a 
        # Since we don't have adag_a * |i><j| and <00|op|00> is already 0, 
        # we just need to calculate one matrix element.
        state_kwargs = dict(
            res_mode_idx = res_mode_idx, qubit_mode_idx = qubit_mode_idx,
            res_trunc_dim = res_trunc_dim, qubit_trunc_dim = qubit_trunc_dim,
            res_state_1 = 1, qubit_state_1 = 0,
            res_state_2 = 1, qubit_state_2 = 0,
        )
        adag_a_mat_elem = calc_mat_elem(
            sys_bath_inter_op, **state_kwargs
        )
        adag_a_freq_diff = calc_freq_diff(
            sweep, idx, **state_kwargs
        )
        jump_adag_a = spec_dens(adag_a_freq_diff) * np.abs(adag_a_mat_elem)**2
        sweep[f"jump_adag_a"][idx] = sweep[f"jump_adag_a"][idx] + np.real(jump_adag_a)

        # a/adag * |i><j|
        a_q_matelems = np.zeros(
            (qubit_trunc_dim, qubit_trunc_dim),
            dtype=complex
        )
        a_q_freq_diffs = np.zeros(
            (qubit_trunc_dim, qubit_trunc_dim),
            dtype=complex
        )
        adag_q_matelems = np.zeros(
            (qubit_trunc_dim, qubit_trunc_dim),
            dtype=complex
        )
        adag_q_freq_diffs = np.zeros(
            (qubit_trunc_dim, qubit_trunc_dim),
            dtype=complex
        )
        for i, j in np.ndindex(qubit_trunc_dim, qubit_trunc_dim):
            a_q_state_kwargs = dict(
                res_mode_idx = res_mode_idx, qubit_mode_idx = qubit_mode_idx,
                res_trunc_dim = res_trunc_dim, qubit_trunc_dim = qubit_trunc_dim,
                res_state_1 = 0, qubit_state_1 = i,
                res_state_2 = 1, qubit_state_2 = j,
            )
            a_q_freq_diff = calc_freq_diff(
                sweep, idx, **a_q_state_kwargs
            )
            a_q_mat_elem = calc_mat_elem(
                sys_bath_inter_op, **a_q_state_kwargs
            )
            a_q_freq_diffs[i, j] = a_q_freq_diff
            a_q_matelems[i, j] = a_q_mat_elem
            
            adag_q_state_kwargs = dict(
                res_mode_idx = res_mode_idx, qubit_mode_idx = qubit_mode_idx,
                res_trunc_dim = res_trunc_dim, qubit_trunc_dim = qubit_trunc_dim,
                res_state_1 = 1, qubit_state_1 = i,
                res_state_2 = 0, qubit_state_2 = j,
            )
            adag_q_freq_diff = calc_freq_diff(
                sweep, idx, **adag_q_state_kwargs
            )
            adag_q_mat_elem = calc_mat_elem(
                sys_bath_inter_op, **adag_q_state_kwargs
            )
            adag_q_freq_diffs[i, j] = adag_q_freq_diff
            adag_q_matelems[i, j] = adag_q_mat_elem
        
        # identity component: a / adag
        jump_a = spec_dens(a_q_freq_diffs[0, 0]) * np.abs(a_q_matelems[0, 0])**2
        jump_adag = spec_dens(adag_q_freq_diffs[0, 0]) * np.abs(adag_q_matelems[0, 0])**2
        sweep[f"jump_a"][idx] = sweep[f"jump_a"][idx] + np.real(jump_a)
        sweep[f"jump_adag"][idx] = sweep[f"jump_adag"][idx] + np.real(jump_adag)
        
        # other: correlated jumps
        a_q_matelems = (
            a_q_matelems 
            - np.eye(qubit_trunc_dim) * a_q_matelems[0, 0]
        )
        adag_q_matelems = (
            adag_q_matelems 
            - np.eye(qubit_trunc_dim) * adag_q_matelems[0, 0]
        )
        sweep[f"jump_a_ij"][idx] = np.real(
            sweep[f"jump_a_ij"][idx] 
            + spec_dens(a_q_freq_diffs) * np.abs(a_q_matelems)**2
        )
        sweep[f"jump_adag_ij"][idx] = np.real(
            sweep[f"jump_adag_ij"][idx] 
            + spec_dens(adag_q_freq_diffs) * np.abs(adag_q_matelems)**2
        )
        
def compute_rate_by_qubit_ops(
    sweep: ParameterSweep,
    res_mode_idx = 0,
    qubit_mode_idx = 1,
    res_trunc_dim = 5,
    qubit_trunc_dim = 2,
    op_name: str = "n_operator",
    **kwargs
):
    """
    From a transformed qubit operator, calculate the rates for jump operators
    appears in second-order perturbative expansion of the system-bath interaction:
    - |i><j|
    - a / adag
    - a / adag * |i><j| 
        
    It performs loop over parameters, not supposed to work with add_sweep.
    
    Up to the second order of the dispersive approximation, the jump operators
    are adag_a and a/adag * qubit ij.
    """

    for idx in np.ndindex(sweep.parameters.counts):
        try:
            spectral_density = sweep["spectral_density"][idx]
            spec_dens = spectral_density[f"qubit_{op_name}"]
        except KeyError:
            return
    
        sys_bath_inter_op = sweep[f"qubit_{op_name}"][idx].copy()
        # set <00|op|00> to 0
        sys_bath_inter_op = (
            sys_bath_inter_op 
            - qt.qeye_like(sys_bath_inter_op) * sys_bath_inter_op[0, 0]
        )
        
        # qubit |i><j|
        jump_q_matelems = np.zeros(
            (qubit_trunc_dim, qubit_trunc_dim),
            dtype=complex
        )
        for i, j in np.ndindex(qubit_trunc_dim, qubit_trunc_dim):
            state_kwargs = dict(
                res_mode_idx = res_mode_idx, qubit_mode_idx = qubit_mode_idx,
                res_trunc_dim = res_trunc_dim, qubit_trunc_dim = qubit_trunc_dim,
                res_state_1 = 0, qubit_state_1 = i,
                res_state_2 = 0, qubit_state_2 = j,
            )
            q_mat_elem = calc_mat_elem(
                sys_bath_inter_op, **state_kwargs
            )
            q_freq_diff = calc_freq_diff(
                sweep, idx, **state_kwargs
            )
            jump_q_matelems[i, j] = spec_dens(q_freq_diff) * np.abs(q_mat_elem)**2
            
        # <00|op|00> should already be 0, so the following is not needed
        # q_matelems = q_matelems - np.eye(qubit_trunc_dim) * q_matelems[0, 0]
        sweep[f"jump_ij"][idx] = sweep[f"jump_ij"][idx] + np.real(jump_q_matelems)
        
        # a/adag * |i><j|
        a_q_matelems = np.zeros(
            (qubit_trunc_dim, qubit_trunc_dim),
            dtype=complex
        )
        a_q_freq_diffs = np.zeros(
            (qubit_trunc_dim, qubit_trunc_dim),
            dtype=complex
        )
        adag_q_matelems = np.zeros(
            (qubit_trunc_dim, qubit_trunc_dim),
            dtype=complex
        )
        adag_q_freq_diffs = np.zeros(
            (qubit_trunc_dim, qubit_trunc_dim),
            dtype=complex
        )
        for i, j in np.ndindex(qubit_trunc_dim, qubit_trunc_dim):
            a_q_state_kwargs = dict(
                res_mode_idx = res_mode_idx, qubit_mode_idx = qubit_mode_idx,
                res_trunc_dim = res_trunc_dim, qubit_trunc_dim = qubit_trunc_dim,
                res_state_1 = 0, qubit_state_1 = i,
                res_state_2 = 1, qubit_state_2 = j,
            )
            a_q_freq_diff = calc_freq_diff(
                sweep, idx, **a_q_state_kwargs
            )
            a_q_mat_elem = calc_mat_elem(
                sys_bath_inter_op, **a_q_state_kwargs
            )
            a_q_freq_diffs[i, j] = a_q_freq_diff
            a_q_matelems[i, j] = a_q_mat_elem
            
            adag_q_state_kwargs = dict(
                res_mode_idx = res_mode_idx, qubit_mode_idx = qubit_mode_idx,
                res_trunc_dim = res_trunc_dim, qubit_trunc_dim = qubit_trunc_dim,
                res_state_1 = 1, qubit_state_1 = i,
                res_state_2 = 0, qubit_state_2 = j,
            )
            adag_q_freq_diff = calc_freq_diff(
                sweep, idx, **adag_q_state_kwargs
            )
            adag_q_mat_elem = calc_mat_elem(
                sys_bath_inter_op, **adag_q_state_kwargs
            )
            adag_q_freq_diffs[i, j] = adag_q_freq_diff
            adag_q_matelems[i, j] = adag_q_mat_elem
        
        # identity component: a / adag
        jump_a = spec_dens(a_q_freq_diffs[0, 0]) * np.abs(a_q_matelems[0, 0])**2
        jump_adag = spec_dens(adag_q_freq_diffs[0, 0]) * np.abs(adag_q_matelems[0, 0])**2
        sweep[f"jump_a"][idx] = sweep[f"jump_a"][idx] + np.real(jump_a)
        sweep[f"jump_adag"][idx] = sweep[f"jump_adag"][idx] + np.real(jump_adag)
        
        # other: correlated jumps
        a_q_matelems = (
            a_q_matelems 
            - np.eye(qubit_trunc_dim) * a_q_matelems[0, 0]
        )
        adag_q_matelems = (
            adag_q_matelems 
            - np.eye(qubit_trunc_dim) * adag_q_matelems[0, 0]
        )
        sweep[f"jump_a_ij"][idx] = np.real(
            sweep[f"jump_a_ij"][idx] 
            + spec_dens(a_q_freq_diffs) * np.abs(a_q_matelems)**2
        )
        sweep[f"jump_adag_ij"][idx] = np.real(
            sweep[f"jump_adag_ij"][idx] 
            + spec_dens(adag_q_freq_diffs) * np.abs(adag_q_matelems)**2
        )
        
def batched_sweep_jump_rates(
    sweep: ParameterSweep,
    res_mode_idx = 0, qubit_mode_idx = 1,
    res_trunc_dim = 5,
    qubit_trunc_dim = 2,
    qubit_op_names: List[str] = [],
    res_op_names: List[str] = [],
    **kwargs
):
    zeros = np.zeros(
        sweep.parameters.counts
    )
    zeros_qij = np.zeros(
        sweep.parameters.counts + (qubit_trunc_dim, qubit_trunc_dim),
    )
    sweep.store_data(
        # resonator jumps
        jump_a = zeros.copy(),
        jump_adag = zeros.copy(),
        jump_adag_a = zeros.copy(),
        
        # qubit jumps
        jump_ij = zeros_qij.copy(), # depolarization + dephasing (11)
        
        # resonator-qubit jumps
        jump_a_ij = zeros_qij.copy(),
        jump_adag_ij = zeros_qij.copy(),
    )
    
    sweep.add_sweep(
        sweep_spectral_density,
        sweep_name = "spectral_density",
        qubit_mode_idx = qubit_mode_idx,
        qubit_op_names = qubit_op_names,
        res_op_names = res_op_names,
    )
    
    compute_rate_by_a_m_adag(
        sweep,
        res_mode_idx = res_mode_idx, qubit_mode_idx = qubit_mode_idx,
        res_trunc_dim = res_trunc_dim, qubit_trunc_dim = qubit_trunc_dim,
    )
    compute_rate_by_adag_a(
        sweep,
        res_mode_idx = res_mode_idx, qubit_mode_idx = qubit_mode_idx,
        res_trunc_dim = res_trunc_dim, qubit_trunc_dim = qubit_trunc_dim,
    )
    for op_name in qubit_op_names:
        compute_rate_by_qubit_ops(
            sweep,
            res_mode_idx = res_mode_idx, qubit_mode_idx = qubit_mode_idx,
            res_trunc_dim = res_trunc_dim, qubit_trunc_dim = qubit_trunc_dim,
            op_name = op_name,
        )
        
    # for fitting into `cat_tree.py`, we need to store some dict values 
    # with a different name
    sweep.store_data(
        kappa_s = sweep["jump_a"],
        K_s = sweep["kerr"][res_mode_idx, res_mode_idx] * np.pi * 2,
    )

# We can stop here and get the master equation and simulate it numerically

# Step 3: Compute the grouped jump propability
# #########################################################################
def sweep_jump_ops(
    sweep: ParameterSweep, idx,
    res_mode_idx = 0,
    res_trunc_dim = 5,
    qubit_trunc_dim = 2,
    **kwargs
):
    ops = get_jump_ops(
        res_mode_idx = res_mode_idx,
        res_trunc_dim = res_trunc_dim,
        qubit_trunc_dim = qubit_trunc_dim,
    )
    
    return ops


def sweep_1jump_prob(
    sweep: ParameterSweep, idx,
    **kwargs
):
    jump_ops = sweep["jump_ops"][idx]
    
    
    pass
    
    

def batched_sweep_jump_prob(
    sweep: ParameterSweep,
    res_mode_idx = 0,
    res_trunc_dim = 5,
    qubit_trunc_dim = 2,
    **kwargs
):
    sweep.add_sweep(
        sweep_jump_ops,
        sweep_name = "jump_ops",
        res_mode_idx = res_mode_idx,
        res_trunc_dim = res_trunc_dim,
        qubit_trunc_dim = qubit_trunc_dim,
    )
    
    sweep.add_sweep(
        sweep_1jump_prob,
        sweep_name = "1jump_prob",
    )