import numpy as np
import qutip as qt

from scqubits.core.hilbert_space import HilbertSpace
from scqubits.core.param_sweep import ParameterSweep

from chencrafts.cqed.mode_assignment import single_mode_dressed_esys
from chencrafts.cqed.states_n_oprts import oprt_in_basis

from typing import List, Tuple, Literal
import warnings

# ##############################################################################
def n_crit_by_diag(
    hilbertspace: HilbertSpace,
    res_mode_idx: int,
    state_label: Tuple[int] | List[int],
    dressed_indices: np.ndarray | None = None,
) -> int:
    """
    It returns the maximum n (aka critical photon number)
    that making the overlap between a dressed state (labeled by (n, ...))
    with its corresponding bare state larger than a threshold. 

    Parameters
    ----------
    hilberspace:
        scq.HilberSpace object which include the desired mode for calculating n_crit
    res_mode_idx:
        The index of the resonator mode in the hilberspace's subsystem_list
    state_label:
        n_crit is calculated with other modes staying at bare state. 
        For example, assume we want to evaluate the n_crit for the first mode in a three 
        mode system, we can set state_label to be (<any number>, 0, 1), indicating the 
        n_crit with other two modes at state (0, 1). 
    dressed_indeces: 
        An array mapping the FLATTENED bare state label to the eigenstate label. Usually
        given by `ParameterSweep["dressed_indices"]`.

    Returns
    -------
    Critical photon number as requested

    Note
    ----
    To match this n_crit with the analytical method, remember to set 
    scq.settings.OVERLAP_THRESHOLD = 0.853 before sweeping
    """

    # Use a dummy esys is enough to get a n_crit
    dummy_esys = np.zeros((2, hilbertspace.dimension - 1))

    result_dummy_evals, _ = single_mode_dressed_esys(
        hilbertspace,
        res_mode_idx,
        state_label,
        dressed_indices,
        dummy_esys,
        adjust_phase=False,     # as we input a dummy esys, we don't need to adjust phase
    )

    return len(result_dummy_evals)

def sweep_n_crit_by_diag(
    ps: ParameterSweep, paramindex_tuple, paramvals_tuple, 
    res_mode_idx: int,
    state_label: Tuple[int] | List[int],
) -> int:
    """
    It's a function for ParameterSweep.add_sweep

    It returns the maximum n that making the overlap between a dressed state (labeled by (n, ...))
    with its corresponding bare state larger than a threshold. 

    Keyword Arguments
    -----------------
    res_mode_idx:
        The index of the resonator mode in the hilberspace's subsystem_list
    state_label:
        n_crit is calculated with other modes staying at bare state. Put any number for 
        the resonator mode. 
        For example, assume we want to evaluate the n_crit for the first mode in a three 
        mode system, we can set state_label to be (10, 0, 1), indicating the n_crit with 
        other two modes at state (0, 1)

    Note
    ----
    To match this n_crit with the analytical method, remember to set 
    scq.settings.OVERLAP_THRESHOLD = 0.853 before sweeping
    """
    if ps._evals_count < ps.hilbertspace.dimension - 1:
        warnings.warn("The n_crit may not reach the max possible number (oscillator."
                      "truncated_dim), because only "
                      f"{ps._evals_count} eigenstates are calculated.", Warning)

    dressed_indices = ps["dressed_indices"][paramindex_tuple]
    
    n_crit = n_crit_by_diag(
        ps.hilbertspace,
        res_mode_idx,
        state_label,
        dressed_indices,
    )

    return n_crit

# ##############################################################################
def n_crit_1st(detuning, mat_elem, scaling=1):
    return detuning**2 / np.abs(mat_elem)**2 / 4 * scaling

def n_crit_2nd(detuning_1, mat_elem_1, detuning_2, mat_elem_2, scaling=1):
    """
    detuning and mat elem should corresponding to the same two bare qubit states
    detuning 1 and mat elem 1 should related to the unperturbed state
    """
    return np.abs(
        (detuning_1 + detuning_2) * (detuning_1) 
        / 4 / mat_elem_1 / mat_elem_2
    ) * scaling

def n_crit_by_pert_1st(
    hilbertspace: HilbertSpace,
):
    pass

def sweep_n_crit_by_pert(
    ps: ParameterSweep, paramindex_tuple, paramvals_tuple, 
    qubit_mode_idx: int, res_mode_idx: int, 
    interaction_idx: int, qubit_idx_in_interaction: int,
):
    # detuning
    qubit_evals = ps["bare_evals"][qubit_mode_idx][paramindex_tuple]
    qubit_evecs = ps["bare_evecs"][qubit_mode_idx][paramindex_tuple]

    res_evals = ps["bare_evals"][res_mode_idx][paramindex_tuple]
    res_freq = res_evals[1] - res_evals[0]

    detuning_0 = np.abs(qubit_evals - qubit_evals[0]) - res_freq
    detuning_1 = np.abs(qubit_evals - qubit_evals[1]) - res_freq

    # mat elem
    res_mode_in_interaction = 1 - qubit_idx_in_interaction

    interaction = ps.hilbertspace.interaction_list[interaction_idx]

    # interaction.operator_list[res_mode_in_interaction][1] is the matrix
    res_oprt = interaction.operator_list[res_mode_in_interaction][1]
    qubit_oprt = interaction.operator_list[qubit_idx_in_interaction][1]
    mat_elem_factor = interaction.g_strength * res_oprt[0, 1]

    qubit_operator = oprt_in_basis(qubit_oprt, qubit_evecs.transpose())

    n_crit = n_crit_1st(
        np.array([detuning_0, detuning_1]), 
        qubit_operator[0:2, :] * mat_elem_factor, 
        scaling=1, 
    )

    n_crit[0, 0] = np.inf
    n_crit[1, 1] = np.inf

    return n_crit
