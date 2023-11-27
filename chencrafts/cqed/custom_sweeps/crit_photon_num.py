import numpy as np
import qutip as qt

import scqubits as scq
from scqubits.core.hilbert_space import HilbertSpace
from scqubits.core.param_sweep import ParameterSweep

from chencrafts.cqed.mode_assignment import single_mode_dressed_esys
from chencrafts.cqed.crit_photon_num import (
    n_crit_by_diag, 
    n_crit_by_1st_pert_w_hilbertspace,
    n_crit_by_diag_subspace_w_hilbertspace,
)
from typing import List, Tuple, Literal
import warnings


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
                      f"{ps._evals_count} eigenstates are calculated.\n", Warning)

    dressed_indices = ps["dressed_indices"][paramindex_tuple]
    
    n_crit = n_crit_by_diag(
        ps.hilbertspace,
        res_mode_idx,
        state_label,
        dressed_indices,
    )

    return n_crit


def sweep_n_crit_by_1st_pert(
    ps: ParameterSweep, paramindex_tuple, paramvals_tuple, 
    qubit_mode_idx: int, res_mode_idx: int, 
    interaction_idx: int, qubit_idx_in_interaction: int,
    qubit_state_label: int,
):
    """
    It's a function for ParameterSweep.add_sweep

    """

    qubit_evals = ps["bare_evals"][qubit_mode_idx][paramindex_tuple]
    qubit_evecs = ps["bare_evecs"][qubit_mode_idx][paramindex_tuple]

    return n_crit_by_1st_pert_w_hilbertspace(
        ps.hilbertspace, 
        qubit_mode_idx, res_mode_idx,
        qubit_state_label,
        interaction_idx, 
        qubit_idx_in_interaction,
        qubit_bare_esys=(qubit_evals, qubit_evecs),
    )


def sweep_n_crit_by_diag_subspace(
    ps: ParameterSweep, paramindex_tuple, paramvals_tuple, 
    qubit_mode_idx_list: List[int], res_mode_idx: int, 
    interaction_idx_list, qubit_idx_in_interaction_list: List[int],
    state_label: List[int],
    res_state_range: int = 2,
    qubit_state_range_list: List[int] | None = None,
    max_detuning: float | None = None,
    overlap_threshold: float | None = None,
    print_hybridization: bool = False,
    max_n_crit: int = 5000,
) -> int:
    """
    It's a function for ParameterSweep.add_sweep
    """    
    qubit_bare_esys_list = []
    for qubit_mode_idx in qubit_mode_idx_list:
        qubit_bare_evals = ps["bare_evals"][qubit_mode_idx][paramindex_tuple]
        qubit_bare_evecs = ps["bare_evecs"][qubit_mode_idx][paramindex_tuple]
        qubit_bare_esys_list.append((qubit_bare_evals, qubit_bare_evecs))

    n_crit = n_crit_by_diag_subspace_w_hilbertspace(
        hilbertspace = ps.hilbertspace, 
        qubit_mode_idx_list = qubit_mode_idx_list,
        res_mode_idx=res_mode_idx,
        interaction_idx_list=interaction_idx_list, 
        qubit_idx_in_interaction_list=qubit_idx_in_interaction_list,
        state_label=state_label,
        qubit_bare_esys_list = None,
        res_state_range = res_state_range,
        qubit_state_range_list = qubit_state_range_list,
        max_detuning = max_detuning,
        overlap_threshold = overlap_threshold,
        print_hybridization=print_hybridization,
        max_n_crit=max_n_crit,
    )

    return n_crit