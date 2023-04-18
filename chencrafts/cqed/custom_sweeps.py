import numpy as np
import scqubits as scq
from scqubits.core.hilbert_space import HilbertSpace
from scqubits.core.param_sweep import ParameterSweep

import qutip as qt

from chencrafts.cqed.scq_helper import label_convert

from typing import List, Tuple, Callable
import warnings
import inspect

# ##############################################################################
def single_mode_dressed_esys(
    hilbertspace: HilbertSpace,
    mode_idx: int,
    state_label: Tuple[int] | List[int],
    dressed_indices: np.ndarray | None = None,
    eigensys = None,
):
    """
    It returns a subset of eigenenergies and dressed states with one of the bare labels 
    varying and the rest fixed. 
    
    For example, we are looking for eigensystem for the first 
    mode in a three mode system with the rest of two modes fixed at bare state 0 and 1, 
    we can set state_label to be (<any number>, 0, 1).

    Parameters
    ----------
    hilberspace:
        scq.HilberSpace object which include the desired mode
    mode_idx:
        The index of the interested mode in the hilberspace's subsystem_list
    state_label:
        the subset of the eigensys is calculated with other modes staying at bare state. 
        For example, we are looking for eigensystem for the first 
        mode in a three mode system with the rest of two modes fixed at bare state 0 and 1, 
        we can set state_label to be (<any number>, 0, 1).
    dressed_indeces: 
        An array mapping the FLATTENED bare state label to the eigenstate label. Usually
        given by `ParameterSweep["dressed_indices"]`.

    Returns
    -------
    A subset of eigensys with one of the bare labels varying and the rest fixed. 
    """
    if dressed_indices is None:
        hilbertspace.generate_lookup()
        drs_idx_map = hilbertspace.dressed_index
    else:
        def drs_idx_map(bare_index_tuple):
            flattened_bare_index = label_convert(bare_index_tuple, hilbertspace)
            return dressed_indices[flattened_bare_index]
        
    if eigensys is None:
        evals, evecs = hilbertspace.eigensys(hilbertspace.dimension - 1)
    else:
        evals, evecs = eigensys

    sm_evecs = []
    sm_evals = []

    dim_list = hilbertspace.subsystem_dims
    dim_res = dim_list[mode_idx]
    bare_index = np.array(state_label).copy()
    for n in range(dim_res):
        bare_index[mode_idx] = n
        drs_idx = drs_idx_map(tuple(bare_index))
        if drs_idx is None:
            break
        sm_evecs.append(evecs[drs_idx])
        sm_evals.append(evals[drs_idx])

    return (sm_evals, sm_evecs)

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

    dummy_esys = np.zeros((2, hilbertspace.dimension - 1))

    result_dummy_evals, result_dummy_evecs = single_mode_dressed_esys(
        hilbertspace,
        res_mode_idx,
        state_label,
        dressed_indices,
        dummy_esys,
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

def n_crit_by_pert(
    hilbertspace: HilbertSpace,
    res_mode_idx: int,
    state_label: Tuple[int] | List[int],
    dressed_indices: np.ndarray | None = None,
):


# ##############################################################################
def sweep_convergence(
    paramsweep: ParameterSweep, paramindex_tuple, paramvals_tuple, mode_idx
):
    bare_evecs = paramsweep["bare_evecs"]["subsys": mode_idx][paramindex_tuple]
    return np.max(np.abs(bare_evecs[-3:, :]))

# ##############################################################################
def purcell_factor(
    hilbertspace: HilbertSpace,
    osc_mode_idx: int, qubit_mode_idx: int, 
    osc_state_func: Callable | int = 0, qubit_state_index: int = 0,
    collapse_op_list: List[qt.Qobj] = [],
    dressed_indices: np.ndarray | None = None, eigensys = None,
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

    Returns
    -------
    Purcell factors
    """

    # obtain collapse operators
    if collapse_op_list == []:
        collapse_op_list.append(
            hilbertspace.annihilate(hilbertspace.subsystem_list[osc_mode_idx]))
        collapse_op_list.append(
            hilbertspace.hubbard_operator(0, 1, hilbertspace.subsystem_list[qubit_mode_idx]))

    # Construct the desired state
    state_label = np.zeros_like(hilbertspace.subsystem_dims, dtype=int)
    state_label[qubit_mode_idx] = qubit_state_index

    _, osc_evecs = single_mode_dressed_esys(
        hilbertspace,
        osc_mode_idx,
        tuple(state_label),
        dressed_indices,
        eigensys,
    )
    try: 
        if callable(osc_state_func):
            state = osc_state_func(osc_evecs)
        else:
            state = osc_evecs[osc_state_func]
    except RuntimeError | IndexError:
        # if the state is invalid
        return [np.nan] * len(collapse_op_list)
        
    # calculate expectation value of collapse operators
    factor_list = []
    for op in collapse_op_list:
        factor_list.append(
            qt.expect(op.dag() * op, state)
        )

    return factor_list

def sweep_purcell_factor(
    ps: ParameterSweep, paramindex_tuple, paramvals_tuple, 
    osc_mode_idx: int, qubit_mode_idx: int, 
    osc_state_func: Callable | int = 0, qubit_state_index: int = 0,
    collapse_op_list: List[qt.Qobj] = [],
) -> np.ndarray:
    """
    It returns some factors between two mode: osc and qubit, in order to 
    calculate the decay rate of the state's occupation probability. The returned numbers, 
    say, n_osc and n_qubit, can be used in this way:  
     - state's decay rate = n_osc * osc_decay_rate + n_qubit * qubit_decay_rate

    Keyword Arguments
    -----------------
    osc_mode_idx, qubit_mode_idx:
        The index of the two modes in the hilberspace's subsystem_list
    osc_state_func, qubit_state_index:
        The purcell decay rate of a joint system when the joint state can be described by 
        some bare product state of osc and B. Those two arguments can be an integer
        (default, 0), indicating a bare fock state. Additionally, A_state_func can
        also be set to a function with signature `osc_state_func(<some basis of osc mode>, 
        **kwargs)`. Such a fuction should check the validation of the basis, and raise a
        RuntimeError if invalid.
    collapse_op_list:
        If empty, the purcell factors will be evaluated assuming the collapse operators
        are osc mode's annilation operator and qubit mode's sigma_minus operator. Otherwise,
        will calculate the factors using operators specified by the user by:
         - factor = qutip.expect(operator, state) for all operators

    Returns
    -------
    Purcell factors
    """
    
    dressed_indices = ps["dressed_indices"][paramindex_tuple]

    evals = ps["evals"][paramindex_tuple]
    evecs = ps["evecs"][paramindex_tuple]

    factors = purcell_factor(
        ps.hilbertspace,
        osc_mode_idx, qubit_mode_idx, osc_state_func, qubit_state_index,
        collapse_op_list,
        dressed_indices, (evals, evecs),
    )

    return np.array(factors)

# ##############################################################################
def sweep_gamma_1(
    ps: ParameterSweep, paramindex_tuple, paramvals_tuple, 
    mode_idx: int, channel_name: str,
    i_list: int | List | np.ndarray = 1, j_list: int | List | np.ndarray = 0,
    **kwargs, 
) -> np.ndarray:
    """
    Sweep the qubit t1 rates.

    Support channel_name in ["t1_capacitive", "t1_inductive", "t1_charge_impedance", 
    "t1_flux_bias_line", "t1_quasiparticle_tunneling"]

    """

    bare_evecs = ps["bare_evecs"][mode_idx][paramindex_tuple]
    bare_evals = ps["bare_evals"][mode_idx][paramindex_tuple]

    qubit = ps.hilbertspace.subsystem_list[mode_idx]

    # proecess the shape of the result
    actual_shape = []
    if isinstance(i_list, List | np.ndarray):
        actual_shape.append(len(i_list))
    if isinstance(j_list, List | np.ndarray):
        actual_shape.append(len(j_list))
    actual_shape = tuple(actual_shape)

    i_array = np.array(i_list, dtype=int).ravel()
    j_array = np.array(j_list, dtype=int).ravel()
    data_shape = i_array.shape + j_array.shape
    rate = np.zeros(data_shape)

    # get relaxation function
    try:
        rate_func = getattr(qubit, channel_name)
    except AttributeError:
        return rate.reshape(actual_shape)   # zero
    
    # collect keyword argument
    input_kwargs = {}
    for arg in inspect.signature(rate_func).parameters.keys():
        if arg in kwargs.keys():
            input_kwargs[arg] = kwargs[arg]

    # calculate    
    for idx in np.ndindex(data_shape):
        i = i_array[idx[0]]
        j = j_array[idx[1]]
        if i == j:
            continue
        rate[idx] = rate_func(
            i = i, 
            j = j, 
            get_rate = True, 
            total = False,
            esys = (bare_evals, bare_evecs),    
            **input_kwargs    
        )

    return rate.reshape(actual_shape)

def sweep_gamma_phi(
    ps: ParameterSweep, paramindex_tuple, paramvals_tuple, 
    mode_idx: int, channel_name: str,
    i: int = 1, j: int = 0,
    **kwargs,
) -> float:

    bare_evecs = ps["bare_evecs"][mode_idx][paramindex_tuple]
    bare_evals = ps["bare_evals"][mode_idx][paramindex_tuple]

    qubit = ps.hilbertspace.subsystem_list[mode_idx]

    # get relaxation function
    try:
        rate_func = getattr(qubit, channel_name)
    except AttributeError:
        return 0
    
    # collect keyword argument
    input_kwargs = {}
    for arg in inspect.signature(rate_func).parameters.keys():
        if arg in kwargs.keys():
            input_kwargs[arg] = kwargs[arg]

    return rate_func(
        i=i, 
        j=j, 
        get_rate=True, 
        esys=(bare_evals, bare_evecs),
        **input_kwargs
    )
