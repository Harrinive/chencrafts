import numpy as np
import qutip as qt

from scqubits.core.hilbert_space import HilbertSpace
from scqubits.core.param_sweep import ParameterSweep

from chencrafts.cqed.mode_assignment import single_mode_dressed_esys
from chencrafts.cqed.custom_sweeps.utils import fill_in_kwargs_during_custom_sweep

from typing import List, Callable
import inspect

# ##############################################################################
def sweep_gamma_1(
    ps: ParameterSweep, paramindex_tuple, paramvals_tuple, 
    mode_idx: int, channel_name: str,
    i_list: int | List | np.ndarray = 1, j_list: int | List | np.ndarray = 0,
    **kwargs, 
) -> np.ndarray:
    """
    Sweep the qubit t1 rates. Hilbertspace must be updated 

    Support channel_name in ["t1_capacitive", "t1_inductive", "t1_charge_impedance", 
    "t1_flux_bias_line", "t1_quasiparticle_tunneling"]

    """

    bare_evecs = ps["bare_evecs"][mode_idx][paramindex_tuple]
    bare_evals = ps["bare_evals"][mode_idx][paramindex_tuple]

    qubit = ps.hilbertspace.subsystem_list[mode_idx]

    # proecess the shape of the result
    actual_shape = []
    if isinstance(i_list, List | np.ndarray | range):
        actual_shape.append(len(i_list))
    if isinstance(j_list, List | np.ndarray | range):
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
    input_kwargs = fill_in_kwargs_during_custom_sweep(
        ps, paramindex_tuple, paramvals_tuple, rate_func, kwargs,
        ignore_kwargs=[
            "i", "j", "A_noise", "total", "esys", "get_rate", 
            "Y_qp", "Delta"         # for t1_quasiparticle_tunneling
        ],
    )
    
    # calculate    
    try: 
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
    except RuntimeError:
        return rate.reshape(actual_shape)   # zero

    return rate.reshape(actual_shape)

def sweep_gamma_phi(
    ps: ParameterSweep, paramindex_tuple, paramvals_tuple, 
    mode_idx: int, channel_name: str,
    i: int = 1, j: int = 0,
    **kwargs,
) -> float:
    """
    Sweep the qubit tphi rates. Hilbertspace must be updated 

    Support channel_name in ["tphi_1_over_f_flux", "tphi_1_over_f_ng", "tphi_1_over_f_cc"]
    """

    bare_evecs = ps["bare_evecs"][mode_idx][paramindex_tuple]
    bare_evals = ps["bare_evals"][mode_idx][paramindex_tuple]

    qubit = ps.hilbertspace.subsystem_list[mode_idx]

    # get relaxation function
    try:
        rate_func = getattr(qubit, channel_name)
    except AttributeError:
        return 0
    
    # collect keyword argument
    input_kwargs = fill_in_kwargs_during_custom_sweep(
        ps, paramindex_tuple, paramvals_tuple, rate_func, kwargs,
        ignore_kwargs=["i", "j", "esys", "get_rate", "kwargs", "A_noise"],
    )
    special_kwarg = "A_" + channel_name.removeprefix("tphi_1_over_f_")
    if special_kwarg in ps.parameters.names:
        input_kwargs["A_noise"] = paramvals_tuple[ps.parameters.index_by_name[special_kwarg]]
    elif special_kwarg in kwargs.keys():
        input_kwargs["A_noise"] = kwargs[special_kwarg]

    try:
        return rate_func(
            i=i, 
            j=j, 
            get_rate=True, 
            esys=(bare_evals, bare_evecs),
            **input_kwargs
        )
    except RuntimeError:
        return 0

# ##############################################################################
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

def sweep_purcell_factor(
    ps: ParameterSweep, paramindex_tuple, paramvals_tuple, 
    res_mode_idx: int, qubit_mode_idx: int, 
    res_state_func: Callable | int = 0, qubit_state_index: int = 0,
    collapse_op_list: List[qt.Qobj] = [],
    **kwargs
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
        also be set to a function with signature `osc_state_func(basis, 
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

    if callable(res_state_func):
        input_kwargs = fill_in_kwargs_during_custom_sweep(
            ps, paramindex_tuple, paramvals_tuple, res_state_func, kwargs,
            ignore_kwargs=["basis"]
        )
    else:
        input_kwargs = {}

    factors = purcell_factor(
        ps.hilbertspace,
        res_mode_idx, qubit_mode_idx, res_state_func, qubit_state_index,
        collapse_op_list,
        dressed_indices, (evals, evecs),
        **input_kwargs
    )

    return np.array(factors)