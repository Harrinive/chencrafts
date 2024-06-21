import numpy as np
import qutip as qt

from scqubits.core.hilbert_space import HilbertSpace
from scqubits.core.qubit_base import QuantumSystem
from scqubits.core.spec_lookup import MixinCompatible, SpectrumLookupMixin
from scqubits.core.namedslots_array import NamedSlotsNdarray

import copy
from warnings import warn
import itertools
from typing import List, Tuple, Any, overload, Optional, Literal

@overload
def label_convert(
    idx: Tuple[int, ...] | List[int],
    h_space: HilbertSpace | None = None, 
    dims: Tuple[int, ...] | List[int] | None = None
) -> np.intp:
    ...

@overload
def label_convert(
    idx: int,
    h_space: HilbertSpace | None = None, 
    dims: Tuple[int, ...] | List[int] | None = None
) -> Tuple[np.intp, ...]:
    ...

def label_convert(
    idx: Tuple[int, ...] | List[int] | int, 
    h_space: HilbertSpace | None = None, 
    dims: Tuple[int, ...] | List[int] | None = None
) -> np.intp | Tuple[np.intp, ...]:
    """
    Convert between a tuple/list of bare state label and the corresponding FLATTENED
    index. It's the combination of `np.ravel_multi_index` and `np.unravel_index`.
    """
    if dims is None:
        assert h_space is not None, "Either HilbertSpace or dims should be given."
        dims = h_space.subsystem_dims

    if isinstance(idx, tuple | list):
        return np.ravel_multi_index(idx, dims)
    
    elif isinstance(idx, int):
        return np.unravel_index(idx, dims)

    else:
        raise ValueError(f"Only support list/tuple/int as an index.")


def organize_dressed_esys(
    hilbertspace: HilbertSpace,
    dressed_indices: np.ndarray | None = None,
    eigensys = None,
    adjust_phase: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    It returns organized eigenenergies and dressed states using two multi-dimensional arrays.
    If a bare label cannot be found, the corresponding evals and evecs will be np.nan and None.

    Parameters
    ----------
    hilberspace:
        scq.HilberSpace object.
    dressed_indeces: 
        An array mapping the FLATTENED bare state label to the eigenstate label. Usually
        given by `ParameterSweep["dressed_indices"]`.
    eigensys:
        The eigenenergies and eigenstates of the bare Hilbert space. Usually given by
        `ParameterSweep["evals"]` and `ParameterSweep["evecs"]`. eigensys and 
        dressed_indices should be given together. Can also be a string "stored" indicating
        that the eigensys is stored inside the hilbertspace object.
    adjust_phase:
        If True, the phase of "bare element" of each eigenstate will be adjusted to be 0.

    Returns
    -------
    evals, evecs 
        organized by bare index labels in multi-dimensional arrays.
    """
    if eigensys is None:
        evals, evecs = hilbertspace.eigensys(hilbertspace.dimension)
    elif eigensys == "stored":
        evals, evecs = hilbertspace["evals"][0], hilbertspace["evecs"][0]
    else:
        evals, evecs = eigensys

    if dressed_indices is None:
        hilbertspace.generate_lookup(dressed_esys=(evals, evecs))
        drs_idx_map = hilbertspace.dressed_index
    else:
        def drs_idx_map(bare_index_tuple):
            flattened_bare_index = label_convert(bare_index_tuple, hilbertspace)
            return dressed_indices[flattened_bare_index]
        
    dim_list = hilbertspace.subsystem_dims

    organized_evals: np.ndarray = np.ndarray(dim_list, dtype=float)
    organized_evecs: np.ndarray = np.ndarray(dim_list, dtype=qt.Qobj)
    for idx, bare_idx in enumerate(np.ndindex(tuple(dim_list))):

        drs_idx = drs_idx_map(bare_idx)   

        eval = np.nan
        evec = None      
        if drs_idx is not None:
            if drs_idx < len(evals):
                evec = evecs[drs_idx]
                eval = evals[drs_idx]

                if adjust_phase:
                    # make the "principle_val" have zero phase
                    principle_val = evec.full()[idx, 0]
                    principle_val_phase = (principle_val) / np.abs(principle_val)
                    evec /= principle_val_phase
        organized_evals[bare_idx] = eval
        organized_evecs[bare_idx] = evec            

    return organized_evals, organized_evecs

def single_mode_dressed_esys(
    hilbertspace: HilbertSpace,
    mode_idx: int,
    state_label: Tuple[int, ...] | List[int],
    dressed_indices: np.ndarray | None = None,
    eigensys = None,
    adjust_phase: bool = True,
) -> Tuple[List[float], List[qt.Qobj]]:
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
        The index of the resonator mode of interest in the hilberspace's subsystem_list
    state_label:
        the subset of the eigensys is calculated with other modes staying at bare state. 
        For example, we are looking for eigensystem for the first 
        mode in a three mode system with the rest of two modes fixed at bare state 0 and 1, 
        we can set state_label to be (<any number>, 0, 1).
    dressed_indeces: 
        An array mapping the FLATTENED bare state label to the eigenstate label. Usually
        given by `ParameterSweep["dressed_indices"]`.
    eigensys:
        The eigenenergies and eigenstates of the bare Hilbert space. Usually given by
        `ParameterSweep["evals"]` and `ParameterSweep["evecs"]`. Can also be a string
        "stored" indicating that the eigensys is stored inside the hilbertspace object.
    adjust_phase:
        If True, the phase of "bare element" of each eigenstate will be adjusted to be 0.

    Returns
    -------
    A subset of eigensys with one of the bare labels varying and the rest fixed. 
    """
    sm_evals = []
    sm_evecs = []

    ornagized_evals, organized_evecs = organize_dressed_esys(
        hilbertspace, dressed_indices, eigensys, adjust_phase
    )

    dim_list = hilbertspace.subsystem_dims
    dim_mode = dim_list[mode_idx]
    bare_index = np.array(state_label).copy()
    for n in range(dim_mode):
        bare_index[mode_idx] = n

        eval = ornagized_evals[tuple(bare_index)]
        evec = organized_evecs[tuple(bare_index)]

        if evec is None or np.isnan(eval):
            break
        
        sm_evecs.append(evec)
        sm_evals.append(eval)

    return (sm_evals, sm_evecs)

def two_mode_dressed_esys(
    hilbertspace: HilbertSpace,
    res_mode_idx: int, qubit_mode_idx: int,
    state_label: Tuple[int, ...] | List[int],
    res_truncated_dim: int | None = None, qubit_truncated_dim: int = 2,
    dressed_indices: np.ndarray | None = None,
    eigensys = None,
    adjust_phase: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    It will return a truncated eigenenergies and dressed states, organized by the bare
    state label of the resonator and qubit. If a bare label cannot be found, the
    resonator mode's truncation will be set to the index of the first nan eigenvalue.

    Parameters
    ----------
    hilberspace:
        scq.HilberSpace object which include the desired mode
    res_mode_idx, qubit_mode_idx: int
        The index of the resonator / qubit mode in the HilbertSpace
    state_label:
        the subset of the eigensys is calculated with other modes staying at the specified
        bare state. For example, we are looking for eigensystem for the first two
        modes in a three mode system with the rest of two modes fixed at bare state 0,
        we can set state_label to be (<any number>, <any number>, 1).
    res_truncated_dim: int | None
        The truncated dimension of the resonator mode. If None, the resonator mode will 
        not be truncated unless a nan eigenvalue is found.
    qubit_truncated_dim: int
        The truncated dimension of the qubit mode. 
    dressed_indeces:
        An array mapping the FLATTENED bare state label to the eigenstate label. Usually
        given by `ParameterSweep["dressed_indices"]`.
    eigensys:
        The eigenenergies and eigenstates of the bare Hilbert space. Usually given by
        `ParameterSweep["evals"]` and `ParameterSweep["evecs"]`. Can also be a string
        "stored" indicating that the eigensys is stored inside the hilbertspace object.
    adjust_phase:
        If True, the phase of "bare element" of each eigenstate will be adjusted to be 0.
    
    Returns
    -------
    eval_array, evec_array
        Those 2-D arrays contains truncated eigenenergies and dressed states, organized by
        the bare state label of the resonator and qubit.
    """
    dim_list = hilbertspace.subsystem_dims

    # make sure the truncated dim < actual dim
    qubit_truncated_dim = int(np.min([qubit_truncated_dim, dim_list[qubit_mode_idx]]))
    if res_truncated_dim is None:
        res_truncated_dim = dim_list[res_mode_idx]
    res_truncated_dim = int(np.min([res_truncated_dim, dim_list[res_mode_idx]]))

    # get the organized evals and evecs
    organized_evals, organized_evecs = organize_dressed_esys(
        hilbertspace, dressed_indices, eigensys, adjust_phase
    )

    # truncation of the qubit mode
    trunc_slice_1: List[Any] = list(state_label).copy()
    trunc_slice_1[qubit_mode_idx] = slice(0, qubit_truncated_dim)
    trunc_slice_1[res_mode_idx] = slice(0, res_truncated_dim)

    truncated_evals = organized_evals[tuple(trunc_slice_1)]
    truncated_evecs = organized_evecs[tuple(trunc_slice_1)]

    # order the modes
    if res_mode_idx > qubit_mode_idx:
        truncated_evals = truncated_evals.T
        truncated_evecs = truncated_evecs.T

    # res mode further truncation: detect nan eigenvalues
    futher_truncated_dim = res_truncated_dim
    for idx in range(res_truncated_dim):
        if np.any(np.isnan(organized_evals[idx, :])):
            futher_truncated_dim = idx
            break
    trunc_slice_2 = (slice(0, futher_truncated_dim), slice(None))
    truncated_evals = truncated_evals[trunc_slice_2]
    truncated_evecs = truncated_evecs[trunc_slice_2]

    return truncated_evals, truncated_evecs

def dressed_state_component(
    hilbertspace: HilbertSpace, 
    state_label: Tuple[int, ...] | List[int] | int,
    eigensys = None,
    truncate: int | None = None,
) -> Tuple[List[int], List[float]]:
    """
    For a dressed state with bare_label, will return the bare state conponents and the 
    corresponding occupation probability. 
    They are sorted by the probability in descending order.

    Parameters
    ----------
    hilbertspace:
        scq.HilbertSpace object
    state_label:
        The bare label of the dressed state of interest. Could be 
            - a tuple/list of bare labels (int)
            - a single dressed label (int)
    eigensys:
        The eigenenergies and eigenstates of the bare Hilbert space in 
        a tuple. Usually given by
        `ParameterSweep["evals"]` and `ParameterSweep["evecs"]`. Can also
        be a string "stored" indicating that the eigensys is stored inside
        the hilbertspace object.
    truncate:
        The number of components to be returned. If None, all components 
        will be returned.
    """
    if eigensys is None:
        eigensys = hilbertspace.eigensys(hilbertspace.dimension)
    elif eigensys == "stored":
        eigensys = hilbertspace["evals"][0], hilbertspace["evecs"][0]
        
    _, evecs = eigensys

    try:
        hilbertspace.generate_lookup(dressed_esys=eigensys)
    except TypeError:
        # TypeError: HilbertSpace.generate_lookup() got an unexpected 
        # keyword argument 'dressed_esys'
        # meaning that it's not in danyang branch
        warn("Not in danyang's branch of scqubits. Generate lookup without "
             "the eigensys if given.\n")
        hilbertspace.generate_lookup()

    if isinstance(state_label, tuple | list): 
        drs_idx = hilbertspace.dressed_index(tuple(state_label))
        if drs_idx is None:
            raise IndexError(f"no dressed state found for bare label {state_label}")
    elif isinstance(state_label, int):
        drs_idx = state_label

    evec_1 = evecs[drs_idx]
    largest_occupation_label = np.argsort(np.abs(evec_1.full()[:, 0]))[::-1]

    bare_label_list = []
    prob_list = []
    for idx in range(evec_1.shape[0]):
        drs_label = int(largest_occupation_label[idx])
        state_label = label_convert(drs_label, hilbertspace)
        prob = (np.abs(evec_1.full()[:, 0])**2)[drs_label]

        bare_label_list.append(state_label)
        prob_list.append(prob)

    if truncate is not None:
        bare_label_list = bare_label_list[:truncate]
        prob_list = prob_list[:truncate]

    return bare_label_list, prob_list

def _excite_op(
    hilbertspace: HilbertSpace,
    mode: int | QuantumSystem,
):
    if isinstance(mode, int):
        mode_idx = mode
        mode = hilbertspace.subsystem_list[mode]
    else:
        mode_idx = hilbertspace.subsystem_list.index(mode)
        
    if mode in hilbertspace.osc_subsys_list:
        return hilbertspace.annihilate(mode).dag()
    else:
        # annhilation operator
        # return hilbertspace.annihilate(mode).dag()
        
        # a sum of |j>+1<j|
        dims = hilbertspace.subsystem_dims
        eyes = [qt.qeye(dim) for dim in dims]
        eyes[mode_idx] = qt.qdiags(
            np.ones(dims[mode_idx] - 1),
            -1,
        )
        return qt.tensor(*eyes)

def _branch_analysis_DF_step(
    self: SpectrumLookupMixin,   # hilbertspace
    mode_priority: List[int],
    recusion_depth: int,
    init_drs_idx: int, init_state: qt.Qobj, 
    remaining_drs_indices: List[int], remaining_evecs: List[qt.Qobj], 
):
    """
    Perform a single branch analysis according to Dumas et al. (2024). This 
    is a core function to be run recursively, which realized a depth-first
    search.

    In a nutshell, the function will:
    1. Start from the "ground" state / starting point the branch, find
    all of the branch states
    2. Remove the found states from the remaining candidates
    3. [If at the end of the depth-first search] Return the branch states
    4. [If not at the end] For each branch state, use it as an init state to 
    start such search again, which will return a (nested) list of branch 
    states. Combine the list of branch states and return a nested list of
    those states

    In such way, the function will recursively go through this multi-dimensional
    Hilbert space and assign the eigenstates to their labels.

    Parameters
    ----------
    self:
        SpectrumLookupMixin object, could be a `ParameterSweep` object or 
        `HilbertSpace` object.
    mode_priority:
        A permutation of the mode indices. 
        It represents the depth of the mode labels to be traversed. The later
        the mode appears in the list, the deeper it is in the recursion.
        For the last mode in the list, its states will be organized in a 
        single branch - the innermost part of the nested list. 
    recusion_depth:
        The current depth of the recursion. It should be 0 at the beginning.
    init_drs_idx:
        The dressed index of the initial state of this branch.
    init_state:
        The initial state of this branch.
    remaining_drs_indices:
        The list of the remaining dressed indices to be assigned.
    remaining_evecs:
        The list of the remaining eigenstates to be assigned.
    
    Returns
    -------
    branch_drs_indices, branch_states
        The (nested) list of the branch states and their dressed indices.
    """

    hspace = self.hilbertspace
    mode_index = mode_priority[recusion_depth]
    mode = hspace.subsystem_list[mode_index]
    terminate_branch_length = hspace.subsystem_dims[mode_index]

    # photon addition operator
    excite_op = _excite_op(hspace, mode)

    # loop over and find all states that matches the excited initial state
    current_state = init_state
    current_drs_idx = init_drs_idx
    branch_drs_indices = []
    branch_states = []
    while True:
        if recusion_depth == len(mode_priority) - 1:
            # we are at the end of the depth-first search:
            # just add the state to the branch
            branch_drs_indices.append(current_drs_idx)
            branch_states.append(current_state)
        else:
            # continue the depth-first search:
            # recursively call the function and append all the branch states
            (
                _branch_drs_indices, _branch_states
            ) = _branch_analysis_DF_step(
                self, 
                mode_priority, 
                recusion_depth + 1,
                current_drs_idx,
                current_state, 
                remaining_drs_indices,
                remaining_evecs, 
            )
            branch_drs_indices.append(_branch_drs_indices)
            branch_states.append(_branch_states)

        # if the branch is long enough, terminate the loop
        if len(branch_states) == terminate_branch_length:
            break

        # find the closest state to the excited current state
        if len(remaining_evecs) == 0:
            raise ValueError(
                "No more states to assign. It's likely that the eignestates "
                "are not complete. Please try obtain a complete set of "
                "eigenstates using generate_lookup."
            )

        excited_state = (excite_op * current_state).unit()
        overlaps = [np.abs(excited_state.overlap(evec)) for evec in remaining_evecs]
        max_overlap_index = np.argmax(overlaps)

        current_state = remaining_evecs[max_overlap_index]
        current_drs_idx = remaining_drs_indices[max_overlap_index]

        # remove the state from the remaining states
        remaining_evecs.pop(max_overlap_index)
        remaining_drs_indices.pop(max_overlap_index)

    return branch_drs_indices, branch_states

def branch_analysis_DF(
    self: SpectrumLookupMixin,   # hilbertspace
    param_indices: Tuple[int, ...],
    mode_priority: Optional[List[int]] = None,
    transpose: bool = False,
):
    """
    Perform a full branch analysis according to Dumas et al. (2024) for 
    a single parameter point using depth-first search. This
    function will organize the eigenstates into a multi-dimensional array
    according to the mode_priority. 

    Parameters
    ----------
    self:
        SpectrumLookupMixin object, could be a `ParameterSweep` object or 
        `HilbertSpace` object.
    param_indices:
        The indices of the parameter sweep to be analyzed.
    mode_priority:
        A permutation of the mode indices. 
        It represents the depth of the mode labels to be traversed. The later
        the mode appears in the list, the deeper it is in the recursion.
        For the last mode in the list, its states will be organized in a 
        single branch - the innermost part of the nested list.
    transpose:
        If True, the returned array will be transposed according to the
        mode_priority. 

    Returns
    -------
    branch_drs_indices
        The multi-dimensional array of the dressed indices organized by 
        the mode_priority. If the dimensions of the subsystems are
        D0, D1 and D2, the returned array will have the shape (D0, D1, D2).
        If transposed is True, the array will be transposed according to
        the mode_priority.
    """
    if mode_priority is None:
        mode_priority = list(range(self.hilbertspace.subsystem_count))
    
    # we assume that the ground state always has bare label (0, 0, ...)
    evecs = self._data["evecs"][param_indices]
    init_state = evecs[0]
    remaining_evecs = list(evecs[1:])
    remaining_drs_indices = list(range(1, self.hilbertspace.dimension))

    branch_drs_indices, _ = _branch_analysis_DF_step(
        self, mode_priority, 
        0, 
        0, init_state,
        remaining_drs_indices, remaining_evecs
    )
    branch_drs_indices = np.array(branch_drs_indices)

    if not transpose:
        reversed_permutation = np.argsort(mode_priority)
        return np.transpose(
            branch_drs_indices, reversed_permutation
        )

    return branch_drs_indices

def branch_analysis_EF(
    self: SpectrumLookupMixin,   # hilbertspace
    param_indices: Tuple[int, ...],
    truncate: int | None = None,
    check_all_prev_states: bool = False,
):
    """
    Perform a full branch analysis using energy-first traversal, as the lower
    (bare) energy states are assigned first.
    """
    hspace = self.hilbertspace
    dims = hspace.subsystem_dims
    
    if truncate is None:
        truncate = len(self._data["evecs"][param_indices])
    elif len(self._data["evecs"][param_indices]) < truncate:
        truncate = len(self._data["evecs"][param_indices])
        warn(
            "evals_count is less than truncate, truncate is set to "
            f"{len(self._data['evecs'][param_indices])}."
        )
    
    # get the associated excitation operators
    excite_op_list = [_excite_op(hspace, mode) for mode in hspace.subsystem_list]
    
    # generate a list of their bare energies
    bare_evals_by_sys = self._data["bare_evals"]
    bare_evals = np.zeros(dims)
    for idx in np.ndindex(tuple(dims)):
        subsys_eval = [
            bare_evals_by_sys[subsys_idx][param_indices][level_idx]
            for subsys_idx, level_idx in enumerate(idx)
        ]
        bare_evals[idx] = np.sum(subsys_eval)
    bare_evals = bare_evals.ravel()
    
    # sort the bare energies
    # which will be the order of state assignment
    sorted_indices = np.argsort(bare_evals)[:truncate]
        
    # mode assignment
    branch_drs_indices = np.ndarray(dims, dtype=object)
    branch_drs_indices.fill(None)
    evecs = self._data["evecs"][param_indices]
    remaining_evecs = list(evecs)
    remaining_drs_indices = list(range(0, self.hilbertspace.dimension))
    
    for raveled_bare_idx in sorted_indices:
        # assign the dressed index for bare_idx
        bare_idx = list(np.unravel_index(raveled_bare_idx, dims))
        
        if raveled_bare_idx == 0:
            # the (0, 0, ...) is always assigned the dressed index 0
            branch_drs_indices[tuple(bare_idx)] = 0
            remaining_drs_indices.pop(0)
            remaining_evecs.pop(0)
            continue
        
        # get previously assigned states (one less excitation) 
        # By comparing the excited states with the dressed states,
        # we can find the dressed index of the current state
        prev_bare_indices = []
        potential_drs_indices = []
        for subsys_idx in range(len(dims)):
            
            # obtain the a bare index with one less excitation
            prev_idx = copy.copy(bare_idx)
            if prev_idx[subsys_idx] == 0:
                continue
            prev_idx[subsys_idx] -= 1
            prev_drs_idx = branch_drs_indices[tuple(prev_idx)]
            
            # No need to check actually, because all lower energy states
            # have been assigned
            # if prev_drs_idx is None:
            #     print(f"skipped {prev_idx} for {bare_idx}")
            #     continue
            
            prev_bare_indices.append(prev_idx)
            
            # state vector
            prev_state = evecs[prev_drs_idx]
            excited_state = excite_op_list[subsys_idx] * prev_state
            excited_state = excited_state.unit()
            
            # find the dressed index
            overlaps = [np.abs(excited_state.overlap(evec)) for evec in remaining_evecs]
            max_overlap_index = np.argmax(overlaps)
            
            potential_drs_indices.append(remaining_drs_indices[max_overlap_index])
            
            # if not check_all_prev_states and len(potential_drs_indices) == 1:
            #     # already find one, that's good enough
            #     continue
            
        # do a majority vote, if equal, chose the first one
        unique_votes, counts = np.unique(potential_drs_indices, return_counts=True)
        vote_result = np.argmax(counts)
        drs_idx = unique_votes[vote_result]
        idx_in_remaining_list = remaining_drs_indices.index(drs_idx)
        
        # # print out the result if unique_votes is more than 1
        # if len(unique_votes) > 1:
        #     print(f"{bare_idx}: {unique_votes}, counts: {counts}")
        
        # remove the state from the remaining states
        remaining_evecs.pop(idx_in_remaining_list)
        remaining_drs_indices.pop(idx_in_remaining_list)
        
        branch_drs_indices[tuple(bare_idx)] = drs_idx
        
    return branch_drs_indices

def branch_analysis(
    self: SpectrumLookupMixin,   # hilbertspace
    mode: Literal["DF", "EF"] = "EF",
    mode_priority: Optional[List[int]] = None,
    transpose: bool = False,
    truncate: int | None = None,
    check_all_prev_states: bool = False,
) -> NamedSlotsNdarray:
    """
    Perform a full branch analysis for all parameter points.
    """
    dressed_indices = np.empty(shape=self._parameters.counts, dtype=object)

    param_indices = itertools.product(*map(range, self._parameters.counts))
    
    for index in param_indices:
        if mode == "DF":
            dressed_indices[index] = branch_analysis_DF(
                self, index, mode_priority, transpose,
            )
        elif mode == "EF":
            dressed_indices[index] = branch_analysis_EF(
                self, index, truncate, check_all_prev_states,
            )
        else:
            raise ValueError(f"Mode {mode} is not supported.")
        
    dressed_indices = np.asarray(dressed_indices[:].tolist())

    parameter_dict = self._parameters.ordered_dict.copy()
    return NamedSlotsNdarray(dressed_indices, parameter_dict)

