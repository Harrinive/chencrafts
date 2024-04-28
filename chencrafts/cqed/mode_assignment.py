import numpy as np
import qutip as qt

from scqubits.core.hilbert_space import HilbertSpace

from warnings import warn
from typing import List, Tuple, Any, overload

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
