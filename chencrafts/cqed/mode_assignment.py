import numpy as np
import qutip as qt

from scqubits.core.hilbert_space import HilbertSpace

from typing import List, Tuple

def label_convert(idx: Tuple | List | int, h_space: HilbertSpace):

    dims = h_space.subsystem_dims

    if isinstance(idx, tuple | list):
        assert (np.array(idx) < np.array(dims)).all(), f"index is not valid for system dimension {dims}"

        drs_idx = 0
        for dim_idx, bare_idx in enumerate(idx):
            drs_idx += np.prod(dims[dim_idx+1:]) * bare_idx

        return int(drs_idx)
    
    elif isinstance(idx, int):
        assert (idx < np.prod(dims)).all(), f"index is not valid for system size {np.prod(dims)}"

        bare_idx_list = []
        for dim_idx in range(len(dims)):
            bare_idx_list.append(int(idx / np.prod(dims[dim_idx+1:])))
            idx = idx % int(np.prod(dims[dim_idx+1:]))

        return tuple(bare_idx_list)

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

    Parameters
    ----------
    hilberspace:
        scq.HilberSpace object.
    dressed_indeces: 
        An array mapping the FLATTENED bare state label to the eigenstate label. Usually
        given by `ParameterSweep["dressed_indices"]`.
    eigensys:
        The eigenenergies and eigenstates of the bare Hilbert space. Usually given by
        `ParameterSweep["evals"]` and `ParameterSweep["evecs"]`. eigensys and dressed_indices should be given together.
    adjust_phase:
        If True, the phase of "bare element" of each eigenstate will be adjusted to be 0.

    Returns
    -------
    Evals and evecs organized with bare index labels in two multi-dimensional arrays.
    """
    if eigensys is None:
        evals, evecs = hilbertspace.eigensys(hilbertspace.dimension)
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

        if drs_idx is not None and drs_idx < len(evals):
            evec = evecs[drs_idx]
            eval = evals[drs_idx]

            if adjust_phase:
                # make the "principle_val" have zero phase
                principle_val = evec.data[idx, 0]
                principle_val_phase = (principle_val) / np.abs(principle_val)
                evec /= principle_val_phase

            organized_evals[bare_idx] = eval
            organized_evecs[bare_idx] = evec
        else:
            organized_evals[bare_idx] = np.nan
            organized_evecs[bare_idx] = None

    return organized_evals, organized_evecs

def single_mode_dressed_esys(
    hilbertspace: HilbertSpace,
    mode_idx: int,
    state_label: Tuple[int, ...] | List[int],
    dressed_indices: np.ndarray | None = None,
    eigensys = None,
    adjust_phase: bool = True,
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
        `ParameterSweep["evals"]` and `ParameterSweep["evecs"]`.
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
    dim_res = dim_list[mode_idx]
    bare_index = np.array(state_label).copy()
    for n in range(dim_res):
        bare_index[mode_idx] = n

        eval = ornagized_evals[tuple(bare_index)]
        evec = organized_evecs[tuple(bare_index)]

        if evec is None or np.isnan(eval):
            break
        
        sm_evecs.append(evec)
        sm_evals.append(eval)

    return (sm_evals, sm_evecs)