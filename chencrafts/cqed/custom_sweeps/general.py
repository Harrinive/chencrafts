__all__ = [
    'sweep_convergence',
    'standardize_evec_sign',
    'sweep_hamiltonian',
]

import numpy as np
import qutip as qt
from scipy.sparse import csr_matrix
from scqubits.core.param_sweep import ParameterSweep

from typing import List, Tuple

# ##############################################################################
def sweep_convergence(
    paramsweep: ParameterSweep, paramindex_tuple, paramvals_tuple, mode_idx
):
    bare_evecs = paramsweep["bare_evecs"]["subsys": mode_idx][paramindex_tuple]
    return np.max(np.abs(bare_evecs[-3:, :]))

def standardize_evec_sign(
    ps: ParameterSweep,
    idx,
    state_labels: List[Tuple[int, ...]],
):
    """
    Standardize the sign of eigenvectors. 
    
    Parameters
    ----------
    ps : scqubits.ParameterSweep
        The parameter sweep object.
    idx : int
        The index of the parameter set to sweep.
    state_labels : List[Tuple[int, ...]]
        The bare labels of the states to be standardized. It is assumed 
        that the dressed states are very close to the bare states.
        
    Returns
    -------
    evecs_std : np.ndarray
        The standardized eigenvectors.
    """
    evecs = ps["evecs"][idx].copy()
    
    dims = tuple(ps.hilbertspace.subsystem_dims)
    for bare_label in state_labels:
        raveled_bare_label = np.ravel_multi_index(bare_label, dims)
        drs_label = ps["dressed_indices"][idx][raveled_bare_label]
        
        evec_to_standardize = evecs[drs_label]
        evec_arr = evec_to_standardize.full()
        
        # extract the sign of the "principal" state component
        principal_component = evec_arr[raveled_bare_label, 0]
        principal_sign = principal_component / np.abs(principal_component)
        
        # standardize the sign
        evecs[drs_label] = evec_to_standardize * principal_sign
        
    return evecs

def sweep_hamiltonian(
    ps:ParameterSweep,
    paramindex_tuple, paramvals_tuple,
):
    bare_esys = {
        subsys_index: [
            ps["bare_evals"][subsys_index][paramindex_tuple],
            ps["bare_evecs"][subsys_index][paramindex_tuple],
        ]
        for subsys_index, _ in enumerate(ps.hilbertspace)
    }
    
    ham = ps.hilbertspace.hamiltonian(bare_esys=bare_esys)
    dims = ham.dims
    
    ham = qt.Qobj(csr_matrix(ham.full()), dims=dims)
    
    return ham
