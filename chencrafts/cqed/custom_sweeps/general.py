import numpy as np
from scqubits.core.param_sweep import ParameterSweep

# ##############################################################################
def sweep_convergence(
    paramsweep: ParameterSweep, paramindex_tuple, paramvals_tuple, mode_idx
):
    bare_evecs = paramsweep["bare_evecs"]["subsys": mode_idx][paramindex_tuple]
    return np.max(np.abs(bare_evecs[-3:, :]))