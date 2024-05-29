import numpy as np
import qutip as qt
import scqubits as scq
from scqubits.core.param_sweep import ParameterSweep
from scqubits.core.namedslots_array import NamedSlotsNdarray

from chencrafts.bsqubits.batched_custom_sweeps import (
    batched_sweep_general,
)
from chencrafts.cqed.qt_helper import oprt_in_basis
from chencrafts.cqed.mode_assignment import two_mode_dressed_esys

from typing import List, Tuple, Any

# Step1: Transform to the desired Lindblad Master Equation #############
# ######################################################################
# Requirements: 
# 1. Two modes: cavity and qubit. 
# 2. Dimension of the cavity: same as the original sweep, 
#    Dimension of the qubit: 2, 
# 3. Linearize the caivity jump operators and assmue it still has the form of a

def sweep_organized_evecs(
    sweep: ParameterSweep, idx, 
    res_mode_idx = 0, qubit_mode_idx = 1,
    qubit_trunc_dim = 2,
):
    hilbertspace = sweep.hilbertspace

    _, evecs = two_mode_dressed_esys(
        hilbertspace=hilbertspace,
        res_mode_idx=res_mode_idx, qubit_mode_idx=qubit_mode_idx,
        state_label=(-1, -1),
        res_truncated_dim=None, qubit_truncated_dim=qubit_trunc_dim,
        dressed_indices=sweep["dressed_indices"][idx],
        eigensys=(sweep["evals"][idx], sweep["evecs"][idx]),
        adjust_phase=True,
    )
    return evecs.ravel()

def sweep_dressed_qubit_projs(
    sweep: ParameterSweep, idx, 
    qubit_mode_idx = 1, 
    qubit_trunc_dim = 2,
):
    """
    Sweep must contains org_evecs key, from sweep_organized_evecs function.
    """
    hilbertspace = sweep.hilbertspace
    qubit = hilbertspace.subsystem_list[qubit_mode_idx]
    
    qubit_projs = np.ndarray((qubit_trunc_dim, qubit_trunc_dim), dtype=qt.Qobj)
    for i in range(qubit_trunc_dim):
        for j in range(qubit_trunc_dim):
            if i <= j:
                proj = hilbertspace.hubbard_operator(i, j, qubit)
                qubit_projs[i, j] = oprt_in_basis(proj, sweep["org_evecs"][idx])

            else:
                qubit_projs[i, j] = qubit_projs[j, i].dag()

    return qubit_projs 

def sweep_dressed_a_op(
    sweep: ParameterSweep, idx, 
    res_mode_idx = 0, 
):
    hilbertspace = sweep.hilbertspace
    res = hilbertspace.subsystem_list[res_mode_idx]
    
    a_op = hilbertspace.annihilate(res)
    a_op_dressed = oprt_in_basis(a_op, sweep["org_evecs"][idx])

    return a_op_dressed

def batched_sweep_dressed_op(
    sweep: ParameterSweep, 
    res_mode_idx = 0, qubit_mode_idx = 1,
    qubit_trunc_dim = 2,
    **kwargs
):
    """
    Like "batched_sweep_purcell_cats", but more general.
    """
    sweep.add_sweep(
        sweep_organized_evecs, 
        sweep_name="org_evecs",
        res_mode_idx=res_mode_idx, qubit_mode_idx=qubit_mode_idx,
        qubit_trunc_dim=qubit_trunc_dim,
    )
    sweep.add_sweep(
        sweep_dressed_qubit_projs, 
        sweep_name="qubit_projs",
        qubit_mode_idx=qubit_mode_idx,
        qubit_trunc_dim=qubit_trunc_dim,
    )
    sweep.add_sweep(
        sweep_dressed_a_op, 
        sweep_name="a_op",
        res_mode_idx=res_mode_idx,
    )

# ----------------------------------------------------------------------
def sweep_qubit_jumps(
    sweep: ParameterSweep, idx,
    qubit_mode_idx = 1,
):
    """
    Qubit jump rates
    """
    pass

def batched_sweep_jump_rates(
    sweep: ParameterSweep, 
    **kwargs
):
    pass