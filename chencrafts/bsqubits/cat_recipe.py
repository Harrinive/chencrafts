import numpy as np
import qutip as qt
import scqubits as scq
from scqubits.core.param_sweep import ParameterSweep
from scqubits.core.namedslots_array import NamedSlotsNdarray
from scqubits.core.hilbert_space import HilbertSpace

from chencrafts.bsqubits.batched_custom_sweeps import (
    batched_sweep_general, batched_sweep_pulse
)
from chencrafts.bsqubits import cat_ideal as cat_ideal
from chencrafts.cqed.qt_helper import oprt_in_basis
from chencrafts.cqed.mode_assignment import two_mode_dressed_esys
from chencrafts.cqed.flexible_sweep import FlexibleSweep

from typing import List, Tuple, Any, Dict, Callable
Esys = Tuple[np.ndarray, np.ndarray]


def get_jump_ops(
    res_mode_idx = 0, 
    res_trunc_dim = 5,
    qubit_trunc_dim = 2,
) -> Dict[str, qt.Qobj | np.ndarray]:
    """
    Compute the jump operators.
    """
    # compute the jump operators
    a_op = cat_ideal.res_destroy(
        res_trunc_dim, qubit_trunc_dim, res_mode_idx,
    )
    ij_ops = np.ndarray(
        (qubit_trunc_dim, qubit_trunc_dim), 
        dtype=object
    )
    for i, j in np.ndindex(qubit_trunc_dim, qubit_trunc_dim):
        ij_ops[i, j] = cat_ideal.qubit_proj(
            res_trunc_dim, qubit_trunc_dim, res_mode_idx,
            qubit_state = i, qubit_state_2 = j,
        )
    vectorized_multiply = np.vectorize(lambda x, y: x * y, otypes=[object])
    a_ij_ops = vectorized_multiply(a_op, ij_ops)
    adag_ij_ops = vectorized_multiply(a_op.dag(), ij_ops)
    
    jump_ops = {
        "jump_a": a_op,
        "jump_adag": a_op.dag(),
        "jump_adag_a": a_op.dag() * a_op,
        
        "jump_ij": ij_ops,
        
        "jump_a_ij": a_ij_ops,
        "jump_adag_ij": adag_ij_ops,
    }
     
    return jump_ops


def cavity_ancilla_me_ingredients(
    fsweep: FlexibleSweep,
    res_mode_idx: int, qubit_mode_idx: int, 
    res_dim: int = 5, qubit_dim: int = 2, 
    res_me_dim: int = 5, qubit_me_dim: int = 2, 
    in_rot_frame: bool = True,
) -> Tuple[qt.Qobj, List[qt.Qobj], Esys, qt.Qobj]:
    """
    Generate hamiltonian and collapse operators for a cavity-ancilla system. The operators
    will be truncated to two modes only with the specified dimension.

    I will use the "cheating" master equation, assuming the jump operators are a, a^dag,
    a^dag a, sigma_p, sigma_m, and sigma_z. 

    Parameters
    ----------
    fsweep: FlexibleSweep
        The flexible sweep object that contains the parameters.
    res_mode_idx, qubit_mode_idx: int
        The index of the resonator / qubit mode in the HilbertSpace
    init_res_state_func: Callable | int
        The initial state of the resonator. It can be a callable function that takes the
        a list of basis as input and returns a state vector, or an integer that specifies
        a fock state. Additionally, the function should have signature `osc_state_func(basis, 
        **kwargs)`. Such a fuction should check the validation of the basis, and raise a
        RuntimeError if invalid. The kwargs will be filled in by the swept parameters or 
        the kwargs of this function. 
    init_qubit_state_index: int
        The initial state of the qubit. Usually 0.
    qubit_dim: int | None
        The truncated dimension of the qubit mode in the . 
        If None, it will be set to
        init_qubit_state_index + 2.
    res_dim: int | None
        The truncated dimension of the resonator mode. If None, the resonator mode will
        not be truncated unless a nan eigenvalue is found.
    in_rot_frame: bool
        If True, the hamiltonian will be transformed into the rotating frame of the
        resonator and qubit 01 frequency. The collapse operators will be transformed 
        accordingly (though the transformaiton is just a trivial phase factor and get 
        cancelled out).

    Returns
    -------
    hamiltonian
        The hamiltonian in the truncated basis. It has dims=[[res_dim, qubit_dim], [res_dim, qubit_dim]].
    c_ops: qt.Qobj, List[qt.Qobj]
        The collapse operators in the truncated basis. 
    eigensys: Tuple[np.ndarray, np.ndarray]
        The eigenenergies and eigenstates of the truncated basis.
    frame_hamiltonian: qt.Qobj
        The hamiltonian that defines a rotating frame transformation the hamiltonian TO the 
        current frame. H_new_frame = H_old_frame - frame_hamiltonian
    """
    if not fsweep.sweep.all_params_fixed(fsweep.sweep._current_param_indices):
        raise ValueError("It's a multi-parameter sweep and the slice is not "
                         "specified. Please try a single-parameter sweep or "
                         "specify the slice.")
    
    hilbertspace = fsweep.hilbertspace
    dims = hilbertspace.subsystem_dims
    if len(dims) > 2:
        raise ValueError("More than 2 subsystems detected. The 'smart truncation' is not "
                         "smart for more than 2 subsystems. It can't determine when to "
                         "truncate for other subsystems and keep the ground state for the mode "
                         "only. It's also not tested."
                         "Please specify the truncation when initialize the HilbertSpace obj.\n")

    # truncate the basis
    # 1. for qubit mode, keep up to the next excited state of the qubit initial state
    # 2. for res mode, keep all levels unless the bare label are not found (eval=nan)
    # 3. for other modes, keep only ground states
    truncated_evals, truncated_evecs = two_mode_dressed_esys(
        hilbertspace, res_mode_idx, qubit_mode_idx,
        state_label=list(np.zeros_like(dims).astype(int)),
        res_truncated_dim=res_dim, qubit_truncated_dim=qubit_dim,
        dressed_indices=fsweep["dressed_indices"], 
        eigensys=(fsweep["evals"], fsweep["evecs"]),
        adjust_phase=True,
    )
    truncated_dims = list(truncated_evals.shape)
    
    # hamiltonian in this basis
    flattend_evals = truncated_evals.ravel() - truncated_evals.ravel()[0]
    hamiltonian = qt.Qobj(np.diag(flattend_evals), dims=[truncated_dims, truncated_dims])

    if in_rot_frame:
        # in the dispersice regime, the transformation hamiltonian is 
        # freq * a^dag a * identity_qubit + identity_res * freq_qubit * qubit^dag qubit
        mode0_freq = truncated_evals[1, 0] - truncated_evals[0, 0]
        mode1_freq = truncated_evals[0, 1] - truncated_evals[0, 0]

        rot_hamiltonian = (
            qt.tensor(mode0_freq * qt.num(truncated_dims[0]), qt.qeye(truncated_dims[1]))
            + qt.tensor(qt.qeye(truncated_dims[0]), mode1_freq * qt.num(truncated_dims[1]))
        )
    else:
        rot_hamiltonian = qt.Qobj(np.zeros_like(hamiltonian.data), dims=hamiltonian.dims)

    hamiltonian -= rot_hamiltonian

    jump_ops = get_jump_ops(res_mode_idx, res_dim, qubit_dim)
    jump_ops_w_rate = {}
    for key, val in jump_ops.items():
        rate = fsweep[key]
        
        if key.endswith("ij"):
            for i, j in np.ndindex(qubit_me_dim, qubit_me_dim):
                key = key[:-2] + f"{i}{j}"
                jump_ops_w_rate[key] = val[i, j] * np.sqrt(rate[i, j])
        else:
            jump_ops_w_rate[key] = val * np.sqrt(rate)
        
    return (
        hamiltonian, 
        list(jump_ops_w_rate.values()), 
        (truncated_evals.ravel(), truncated_evecs.ravel()),
        rot_hamiltonian,
    )
