import numpy as np
import qutip as qt
import copy

import scqubits as scq
from scqubits.core.hilbert_space import HilbertSpace
from scqubits.core.param_sweep import ParameterSweep
from scqubits.core.namedslots_array import NamedSlotsNdarray

from chencrafts.cqed.mode_assignment import two_mode_dressed_esys
from chencrafts.cqed.qt_helper import oprt_in_basis, direct_sum
from chencrafts.cqed.pulses import DRAGGaussian, Gaussian
from chencrafts.settings import QUTIP_VERSION

from typing import Dict, List, Tuple, Callable, Any, Literal
import warnings

Esys = Tuple[np.ndarray, np.ndarray]

# ##############################################################################
def _collapse_operators_by_rate(
    hilbertspace: HilbertSpace,
    mode_idx: int, 
    collapse_parameters: Dict[str, Any] = {},
    basis: List[qt.Qobj] | np.ndarray | None = None,
) -> List[qt.Qobj]:
    """
    Generate a dict of collapse operators given the collapse parameters. 

    Parameters
    ----------
    hilbertspace: HilbertSpace
        scq.HilbertSpace object that contains the desired mode
    mode_idx: int
        The index of the mode in the HilbertSpace.
    collapse_parameters: Dict[str, float]
        A dictionary of collapse parameters. Certain channels will be added if the 
        corresponding key exists. The accepted keys are:  
        - "res_decay": The depolarization rate of the resonator. jump operator: a
        - "res_excite": The excitation rate of the resonator. jump operator: a^dag
        - "res_dephase": The pure dephasing rate of the resonator. jump operator: a^dag a
        - "qubit_decay": The depolarization rate of the qubit. The dict value should be a 2D 
        array `arr`, its element `arr[i, j]` should be the rate for transition from 
        state i to state j. jump operator: |j><i|
        - "qubit_dephase": The pure dephasing rate of the qubit. The dict value should be
        a 1D array `arr`, its element `arr[i]` should be the pure dephasing rate for state 
        i. jump operator: |i><i|
    basis: List[qt.Qobj] | np.ndarray | None
        The basis to transform the jump operators. If None, no basis transformation will be
        done.
    """

    hilbertspace = copy.deepcopy(hilbertspace)
    mode = hilbertspace.subsystem_list[mode_idx]
    dim = hilbertspace.subsystem_dims[mode_idx]

    a_oprt = hilbertspace.annihilate(mode)
    if basis is not None:
        a_oprt = oprt_in_basis(a_oprt, basis)

    collapse_ops = []
    if "res_decay" in collapse_parameters.keys():
        collapse_ops.append(np.sqrt(collapse_parameters["res_decay"]) * a_oprt)

    if "res_excite" in collapse_parameters.keys():
        collapse_ops.append(np.sqrt(collapse_parameters["res_excite"]) * a_oprt.dag())

    if "res_dephase" in collapse_parameters.keys():
        collapse_ops.append(
            np.sqrt(collapse_parameters["res_dephase"]) * a_oprt.dag() * a_oprt)

    if "qubit_decay" in collapse_parameters.keys():
        rate_arr = np.array(collapse_parameters["qubit_decay"])
        if len(rate_arr.shape) != 2:
            raise ValueError("The qubit decay rate should be a 2D array.")
        for idx, value in np.ndenumerate(rate_arr):
            # no self transition, neglect small rates
            if idx[0] == idx[1] or value < 1e-14:
                continue

            oprt_ij = hilbertspace.hubbard_operator(idx[1], idx[0], mode)
            if basis is not None:
                oprt_ij = oprt_in_basis(oprt_ij, basis)

            collapse_ops.append(np.sqrt(value) * oprt_ij)

    if "qubit_dephase" in collapse_parameters.keys():
        rate_arr = np.array(collapse_parameters["qubit_dephase"])
        if len(rate_arr.shape) != 1:
            raise ValueError("The qubit pure dephasing rate should be a 1D array.")
        
        diag_elem = np.zeros(dim)
        len_rate = len(rate_arr)
        if dim > len_rate:
            diag_elem[:len_rate] = np.sqrt(rate_arr)
        else:
            diag_elem = np.sqrt(rate_arr[:dim])

        oprt_ii = scq.identity_wrap(
            np.diag(diag_elem), mode, hilbertspace.subsystem_list, 
            op_in_eigenbasis=True,
        )
        if basis is not None:
            oprt_ii = oprt_in_basis(oprt_ii, basis)

        collapse_ops.append(oprt_ii)

    return collapse_ops
    
def cavity_ancilla_me_ingredients(
    hilbertspace: HilbertSpace,
    res_mode_idx: int, qubit_mode_idx: int, 
    res_truncated_dim: int | None = None, qubit_truncated_dim: int = 2, 
    dressed_indices: np.ndarray | None = None, eigensys = None,
    collapse_parameters: Dict[str, Any] = {},
    res_n_bar: int | None = None,
    in_rot_frame: bool = True,
) -> Tuple[qt.Qobj, List[qt.Qobj], Esys, qt.Qobj]:
    """
    Generate hamiltonian and collapse operators for a cavity-ancilla system. The operators
    will be truncated to two modes only with the specified dimension.

    I will use the "cheating" master equation, assuming the jump operators are a, a^dag,
    a^dag a, sigma_p, sigma_m, and sigma_z. 

    Parameters
    ----------
    hilbertspace: HilbertSpace
        scq.HilbertSpace object that contains a qubit and a resonator
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
    qubit_truncated_dim: int | None
        The truncated dimension of the qubit mode. If None, it will be set to
        init_qubit_state_index + 2.
    res_truncated_dim: int | None
        The truncated dimension of the resonator mode. If None, the resonator mode will
        not be truncated unless a nan eigenvalue is found.
    dressed_indices: np.ndarray | None
        An array mapping the FLATTENED bare state label to the eigenstate label. Usually
        given by `ParameterSweep["dressed_indices"]`.
    eigensys:
        The eigenenergies and eigenstates of the bare Hilbert space. Usually given by
        `ParameterSweep["evals"]` and `ParameterSweep["evecs"]`.
    collapse_parameters: Dict[str, float]
        A dictionary of collapse parameters. Certain channels will be added if the 
        corresponding key exists. The accepted keys are:  
        - "res_decay": The depolarization rate of the resonator. jump operator: a
        - "res_excite": The excitation rate of the resonator. jump operator: a^dag
        - "res_dephase": The pure dephasing rate of the resonator. jump operator: a^dag a
        - "qubit_decay": The depolarization rate of the qubit. The dict value should be a 2D 
        array `arr`, its element `arr[i, j]` should be the rate for transition from 
        state i to state j. jump operator: |j><i|
        - "qubit_dephase": The pure dephasing rate of the qubit. The dict value should be
        a 1D array `arr`, its element `arr[i]` should be the pure dephasing rate for state 
        i. jump operator: |i><i|
    res_n_bar: int | None
        The average photon number of the resonator. If provided, the qubit will be 
        brought into a reference frame that the frequency is shifted by res_n_bar * chi.
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
    # prepare
    hilbertspace = copy.deepcopy(hilbertspace)
    dims = hilbertspace.subsystem_dims
    if len(dims) > 2:
        warnings.warn("More than 2 subsystems detected. The 'smart truncation' is not "
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
        res_truncated_dim=res_truncated_dim, qubit_truncated_dim=qubit_truncated_dim,
        dressed_indices=dressed_indices, eigensys=eigensys,
        adjust_phase=True,
    )
    truncated_dims = list(truncated_evals.shape)

    # hamiltonian in this basis
    flattend_evals = truncated_evals.ravel() - truncated_evals.ravel()[0]
    hamiltonian = qt.Qobj(np.diag(flattend_evals), dims=[truncated_dims, truncated_dims])

    if in_rot_frame:
        if res_n_bar is None:
            res_n_bar = 0

        # in the dispersice regime, the transformation hamiltonian is 
        # freq * a^dag a * identity_qubit + identity_res * freq_qubit * qubit^dag qubit
        res_freq = truncated_evals[1, 0] - truncated_evals[0, 0]
        qubit_freq = truncated_evals[0, 1] - truncated_evals[0, 0]

        rot_hamiltonian = (
            qt.tensor(res_freq * qt.num(truncated_dims[0]), qt.qeye(qubit_truncated_dim))
            + qt.tensor(qt.qeye(truncated_dims[0]), qubit_freq * qt.num(qubit_truncated_dim))
        )
    else:
        rot_hamiltonian = qt.Qobj(np.zeros_like(hamiltonian.data), dims=hamiltonian.dims)

    hamiltonian -= rot_hamiltonian

    # Construct the collapse operators in this basis
    res_collapse_parameters = {
        key: collapse_parameters[key] for key in collapse_parameters.keys()
        if key.startswith("res")
    }
    res_collapse_operators = _collapse_operators_by_rate(
        hilbertspace, res_mode_idx, res_collapse_parameters, basis=truncated_evecs.ravel()
    )
    qubit_collapse_parameters = {
        key: collapse_parameters[key] for key in collapse_parameters.keys()
        if key.startswith("qubit")
    }
    qubit_collapse_operators = _collapse_operators_by_rate(
        hilbertspace, qubit_mode_idx, qubit_collapse_parameters, basis=truncated_evecs.ravel()
    )
    c_ops = [
        qt.Qobj(op, dims=[truncated_dims, truncated_dims])
        for op in res_collapse_operators + qubit_collapse_operators
    ]       # change the dims of the collapse operators
    
    return (
        hamiltonian, 
        c_ops, 
        (truncated_evals.ravel(), truncated_evecs.ravel()),
        rot_hamiltonian,
    )

def idling_propagator(
    hamiltonian: qt.Qobj, 
    c_ops: List[qt.Qobj],
    time: float,
) -> qt.Qobj:
    """
    Run the idling process for a given time.

    Parameters
    ----------
    hamiltonian: qt.Qobj
        The hamiltonian of the system.
    c_ops: List[qt.Qobj]
        The collapse operators of the system.
    idling_time: float | List[float] | np.ndarray
        The idling time. If a list or array is given, will return a list of final states.

    Returns
    -------
    final_states: List[qt.Qobj]
    """
    liouv = qt.liouvillian(hamiltonian, c_ops)

    return (liouv * time).expm()

from chencrafts.bsqubits.cat_ideal import qubit_projectors as ideal_qubit_projectors
def qubit_projectors(
    res_dim: int, qubit_dim: int,
    res_mode_idx: Literal[0, 1] = 0,
    confusion_matrix: np.ndarray | List = None,
    ensamble_average: bool = False,
    superop: bool = True,
) -> List[qt.Qobj]:
    """
    A senario of qubit measurement with the assignment error - when applying 
    the projective measurement, the outcome has the probability to be different
    from the projectors applied. In this case, a set of measurement operators 
    M_j determined by the confusion matrix. 
    
    The confusion matrix C_ij is the 
    probability of measuring state i and getting 
    result j. It is a square matrix with dimension qubit_dim.
    
    The measurement operators are given by M_j = sum_i sqrt(C_ij) |i><i|, 
    corresponding to the measurement result j. Those M_j satisfy the completeness 
    relation: sum_j M_j^\dag M_j = I. 
    
    """
    projs = ideal_qubit_projectors(
        res_dim=res_dim, qubit_dim=qubit_dim,
        res_mode_idx=res_mode_idx,
        superop=False,
    )

    if confusion_matrix is None:
        confusion_matrix = np.eye(qubit_dim)
    else:
        # check the normalization of the confusion matrix
        confusion_matrix = np.array(confusion_matrix)
        if not np.allclose(np.sum(confusion_matrix, axis=1), 1):
            raise ValueError("Each row of the confusion matrix should be "
                             "normalized to 1.")
        
        # check the dimension of the confusion matrix
        if confusion_matrix.shape != (qubit_dim, qubit_dim):
            raise ValueError("The confusion matrix should be a square matrix "
                             "with dimension qubit_dim.")
        
    # construct the measurement operators
    measurement_ops = np.empty_like(confusion_matrix, dtype=object)
    for idx, prob in np.ndenumerate(confusion_matrix):
        op = np.sqrt(prob) * projs[idx[0]]
        if superop:
            op = qt.to_super(op)
        measurement_ops[idx] = op

    if ensamble_average:
        measurement_ops = np.sum(measurement_ops, axis=0)
    else:
        measurement_ops = measurement_ops.ravel()

    return measurement_ops.tolist()
    
def qubit_gate(
    hilbertspace: HilbertSpace,
    res_mode_idx: int, qubit_mode_idx: int,
    res_truncated_dim: int | None = None, qubit_truncated_dim: int = 2,
    dressed_indices: np.ndarray | None = None, eigensys = None,
    rotation_angle: float = np.pi / 2,
    gate_params: Dict[str, Any] = {},
    num_cpus: int = 8,
):
    """
    qubit gate propagator in the lab frame.

    Parameters
    ----------
    hilbertspace: HilbertSpace
        scq.HilbertSpace object that contains a qubit and a resonator
    res_mode_idx, qubit_mode_idx: int
        The index of the resonator / qubit mode in the HilbertSpace
    res_truncated_dim: int | None
        The truncated dimension of the resonator mode. If None, the resonator mode will not be truncated unless a nan eigenvalue is found.
    """
    qubit = hilbertspace.subsystem_list[qubit_mode_idx]
    qubit_type = qubit.__class__.__name__

    # hamiltonian
    H0 = hilbertspace.hamiltonian() * np.pi * 2
    if qubit_type == "Transmon":
        H1 = scq.identity_wrap(
            qubit.n_operator(), 
            qubit, 
            hilbertspace.subsystem_list
        ) * np.pi * 2
    elif qubit_type == "Fluxonium":
        H1 = scq.identity_wrap(
            qubit.phi_operator(), 
            qubit, 
            hilbertspace.subsystem_list
        ) * np.pi * 2
    else:
        raise ValueError(f"Unknown qubit type {qubit_type}")

    # gate params
    res_n_bar = np.round(gate_params["disp"]**2).astype(int)
    evals, evec_arr = two_mode_dressed_esys(
        hilbertspace, 
        res_mode_idx, qubit_mode_idx, 
        state_label=(0, 0),
        res_truncated_dim=res_truncated_dim,
        qubit_truncated_dim=qubit_truncated_dim,
        dressed_indices=dressed_indices, eigensys=eigensys,
    )

    evals = evals * np.pi * 2
    bare_angular_freq = (evals[res_n_bar, 1] - evals[res_n_bar, 0])
    non_lin = (evals[res_n_bar, 2] - 2*evals[res_n_bar, 1] + evals[res_n_bar, 0])

    gate_in_subspace = oprt_in_basis(
        H1, evec_arr[res_n_bar, 0:qubit_truncated_dim]
    )
    tgt_mat_elem = gate_in_subspace[0, 1]
    leaking_mat_elem = gate_in_subspace[0, 2]
    
    # "normalize" Hamiltonian
    H0 = H0 - evals[0, 0]
    H1 = H1 / (tgt_mat_elem / np.abs(tgt_mat_elem))     # remove phase factor

    # scale the pulse duration for different rotation angle
    sigma = gate_params["sigma"] * np.abs(rotation_angle) / (np.pi / 2)
    duration = gate_params["tau_p_eff"] * np.abs(rotation_angle) / (np.pi / 2)

    if qubit_type == "Transmon":
        pulse = DRAGGaussian(
            base_angular_freq = bare_angular_freq,
            duration = duration,
            sigma = sigma, 
            non_lin = non_lin,
            order = 2, 
            rotation_angle = rotation_angle, 
            tgt_mat_elem = tgt_mat_elem,
            leaking_mat_elem = leaking_mat_elem,
            dynamic_drive_freq = False,
        )
    elif qubit_type == "Fluxonium":
        pulse = Gaussian(
            base_angular_freq = bare_angular_freq,
            duration = duration,
            sigma = sigma, 
            rotation_angle = rotation_angle, 
            tgt_mat_elem = tgt_mat_elem,
        )

    # simulate subspace by subspace
    def calculate_prop(basis):
        H0_ss = oprt_in_basis(H0, basis)
        H1_ss = oprt_in_basis(H1, basis)

        # global phase -> global phase % (2 * np.pi)
        phase_quantum = np.pi * 2 / pulse.duration
        cycle_num = np.round(H0_ss[1, 1] / phase_quantum)
        H0_ss = H0_ss - phase_quantum * cycle_num

        try:
            pulse.reset()
        except AttributeError:
            pass

        options = dict(
            nsteps=1000000,
            atol=1e-10,
        )
        if QUTIP_VERSION[0] < 5:
            options = qt.Options(**options)

        prop = qt.propagator(
            H = lambda t, args: H0_ss + H1_ss * pulse(t),
            t = pulse.duration,
            options = options,
        )

        return prop
    
    if num_cpus > 1:
        try:
            from multiprocess import Pool
        except ImportError:
            raise ImportError(
                "multiprocess is a optional dependency for bsqubits module."
                "Please install it via 'pip install multiprocess' or 'conda install multiprocess'."
            )
        with Pool(num_cpus) as pool:
            prop_list = list(pool.map(
                calculate_prop, evec_arr
            ))
    else:
        prop_list = list(map(
            calculate_prop, evec_arr
        ))
    
    prop_ds = direct_sum(*prop_list)
    # may not be correct - two_mode_dressed_esys function may return a 
    # partial baiss
    prop_ds.dims = [[res_truncated_dim, qubit_truncated_dim], [res_truncated_dim, qubit_truncated_dim]]

    return prop_ds
