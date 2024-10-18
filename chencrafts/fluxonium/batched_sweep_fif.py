import sympy as sp
import scqubits as scq
import numpy as np
import qutip as qt
import copy
import warnings

from qutip.solver.integrator.integrator import IntegratorException
from chencrafts.cqed.floquet import FloquetBasis
from chencrafts.cqed.qt_helper import oprt_in_basis, qobj_submatrix, Stinespring_to_Kraus
from chencrafts.cqed.custom_sweeps.general import standardize_evec_sign
from chencrafts.cqed.pulses import Gaussian
from chencrafts.toolbox.gadgets import mod_c
from chencrafts.toolbox.optimize import Optimization
from chencrafts.fluxonium.batched_sweep_frf import single_q_eye, eye2_wrap
from chencrafts.projects.nonstandard_2qbasis_gates.synth import OneLayerSynth

from typing import List, Tuple, Dict, Literal

# Static properties ====================================================
from chencrafts.fluxonium.batched_sweep_frf import (
    sweep_comp_drs_indices, 
    sweep_comp_bare_overlap, 
    sweep_static_zzz,
    sweep_hybridization,
)

Q1, Q2, Q3, θ1, θ2, θ3 = sp.symbols("Q1 Q2 Q3 θ1 θ2 θ3")

def sweep_sym_hamiltonian(
    ps: scq.ParameterSweep,
    **kwargs
):
    """
    Sweep the symbolic Hamiltonian and merge the kinetic and potential
    parts.
    
    Must be stored with key "sym_ham".
    """
    circ = ps.hilbertspace.subsystem_list[0].parent
    trans_mat = circ.transformation_matrix
    circ.configure(transformation_matrix=trans_mat) # recalc the hamiltonian
    ham = circ.sym_hamiltonian(return_expr=True)
    
    # ham has two parts and we need assemble them 
    terms = ham.as_ordered_terms()
    return terms[0] + terms[1]

def sweep_1q_ham_params(
    ps: scq.ParameterSweep,
    idx,
    q_idx: int,
    **kwargs
):
    """
    Must have "sym_ham" in the parameter sweep. Calculate the 
    Hamiltonian parameters for a single qubit, ordered by 
    [EC, EL].
    """
    ham = ps["sym_ham"][idx]
    
    Q_expr, θ_expr = sp.symbols(f"Q{q_idx+1} θ{q_idx+1}")
    EC = float(ham.coeff(Q_expr**2)) / 4
    EL = float(ham.coeff(θ_expr**2)) * 2
    
    return np.array([EC, EL])
    
def sweep_2q_ham_params(
    ps: scq.ParameterSweep,
    idx,
    q1_idx: int,
    q2_idx: int,
    **kwargs
):
    """
    Must have "sym_ham" in the parameter sweep. Calculate the 
    Hamiltonian parameters for two qubits, ordered by [JC, JL].
    """
    ham = ps["sym_ham"][idx]
    
    Q1_expr, Q2_expr = sp.symbols(f"Q{q1_idx+1} Q{q2_idx+1}")
    θ1_expr, θ2_expr = sp.symbols(f"θ{q1_idx+1} θ{q2_idx+1}")
    JCAB = float(ham.coeff(Q1_expr * Q2_expr))
    JLAB = float(ham.coeff(θ1_expr * θ2_expr))
    
    return np.array([JCAB, JLAB])

def sweep_dressed_freq_diff(
    ps: scq.ParameterSweep,
    idx,
    q_idx: int,
    num_q: int,
    num_r: int,
):
    """
    Get the dressed frequency difference between the ground state and the first
    excited state of qubit q_idx.
    
    must be stored with key f"dressed_freq_{q_idx}"
    """
    dims = ps.hilbertspace.subsystem_dims  
    bare_init = [0] * (num_q + num_r)
    raveled_init = np.ravel_multi_index(bare_init, tuple(dims))
    drs_label = ps["dressed_indices"][idx][raveled_init]
    eval_init = ps["evals"][idx][drs_label]
    
    bare_final = [0] * (num_q + num_r)
    bare_final[q_idx] = 1
    raveled_final = np.ravel_multi_index(bare_final, tuple(dims))
    drs_label = ps["dressed_indices"][idx][raveled_final]
    eval_final = ps["evals"][idx][drs_label]
    
    return eval_final - eval_init

def sweep_freq_diff(
    ps: scq.ParameterSweep,
    idx,
    q1_idx: int,
    q2_idx: int,
    mode: str = "dressed",
):
    if mode == "bare":
        evals_q1 = ps[f"bare_evals"][q1_idx][idx]
        evals_q2 = ps[f"bare_evals"][q2_idx][idx]
        freq_q1 = evals_q1[1] - evals_q1[0]
        freq_q2 = evals_q2[1] - evals_q2[0]
    elif mode == "dressed":
        freq_q1 = ps[f"dressed_freq_{q1_idx}"][idx]
        freq_q2 = ps[f"dressed_freq_{q2_idx}"][idx]
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return freq_q1 - freq_q2

def sweep_diff_of_freq_diff(
    ps: scq.ParameterSweep,
    idx,
    q1_idx: int,
    q2_idx: int,
    q3_idx: int,
    mode: str = "dressed",
):
    if mode == "bare":
        evals_q1 = ps[f"bare_evals"][q1_idx][idx]
        evals_q2 = ps[f"bare_evals"][q2_idx][idx]
        evals_q3 = ps[f"bare_evals"][q3_idx][idx]
        
        freq_q1 = evals_q1[1] - evals_q1[0]
        freq_q2 = evals_q2[1] - evals_q2[0]
        freq_q3 = evals_q3[1] - evals_q3[0]
        
    elif mode == "dressed":
        freq_q1 = ps[f"dressed_freq_{q1_idx}"][idx]
        freq_q2 = ps[f"dressed_freq_{q2_idx}"][idx]
        freq_q3 = ps[f"dressed_freq_{q3_idx}"][idx]
        
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return freq_q1 + freq_q3 - 2 * freq_q2

def batched_sweep_CR_static(
    ps: scq.ParameterSweep,
    num_q: int,
    comp_labels: List[Tuple[int, ...]],
    CR_bright_map: Dict[Tuple[int, int], int],
    sweep_ham_params: bool = True,
    **kwargs
):
    """
    Static properties:
    - sym_ham
    - ham_param_Q{q_idx}: [EC, EL]
    - ham_param_Q{q1_idx}_Q{q2_idx}: [JC, JL]
    - comp_drs_indices: the dressed indices of the components
    - comp_bare_overlap: the minimal overlap on bare basis
    - static_zzz
    """
    if sweep_ham_params:
        ps.add_sweep(sweep_sym_hamiltonian, "sym_ham")
        
        for q_idx in range(num_q):
            ps.add_sweep(sweep_1q_ham_params, f"ham_param_Q{q_idx}", q_idx = q_idx)

        for q1_idx in range(num_q):
            for q2_idx in range(q1_idx + 1, num_q):
                ps.add_sweep(sweep_2q_ham_params, f"ham_param_Q{q1_idx}_Q{q2_idx}", q1_idx = q1_idx, q2_idx = q2_idx)
            
    if "comp_drs_indices" not in ps.keys():
        ps.add_sweep(
            sweep_comp_drs_indices,
            sweep_name = 'comp_drs_indices',
            comp_labels = comp_labels,
        )
    if "hybridization" not in ps.keys():
        ps.add_sweep(
            sweep_hybridization,
            sweep_name = 'hybridization',
        )
    if "comp_bare_overlap" not in ps.keys():
        ps.add_sweep(
            sweep_comp_bare_overlap,
            sweep_name = 'comp_bare_overlap',
            comp_labels = comp_labels,
        )
    if "static_zzz" not in ps.keys():
        ps.add_sweep(
            sweep_static_zzz,
            sweep_name = 'static_zzz',
            comp_labels = comp_labels,
        )
    
    for q_idx in range(num_q):
        ps.add_sweep(
            sweep_dressed_freq_diff,
            sweep_name = f"dressed_freq_{q_idx}",
            q_idx = q_idx,
            num_q = num_q,
            num_r = num_r,  
        )

    for q1_idx, q2_idx in CR_bright_map.keys():
        ps.add_sweep(
            sweep_freq_diff,
            sweep_name = f"freq_diff_{q1_idx}_{q2_idx}",
            q1_idx = q1_idx,
            q2_idx = q2_idx,
            mode = "dressed",
        )
        
    # find all three-qubit chains
    for q_c in range(num_q):
        for q_l in range(num_q):
            for q_r in range(num_q):
                if (
                    (q_l, q_c) in CR_bright_map.keys() 
                    or (q_c, q_l) in CR_bright_map.keys() 
                ) and (
                    (q_c, q_r) in CR_bright_map.keys() 
                    or (q_r, q_c) in CR_bright_map.keys()
                ):
                    if (
                        f"diff_of_freq_diff_{q_l}_{q_c}_{q_r}" in ps.keys()
                        or f"diff_of_freq_diff_{q_r}_{q_c}_{q_l}" in ps.keys()
                    ):
                        continue
                
                    ps.add_sweep(
                        sweep_diff_of_freq_diff,
                        sweep_name = f"diff_of_freq_diff_{q_l}_{q_c}_{q_r}",
                        q1_idx = q_l,
                        q2_idx = q_c,
                        q3_idx = q_r,
                        mode = "dressed",
                    )

        
# Gate ingredients =====================================================
from chencrafts.fluxonium.batched_sweep_frf import fill_in_target_transitions

def sweep_default_target_transitions(
    ps: scq.ParameterSweep, 
    q1_idx: int, 
    q2_idx: int, 
    bright_state_label: int,
    num_q: int,
    num_r: int,
    **kwargs
):
    """
    Default target transitions: (1, 0, 1) -> (1, 1, 1) like.
    
    Must be saved with key f'target_transitions_{q1_idx}_{q2_idx}'
    
    Parameters
    ----------
    ps : scqubits.ParameterSweep
        The parameter sweep object.
    idx : int
        The index of the parameter set to sweep.
    q1_idx : int
        The index of the first qubit, starts from 0. It's the one to be 
        driven.
    q2_idx : int
        The index of the second qubit, starts from 0, whose frequency will
        be the drive frequency
    bright_state_label : int
        The label of the "bright" state, 0 or 1.
    num_q : int
        The number of qubits.
    num_r : int
        The number of resonators / spurious modes.
        
    Returns
    -------
    transitions_to_drive : np.ndarray
        A 3D array of init and final state pairs, dimensions: 
        0. different spectator states, 
        1. init & final state
        2. state label
    """
    # all init and final state pairs -----------------------------------
    # (actually final states are just intermediate states)
    
    all_q_id = range(num_q)
    q_spec = [q for q in all_q_id if q != q1_idx and q != q2_idx]

    # transitions_to_drive is a 4D array, dimensions: 
    # 0. [length-(2**(num_q-2))] different spectator states,
    # 1. [length-2] bright & dark transitions
    # 2. [length-2] init & final state
    # 3. [length-(num_q+num_r)] state label
    transitions_to_drive = []
    for q_spec_idx in np.ndindex((2,) * len(q_spec)):
        # qubit states, we specify states for q1 and q2,
        # and vary different spectator qubits' states
        # something like (000) and (010) if q1_idx = 0 and q2_idx = 1, spectator = 2
        init_qubit_bright_state = [0] * num_q
        init_qubit_bright_state[q1_idx] = bright_state_label
        init_qubit_bright_state[q2_idx] = 0
        for q_spec_id, q_spec_state in enumerate(q_spec_idx):
            init_qubit_bright_state[q_spec[q_spec_id]] = q_spec_state

        # add suprious modes, becomes something like (000 00)
        init_bright_state = init_qubit_bright_state + [0] * num_r 
        
        # final state
        final_bright_state = copy.copy(init_bright_state)
        final_bright_state[q2_idx] = 1
        
        # dark states
        init_dark_state = copy.copy(init_bright_state)
        final_dark_state = copy.copy(final_bright_state)
        init_dark_state[q1_idx] = 1 - bright_state_label    # 0 <-> 1
        final_dark_state[q1_idx] = 1 - bright_state_label    

        transitions_to_drive.append([
            [init_bright_state, final_bright_state],
            [init_dark_state, final_dark_state]
        ])

    return np.array(transitions_to_drive)

def sweep_drs_target_trans(
    ps: scq.ParameterSweep, 
    idx,
    q1_idx: int, 
    q2_idx: int, 
    **kwargs
):
    """
    Get the dressed target transitions, must be called after 
    sweep_default_target_transitions or any other sweeps that get
    target_transitions.
    
    Must be saved with key f'drs_target_trans_{q1_idx}_{q2_idx}'.
    """
    target_transitions = ps[f"target_transitions_{q1_idx}_{q2_idx}"][idx]
    
    # drs_targ_trans is a 3D array, dimensions: 
    # 0. different spectator states, 
    # 1. bright & dark transitions
    # 2. init & final state (scaler)
    drs_targ_trans = []
    for transitions in target_transitions:
        drs_targ_trans.append([])
        for init, final in transitions:
            raveled_init = np.ravel_multi_index(init, tuple(ps.hilbertspace.subsystem_dims))
            raveled_final = np.ravel_multi_index(final, tuple(ps.hilbertspace.subsystem_dims))
            drs_targ_trans[-1].append(
                [
                    ps["dressed_indices"][idx][raveled_init], 
                    ps["dressed_indices"][idx][raveled_final]
                ]
            )

    return np.array(drs_targ_trans)

def sweep_target_freq(
    ps: scq.ParameterSweep,
    idx,
    q1_idx: int,
    q2_idx: int,
):
    """
    The target transition frequency, must be called after 
    sweep_drs_target_trans. 
    
    Must be saved with key f'target_freq_{q1_idx}_{q2_idx}'.
    """  
    drs_trans = ps[f"drs_target_trans_{q1_idx}_{q2_idx}"][idx]
    evals = ps["evals"][idx]
    
    # freqs is a 2D array, dimensions: 
    # 0. different spectator states, 
    # 1. bright & dark transition frequency
    freqs = []
    for bright_and_dark_trans in drs_trans:
        freqs.append([])
        for init, final in bright_and_dark_trans:
            eval_i = evals[init]
            eval_f = evals[final]
            freqs[-1].append(eval_f - eval_i)
        
    return np.array(freqs)

def sweep_drive_freq(
    ps: scq.ParameterSweep, 
    idx,
    q1_idx,
    q2_idx,
):
    """
    maybe there will other methods to make drive freq more accurate
    e.g. try to make the Floquet modes to be an equal superposition of 
    the driven states. 
    
    For the moment, it's just the average of all bright transition freqs.
    
    Must be saved with key f'drive_freq_{q1_idx}_{q2_idx}'.
    """
    # base drive freq = average of all bright transition freqs
    drive_freq = np.average(
        ps[f"target_freq_{q1_idx}_{q2_idx}"][idx][:, 0]
    ) * np.pi * 2   # average over different spectator states
    
    # maybe there will other methods to make drive freq more accurate
    # e.g. try to make the Floquet modes to be an equal superposition of 
    # the driven states
    
    return drive_freq

def sweep_drive_op(
    ps: scq.ParameterSweep,
    idx,
    q_idx,
    num_q, 
    num_r,
    trunc: int = 30,
    **kwargs
):
    """
    Calculate the drive operator for a single qubit.
    
    Must be saved with key f'drive_op_{q_idx}'.
    """
    try:
        qubit = ps.hilbertspace.subsystem_list[q_idx]
        qubit_n_op_bare = qubit.n_operator()
        qubit_n_op = scq.identity_wrap(
            qubit_n_op_bare, qubit, ps.hilbertspace.subsystem_list
        ),
    except AttributeError:
        # obtain the opertor directly from the circuit
        # NOTE: DO NOT OBTAIN THE OPERATOR FROM THE SUBSYSTEM:
        #       e.g. hspace.subsystem_list[q_idx].Q1_operator() is not 
        #       updated during the sweep
        hspace = ps.hilbertspace
        circ = hspace.subsystem_list[0].parent
        dims = hspace.subsystem_dims
        
        var_q_idx = q_idx + 1
        Q_str = f"Q{var_q_idx}"
        op_name = str(f"{Q_str}_operator")
        qubit_n_op_sparse = getattr(circ, op_name)()  
        
        qubit_n_op = qt.Qobj(qubit_n_op_sparse, dims = [dims, dims])
        
    drive_op = oprt_in_basis(
        qubit_n_op,
        ps["evecs"][idx][:trunc]
    )
    
    # standardize the sign: 0 -> 1 transition must be 1j (like sigma_y)
    dims = tuple(ps.hilbertspace.subsystem_dims)
    bare_label_0 = [0] * (num_q + num_r)
    bare_label_1 = copy.copy(bare_label_0)
    bare_label_1[q_idx] = 1
    raveled_bare_label_0 = np.ravel_multi_index(bare_label_0, dims)
    raveled_bare_label_1 = np.ravel_multi_index(bare_label_1, dims)
    drs_label_0 = ps["dressed_indices"][idx][raveled_bare_label_0]
    drs_label_1 = ps["dressed_indices"][idx][raveled_bare_label_1]
    
    drive_op_mat_elem = drive_op[drs_label_0, drs_label_1]
    drive_op_phase = drive_op_mat_elem / np.abs(drive_op_mat_elem)

    return drive_op / drive_op_phase * 1j

def sweep_drive_mat_elem(
    ps: scq.ParameterSweep,
    idx,
    q1_idx,
    q2_idx,
):
    """
    A 2 * 2 matrix representing:
    
    [
        [Q1_bright, Q2_bright],
        [Q1_dark, Q2_dark]
    ]
    
    Must be saved with key f'drive_mat_elem_{q1_idx}_{q2_idx}'.
    """
    Q1_op = ps[f"drive_op_{q1_idx}"][idx]
    Q2_op = ps[f"drive_op_{q2_idx}"][idx]
    
    mat_elem = []
    # mat_elem is a 3D array, dimensions: 
    # 0. different spectator states, 
    # 1. bright & dark transitions
    # 2. Q1 & Q2 operator
    for trans in ps[f"drs_target_trans_{q1_idx}_{q2_idx}"][idx]:
        (
            (bright_init, bright_final), (dark_init, dark_final)
        ) = trans
        
        mat_elem.append([
            [
                Q1_op[bright_init, bright_final], 
                Q2_op[bright_init, bright_final],
            ], [
                Q1_op[dark_init, dark_final],
                Q2_op[dark_init, dark_final],
            ]
        ])
        
    return np.array(mat_elem)

def sweep_drive_amp(
    ps: scq.ParameterSweep, 
    idx,
    q1_idx,
    q2_idx,
):
    """
    Must be saved with key f'drive_amp_{q1_idx}_{q2_idx}'.
    """
    param_mesh = ps.parameters.meshgrid_by_name()
    
    try:
        amp = param_mesh[f"amp_{q1_idx}_{q2_idx}"][idx]
        
        if "amp" in param_mesh.keys():
            warnings.warn(f"Both of 'amp_{q1_idx}_{q2_idx}' and 'amp' are "
                          f"in the parameters, take 'amp_{q1_idx}_{q2_idx}' "
                          f"as the amplitude.")
    except KeyError:
        amp = param_mesh["amp"][idx]
        
    drive_mat_elem = np.average(
        ps[f"drive_mat_elem_{q1_idx}_{q2_idx}"][idx], axis=0
    )       # average over different spectator states
    
    # trying to cancel out the dark transition drive amp,
    # they are purely imaginary
    amp_q1, amp_q2 = np.linalg.solve(drive_mat_elem, [amp, 0])
    
    return np.array([amp_q1.imag, amp_q2.imag])

def sweep_sum_drive_op(
    ps: scq.ParameterSweep,
    idx,
    q1_idx,
    q2_idx,
):
    """
    Must be saved with key f'sum_drive_op_{q1_idx}_{q2_idx}'.
    """
    drive_op1 = ps[f"drive_op_{q1_idx}"][idx]
    drive_op2 = ps[f"drive_op_{q2_idx}"][idx]
    amp_q1, amp_q2 = ps[f"drive_amp_{q1_idx}_{q2_idx}"][idx]
    return amp_q1 * drive_op1 + amp_q2 * drive_op2

def batched_sweep_CR_ingredients(
    ps: scq.ParameterSweep,
    num_q: int,
    num_r: int,
    trunc: int,
    comp_labels: List[Tuple[int, ...]],
    CR_bright_map: Dict[Tuple[int, int], int],
    add_default_target: bool = True,
    **kwargs
):
    """
    Get the target transition frequency, must be called after 
    sweep_drs_target_trans.
    """
    # standardize the sign of eigenvectors
    ps.add_sweep(
        standardize_evec_sign,
        sweep_name = "evecs",
        state_labels = comp_labels,
    )
    
    for q_idx in range(num_q):
        ps.add_sweep(
            sweep_drive_op,
            sweep_name = f'drive_op_{q_idx}',
            q_idx = q_idx,
            num_q = num_q,
            num_r = num_r,
            trunc = trunc,
        )
        
    for (q1_idx, q2_idx), bright_state_label in CR_bright_map.items():
        if add_default_target:
            ps.add_sweep(
                sweep_default_target_transitions,
                sweep_name = f'target_transitions_{q1_idx}_{q2_idx}',
                q1_idx = q1_idx,
                q2_idx = q2_idx,
                bright_state_label = bright_state_label,
                num_q = num_q,
                num_r = num_r,
            )
        
        ps.add_sweep(
            sweep_drs_target_trans,
            sweep_name = f'drs_target_trans_{q1_idx}_{q2_idx}',
            q1_idx = q1_idx,
            q2_idx = q2_idx,
        )
        ps.add_sweep(
            sweep_target_freq,
            sweep_name = f'target_freq_{q1_idx}_{q2_idx}',
            q1_idx = q1_idx,
            q2_idx = q2_idx,
        )
        
        target_freq = ps[f'target_freq_{q1_idx}_{q2_idx}']
        ps.store_data(**{
            "dynamical_zzz_" + f'{q1_idx}_{q2_idx}': np.std(target_freq, axis=-2)
        })  
    
        ps.add_sweep(
            sweep_drive_mat_elem,
            sweep_name = f'drive_mat_elem_{q1_idx}_{q2_idx}',
            q1_idx = q1_idx,
            q2_idx = q2_idx,
        )
        
        ps.add_sweep(
            sweep_drive_amp,
            sweep_name = f'drive_amp_{q1_idx}_{q2_idx}',
            q1_idx = q1_idx,
            q2_idx = q2_idx,
        )
        ps.add_sweep(
            sweep_sum_drive_op,
            sweep_name = f'sum_drive_op_{q1_idx}_{q2_idx}',
            q1_idx = q1_idx,
            q2_idx = q2_idx,
        )
        ps.add_sweep(
            sweep_drive_freq,
            sweep_name = f'drive_freq_{q1_idx}_{q2_idx}',
            q1_idx = q1_idx,
            q2_idx = q2_idx,
        )   
        

# single qubit gate =======================================================
def sweep_single_qubit_gate(
    ps: scq.ParameterSweep,
    idx,
    q_idx,
    num_q: int,
    num_r: int,
    trunc: int = 30,
    max_amp_1Q: float = 0.1,    # the maximum amplitude of the drive pulse
    min_larmor_period: int = 5,  # the minimum number of qubit larmor periods in a gate
):
    # transition to drive ----------------------------------------------
    dims = ps.hilbertspace.subsystem_dims  
    bare_init = [0] * (num_q + num_r)
    raveled_init = np.ravel_multi_index(bare_init, tuple(dims))
    drs_init = ps["dressed_indices"][idx][raveled_init]
    eval_init = ps["evals"][idx][drs_init]
    
    bare_final = [0] * (num_q + num_r)
    bare_final[q_idx] = 1
    raveled_final = np.ravel_multi_index(bare_final, tuple(dims))
    drs_final = ps["dressed_indices"][idx][raveled_final]
    eval_final = ps["evals"][idx][drs_final]

    # pulse parameters -------------------------------------------------
    ham = qt.qdiags(ps["evals"][idx][:trunc], 0) * np.pi * 2
    drive_freq = eval_final - eval_init
    drive_op = ps[f"drive_op_{q_idx}"][idx]
    drive_amp = np.min([drive_freq / min_larmor_period, max_amp_1Q])
    
    drive_op = drive_op / np.abs(drive_op[drs_init, drs_final]) * drive_amp
    
    # construct a time-dependent hamiltonian
    ham_t = [
        ham,
        [
            sum_drive_op, 
            f"cos({drive_freq}*t)"
        ]
    ]
    
    # Floquet analysis and gate calibration ----------------------------
    T = np.pi * 2 / drive_freq
    try:
        fbasis = FloquetBasis(
            H = ham_t, 
            T = T,
            options = {
                "rtol": 1e-10,
                "atol": 1e-10,
                "nsteps": 10000000,
            }
        )
    except IntegratorException as e:
        warnings.warn(f"At idx: {idx}, q1_idx: {q1_idx}, q2_idx: {q2_idx}, "
                     f"Floquet basis integration failed with error: {e}")
        return np.array([np.nan, None, None], dtype=object)
    
    fevals = fbasis.e_quasi
    fevecs = fbasis.mode(0)
    
    # Rabi amplitude for bright states
    Rabi_amp_list = []

    for init, final in drs_trans[:, 0, :]:
        drs_state_init = qt.basis(ham.shape[0], init)
        drs_state_final = qt.basis(ham.shape[0], final)
        drs_plus = (drs_state_init + 1j * drs_state_final).unit()   # 1j comes from driving charge matrix (sigma_y)
        drs_minus = (drs_state_init - 1j * drs_state_final).unit()
        f_idx_plus, _ = fbasis._closest_state(fevecs, drs_plus)  # we put the |+> state in the qubit state list
        f_idx_minus, _ = fbasis._closest_state(fevecs, drs_minus) # we put the |1> state in the resonator list 
        
        if (
            init is None 
            or final is None 
            or f_idx_plus is None 
            or f_idx_minus is None
        ):
            warnings.warn(
                f"At idx: {idx}, q1_idx: {q1_idx}, q2_idx: {q2_idx}, init "
                "state: {init}, final state: {final}. "
                "Driven state identification failed. It's usually due to "
                "strongly driving / coupling to the unwanted transitions. Please check "
                "the system config."
            )
            Rabi_amp_list.append(np.nan)
            continue
        
        # it could be used to calibrate a gate time to complete a rabi cycle
        Rabi_amp = mod_c(
            fevals[f_idx_minus] - fevals[f_idx_plus],
            drive_freq
        )
        Rabi_amp_list.append(np.abs(Rabi_amp))
        
    # gate time
    gate_time = np.pi / np.average(Rabi_amp_list)
    
    # full unitary -----------------------------------------------------
    try:
        unitary = fbasis.propagator(gate_time)
    except IntegratorException as e:
        warnings.warn(f"At idx: {idx}, q1_idx: {q1_idx}, q2_idx: {q2_idx}, "
                     f"Floquet propagator integration failed with error: {e}")
        return np.array([np.nan, None, None], dtype=object) 
    
    # rotating frame
    rot_unit = (-1j * ham * gate_time).expm()
    rot_prop = rot_unit.dag() * unitary

    return np.array([gate_time, fbasis, rot_prop], dtype=object)


def batched_sweep_1Q_gate(
    ps: scq.ParameterSweep,
    num_q: int,
    num_r: int,
):
    pass

# CR gate ==============================================================
def sweep_CR_propagator(
    ps: scq.ParameterSweep,
    idx,
    q1_idx,
    q2_idx,
    trunc: int = 30,
):
    drs_trans = ps[f"drs_target_trans_{q1_idx}_{q2_idx}"][idx]

    # pulse parameters -------------------------------------------------
    ham = qt.qdiags(ps["evals"][idx][:trunc], 0) * np.pi * 2
    drive_freq = ps[f"drive_freq_{q1_idx}_{q2_idx}"][idx]
    sum_drive_op = ps[f"sum_drive_op_{q1_idx}_{q2_idx}"][idx]
    
    if drive_freq < 0:
        warnings.warn(f"At idx: {idx}, q1_idx: {q1_idx}, q2_idx: {q2_idx}, "
                      f"Found negative drive freqs, potentially due to "
                      f"wrong dressed label assignment.")
        return np.array([np.nan, None, None], dtype=object) 

    # construct a time-dependent hamiltonian
    ham_t = [
        ham,
        [
            sum_drive_op, 
            f"cos({drive_freq}*t)"
        ]
    ]
    
    # Floquet analysis and gate calibration ----------------------------
    T = np.pi * 2 / drive_freq
    try:
        fbasis = FloquetBasis(
            H = ham_t, 
            T = T,
            options = {
                "rtol": 1e-10,
                "atol": 1e-10,
                "nsteps": 10000000,
            }
        )
    except IntegratorException as e:
        warnings.warn(f"At idx: {idx}, q1_idx: {q1_idx}, q2_idx: {q2_idx}, "
                     f"Floquet basis integration failed with error: {e}")
        return np.array([np.nan, None, None], dtype=object)
    
    fevals = fbasis.e_quasi
    fevecs = fbasis.mode(0)
    
    # Rabi amplitude for bright states
    Rabi_amp_list = []

    for init, final in drs_trans[:, 0, :]:
        drs_state_init = qt.basis(ham.shape[0], init)
        drs_state_final = qt.basis(ham.shape[0], final)
        drs_plus = (drs_state_init + 1j * drs_state_final).unit()   # 1j comes from driving charge matrix (sigma_y)
        drs_minus = (drs_state_init - 1j * drs_state_final).unit()
        f_idx_plus, _ = fbasis._closest_state(fevecs, drs_plus)  # we put the |+> state in the qubit state list
        f_idx_minus, _ = fbasis._closest_state(fevecs, drs_minus) # we put the |1> state in the resonator list 
        
        if (
            init is None 
            or final is None 
            or f_idx_plus is None 
            or f_idx_minus is None
        ):
            warnings.warn(
                f"At idx: {idx}, q1_idx: {q1_idx}, q2_idx: {q2_idx}, init "
                "state: {init}, final state: {final}. "
                "Driven state identification failed. It's usually due to "
                "strongly driving / coupling to the unwanted transitions. Please check "
                "the system config."
            )
            Rabi_amp_list.append(np.nan)
            continue
        
        # it could be used to calibrate a gate time to complete a rabi cycle
        Rabi_amp = mod_c(
            fevals[f_idx_minus] - fevals[f_idx_plus],
            drive_freq
        )
        Rabi_amp_list.append(np.abs(Rabi_amp))
        
    # gate time
    gate_time = np.pi / np.average(Rabi_amp_list)
    
    # full unitary -----------------------------------------------------
    try:
        unitary = fbasis.propagator(gate_time)
    except IntegratorException as e:
        warnings.warn(f"At idx: {idx}, q1_idx: {q1_idx}, q2_idx: {q2_idx}, "
                     f"Floquet propagator integration failed with error: {e}")
        return np.array([np.nan, None, None], dtype=object) 
    
    # rotating frame
    rot_unit = (-1j * ham * gate_time).expm()
    rot_prop = rot_unit.dag() * unitary

    return np.array([gate_time, fbasis, rot_prop], dtype=object)

def CR_Gaussian_solve(
    ps: scq.ParameterSweep,
    idx,
    q1_idx,
    q2_idx,
    gate_time,
    detuning,
    gate_time_by_sigma,
    max_rabi_amp,
    trunc: int = 30,
    return_type: Literal["pop", "U"] = "pop",
):
    drs_trans = ps[f"drs_target_trans_{q1_idx}_{q2_idx}"][idx]

    # pulse parameters -------------------------------------------------
    ham = qt.qdiags(ps["evals"][idx][:trunc], 0) * np.pi * 2
    drive_freq = ps[f"drive_freq_{q1_idx}_{q2_idx}"][idx]
    sum_drive_op = ps[f"sum_drive_op_{q1_idx}_{q2_idx}"][idx]
    
    # we just pick one of the bright transition that's going to be driven
    init_drs_idx, final_drs_idx = drs_trans[0][0]

    # construct a time-dependent hamiltonian
    pulse = Gaussian(
        base_angular_freq = drive_freq + detuning,
        duration = gate_time,
        sigma = gate_time / gate_time_by_sigma,
        rotation_angle = np.pi,
        tgt_mat_elem = sum_drive_op[final_drs_idx, init_drs_idx],
    )
    pulse.rabi_amp = max_rabi_amp    # rabi_amp ~ drive_amp * tgt_mat_elem
    
    ham_t = lambda t, *args: ham + pulse(t) * sum_drive_op
    
    # Time dynamical simulation ----------------------------------------
    
    if return_type == "pop":
        # return the population of the final state, 
        # for optimization
        init_state = qt.basis(trunc, init_drs_idx)
        final_state = qt.basis(trunc, final_drs_idx)
        
        res = qt.sesolve(
            H = ham_t,
            psi0 = init_state,
            tlist = np.linspace(0, gate_time, 2),
            e_ops = [final_state * final_state.dag()],
            options = {"nsteps": 1000000}
        )
        
        return res.expect[0][-1]
    
    else:
        # return the propagator, for verification
        res = qt.propagator(
            H = ham_t,
            t = gate_time,
            options = {"nsteps": 1000000}
        )
        
        # rotating frame
        rot_unit = (-1j * ham * gate_time).expm()
        rot_prop = rot_unit.dag() * res
        
        return rot_prop

def sweep_Gaussian_params(
    ps: scq.ParameterSweep,
    idx,
    q1_idx,
    q2_idx,
    gate_time_by_sigma = 4,
    trunc: int = 30,
):
    # Since we specify the gate parameter using amp for a square pulse, 
    # we borrow it to the Gaussian. We first calculate what is the gate 
    # time for a square pulse and fix it.
    param_mesh = ps.parameters.meshgrid_by_name()
    try:
        amp = param_mesh[f"amp_{q1_idx}_{q2_idx}"][idx]
        if "amp" in param_mesh.keys():
            warnings.warn(f"Both of 'amp_{q1_idx}_{q2_idx}' and 'amp' are "
                          f"in the parameters, take 'amp_{q1_idx}_{q2_idx}' "
                          f"as the amplitude.")
    except KeyError:
        amp = param_mesh["amp"][idx]
    gate_time = np.pi / amp
    
    # to get a sense of the max_rabi_amp, we can integrate the area under the
    # pulse envelope. It's realized in Gaussian pulse class already:
    temp_pulse = Gaussian(
        base_angular_freq = 1,
        duration = gate_time,
        sigma = gate_time / gate_time_by_sigma,
        rotation_angle = np.pi,
    )
    approx_gaussian_amp = temp_pulse.rabi_amp
    
    # construct optimization parameters
    fixed_param = {
        "gate_time": gate_time,
        "gate_time_by_sigma": gate_time_by_sigma,
        "detuning": 0,
    }
    
    free_param_range = {
        "max_rabi_amp": [0.95*approx_gaussian_amp, 1.05*approx_gaussian_amp],       
        # "detuning": [-1e-2, 1e-2],  # ns^-1
    }
    
    def target(params):
        return 1 - CR_Gaussian_solve(
            ps = ps, 
            idx = idx, 
            q1_idx = q1_idx, 
            q2_idx = q2_idx, 
            gate_time = gate_time, 
            gate_time_by_sigma = gate_time_by_sigma, 
            trunc = trunc,
            max_rabi_amp = params["max_rabi_amp"],
            detuning = params["detuning"],
            return_type = "pop",
        )
    
    opt = Optimization(
        fixed_variables = fixed_param,
        free_variable_ranges = free_param_range,
        target_func = target,
        opt_options = {
            "disp": False,
        }
    )
    
    traj = opt.run()
    
    return traj.final_full_para
    
def sweep_CR_propagator_Gaussian(
    ps: scq.ParameterSweep,
    idx,
    q1_idx,
    q2_idx,
    trunc: int = 30,
):
    g_param = ps[f"gaussian_params_{q1_idx}_{q2_idx}"][idx]
    prop = CR_Gaussian_solve(
        ps = ps, 
        idx = idx, 
        q1_idx = q1_idx, 
        q2_idx = q2_idx, 
        return_type = "U",
        trunc = trunc,
        **g_param
    )

    return prop

def sweep_CR_comp(
    ps: scq.ParameterSweep,
    idx,
    q1_idx,
    q2_idx,
):
    rot_prop = ps[f"full_CR_{q1_idx}_{q2_idx}"][idx]
    
    if rot_prop is None:
        # some error occured in fbasis calculation
        return None
    
    # truncate to computational basis
    trunc = rot_prop.shape[0]
    comp_drs_indices = ps[f"comp_drs_indices"][idx]
    comp_drs_states = [
        qt.basis(trunc, index)
        for index in comp_drs_indices
    ]
    trunc_rot_unitary = oprt_in_basis(
        rot_prop,
        comp_drs_states,
    )

    return trunc_rot_unitary

def sweep_pure_CR(
    ps: scq.ParameterSweep, 
    idx, 
    q1_idx, 
    q2_idx,
    num_q,
):
    unitary = ps[f"CR_{q1_idx}_{q2_idx}"][idx]
    if unitary is None:
        # some error occured in fbasis calculation
        return None
    
    unitary.dims = [[2] * num_q] * 2
    
    eye_full = qt.tensor([single_q_eye] * num_q)
    phase_ops = [eye2_wrap(qt.projection(2, 1, 1), q_idx, num_q) for q_idx in range(num_q)]
    
    # remove global phase & dark phase ---------------------------------
    # mat_elem_col_by_row is a list of the column index of the largest 
    # element in each row
    unitary_arr = unitary.full()
    mat_elem_col_by_row = np.argmax(np.abs(unitary_arr), axis=1)
    phase = np.angle(unitary_arr[np.arange(unitary_arr.shape[0]), mat_elem_col_by_row])
    
    # remove global phase
    global_phase = phase[0]
    phase = phase - global_phase
    
    # Remove single qubit phase component for dark states. 
    # It should be done before dealing with bright states, as the correction
    # applied to dark states will change the phase for bright states.
    dark_phase_to_correct = []
    dark_correction_ops = []
    
    for q_idx in range(num_q):
        # we look at phase for states like (0, 0, 1, 0, 0, ...) that
        # have only one q_idx being 1
        state_label = [0] * num_q
        state_label[q_idx] = 1
        raveled_state_label = np.ravel_multi_index(state_label, (2,) * num_q)
        
        # in the transition unitary, we determine which state transfer to
        # |raveled_state_label>
        major_source_label = mat_elem_col_by_row[raveled_state_label]
        if major_source_label != raveled_state_label:
            # bright transitions
            continue

        ideal_phase = 0 
        dark_phase_to_correct.append(
            phase[raveled_state_label] - ideal_phase
        )
        dark_correction_ops.append(phase_ops[q_idx])
        
    # remove the global phase 
    unitary = (-1j * global_phase * eye_full).expm() * unitary
    
    # remove the dark phase
    for dark_phase, dark_op in zip(dark_phase_to_correct, dark_correction_ops):
        phase_dark_op = (-1j * dark_phase * dark_op / 2).expm() 
        unitary = (
            phase_dark_op
            * unitary
            * phase_dark_op
        )
    
    # remove single qubit phase for bright states ----------------------
    # calculate the remaining phase again
    unitary_arr = unitary.full()
    phase = np.angle(unitary_arr[np.arange(unitary_arr.shape[0]), mat_elem_col_by_row])
    
    bright_phase_to_correct = []
    bright_correction_ops = []
    for q_idx in range(num_q):
        # we look at phase for states like (0, 0, 1, 0, 0, ...) that
        # have only one q_idx being 1
        state_label = [0] * num_q
        state_label[q_idx] = 1
        raveled_state_label = np.ravel_multi_index(state_label, (2,) * num_q)
        
        # in the transition unitary, we determine which state transfer to
        # |raveled_state_label>
        major_source_label = mat_elem_col_by_row[raveled_state_label]
        if major_source_label == raveled_state_label:
            # dark state
            continue
            
        elif major_source_label > raveled_state_label:
            # bright state, this matrix element corresponds to sigma_y[0, 1]
            ideal_phase = - np.pi / 2
            actual_phase = mod_c(
                phase[raveled_state_label], 
                np.pi * 2, 
                ideal_phase
            )
            bright_phase_to_correct.append(actual_phase - ideal_phase)
            bright_correction_ops.append(phase_ops[q_idx])
        else:
            # bright state, this matrix element corresponds to sigma_y[1, 0]
            ideal_phase = np.pi / 2
            actual_phase = mod_c(
                phase[raveled_state_label], 
                np.pi * 2, 
                ideal_phase
            )
            bright_phase_to_correct.append(actual_phase - ideal_phase)
            bright_correction_ops.append(phase_ops[q_idx])

    # remove the bright phase
    for bright_phase, bright_op in zip(bright_phase_to_correct, bright_correction_ops):
        unitary = (
            (-1j * bright_phase * bright_op).expm() * unitary
        )

    return unitary

def sweep_target_unitary(
    ps: scq.ParameterSweep,
    idx,
    q1_idx,
    q2_idx,
    num_q,
):
    bare_trans = ps[f"target_transitions_{q1_idx}_{q2_idx}"][idx]
    
    # let's contruct the target matrix element by matrix element
    target = np.zeros((2**num_q, 2**num_q), dtype=complex)
    
    # dark states: identity
    for init, final in bare_trans[:, 1, :, :]:
        init_idx = np.ravel_multi_index(init[:num_q], (2,) * num_q)
        final_idx = np.ravel_multi_index(final[:num_q], (2,) * num_q)
        target[init_idx, init_idx] = 1
        target[final_idx, final_idx] = 1
        
    # bright states: sigma y
    for init, final in bare_trans[:, 0, :, :]:
        init_idx = np.ravel_multi_index(init[:num_q], (2,) * num_q)
        final_idx = np.ravel_multi_index(final[:num_q], (2,) * num_q)
        sign = int(init_idx < final_idx)
        target[init_idx, final_idx] = -1j * sign
        target[final_idx, init_idx] = 1j * sign
        
    target = qt.Qobj(target, dims=[[2] * num_q] * 2)
    
    return target

def sweep_fidelity(
    ps: scq.ParameterSweep, 
    idx, 
    q1_idx, 
    q2_idx,
    ignore_phase: bool = False,
):
    unitary = ps[f"pure_CR_{q1_idx}_{q2_idx}"][idx]
    if unitary is None:
        return np.nan
    
    target = ps[f"target_CR_{q1_idx}_{q2_idx}"][idx]
    
    if ignore_phase:
        warnings.warn("Phase is ignored in fidelity calculation.")
        unitary = qt.Qobj(np.abs(unitary.full()))
        target = qt.Qobj(np.abs(target.full()))

    # compute fidelity
    fidelity = qt.process_fidelity(
        unitary,
        target,
    )

    return fidelity

def sweep_unitarity(
    ps: scq.ParameterSweep,
    idx,
    q1_idx,
    q2_idx,
):
    pure_CR = ps[f"pure_CR_{q1_idx}_{q2_idx}"][idx]
    if pure_CR is None:
        return np.nan
    return qt.unitarity(pure_CR)

def sweep_reduced_CR(
    ps: scq.ParameterSweep,
    idx,
    q1_idx,
    q2_idx,
    num_q,
):
    pure_CR = ps[f"pure_CR_{q1_idx}_{q2_idx}"][idx]
    num_spectators = num_q - 2
    if pure_CR is None:
        if num_spectators == 0:
            return np.array(None, dtype=object)
        res = np.ndarray((2,) * num_spectators, dtype=object)
        return res
    
    if num_spectators == 0:
        return np.array(pure_CR, dtype=object)
    
    reduced_CR = np.ndarray((2,) * num_spectators, dtype=object)
    for spec_state_label in np.ndindex((2,) * num_spectators):
        Kraus_ops = Stinespring_to_Kraus(
            sys_env_prop = pure_CR,
            sys_indices = [q1_idx, q2_idx],
            env_state_label = spec_state_label,
        )
        reduced_CR[spec_state_label] = sum([qt.to_super(kop) for kop in Kraus_ops])
    
    return reduced_CR

def sweep_reduced_unitarity(
    ps: scq.ParameterSweep,
    idx,
    q1_idx,
    q2_idx,
):
    reduced_CR = ps[f"reduced_CR_{q1_idx}_{q2_idx}"][idx].ravel()
    if reduced_CR[0] is None:
        return np.nan
    unitarity = np.array([qt.unitarity(op) for op in reduced_CR])
    return np.mean(unitarity)

def sweep_CR_synthesis(
    ps: scq.ParameterSweep,
    idx,
    q1_idx,
    q2_idx,
    num_q,
):
    original_U = ps[f"pure_CR_{q1_idx}_{q2_idx}"][idx]
    target_U = ps[f"target_CR_{q1_idx}_{q2_idx}"][idx]
    if original_U is None:
        return None
    
    synth = OneLayerSynth(
        original_U, target_U,
        qubit_pair = (q1_idx, q2_idx),
        num_qubits = num_q,
    )
    synth.run()
    return synth

def sweep_synth_params(
    ps: scq.ParameterSweep,
    idx,
    q1_idx,
    q2_idx,
):
    # this one can't be vectorized like grabing other synth attributes
    synth = ps[f"synth_{q1_idx}_{q2_idx}"][idx]
    if synth is None:
        return np.zeros(8)
    return synth.params

def batched_sweep_CR(
    ps: scq.ParameterSweep,
    num_q: int,
    trunc: int,
    CR_bright_map: Dict[Tuple[int, int], int],
    ignore_phase: bool = False,
    gaussian_pulse: bool = False,
    **kwargs
):
    for (q1_idx, q2_idx), _ in CR_bright_map.items():
        # run the pulse
        if gaussian_pulse:
            ps.add_sweep(
                sweep_Gaussian_params,
                sweep_name = f'gaussian_params_{q1_idx}_{q2_idx}',
                q1_idx = q1_idx,
                q2_idx = q2_idx,
                trunc = trunc,
                gate_time_by_sigma = 4,
            )
            
            grab_gate_time = np.vectorize(lambda d: d['gate_time'])
            ps.store_data(**{
                f"gate_time_{q1_idx}_{q2_idx}": grab_gate_time(ps[f"gaussian_params_{q1_idx}_{q2_idx}"])
            })
            
            ps.add_sweep(
                sweep_CR_propagator_Gaussian,
                sweep_name = f'full_CR_{q1_idx}_{q2_idx}',
                q1_idx = q1_idx,
                q2_idx = q2_idx,
                trunc = trunc,
            )
        else:
            ps.add_sweep(
                sweep_CR_propagator,
                sweep_name = f'CR_results_{q1_idx}_{q2_idx}',
                q1_idx = q1_idx,
                q2_idx = q2_idx,
                trunc = trunc,
            )
            
            ps.store_data(**{
                f"gate_time_{q1_idx}_{q2_idx}": ps[f"CR_results_{q1_idx}_{q2_idx}"][..., 0].astype(float),
                f"full_CR_{q1_idx}_{q2_idx}": ps[f"CR_results_{q1_idx}_{q2_idx}"][..., 2],
            })
        
        # process the propagator for further analysis
        ps.add_sweep(
            sweep_CR_comp,
            sweep_name = f'CR_{q1_idx}_{q2_idx}',
            q1_idx = q1_idx,
            q2_idx = q2_idx,
        )
        ps.add_sweep(
            sweep_pure_CR,
            sweep_name = f'pure_CR_{q1_idx}_{q2_idx}',
            q1_idx = q1_idx,
            q2_idx = q2_idx,
            num_q = num_q,
        )
        ps.add_sweep(
            sweep_target_unitary,
            sweep_name = f'target_CR_{q1_idx}_{q2_idx}',
            q1_idx = q1_idx,
            q2_idx = q2_idx,
            num_q = num_q,
        )
        ps.add_sweep(
            sweep_fidelity,
            sweep_name = f'fidelity_{q1_idx}_{q2_idx}',
            q1_idx = q1_idx,
            q2_idx = q2_idx,
            ignore_phase = ignore_phase,
        )
        ps.add_sweep(
            sweep_unitarity,
            f"unitarity_{q1_idx}_{q2_idx}",
            q1_idx = q1_idx,
            q2_idx = q2_idx,
        )
        ps.add_sweep(
            sweep_reduced_CR,
            f"reduced_CR_{q1_idx}_{q2_idx}",
            q1_idx = q1_idx,
            q2_idx = q2_idx,
            num_q = 3,
        )
        ps.add_sweep(
            sweep_reduced_unitarity,
            f"reduced_unitarity_{q1_idx}_{q2_idx}",
            q1_idx = q1_idx,
            q2_idx = q2_idx,
        )
        
        # synthesize a better 2-qubit gate with 4 single-qubit gates and 
        # the original gate
        ps.add_sweep(
            sweep_CR_synthesis,
            sweep_name = f'synth_{q1_idx}_{q2_idx}',
            q1_idx = q1_idx,
            q2_idx = q2_idx,
            num_q = num_q,
        )
        grab_leakage = np.vectorize(
            lambda synth: synth.leakage 
            if synth is not None else np.nan
        )
        grab_synth_CR = np.vectorize(
            lambda synth: qt.Qobj(
                synth.synth_U, dims=[[2] * num_q] * 2
            ) if synth is not None else None
        )
        grab_synth_fidelity = np.vectorize(
            lambda synth: synth.synth_fidelity 
            if synth is not None else np.nan
        )
        ps.store_data(**{
            f"synth_CR_{q1_idx}_{q2_idx}": grab_synth_CR(ps[f"synth_{q1_idx}_{q2_idx}"]),
            f"leakage_{q1_idx}_{q2_idx}": grab_leakage(ps[f"synth_{q1_idx}_{q2_idx}"]),
            f"synth_fidelity_{q1_idx}_{q2_idx}": grab_synth_fidelity(ps[f"synth_{q1_idx}_{q2_idx}"]),
        })
        
        ps.add_sweep(
            sweep_synth_params,
            sweep_name = f'synth_params_{q1_idx}_{q2_idx}',
            q1_idx = q1_idx,
            q2_idx = q2_idx,
        )
            
# Cost function... =====================================================
from chencrafts.fluxonium.batched_sweep_frf import (
    sweep_qubit_coherence, 
    sweep_1Q_gate_time, 
    sweep_1Q_error, 
)

def sweep_CR_incoh_infid(
    ps: scq.ParameterSweep,
    idx,
    q1_idx,
    q2_idx,
):
    qubit_decay_rate = ps[f"qubit_decay"][idx]
    gate_time = ps[f"gate_time_{q1_idx}_{q2_idx}"][idx]
    
    return (
        (qubit_decay_rate[q1_idx] + qubit_decay_rate[q2_idx])
        * gate_time 
    )

def batched_sweep_incoh_infid_CR(
    ps: scq.ParameterSweep,
    num_q,
    CR_bright_map: Dict[Tuple[int, int], int],
    Q_cap = 1e6,
    Q_ind = 1e8,
    T = 0.05,
    cycle_per_gate = 4,
    zz_penalty = 1,
    sqg_tqg_ratio = 4,
    **kwargs,
):
    """
    Incoherent error infidility. Key:
    - off_ZZ_{q1_idx}_{q2_idx}: the off-diagonal ZZ coupling strength
    - qubit_decay: the qubit decay rate
    - tgt_decay_{q1_idx}_{q2_idx}: the target decay rate
    - CZ_incoh_infid_{q1_idx}_{q2_idx}: the incoherent error infidility of the CZ gate
    - 1Q_incoh_infid_{q1_idx}: the incoherent error infidility of the 1Q gate
    - error_{q1_idx}_{q2_idx}: the two-qubit error
    - error: the total error
    
    Parameters
    ----------
    num_q: 
        the number of qubits
    CR_bright_map: Dict[Tuple[int, int], int]
        Key: (q1_idx, q2_idx), drive q1 at q2's freq to realize a CR gate
        Value: bright state index (0 or 1), at the other state, the transition
        is selectively darkened
    Q_cap: 
        the qubit capacitive Q factor
    Q_ind: 
        the qubit inductive Q factor
    T: 
        the temperature
    cycle_per_gate: 
        the number of qubit Lamor cycles per single qubit gate
    sqg_tqg_ratio: 
        the ratio between the number of single qubit gates and the number of
        two qubit gates in an quantum algorithm. Serves as a weight to balance
        the error from single vs two qubit gates.
    zz_penalty: 
        the penalty for the ZZ coupling strength, turn zz in GHz to error
    """
    for q1_idx in range(num_q):
        for q2_idx in range(q1_idx + 1, num_q):
            ps.store_data(**{
                f"off_ZZ_{q1_idx}_{q2_idx}": ps[f"kerr"][q1_idx, q2_idx][..., 1, 1]
            })
    ps.add_sweep(
        sweep_qubit_coherence,
        sweep_name = f'qubit_decay',
        num_q = num_q,
        Q_cap = Q_cap,
        Q_ind = Q_ind,
        T = T,
    )
    ps.add_sweep(
        sweep_1Q_gate_time,
        sweep_name = f'1Q_gate_time',
        num_q = num_q,
        cycle_per_gate = cycle_per_gate,
        update_hilbertspace=False,
    )
    ps.add_sweep(
        sweep_1Q_error,
        sweep_name = f'1Q_error',
        update_hilbertspace=False,
        penalize_zz = False,
    )
        
    for (q1_idx, q2_idx) in CR_bright_map.keys():
        ps.add_sweep(
            sweep_CR_incoh_infid,
            sweep_name = f'CR_incoh_infid_{q1_idx}_{q2_idx}',
            q1_idx = q1_idx,
            q2_idx = q2_idx,
        )
    
    # summarize
    # two qubit error
    tot_error = 0
    for (q1_idx, q2_idx) in CR_bright_map.keys():
        two_q_error = (
            1 - ps[f"fidelity_{q1_idx}_{q2_idx}"]
            + ps[f"CR_incoh_infid_{q1_idx}_{q2_idx}"]
        )
        
        tot_error += two_q_error
        ps.store_data(**{f"error_{q1_idx}_{q2_idx}": two_q_error})
        
    tot_error /= len(CR_bright_map)     # average over all CZ gates
    
    # single qubit error
    single_q_error = np.sum(ps[f"1Q_error"], axis=-1) / num_q * sqg_tqg_ratio
    tot_error += single_q_error
    
    # penalize ZZ
    abs_zz = 0
    for q1_idx in range(num_q):
        for q2_idx in range(q1_idx + 1, num_q):
            abs_zz += np.abs(ps[f"off_ZZ_{q1_idx}_{q2_idx}"])
    cnt = num_q * (num_q - 1) // 2
    tot_error += abs_zz * zz_penalty / cnt
    
    ps.store_data(
        error = tot_error,
    )
    
# overall sweep
def batched_sweep_fidelity_CR(
    ps: scq.ParameterSweep,
    num_q: int,
    num_r: int,
    comp_labels: List[Tuple[int, ...]],
    CR_bright_map: Dict[Tuple[int, int], int],
    sweep_ham_params: bool = False,
    dynamical_truncation: int = 30,
    gaussian_pulse: bool = False,
    Q_cap = 1e6,
    Q_ind = 1e8,
    T = 0.05,
    cycle_per_gate = 4,
    zz_penalty = 1,
    ignore_phase: bool = False,
    **kwargs,
):
    batched_sweep_CR_static(
        ps,
        num_q = num_q,
        comp_labels = comp_labels,
        CR_bright_map = CR_bright_map,
        sweep_ham_params = sweep_ham_params,
    )
    
    batched_sweep_CR_ingredients(
        ps,
        num_q = num_q,
        num_r = num_r,
        trunc = dynamical_truncation,
        CR_bright_map = CR_bright_map,
        add_default_target = True,
        comp_labels = comp_labels,
    )
    
    batched_sweep_CR(
        ps,
        num_q = num_q,
        trunc = dynamical_truncation,
        CR_bright_map = CR_bright_map,
        ignore_phase = ignore_phase,
        gaussian_pulse = gaussian_pulse,
    )
    
    batched_sweep_incoh_infid_CR(
        ps,
        num_q = num_q,
        CR_bright_map = CR_bright_map,
        Q_cap = Q_cap,
        Q_ind = Q_ind,
        T = T,
        cycle_per_gate = cycle_per_gate,
        zz_penalty = zz_penalty,
    )
    
    