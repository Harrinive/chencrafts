import sympy as sp
import scqubits as scq
import numpy as np
import qutip as qt
import copy
import warnings

from qutip.solver.integrator.integrator import IntegratorException
from chencrafts.cqed.floquet import FloquetBasis
from chencrafts.cqed.qt_helper import oprt_in_basis
from chencrafts.toolbox.gadgets import mod_c
from chencrafts.fluxonium.batched_sweep_frf import single_q_eye, eye2_wrap

from typing import List, Tuple, Dict

# Static properties ====================================================
from chencrafts.fluxonium.batched_sweep_frf import sweep_comp_drs_indices, sweep_comp_bare_overlap, sweep_static_zzz

Q1, Q2, Q3, θ1, θ2, θ3 = sp.symbols("Q1 Q2 Q3 θ1 θ2 θ3")

def sweep_sym_hamiltonian(
    ps: scq.ParameterSweep,
    **kwargs
):
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
    Must have "sym_ham" in the parameter sweep
    
    Order: EC, EL
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
    Must have "sym_ham" in the parameter sweep
    
    Order: JC, JL
    """
    ham = ps["sym_ham"][idx]
    
    Q1_expr, Q2_expr = sp.symbols(f"Q{q1_idx+1} Q{q2_idx+1}")
    θ1_expr, θ2_expr = sp.symbols(f"θ{q1_idx+1} θ{q2_idx+1}")
    JCAB = float(ham.coeff(Q1_expr * Q2_expr))
    JLAB = float(ham.coeff(θ1_expr * θ2_expr))
    
    return np.array([JCAB, JLAB])

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
    - ham_param_Q{q_idx}
    - ham_param_Q{q1_idx}_Q{q2_idx}
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
    trunc: int = 30,
):
    qubit = ps.hilbertspace.subsystem_list[q_idx]
    
    try:
        qubit_n_op = qubit.n_operator()
    except AttributeError:
        q_idx = q_idx + 1
        Q_str = f"Q{q_idx}"
        op_name = str(f"{Q_str}_operator")
        qubit_n_op = getattr(qubit, op_name)()   
         
    drive_op = oprt_in_basis(
        scq.identity_wrap(qubit_n_op, qubit, ps.hilbertspace.subsystem_list),
        ps["evecs"][idx][:trunc]
    )
    
    return drive_op

def sweep_drive_mat_elem(
    ps: scq.ParameterSweep,
    idx,
    q1_idx,
    q2_idx,
):
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
    for q_idx in range(num_q):
        ps.add_sweep(
            sweep_drive_op,
            sweep_name = f'drive_op_{q_idx}',
            q_idx = q_idx,
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
        fbasis = FloquetBasis(ham_t, T)
    except IntegratorException:
        warnings.warn(f"At idx: {idx}, q1_idx: {q1_idx}, q2_idx: {q2_idx}, "
                     "Floquet basis integration failed.")
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
    unitary = fbasis.propagator(gate_time)
    
    # rotating frame
    rot_unit = (-1j * ham * gate_time).expm()
    rot_prop = rot_unit.dag() * unitary

    return np.array([gate_time, fbasis, rot_prop], dtype=object)

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
    num_q,
):
    unitary = ps[f"pure_CR_{q1_idx}_{q2_idx}"][idx]
    if unitary is None:
        return np.nan
    
    target = ps[f"target_CR_{q1_idx}_{q2_idx}"][idx]

    # compute fidelity
    fidelity = qt.process_fidelity(
        unitary,
        target,
    )

    return fidelity

def batched_sweep_CR(
    ps: scq.ParameterSweep,
    num_q: int,
    trunc: int,
    CR_bright_map: Dict[Tuple[int, int], int],
    **kwargs
):
    for (q1_idx, q2_idx), _ in CR_bright_map.items():
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
            num_q = num_q,
        )