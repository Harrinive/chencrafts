import numpy as np
import scqubits as scq
import qutip as qt
import copy

from chencrafts.cqed.qt_helper import oprt_in_basis, process_fidelity
from chencrafts.cqed.floquet import FloquetBasis
from chencrafts.toolbox.gadgets import mod_c
from typing import List, Tuple

# static properties ====================================================
def sweep_comp_drs_indices(
    ps: scq.ParameterSweep, 
    idx,
    comp_labels: List[Tuple[int, ...]]
):
    comp_drs_indices = [
        int(ps.dressed_index(label)[idx])
        for label in comp_labels
    ]
    return np.array(comp_drs_indices)

def sweep_comp_bare_overlap(
    ps: scq.ParameterSweep, 
    idx,
    comp_labels: List[Tuple[int, ...]]
):
    comp_drs_indices = ps["comp_drs_indices"][idx]
    
    overlaps = []
    for raveled_comp_label, drs_idx in enumerate(comp_drs_indices):
        dressed_states = ps["evecs"][idx][drs_idx]

        # raveled_comp_label runs from 0 - 7
        unraveled_bare_label = comp_labels[raveled_comp_label]
        raveled_bare_label = np.ravel_multi_index(unraveled_bare_label, ps.hilbertspace.subsystem_dims)

        overlaps.append(np.abs(dressed_states.full()[raveled_bare_label, 0])**2)

    return np.array(overlaps)

def sweep_static_zzz(
    ps: scq.ParameterSweep, 
    idx, 
    comp_labels: List[Tuple[int, ...]]
) -> float:
    """
    Comp_labels is a list of bare labels, e.g. [(0, 0, 0), (0, 0, 1),
    (1, 0, 0), (1, 0, 1)].
    """
    evals = ps["evals"][idx]
    comp_evals_w_sgn = [
        evals[ps.dressed_index(label)[idx]] * (-1)**np.sum(label)
        for label in comp_labels
    ]
    return np.sum(comp_evals_w_sgn)

def batched_sweep_static(
    ps: scq.ParameterSweep,
    comp_labels: List[Tuple[int, ...]]  
):
    """
    Static properties:
    - comp_drs_indices: the dressed indices of the components
    - comp_bare_overlap: the minimal overlap on bare basis
    - static_zzz: the static ZZ observable
    """
    ps.add_sweep(
    sweep_comp_drs_indices,
        sweep_name = 'comp_drs_indices',
        comp_labels = comp_labels,
    )
    ps.add_sweep(
        sweep_static_zzz,
        sweep_name = 'static_zzz',
        comp_labels = comp_labels,
    )
    ps.add_sweep(
        sweep_comp_bare_overlap,
        sweep_name = 'comp_bare_overlap',
        comp_labels = comp_labels,
    )

# target transitions ===================================================
def fill_in_target_transitions(
    ps: scq.ParameterSweep, 
    transitions_to_drive: List[List[List[int]]] | np.ndarray,
):
    """
    Fill in the target transitions to drive given the init and final states.
    
    Parameters
    ----------
    q1_idx, q2_idx : int
        The indices of the qubits to drive.
    transitions_to_drive : List[List[int]]
        The init and final states to drive. It's a 3D array, where the 
        first dimension is the different spectator states, the second 
        dimension is the init and final state, and the third dimension is 
        the state label.
    num_q, num_r : int
        The number of qubits and resonators.
        
    Returns
    -------
    target_transitions : np.ndarray
        A 3D array of init and final state pairs, dimensions: 
        0. different spectator states, 
        1. init & final state
        2. state label
    """
    return np.array(transitions_to_drive)
    

def sweep_default_target_transitions(
    ps: scq.ParameterSweep, 
    q1_idx: int, 
    q2_idx: int, 
    r_idx: int, 
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
        The index of the first qubit, starts from 0.
    q2_idx : int
        The index of the second qubit, starts from 0.
    r_idx : int
        The index of the resonator to drive, starts from num_q.
    num_q : int
        The number of qubits.
    num_r : int
        The number of resonators.
        
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

    # transitions_to_drive is a 3D array, dimensions: 
    # 0. different spectator states, 
    # 1. init & final state
    # 2. state label
    transitions_to_drive = []
    for q_spec_idx in np.ndindex((2,) * len(q_spec)):
        # qubit states, with q1 and q2 excited and different spectator states
        # something like (111) and (110) if q1 = 0 and q2 = 1, spectator is 2
        init_q_state = [0] * num_q
        init_q_state[q1_idx] = 1
        init_q_state[q2_idx] = 1
        for q_spec_id, q_spec_state in enumerate(q_spec_idx):
            init_q_state[q_spec[q_spec_id]] = q_spec_state

        # add resonators, becomes something like (11100)
        init_state = init_q_state + [0] * num_r 
        final_state = copy.copy(init_state)
        final_state[r_idx+num_q] = 1

        transitions_to_drive.append([init_state, final_state])

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
    
    # drs_targ_trans is a 2D array, dimensions: 
    # 0. different spectator states, 
    # 1. init & final state (scaler)
    drs_targ_trans = []
    for init, final in target_transitions:
        raveled_init = np.ravel_multi_index(init, tuple(ps.hilbertspace.subsystem_dims))
        raveled_final = np.ravel_multi_index(final, tuple(ps.hilbertspace.subsystem_dims))
        drs_targ_trans.append(
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
    
    freqs = []
    for init, final in drs_trans:
        eval_i = evals[init]
        eval_f = evals[final]
        freqs.append(eval_f - eval_i)
        
    return np.array(freqs)

def batched_sweep_target_transition(
    ps: scq.ParameterSweep,
    q1_idx: int,
    q2_idx: int,
    r_idx: int,
    num_q: int,
    num_r: int,
    add_default_target: bool = True,
):
    """
    Target transition related sweeps:
    - target_transitions_{q1_idx}_{q2_idx}: the target transitions
    - drs_target_trans_{q1_idx}_{q2_idx}: the dressed target transitions
    - target_freq_{q1_idx}_{q2_idx}: the target transition frequency
    """
    if add_default_target:
        ps.add_sweep(
            sweep_default_target_transitions,
            sweep_name = f'target_transitions_{q1_idx}_{q2_idx}',
            q1_idx = q1_idx,
            q2_idx = q2_idx,
            r_idx = r_idx,
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
        "dynamical_zzz_" + f'{q1_idx}_{q2_idx}': np.std(target_freq, axis=-1)
    })  
    
# nearby & unwanted transitions ========================================
def sweep_drive_op(
    ps: scq.ParameterSweep,
    idx,
    r_idx,
    num_q,
    trunc: int = 30,
):
    res = ps.hilbertspace.subsystem_list[num_q + r_idx]
    try:
        res_n_op = res.n_operator()
    except AttributeError:
        op_name = str(res.vars['extended']['momentum'][0]) + "_operator"
        res_n_op = getattr(res, op_name)()    
    drive_op = oprt_in_basis(
        scq.identity_wrap(res_n_op, res, ps.hilbertspace.subsystem_list),
        ps["evecs"][idx][:trunc]
    ) * np.pi * 2
    
    return drive_op

def sweep_nearby_trans(
    ps: scq.ParameterSweep, 
    idx, 
    q1_idx, 
    q2_idx,
    comp_labels: List[Tuple[int, ...]],
    n_matelem_fraction_thres: float = 1e-1,
    freq_thres_GHz: float = 0.3,
    num_thres: int = 30,
):
    """
    Identify transitions that are close to the target transition. 
    
    Parameters
    ----------
    n_matelem_fraction_thres: float, default = 1e-1
        The threshold for drive operator matrix element for including a 
        particular transition. It's set to be a fraction of the 
        target drive matrix element.
    freq_thres_GHz: float, default = 0.3
        The threshold for the frequency difference between the target transition
        and the nearby transition.
    num_thres: int, default = 30
        The total number of states considered
    """
    evals = ps["evals"][idx]
    target_transitions = ps[f"target_transitions_{q1_idx}_{q2_idx}"][idx]
    drs_target_trans = ps[f"drs_target_trans_{q1_idx}_{q2_idx}"][idx]
    
    res_n_op = ps[f"drive_op_{q1_idx}_{q2_idx}"][idx]
    n_matelem_thres = np.average([
        np.abs(res_n_op[init, final])
        for init, final in drs_target_trans
    ], axis=0) * n_matelem_fraction_thres
    
    drs_target_freq = np.average(ps[f"target_freq_{q1_idx}_{q2_idx}"][idx])
    
    # initial states: all possible computational states and the target state
    init_states = np.array(list(comp_labels) + list(target_transitions[:, 1]))
    
    # final states: 
    final_states = np.array([list(idx) for idx in np.ndindex(tuple(ps.hilbertspace.subsystem_dims))])
        
    # near_trans is a 3D array, dimensions: 
    # 0. near-by transition, 
    # 1. init & final state, 
    # 2. state label.
    near_trans = []
    for init in init_states:
        for final in final_states:            
            # skip the same init / final state
            if np.all(init == final):
                continue
            
            # # skip the state with two excitations on a mode
            # if ((final - init) >= 2).any():
            #     continue
            
            # skip the transition that doesn't have a label
            raveled_init = np.ravel_multi_index(init, tuple(ps.hilbertspace.subsystem_dims))
            raveled_final = np.ravel_multi_index(final, tuple(ps.hilbertspace.subsystem_dims))
            drs_i = ps["dressed_indices"][idx][raveled_init]
            drs_f = ps["dressed_indices"][idx][raveled_final]
            if drs_i is None or drs_f is None:
                continue
            
            # skip the transitions with very different frequency
            freq = evals[drs_f] - evals[drs_i]
            if np.abs(freq - drs_target_freq) > freq_thres_GHz:
                continue
            
            # skip the state with small drive matrix element
            n_matelem = np.abs(res_n_op[drs_i, drs_f]) 
            if n_matelem < n_matelem_thres:
                continue

            near_trans.append([init, final])
    
    # pad zeros to the near_trans array to make the first dimension = num_thres
    padded_near_trans = np.ndarray((num_thres, 2, ps.hilbertspace.subsystem_count), dtype=object)
    padded_near_trans[:len(near_trans)] = np.array(near_trans)
    
    return padded_near_trans

def sweep_nearby_freq(
    ps: scq.ParameterSweep,
    idx,
    q1_idx,
    q2_idx,
):
    bare_trans = ps[f"nearby_trans_{q1_idx}_{q2_idx}"][idx]
    evals = ps["evals"][idx]
    
    # 1D array, dimensions: 
    # 0. near-by transition frequency
    freqs = []
    for init, final in bare_trans:
        if np.any(init == None) or np.any(final == None):
            continue
        
        raveled_init = np.ravel_multi_index(init, tuple(ps.hilbertspace.subsystem_dims))
        raveled_final = np.ravel_multi_index(final, tuple(ps.hilbertspace.subsystem_dims))
        drs_i = ps["dressed_indices"][idx][raveled_init]
        drs_f = ps["dressed_indices"][idx][raveled_final]
        eval_i = evals[drs_i]
        eval_f = evals[drs_f]
        freqs.append(eval_f - eval_i)
        
    padded_freqs = np.zeros(len(bare_trans))
    padded_freqs[:len(freqs)] = np.array(freqs)
        
    return padded_freqs

def batched_sweep_nearby_trans(
    ps: scq.ParameterSweep,
    q1_idx: int,
    q2_idx: int,
    r_idx: int,
    num_q: int,
    comp_labels: List[Tuple[int, ...]],
    trunc: int = 30,
    n_matelem_fraction_thres: float = 1e-1,
    freq_thres_GHz: float = 0.3,
    num_thres: int = 30,
):
    """
    Identify nearby transitions and their frequency
    - drive_op_{q1_idx}_{q2_idx}: the drive operator
    - nearby_trans_{q1_idx}_{q2_idx}: the nearby transitions. For each parameter
        it's a 3D array, dimensions:
        0. near-by transition, 
        1. init & final state
        2. state label
    - nearby_freq_{q1_idx}_{q2_idx}: the nearby transition frequency, for each parameter
        it's an 1D array, dimensions:
        0. near-by transition frequency
    """
    ps.add_sweep(
        sweep_drive_op,
        sweep_name = f"drive_op_{q1_idx}_{q2_idx}",
        r_idx = r_idx,
        num_q = num_q,
        trunc = trunc,
    )
    ps.add_sweep(
        sweep_nearby_trans,
        sweep_name = f"nearby_trans_{q1_idx}_{q2_idx}",
        q1_idx = q1_idx,
        q2_idx = q2_idx,
        comp_labels = comp_labels,
        n_matelem_fraction_thres = n_matelem_fraction_thres,
        freq_thres_GHz = freq_thres_GHz,
        num_thres = num_thres,
    )
    ps.add_sweep(
        sweep_nearby_freq,
        sweep_name = f"nearby_freq_{q1_idx}_{q2_idx}",
        q1_idx = q1_idx,
        q2_idx = q2_idx,
    )

# CZ calibration =======================================================
def sweep_ac_stark_shift(
    ps: scq.ParameterSweep, 
    idx,
    q1_idx,
    q2_idx,
    num_q,
    num_r,
    comp_labels: List[Tuple[int, ...]],
    trunc: int = 30,
):
    bare_trans = ps[f"target_transitions_{q1_idx}_{q2_idx}"][idx]
    drs_trans = ps[f"drs_target_trans_{q1_idx}_{q2_idx}"][idx]

    # pulse parameters -------------------------------------------------
    ham = qt.qdiags(ps["evals"][idx][:trunc], 0) * np.pi * 2

    # drive freq = average of all target transition freqs
    drive_freq = 0.0
    for init, final in drs_trans:
        e101 = ps["evals"][idx][init]
        e111 = ps["evals"][idx][final]
        drive_freq += (e111 - e101) * np.pi * 2
    drive_freq /= len(drs_trans)

    drive_op = ps[f"drive_op_{q1_idx}_{q2_idx}"][idx]
    # "normalize" the drive operator with one of its mat elem
    target_mat_elem = drive_op[drs_trans[0][0], drs_trans[0][1]]    

    param_mesh = ps.parameters.meshgrid_by_name()
    ham_floquet = [
        ham,
        [
            param_mesh["amp"][idx] * drive_op / np.abs(target_mat_elem), 
            f"cos({drive_freq}*t)"
        ],
    ]

    # floquet analysis and calibration for gate time ----------------------
    T = np.pi * 2 / drive_freq
    fbasis = FloquetBasis(ham_floquet, T)
    
    fevals = fbasis.e_quasi
    fevecs = fbasis.mode(0)
    
    # undriven states lookup
    lookup = fbasis.floquet_lookup(0, threshold=0.7)
    raveled_0 = np.ravel_multi_index((0,) * (num_q + num_r), tuple(ps.hilbertspace.subsystem_dims))
    drs_idx_0 = ps["dressed_indices"][idx][raveled_0]
    eval_0 = ps["evals"][idx][drs_idx_0]
    f_idx_0 = lookup[drs_idx_0]
    feval_0 = fevals[f_idx_0]

    # calculate ac-Stark shift
    init_state_bare_labels = bare_trans[:, 0, :].tolist()
    ac_stark_shifts = []    # unit: rad / ns
    for state in comp_labels:
        raveled_state = np.ravel_multi_index(state, tuple(ps.hilbertspace.subsystem_dims))
        drs_idx = ps["dressed_indices"][idx][raveled_state]
        if list(state) in init_state_bare_labels:
            # the second dimension is pair index of transitions, 
            # 0 represent the initial state label
            # if True, the state is half-half hybridized with the final state
            # we will calculate it separately
            ac_stark_shifts.append(np.nan)
            continue
        else:
            # the dressed states are nearly bare states
            f_idx = lookup[drs_idx]
            
        if drs_idx is None or f_idx is None:
            raise ValueError(f"drs_idx: {drs_idx}, f_idx: {f_idx}. Please check the system config.")

        shift = - mod_c(    # minus sign comes from -1j in exp(-1j * theta)
            (fevals[f_idx] - feval_0)
            - (ps["evals"][idx][drs_idx] - eval_0) * np.pi * 2,     
            # go to rotating frame
            drive_freq
        )

        ac_stark_shifts.append(shift)

    ac_stark_shifts = np.array(ac_stark_shifts)

    # driven state lookup
    Rabi_minus_list = []
    Rabi_plus_list = []
    Rabi_rot_frame_list = []

    for init, final in drs_trans:
        drs_state_init = qt.basis(ham.shape[0], init)
        drs_state_final = qt.basis(ham.shape[0], final)
        drs_plus = (drs_state_init + 1j * drs_state_final).unit()   # 1j comes from driving change matrix (sigma_y)
        drs_minus = (drs_state_init - 1j * drs_state_final).unit()
        f_idx_plus, _ = fbasis._closest_state(fevecs, drs_plus)  # we put the |+> state in the qubit state list
        f_idx_minus, _ = fbasis._closest_state(fevecs, drs_minus) # we put the |1> state in the resonator list 
        
        # it could be used to calibrate a gate time to complete a rabi cycle
        Rabi_minus = - mod_c(
            fevals[f_idx_minus] - feval_0,
            drive_freq
        )
        Rabi_plus = - mod_c(
            fevals[f_idx_plus] - feval_0,
            drive_freq
        )
        Rabi_rot_frame = - mod_c(
            (fevals[f_idx_minus] - feval_0)      
            # doesn't matter if we choose Rabi_plus as their phase added up to 2pi
            # it's valid in the limit of small drive_freq variation compared to drive amp
            # if not, we should use the "averaged" phase for a off-resonant Rabi
            - (ps["evals"][idx][init] - eval_0) * np.pi * 2,
            drive_freq
        )
        Rabi_minus_list.append(Rabi_minus)
        Rabi_plus_list.append(Rabi_plus)
        Rabi_rot_frame_list.append(Rabi_rot_frame)

    # ac_stark_shifts and Rabi_rot_frame are just Floquet evals in rotating 
    # frame. So we put them together.
    for ac_shift_idx, state in enumerate(comp_labels):
        if list(state) in init_state_bare_labels:
            bare_trans_idx = init_state_bare_labels.index(list(state))
            ac_stark_shifts[ac_shift_idx] = Rabi_rot_frame_list[bare_trans_idx]

    # just a container holding arrays with different length, 
    # there will be a lot of zero entries, but it does not matter
    freq_shift_data = np.zeros((3, len(ac_stark_shifts)))
    freq_shift_data[0, :] = ac_stark_shifts
    freq_shift_data[1, :len(drs_trans)] = Rabi_minus_list
    freq_shift_data[2, :len(drs_trans)] = Rabi_plus_list

    return freq_shift_data


def sweep_gate_time(ps: scq.ParameterSweep, idx, q1_idx, q2_idx):
    freq_shifts = ps[f"ac_stark_shifts_{q1_idx}_{q2_idx}"][idx]

    # calculate how many transitions we want to drive simultaneously
    # since freq_shifts has second dimension with length 2**num_q, 
    # the number of transitions is 2**(num_q-2)
    len_trans = int(np.round(freq_shifts.shape[1] / 4))

    Rabi_minus = freq_shifts[1, :len_trans]
    Rabi_plus = freq_shifts[2, :len_trans]
    gate_time_list = []

    for i in range(len_trans):
        gate_time = np.abs(np.pi * 2 / (Rabi_minus[i] - Rabi_plus[i]))
        gate_time_list.append(gate_time)

    return np.average(gate_time_list)

def sweep_spurious_phase(
    ps: scq.ParameterSweep, 
    idx, 
    q1_idx, 
    q2_idx,
    num_q,
):
    gate_time = ps[f"gate_time_{q1_idx}_{q2_idx}"][idx]

    # reshape it to num_q D array
    ac_stark_shifts = ps[f"ac_stark_shifts_{q1_idx}_{q2_idx}"][idx][0, :]
    ac_stark_shifts = ac_stark_shifts.reshape((2,) * num_q)

    # ZZ phase for every configuration of spectator qubit(s)
    all_q_id = range(num_q)
    q_spec = [q for q in all_q_id if q != q1_idx and q != q2_idx]
    spurious_phases = []
    for q_spec_idx in np.ndindex((2,) * len(q_spec)):
        slc = [slice(None),] * num_q
        for q_spec_i, q_spec_val in enumerate(q_spec_idx):
            slc[q_spec[q_spec_i]] = q_spec_val

        phase_4lvl = ac_stark_shifts[*slc] * gate_time
        ZZ_phase = (
            phase_4lvl[0, 0] - phase_4lvl[0, 1] 
            - phase_4lvl[1, 0] + phase_4lvl[1, 1]
        )
        spurious_phases.append(mod_c(ZZ_phase - np.pi, np.pi * 2))

    return np.average(spurious_phases)

def batched_sweep_gate_calib(
    ps: scq.ParameterSweep,
    q1_idx: int,
    q2_idx: int,
    r_idx: int,
    num_q: int,
    num_r: int,
    comp_labels: List[Tuple[int, ...]],
    trunc: int = 30,
):
    """
    Calibration of gate time and spurious phase, keys:
    - ac_stark_shifts_{q1_idx}_{q2_idx}: the AC Stark shifts
    - gate_time_{q1_idx}_{q2_idx}: the gate time
    - spurious_phase_{q1_idx}_{q2_idx}: the spurious phase
    """
    ps.add_sweep(
        sweep_drive_op,
        sweep_name = f"drive_op_{q1_idx}_{q2_idx}",
        r_idx = r_idx,
        num_q = num_q,
        trunc = trunc,
    )
    ps.add_sweep(
        sweep_ac_stark_shift,
        sweep_name = f"ac_stark_shifts_{q1_idx}_{q2_idx}",
        q1_idx = q1_idx,
        q2_idx = q2_idx,
        num_q = num_q,
        num_r = num_r,
        comp_labels = comp_labels,
        trunc = trunc,
        update_hilbertspace=False,
    )
    ps.add_sweep(
        sweep_gate_time,
        sweep_name = f"gate_time_{q1_idx}_{q2_idx}",
        q1_idx = q1_idx,
        q2_idx = q2_idx,
        update_hilbertspace=False,
    )
    ps.add_sweep(
        sweep_spurious_phase,
        sweep_name = f"spurious_phase_{q1_idx}_{q2_idx}",
        q1_idx = q1_idx,
        q2_idx = q2_idx,
        num_q = num_q,
        update_hilbertspace=False,
    )
    
# CZ gate ==============================================================
def sweep_CZ_propagator(
    ps: scq.ParameterSweep, 
    idx,
    q1_idx,
    q2_idx,
    trunc = 60,
):
    drs_trans = ps[f"drs_target_trans_{q1_idx}_{q2_idx}"][idx]

    # pulse 1 ----------------------------------------------------------
    ham = qt.qdiags(ps["evals"][idx][:trunc], 0) * np.pi * 2

    # drive freq = average of all target transition freqs
    drive_freq = 0.0
    for init, final in drs_trans:
        e101 = ps["evals"][idx][init]
        e111 = ps["evals"][idx][final]
        drive_freq += (e111 - e101) * np.pi * 2
    drive_freq /= len(drs_trans)

    drive_op = ps[f"drive_op_{q1_idx}_{q2_idx}"][idx]
    # "normalize" the drive operator with one of its mat elem
    target_mat_elem = drive_op[drs_trans[0][0], drs_trans[0][1]]    

    param_mesh = ps.parameters.meshgrid_by_name()
    ham_floquet = [
        ham,
        [
            param_mesh["amp"][idx] * drive_op / np.abs(target_mat_elem), 
            f"cos({drive_freq}*t)"
        ],
    ]

    T = np.pi * 2 / drive_freq
    fbasis = FloquetBasis(ham_floquet, T)

    gate_time = ps[f"gate_time_{q1_idx}_{q2_idx}"][idx]    
    spurious_phase = ps[f"spurious_phase_{q1_idx}_{q2_idx}"][idx]

    # unitary without phase shift
    unitary_1 = fbasis.propagator(gate_time / 2)

    # pulse 2 with phase shift -----------------------------------------
    spurious_phase_sign = "-" if spurious_phase > 0 else "+"
    
    ham_floquet = [
        ham,
        [
            param_mesh["amp"][idx] * drive_op / np.abs(target_mat_elem), 
            f"cos({drive_freq}*t{spurious_phase_sign}{np.abs(spurious_phase)})"
        ],
    ]
    fbasis = FloquetBasis(ham_floquet, T)
    unitary_2 = fbasis.propagator(gate_time, t0=gate_time / 2)

    # full gate: composed of two pulses --------------------------------
    unitary = unitary_2 * unitary_1

    # rotating frame
    rot_unit = (-1j * ham * gate_time).expm()
    rot_prop = rot_unit.dag() * unitary
    
    return rot_prop

def sweep_CZ_comp(
    ps: scq.ParameterSweep,
    idx,
    q1_idx,
    q2_idx,
):
    rot_prop = ps[f"full_CZ_{q1_idx}_{q2_idx}"][idx]
    
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

single_q_eye = qt.qeye(2)
def eye2_wrap(op, which, num_q):
    ops = [single_q_eye] * num_q
    ops[which] = op
    return qt.tensor(ops)

def eye2_wrap_2q(op1, op2, which1, which2, num_q):
    ops = [single_q_eye] * num_q
    ops[which1] = op1
    ops[which2] = op2
    return qt.tensor(ops)

def sweep_pure_CZ(
    ps: scq.ParameterSweep, 
    idx, 
    q1_idx, 
    q2_idx,
    num_q,
):
    eye_full = qt.tensor([single_q_eye] * num_q)
    phase_ops = [eye2_wrap(qt.projection(2, 1, 1), q_idx, num_q) for q_idx in range(num_q)]

    unitary = ps[f"CZ_{q1_idx}_{q2_idx}"][idx]
    unitary.dims = [[2] * num_q] * 2
    
    # remove single qubit gate component:
    phase = np.angle(np.diag(unitary.full()))
    
    global_phase = phase[0]
    phase_to_correct = []
    for q_idx in range(num_q):
        # state label with only q_idx is 1
        state_label = [0] * num_q
        state_label[q_idx] = 1
        raveled_state_label = np.ravel_multi_index(state_label, (2,) * num_q)
        phase_to_correct.append(phase[raveled_state_label] - global_phase)

    unitary = (-1j * global_phase * eye_full).expm() * unitary
    
    for q_idx in range(num_q):
        unitary = (-1j * phase_to_correct[q_idx] * phase_ops[q_idx]).expm() * unitary

    return unitary

def sweep_zzz(
    ps: scq.ParameterSweep, 
    idx, 
    q1_idx, 
    q2_idx,
    num_q,
):
    unitary = ps[f"pure_CZ_{q1_idx}_{q2_idx}"][idx]
    phase = np.angle(np.diag(unitary.full())).reshape((2,) * num_q)

    zzz = 0.0
    for idx, val in np.ndenumerate(phase):
        zzz += val * (-1)**np.sum(idx)

    return mod_c(zzz, np.pi * 2)

def sweep_fidelity(
    ps: scq.ParameterSweep, 
    idx, 
    q1_idx, 
    q2_idx,
    num_q,
):
    eye_full = qt.tensor([single_q_eye] * num_q)
    
    target = (
        - eye2_wrap_2q(qt.sigmaz(), qt.sigmaz(), q1_idx, q2_idx, num_q)
        + eye2_wrap(qt.sigmaz(), q1_idx, num_q)
        + eye2_wrap(qt.sigmaz(), q2_idx, num_q)
        + eye_full
    ) / 2

    unitary = ps[f"pure_CZ_{q1_idx}_{q2_idx}"][idx]
    fidelity = process_fidelity(
        qt.to_super(unitary),
        qt.to_super(target),
    )

    return fidelity

def batched_sweep_CZ(
    ps: scq.ParameterSweep,
    q1_idx,
    q2_idx,
    r_idx,
    num_q,
    trunc = 60,
):
    """
    CZ gate sweep, keys:
    - full_CZ_{q1_idx}_{q2_idx}: the full CZ gate
    - CZ_{q1_idx}_{q2_idx}: the CZ gate
    - pure_CZ_{q1_idx}_{q2_idx}: the pure CZ gate
    - zzz_{q1_idx}_{q2_idx}: the ZZZ spurious phase
    - fidelity_{q1_idx}_{q2_idx}: the fidelity
    """
    ps.add_sweep(
        sweep_CZ_propagator,
        sweep_name = f'full_CZ_{q1_idx}_{q2_idx}',
        q1_idx = q1_idx,
        q2_idx = q2_idx,
        trunc = trunc,
        update_hilbertspace=False,
    )
    ps.add_sweep(
        sweep_CZ_comp,
        sweep_name = f'CZ_{q1_idx}_{q2_idx}',
        q1_idx = q1_idx,
        q2_idx = q2_idx,
        update_hilbertspace=False,
    )
    ps.add_sweep(
        sweep_pure_CZ,
        sweep_name = f'pure_CZ_{q1_idx}_{q2_idx}',     
        q1_idx = q1_idx,
        q2_idx = q2_idx,
        num_q = num_q,
        update_hilbertspace=False,
    )
    ps.add_sweep(
        sweep_zzz,
        sweep_name = f'zzz_{q1_idx}_{q2_idx}',
        q1_idx = q1_idx,
        q2_idx = q2_idx,
        num_q = num_q,
        update_hilbertspace=False,
    )
    ps.add_sweep(
        sweep_fidelity,
        sweep_name = f'fidelity_{q1_idx}_{q2_idx}',
        q1_idx = q1_idx,
        q2_idx = q2_idx,
        num_q = num_q,
        update_hilbertspace=False,
    )