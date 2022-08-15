import numpy as np
import qutip as qt

from chencrafts.bsqubits.ec_systems import initialize_joint_system_tmon
from chencrafts.bsqubits.states import state_sets

def QOC_inputs_tmon(
    omega_s, EJ, EC, ng, g_sa, n_bar,
    sys_dim,
    anc_dim,
    anc_ncut,
    purpose,
    gate_time=None,
    return_array=True,
    rotated_target=True
):
    h_space, hamiltonian, H_controls = initialize_joint_system_tmon(
        omega_s=omega_s, EJ=EJ, EC=EC, ng=ng, g_sa=g_sa,
        sys_dim=sys_dim, anc_ncut=anc_ncut, anc_dim=anc_dim,
        returns=["hilbert_space", "hamiltonian", "H_controls"],
        return_ndarray=return_array
    )

    (
        anc_0, anc_1,
        sys_0_anc_0_even, sys_1_anc_0_even,
        sys_d0_anc_0_even, sys_d1_anc_0_even,
        sys_d20_anc_0_even, sys_d21_anc_0_even,
        sys_1_anc_1_even,
        sys_d1_anc_1_even,
        sys_d21_anc_1_even,
    ) = state_sets(
        h_space, 
        np.sqrt(n_bar), 
        returns = (
            "anc_0", "anc_1",
            "sys_0_anc_0_even", "sys_1_anc_0_even",
            "sys_d0_anc_0_even", "sys_d1_anc_0_even",
            "sys_d20_anc_0_even", "sys_d21_anc_0_even",
            "sys_1_anc_1_even",
            "sys_d1_anc_1_even",
            "sys_d21_anc_1_even",
        ),
        return_1darray = return_array
    )
    encoded_states = [
        sys_0_anc_0_even, sys_1_anc_0_even,
        sys_d0_anc_0_even, sys_d1_anc_0_even,
        # sys_d20_anc_0_even, sys_d21_anc_0_even,
    ]
    decoded_states = [
        # anc_0, anc_1,
        sys_0_anc_0_even, sys_1_anc_1_even,
        sys_d0_anc_0_even, sys_d1_anc_1_even,
        # sys_d20_anc_0_even, sys_d21_anc_1_even,
    ]
    
    # forbiden_states = [
    #     [qt.tensor([qt.coherent(trial_sys_dim, np.sqrt(n_bar)), qt.fock(trial_anc_dim, trial_anc_dim-1)]).data.toarray()[:, 0]] * len(encoded_states),
    #     # [qt.tensor([qt.fock(trial_sys_dim, 4), qt.fock(trial_anc_dim, trial_anc_dim-2)]).data.toarray()[:, 0]] * len(encoded_states),
    #     [qt.tensor([qt.fock(trial_sys_dim, trial_sys_dim-1), qt.fock(trial_anc_dim, 0)]).data.toarray()[:, 0]] * len(encoded_states),
    #     [qt.tensor([qt.fock(trial_sys_dim, trial_sys_dim-1), qt.fock(trial_anc_dim, 1)]).data.toarray()[:, 0]] * len(encoded_states),
    # ]
    forbiden_states = []

    if purpose == "encoding":
        initial_states = [anc_0, anc_1]
        target_states = encoded_states[:2]
    elif purpose == "decoding":
        initial_states = encoded_states
        target_states = decoded_states

    if rotated_target:
        if gate_time is None:
            raise ValueError(f"please specify the gate_time when rotating the target states")

        if return_array:
            qobj_hamiltonian = qt.Qobj(
                hamiltonian,
                dims = [[sys_dim, anc_dim], [sys_dim, anc_dim]]
            )
        else:
            qobj_hamiltonian = hamiltonian

        free_unitary = (1j * qobj_hamiltonian * gate_time).expm()
        target_states = [free_unitary * state for state in target_states]
        
    if return_array:
        initial_states = np.array(initial_states)
        target_states = np.array(target_states)
        forbiden_states = np.array(forbiden_states)

    return (
        hamiltonian,
        H_controls,
        initial_states,
        target_states,
        forbiden_states
    )