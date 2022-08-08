import numpy as np

def unitary_by_pulse(pulse, time_step, H0, H_controls):
    shape = pulse.shape

    unitary = 1
    for t_i in range(shape[1]):
        hamiltonian = H0.copy()
        for i in range(shape[0]):
            hamiltonian = hamiltonian + pulse[i, t_i] * H_controls[i]

        unitary = (-1j * time_step * hamiltonian).expm() * unitary

    return unitary

def state_transfer_process(pulse, total_time, H0, H_controls, init_state, record_num):
    shape = pulse.shape

    time_step = total_time / shape[1]


    record_states = [init_state]
    record_time_steps = np.round(np.linspace(0, shape[1] - 1, record_num)).astype(int)
    remain_record_steps = record_time_steps[1:].copy()

    currrent_state = init_state.copy()
    for t_i in range(shape[1]):
        hamiltonian = H0.copy()
        for i in range(shape[0]):
            hamiltonian = hamiltonian + pulse[i, t_i] * H_controls[i]

        currrent_state = (-1j * time_step * hamiltonian).expm() * currrent_state

        if t_i == remain_record_steps[0]:
            record_states.append(currrent_state)
            remain_record_steps = np.delete(remain_record_steps, [0])

    return record_time_steps, record_states
    
def plot_n_distr(ax, record_states):
    distr = []
    for state in record_states:
        if np.prod(state.dims[1]) == 1:
            np_state = state.full().reshape(-1)
            distr.append((np_state.conj() * np_state).real)
        else:
            distr.append(np.diag(state.full()).real)

    distr = np.array(distr).transpose()
    
    ax.imshow(distr)

def fidelity_with_gate(result_unitary, initial_states, target_states):
    fid = []
    for i in range(len(initial_states)):
        final_test = result_unitary * initial_states[i]
        fid.append(np.abs(target_states[i].overlap(final_test)).real**2)

    return fid