import numpy as np
import qutip as qt

import scqubits as scq
from scqubits.core.hilbert_space import HilbertSpace
import scipy as sp

from chencrafts.cqed.mode_assignment import single_mode_dressed_esys

import itertools
from typing import List, Tuple, Literal

# ##############################################################################
def n_crit_by_diag(
    hilbertspace: HilbertSpace,
    res_mode_idx: int,
    state_label: Tuple[int] | List[int],
    dressed_indices: np.ndarray | None = None,
) -> int:
    """
    It returns the maximum n (aka critical photon number)
    that making the overlap between a dressed state (labeled by (n, ...))
    with its corresponding bare state larger than a threshold. 

    Parameters
    ----------
    hilberspace:
        scq.HilberSpace object which include the desired mode for calculating n_crit
    res_mode_idx:
        The index of the resonator mode in the hilberspace's subsystem_list
    state_label:
        n_crit is calculated with other modes staying at bare state. 
        For example, assume we want to evaluate the n_crit for the first mode in a three 
        mode system, we can set state_label to be (<any number>, 0, 1), indicating the 
        n_crit with other two modes at state (0, 1). 
    dressed_indeces: 
        An array mapping the FLATTENED bare state label to the eigenstate label. Usually
        given by `ParameterSweep["dressed_indices"]`.

    Returns
    -------
    Critical photon number as requested

    Note
    ----
    To match this n_crit with the analytical method, remember to set 
    scq.settings.OVERLAP_THRESHOLD = 0.853 before sweeping
    """

    # Use a dummy esys is enough to get a n_crit

    result_dummy_evals, _ = single_mode_dressed_esys(
        hilbertspace,
        res_mode_idx,
        state_label,
        dressed_indices,
        adjust_phase=False,     # as we input a dummy esys, we don't need to adjust phase
    )

    return len(result_dummy_evals)

# ##############################################################################
def n_crit_1st(detuning, mat_elem, scaling=1):
    return detuning**2 / np.abs(mat_elem)**2 / 4 * scaling

def n_crit_2nd(detuning_1, mat_elem_1, detuning_2, mat_elem_2, scaling=1):
    """
    detuning and mat elem should corresponding to the same two bare qubit states
    detuning 1 and mat elem 1 should related to the unperturbed state
    """
    return np.abs(
        (detuning_1 + detuning_2) * (detuning_1) 
        / 4 / mat_elem_1 / mat_elem_2
    ) * scaling

def n_crit_by_1st_pert(
    qubit, res_freq, g,
    qubit_state_label: int,
    qubit_bare_esys = None,
    qubit_native_operator: str | np.ndarray = "n_operator", 
    res_l_osc = np.sqrt(2),
):
    """
    Resonator mat_elem = np.sqrt(n) * res_l_osc / np.sqrt(2). Note that there is
    a factor of sqrt(2) in the denominator. 
    """

    # detuning
    if qubit_bare_esys is None:
        qubit_evals, qubit_evecs = qubit.eigensys(qubit.truncated_dim)
    else:
        qubit_evals, qubit_evecs = qubit_bare_esys

    detuning = np.abs(qubit_evals - qubit_evals[qubit_state_label]) - res_freq

    # mat elem
    if isinstance(qubit_native_operator, str):
        qubit_oprt_func = getattr(qubit, qubit_native_operator)
        qubit_native_operator_arr = qubit_oprt_func()
    else:
        qubit_native_operator_arr = qubit_native_operator

    try:
        qubit_op_energy_basis: np.ndarray = qubit.process_op(qubit_native_operator_arr, (qubit_evals, qubit_evecs))
    except AttributeError:
        # GenericQubit don't have process_op
        qubit_op_energy_basis = qubit_native_operator_arr

    mat_elem_factor = g * res_l_osc / np.sqrt(2)

    # n_crit
    n_crit = n_crit_1st(
        np.array(detuning), 
        qubit_op_energy_basis[qubit_state_label, :] * mat_elem_factor, 
        scaling=1, 
    )

    return n_crit

def n_crit_by_1st_pert_w_hilbertspace(
    hilbertspace: HilbertSpace, 
    qubit_mode_idx: int, res_mode_idx: int,
    qubit_state_label: int,
    interaction_idx: int, qubit_idx_in_interaction: int,
    qubit_bare_esys = None, 
):
    """
    Qubit mode and resonator mode are included in the hilbertspace object.
    Calculate the n_crit with qubit state = 0 and 1
    """

    res = hilbertspace.subsystem_list[res_mode_idx]
    if not isinstance(res, scq.Oscillator):
        raise TypeError("The resonator mode should be an oscillator mode")
    res_freq = res.E_osc

    # mat elem
    res_mode_in_interaction = 1 - qubit_idx_in_interaction
    interaction = hilbertspace.interaction_list[interaction_idx]
    res_oprt = interaction.operator_list[res_mode_in_interaction][1]
    qubit_oprt = interaction.operator_list[qubit_idx_in_interaction][1]

    n_crit = n_crit_by_1st_pert(
        qubit=hilbertspace.subsystem_list[qubit_mode_idx],
        res_freq=res_freq,
        g=interaction.g_strength,
        qubit_state_label=qubit_state_label,
        qubit_bare_esys=qubit_bare_esys,
        qubit_native_operator=np.array(qubit_oprt),
        res_l_osc=res_oprt[0, 1] * np.sqrt(2),  # sqrt(2) will be devided inside
    )

    return n_crit

# ##############################################################################

def n_crit_by_diag_subspace_base_single_qubit(
    qubit_evals,
    qubit_op_energy_basis,
    res_freq, g,
    qubit_state_label: int,
    res_l_osc: float = np.sqrt(2),
    res_state_range: int = 2,
    qubit_state_range: int = 2,
    max_detuning: float | None = None,
    overlap_threshold: float | None = None,
    print_hybridization: bool = False,
):  
    """
    Here qubit can be any mutil-mode non-lienar system. Just need to provide the coupling 
    operator.

    Parameters
    ----------

    max_detuning:
        If not None, the detuning between the resonator and the qubit is limited to g*20.
        Which approximately gives a upper bound 100 (25**2 / 4) of the n_crit we are able 
        to calculate
    """
    # states are represented by a bare index tuple: (osc_idx, qubit_idx)
    def mat_elem(state_1: Tuple[int, int], state_2: Tuple[int, int]):
        res_idx_1, qubit_idx_1 = state_1
        res_idx_2, qubit_idx_2 = state_2

        # check if the state label are invalid
        if res_idx_1 < 0 or res_idx_2 < 0 or qubit_idx_1 < 0 or qubit_idx_2 < 0:
            raise IndexError(f"invalid resonator index: {state_1} or {state_2}")
        if qubit_idx_1 >= qubit_op_energy_basis.shape[0] or qubit_idx_2 >= qubit_op_energy_basis.shape[0]:
            raise IndexError(f"invalid qubit index: {state_1} or {state_2}")

        # diagnal elements
        if res_idx_1 == res_idx_2 and qubit_idx_1 == qubit_idx_2:
            return res_freq * res_idx_1 + qubit_evals[qubit_idx_1]

        # off-diagnal elements
        if np.abs(res_idx_1 - res_idx_2) != 1:
            return 0
        
        osc_mat_elem = np.sqrt(np.max([res_idx_1, res_idx_2])) * res_l_osc / np.sqrt(2)
        qubit_mat_elem = qubit_op_energy_basis[qubit_idx_1, qubit_idx_2]
        return g * osc_mat_elem * qubit_mat_elem
    
    # construct the Hamiltonian for different number of photons
    def construct_H(photon_num):
        """
        photon_num should be larger than res_state_range
        """
        # collect states
        state_list = [(photon_num, qubit_state_label)]
        for i in range(-res_state_range, res_state_range+1):
            for j in range(-qubit_state_range, qubit_state_range+1):
                if i == 0 and j == 0:
                    # Already put the current state in the first element
                    continue
                state = (photon_num + i, qubit_state_label + j)
                try:
                    mat_elem(state, state)
                except IndexError:
                    continue
                state_list.append(state)

        state_list = np.array(state_list)

        # throw away states with detuning larger than max_detuning
        current_state = (photon_num, qubit_state_label)
        current_state_energy = mat_elem(current_state, current_state)
        energies = np.array([mat_elem(state, state) for state in state_list])
        detuning = np.abs(energies - current_state_energy)

        state_list = state_list[detuning < max_detuning]

        # construct Hamiltonian
        H = np.zeros((len(state_list), len(state_list)), dtype=complex)
        for i, state_1 in enumerate(state_list):
            for j, state_2 in enumerate(state_list):
                if i >= j:
                    H[i, j] = mat_elem(state_1, state_2)
                    H[j, i] = np.conj(H[i, j])

        return H, state_list
    
    # diagonalize the Hamiltonian and compare the dressed state with the bare state
    def diag_n_overlap(H):
        evals, evecs = np.linalg.eigh(H)

        # calculate the overlap between the eigenstates and the first bare state
        # which is the mod square of the first element of the eigenvector
        overlap_list = np.abs(evecs[0, :])**2

        # find the eigenstate with the largest overlap
        max_idx = np.argmax(overlap_list)
        max_overlap = overlap_list[max_idx]
        best_match_evec = evecs[:, max_idx]

        return max_overlap, best_match_evec, overlap_list
    
    # find the critical photon number
    # When increase n from res_state_range, it is the first n that makes the 
    # overlap smaller than overlap_threshold
    n_crit = res_state_range
    while True:
        H, state_list = construct_H(n_crit)
        max_overlap, best_match_evec, overlap_list = diag_n_overlap(H)
        if max_overlap < overlap_threshold:
            break

        if n_crit > 3000:
            print(f"n_crit is not found when n < 3000")
            break

        n_crit += 1

    if print_hybridization:
        # print(state_list) 
        # print(f"overlap_list: {overlap_list}")
        print(f"Dressed state ({n_crit}, {qubit_state_label})'s occupation probability:")
        occup_prob = np.abs(best_match_evec)**2
        overlap_sort = np.argsort(occup_prob)[::-1]
        sorted_prob = occup_prob[overlap_sort]
        for idx, sorted_idx in enumerate(overlap_sort):
            state = state_list[sorted_idx]
            overlap = sorted_prob[idx]
            if overlap < 0.0001:
                break
            print(f"\t{tuple(state)}: {overlap:.4f}")

    return n_crit


def n_crit_by_diag_subspace_base(
    qubit_evals_list,
    qubit_op_energy_basis_list,
    g_list,
    res_freq, 
    state_label,
    res_mode_idx = 0,
    res_state_range: int = 2,
    qubit_state_range_list: List[int] | None = None,
    max_detuning: float | None = None,
    overlap_threshold: float | None = None,
    print_hybridization: bool = False,
    max_n_crit: int = 2500,
):
    """
    When a resonator is coupled to multiple qubits, calculate the crticial photon number 
    given by the diagonalization of the Hamiltonian in the subspace.
    We assume there only exist the resonator-qubit coupling in the form of
        g * (a + a^dagger) * qubit_operator,
    for each g and qubit_operator in the g_list and qubit_op_energy_basis_list.

    In scqubits' convention, the resonator charge operator is 
        i (a^dagger - a) / sqrt(2) / l_osc,
    and the qubit phi operator is
        (a + a^dagger) / sqrt(2) * l_osc.
    Those oscillator length and sqrt(2) should be included in the definition of g.

    Parameters
    ----------
    state_label:
        n_crit is calculated with other modes staying at bare state. 
        For example, assume we want to evaluate the n_crit for the first mode in a three 
        mode system, we can set state_label to be (<any number>, 0, 1), indicating we are calculating
        n_crit with other two modes at state (0, 1). 

    max_detuning:
        If not None, the detuning between the resonator and the qubit is limited to g*100
        when construncting the subspace for diagnolization.
        Which approximately gives a upper bound 2500 (100**2 / 4) of the n_crit we are able 
        to calculate
    """
    # check and process input
    if len(qubit_evals_list) != len(qubit_op_energy_basis_list):
        raise ValueError("The length of qubit_evals_list and qubit_op_energy_basis_list"
                         " should be the same")
    for idx, qubit_evals in enumerate(qubit_evals_list):
        if len(qubit_evals) != qubit_op_energy_basis_list[idx].shape[0]:
            raise ValueError(f"The length of qubit_evals_list[{idx}] and "
                             f"qubit_op_energy_basis_list[{idx}] should be the same")
    if len(qubit_evals_list) != len(g_list):
        raise ValueError("The length of qubit_evals_list and g_list should be the same")
    if len(qubit_evals_list) != len(state_label) - 1:
        raise ValueError("The length of qubit_evals_list and qubit_state_label should "
                         "be the different by 1")
    
    if qubit_state_range_list is None:
        # rename the variable just for avoiding warning
        qubit_range_list = [2] * len(qubit_evals_list) 
    else:
        qubit_range_list = qubit_state_range_list.copy() 

    if len(qubit_evals_list) != len(qubit_range_list):
        raise ValueError("The length of qubit_evals_list and qubit_state_range_list"
                         " should be the same")
    
    max_detuning = np.min(np.abs(g_list)) * 100 if max_detuning is None else max_detuning
    overlap_threshold = scq.settings.OVERLAP_THRESHOLD if overlap_threshold is None else overlap_threshold

    # separate and combine the state label into res_state_label and qubit_state_label
    def separate_state_label(
        state_label: List[int] | np.ndarray | Tuple[int, ...]
    ) -> Tuple[int, List[int]]:
        state_label = list(state_label)
        res_state_label = state_label.pop(res_mode_idx)
        qubit_state_label = state_label.copy()
        return res_state_label, qubit_state_label
    def combine_state_label(
        res_state_label: int, qubit_state_label: List[int]
    ) -> Tuple[int, ...]:
        state_label = qubit_state_label.copy()
        state_label.insert(res_mode_idx, res_state_label)
        return tuple(state_label)
    _, current_qubit_state_label = separate_state_label(state_label)
    
    # dimension of each subsystem
    subsys_dims = [qubit_op_energy.shape[0] for qubit_op_energy in qubit_op_energy_basis_list]
    subsys_dims = combine_state_label(max_n_crit + 1, subsys_dims)

    # Calculate the matrix element of the Hamiltonian
    def mat_elem(state_1, state_2) -> complex:
        state_1 = np.array(state_1, dtype=int)
        state_2 = np.array(state_2, dtype=int)
        # check if the state is valid (larger than 0 and smaller than the dimension)
        if np.any(state_1 < 0) or np.any(state_2 < 0):
            raise IndexError("The state label should be larger than 0") 
        if np.any(state_1 >= subsys_dims) or np.any(state_2 >= subsys_dims):
            raise IndexError(f"The state label {state_1}, {state_2} should be smaller"
                             " than the dimension")
        
        res_state_label_1, qubit_state_label_1 = separate_state_label(state_1)
        res_state_label_2, qubit_state_label_2 = separate_state_label(state_2)
        
        # diagonal elements
        if np.all(state_1 == state_2):
            return np.sum([
                qubit_evals_list[i][qubit_state_label_1[i]] 
                for i in range(len(qubit_evals_list))
            ]) + res_freq * res_state_label_1
        
        # off-diagonal elements for coupling in the form of 
        # g * (a + a^dagger) * qubit_operator
        # resonator matrix element
        if np.abs(res_state_label_1 - res_state_label_2) != 1:
            return 0
        osc_mat_elem = np.sqrt(np.max([res_state_label_1, res_state_label_2]))
        
        # qubit matrix element
        # at most one qubit bare state should be different
        diff_qubit_state = np.array(qubit_state_label_1) != np.array(qubit_state_label_2)
        if np.sum(diff_qubit_state) > 1:
            return 0
        elif np.sum(diff_qubit_state) == 1:
            diff_qubit_idx = np.where(diff_qubit_state)[0][0]
            qubit_op = qubit_op_energy_basis_list[diff_qubit_idx]
            qubit_mat_elem = qubit_op[qubit_state_label_1[diff_qubit_idx], qubit_state_label_2[diff_qubit_idx]]

            return osc_mat_elem * qubit_mat_elem * g_list[diff_qubit_idx]
        elif np.sum(diff_qubit_state) == 0:
            # a qubit operator may have non-zero diagonal elements
            # in this case, the matrix element is the sum of the corresponding 
            # diagonal elements for all qubit operators
            qubit_mat_elem = np.sum([
                qubit_op_energy[qubit_state_label_1[idx], qubit_state_label_2[idx]] * g_list[idx]
                for idx, qubit_op_energy in enumerate(qubit_op_energy_basis_list)
            ])

            return osc_mat_elem * qubit_mat_elem
        
        # can support different coupling scheme by updating this part

        return 0    # not necesary, just to avoid warning

    # build the Hamiltonian following the single qubit code example
    def construct_H(photon_num):
        # collect states within (-res_state_range, res_state_range+1)
        # and (-qubit_state_range, qubit_state_rage) for each qubit. 
        state_range_list = [
            np.arange(-qubit_range, qubit_range+1) + current_qubit_state_label[idx]
            for idx, qubit_range in enumerate(qubit_range_list)
        ]
        state_range_list.insert(
            res_mode_idx, 
            np.arange(-res_state_range, res_state_range+1) + photon_num
        )
        state_list = list(itertools.product(*state_range_list))

        # throw away states with detuning larger than max_detuning
        current_state_label = combine_state_label(photon_num, current_qubit_state_label)
        current_state_energy = mat_elem(current_state_label, current_state_label)
        reduced_state_list = []
        for state in state_list:
            try:
                energy = mat_elem(state, state)
            except IndexError:
                continue
        
            if np.abs(energy - current_state_energy) < max_detuning:
                reduced_state_list.append(state)
        
        current_state_idx_in_list = reduced_state_list.index(tuple(current_state_label))

        # construct Hamiltonian using sprase matrix
        H = sp.sparse.lil_matrix((len(reduced_state_list), len(reduced_state_list)), dtype=complex)
        for i, state_1 in enumerate(reduced_state_list):
            for j, state_2 in enumerate(reduced_state_list):
                H[i, j] = mat_elem(state_1, state_2)
                if i == j:
                    H[i, j] -= current_state_energy

        if not np.allclose(H.toarray(), H.toarray().conj().T):
            raise ValueError("The Hamiltonian is not Hermitian")

        return H.tocsr(), reduced_state_list, current_state_idx_in_list
        
            
     # diagonalize the Hamiltonian and compare the dressed state with the bare state
    def diag_n_overlap(H, bare_state_idx):
        if H.shape[0] <= 15:
            # use regular eigensolver and obtain the full spectra
            _, evecs = sp.linalg.eigh(H.toarray())
        else:
            # use sparse eigensolver
            _, evecs = sp.sparse.linalg.eigsh(H, k=H.shape[0]-2, which='SA')

        # calculate the overlap between the eigenstates and the bare state
        # which is the mod square of the corresponding element of the eigenvector
        overlap_list = np.abs(evecs[bare_state_idx, :])**2

        # find the eigenstate with the largest overlap
        max_idx = np.argmax(overlap_list)
        max_overlap = overlap_list[max_idx]
        best_match_evec = evecs[:, max_idx]

        return max_overlap, best_match_evec, overlap_list
    
    # find the critical photon number
    # When increase n from res_state_range, it is the first n that makes the 
    # overlap smaller than overlap_threshold
    n_crit = res_state_range
    while True:
        
        H, state_list, current_state_idx_in_list = construct_H(n_crit)
        max_overlap, best_match_evec, overlap_list = diag_n_overlap(H, current_state_idx_in_list)

        if max_overlap < overlap_threshold:
            break

        if n_crit + 1 > max_n_crit:
            print(f"n_crit is not found when n < {max_n_crit}")
            break

        n_crit += 1

    if print_hybridization:
        # print(state_list) 
        # print(f"overlap_list: {overlap_list}")
        n_crit_state = combine_state_label(n_crit, current_qubit_state_label)
        print(f"Dressed state {n_crit_state}'s occupation probability:")
        occup_prob = np.abs(best_match_evec)**2
        overlap_sort = np.argsort(occup_prob)[::-1]
        sorted_prob = occup_prob[overlap_sort]
        for idx, sorted_idx in enumerate(overlap_sort):
            state = state_list[sorted_idx]
            overlap = sorted_prob[idx]
            if overlap < 0.0001:
                break
            print(f"\t{tuple(state)}: {overlap:.4f}")

    return n_crit


def n_crit_by_diag_subspace(
    qubit_list, 
    qubit_native_operator_list: List[str | np.ndarray], 
    res_freq, 
    g_list,
    state_label,
    res_mode_idx = 0,
    qubit_bare_esys_list = None,
    res_state_range: int = 2,
    qubit_state_range_list: List[int] | None = None,
    max_detuning: float | None = None,
    overlap_threshold: float | None = None,
    print_hybridization: bool = False,
    max_n_crit: int = 2500,
):
    """
    When a resonator is coupled to multiple qubits, calculate the crticial photon number 
    given by the diagonalization of the Hamiltonian in the subspace.
    We assume there only exist the resonator-qubit coupling in the form of
        g * (a + a^dagger) * qubit_operator,
    for each g and qubit_operator in the g_list and qubit_op_energy_basis_list.

    In scqubits' convention, the resonator charge operator is 
        i (a^dagger - a) / sqrt(2) / l_osc,
    and the qubit phi operator is
        (a + a^dagger) / sqrt(2) * l_osc.
    Those oscillator length and sqrt(2) should be included in the definition of g.

    Parameters
    ----------
    state_label:
        n_crit is calculated with other modes staying at bare state. 
        For example, assume we want to evaluate the n_crit for the first mode in a three 
        mode system, we can set state_label to be (<any number>, 0, 1), indicating we are calculating
        n_crit with other two modes at state (0, 1). 

    max_detuning:
        If not None, the detuning between the resonator and the qubit is limited to g*100
        when construncting the subspace for diagnolization.
        Which approximately gives a upper bound 2500 (100**2 / 4) of the n_crit we are able 
        to calculate
    """
    # check input
    if len(qubit_list) != len(qubit_native_operator_list):
        raise ValueError("The number of qubits and qubit operators are not equal")

    # mat elem of the qubit operator
    if qubit_bare_esys_list is None:
        qubit_bare_esys_list = [
            qubit.eigensys(qubit.truncated_dim) for qubit in qubit_list
        ]

    qubit_op_energy_basis_list = []
    for idx, qubit_native_operator in enumerate(qubit_native_operator_list):
        qubit = qubit_list[idx]
        qubit_esys = qubit_bare_esys_list[idx]
        if isinstance(qubit_native_operator, str):
            qubit_oprt_func = getattr(qubit, qubit_native_operator)
            qubit_native_operator_arr = qubit_oprt_func()
        else:
            qubit_native_operator_arr = qubit_native_operator

        try:
            qubit_op_energy_basis: np.ndarray = qubit.process_op(qubit_native_operator_arr, qubit_esys)
        except AttributeError:
            # GenericQubit don't have process_op
            qubit_op_energy_basis = qubit_native_operator_arr

        qubit_op_energy_basis_list.append(qubit_op_energy_basis)

    qubit_evals_list = [qubit_esys[0] for qubit_esys in qubit_bare_esys_list]

    n_crit = n_crit_by_diag_subspace_base(
        qubit_evals_list=qubit_evals_list,
        qubit_op_energy_basis_list=qubit_op_energy_basis_list,
        state_label=state_label,
        res_freq=res_freq, 
        g_list=g_list,
        res_mode_idx=res_mode_idx,
        res_state_range=res_state_range,
        qubit_state_range_list=qubit_state_range_list,
        max_detuning=max_detuning,
        overlap_threshold=overlap_threshold,
        print_hybridization=print_hybridization,
        max_n_crit=max_n_crit,
    )

    return n_crit

def n_crit_by_diag_subspace_w_hilbertspace(
    hilbertspace: HilbertSpace, 
    qubit_mode_idx_list: List[int], res_mode_idx: int,
    interaction_idx_list: List[int], qubit_idx_in_interaction_list: List[int],
    state_label,
    qubit_bare_esys_list = None,
    res_state_range: int = 2,
    qubit_state_range_list: List[int] | None = None,
    max_detuning: float | None = None,
    overlap_threshold: float | None = None,
    print_hybridization: bool = False,
    max_n_crit: int = 2500,
):
    """
    When a resonator is coupled to multiple qubits, calculate the crticial photon number 
    given by the diagonalization of the Hamiltonian in the subspace.
    Qubit mode and resonator mode are included in the hilbertspace object.

    Parameters
    ----------
    interaction_idx_list:
        Assuming the qubits are coupled to the resonator with only one interaction each,
        interaction_idx_list should be a list of indices of the interactions.

    qubit_mode_idx_list:
        Should be a list of qubit mode indices in the hilbertspace.subsystem_list.
        The order of the qubit mode indices should be the same as the order of the
        interaction_idx_list and qubit_idx_in_interaction_list.

    state_label:
        n_crit is calculated with other modes staying at bare state. 
        For example, assume we want to evaluate the n_crit for the first mode in a three 
        mode system, we can set state_label to be (<any number>, 0, 1), indicating we are calculating
        n_crit with other two modes at state (0, 1). 

    max_detuning:
        If not None, the detuning between the resonator and the qubit is limited to g*100
        when construncting the subspace for diagnolization.
        Which approximately gives a upper bound 2500 (100**2 / 4) of the n_crit we are able 
        to calculate
    """
    # check input
    if len(qubit_mode_idx_list) != len(interaction_idx_list):
        raise ValueError("The number of qubit modes and interaction indices are not equal")
    if len(qubit_mode_idx_list) != len(qubit_idx_in_interaction_list):
        raise ValueError("The number of qubit modes and qubit indices in interaction are not equal")
    
    # resonator
    res = hilbertspace.subsystem_list[res_mode_idx]
    if not isinstance(res, scq.Oscillator):
        raise TypeError("The resonator mode should be an oscillator mode")
    res_freq = res.E_osc

    # generate qubit list, qubit_oprt_list, g_list
    qubit_list = []
    qubit_native_operator_list = []
    g_list = []
    for idx, qubit_mode_idx in enumerate(qubit_mode_idx_list):
        qubit = hilbertspace.subsystem_list[qubit_mode_idx]

        interaction = hilbertspace.interaction_list[interaction_idx_list[idx]]
        qubit_idx_in_interaction = qubit_idx_in_interaction_list[idx]
        res_mode_in_interaction = 1 - qubit_idx_in_interaction

        res_oprt = interaction.operator_list[res_mode_in_interaction][1]
        qubit_oprt = interaction.operator_list[qubit_idx_in_interaction][1]
        if callable(qubit_oprt):
            qubit_oprt = qubit_oprt()
        if callable(res_oprt):
            res_oprt = res_oprt()

        g = interaction.g_strength * np.abs(np.array(res_oprt)[0, 1])   # absorb resonator operator into g

        qubit_list.append(qubit)
        qubit_native_operator_list.append(qubit_oprt)
        g_list.append(g)

    # call n_crit_by_diag_subspace
    n_crit = n_crit_by_diag_subspace(
        qubit_list, 
        qubit_native_operator_list, 
        res_freq, 
        g_list,
        state_label = state_label,
        res_mode_idx = res_mode_idx,
        qubit_bare_esys_list = qubit_bare_esys_list,
        res_state_range = res_state_range,
        qubit_state_range_list = qubit_state_range_list,
        max_detuning = max_detuning,
        overlap_threshold = overlap_threshold,
        print_hybridization = print_hybridization,
        max_n_crit = max_n_crit,
    )

    return n_crit
