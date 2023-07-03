import numpy as np
import qutip as qt
from typing import List, Tuple

import scqubits as scq

def coherent_coef_list(n, alpha) -> np.ndarray:
    coef = np.exp(-alpha*alpha.conjugate()/2)
    list = [coef]
    for idx in range(1, n):
        coef *= alpha / np.sqrt(idx)
        list.append(coef)
    return np.array(list)

def d_coherent_coef_list(n, alpha) -> np.ndarray:
    coef = np.exp(-alpha*alpha.conjugate()/2)
    list = [coef]
    for idx in range(1, n):
        coef *= alpha / np.sqrt(idx)
        list.append(coef * 1j * idx)
    return np.array(list)

def d2_coherent_coef_list(n, alpha) -> np.ndarray:
    coef = np.exp(-alpha*alpha.conjugate()/2)
    list = [coef]
    for idx in range(1, n):
        coef *= alpha / np.sqrt(idx)
        list.append(-coef * idx**2)
    return np.array(list)

def sum_of_basis(basis: List[qt.Qobj], coef_list: List[complex]) -> qt.Qobj:
    dims = basis[0].dims
    N = np.prod(np.array(dims))

    state = qt.zero_ket(N, dims=dims)
    for idx in range(len(coef_list)):
        state = state + basis[idx] * coef_list[idx]
    return state.unit()

def coherent(basis: List[qt.Qobj], alpha: complex) -> qt.Qobj:
    # check all Nones
    available_dim = 0
    available_ket = []
    for ket in basis:
        if ket is not None:
            available_dim += 1
            available_ket.append(ket)
    
    # calcualte coef and generate state
    coef = coherent_coef_list(available_dim, alpha)
    return sum_of_basis(available_ket, coef)

def cat(phase_disp_pair: List[Tuple[complex, complex]], basis: List[qt.Qobj] | None = None) -> qt.Qobj:
    """
    Return a cat state with given phase and displacement.

    Parameters
    ----------
    phase_disp_pair
        for a two-legged cat: [(1, alpha), (1, -alpha)]

    basis
        [ket0, ket1, ket2, ...]. If None, use Fock basis.
    """
    if basis is None:
        disp_list = [disp for phase, disp in phase_disp_pair]
        max_disp = np.max(np.abs(disp_list))
        max_n = int(max_disp**2 + 5 * max_disp)

        basis = [qt.fock(max_n, n) for n in range(max_n)]

    dims = basis[0].dims
    N = np.prod(np.array(dims))
    state = qt.zero_ket(N, dims=dims)
    
    for phase, disp in phase_disp_pair:
        state += phase * coherent(basis, disp)

    return state.unit()

# ##############################################################################
def projector_w_basis(basis: List[qt.Qobj]) -> qt.Qobj:
    """
    Generate a projector onto the subspace spanned by the given basis.
    """

    projector: qt.Qobj = basis[0] * basis[0].dag()
    for ket in basis[1:]:
        projector = projector + ket * ket.dag()
    return projector

def oprt_in_basis(
    oprt: np.ndarray | qt.Qobj, 
    states: List[np.ndarray] | List[qt.Qobj] | np.ndarray
):
    """
    Convert an operator to a matrix representation described by a given set of basis.
    """
    length = len(states)

    # go through all states and oprt, to find a dimension 
    if isinstance(oprt, qt.Qobj):
        dim = oprt.dims[0]
    elif isinstance(states[0], qt.Qobj):
        dim = states[0].dims[0]
    else:
        dim = [oprt.shape[0]]

    # convert to qobj
    if isinstance(oprt, np.ndarray):
        oprt = qt.Qobj(oprt, dims=[dim, dim])
    state_qobj = [qt.Qobj(state, dims=[dim, list(np.ones_like(dim).astype(int))]) for state in states]

    # calculate matrix elements
    data = np.zeros((length, length), dtype=complex)
    for j in range(length):
        for k in range(j, length):
            data[j, k] = oprt.matrix_element(state_qobj[j], state_qobj[k])
            if j != k:
                if oprt.isherm:
                    data[k, j] = data[j, k].conjugate()
                else:
                    data[k, j] = oprt.matrix_element(state_qobj[k], state_qobj[j])

    return qt.Qobj(data)