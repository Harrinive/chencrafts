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

def sum_of_basis(basis: List[qt.Qobj], coef_list) -> qt.Qobj:
    dims = basis[0].dims
    N = np.prod(np.array(dims))

    state = qt.zero_ket(N, dims=dims)
    for idx in range(len(coef_list)):
        state += basis[idx] * coef_list[idx]
    return state.unit()

def coherent(basis: List[qt.Qobj], alpha: float) -> qt.Qobj:
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

def cat(basis: List[qt.Qobj], phase_disp_pair: List[Tuple[int | float, int | float]]) -> qt.Qobj:
    """
    phase_disp_pair for a two-legged cat: [(1, alpha), (1, -alpha)]
    """
    dims = basis[0].dims
    N = np.prod(np.array(dims))
    state = qt.zero_ket(N, dims=dims)
    
    for phase, disp in phase_disp_pair:
        state += phase * coherent(basis, disp)

    return state.unit()

# ##############################################################################
def projector_w_basis(basis: List[qt.Qobj]) -> qt.Qobj:
    projector: qt.Qobj = basis[0] * basis[0].dag()
    for ket in basis[1:]:
        projector = projector + ket * ket.dag()
    return projector

def oprt_in_basis(oprt: np.ndarray | qt.Qobj, states):
    length = len(states)

    if isinstance(oprt, np.ndarray):
        oprt = qt.Qobj(oprt)
    state_qobj = [qt.Qobj(state) for state in states if isinstance(state, np.ndarray)]

    data = np.zeros((length, length), dtype=complex)
    for j in range(length):
        for k in range(j, length):
            elem = oprt.matrix_element(state_qobj[j], state_qobj[k])
            data[j, k] = elem
            data[k, j] = elem.conjugate()

    return qt.Qobj(data)