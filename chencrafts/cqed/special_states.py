import numpy as np
import qutip as qt
from typing import List, Tuple

import scqubits as scq
from chencrafts.settings import QUTIP_VERSION

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

    if QUTIP_VERSION[0] >= 5:
        state = qt.zero_ket(dimensions=dims[0])
    else:
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

def cat(
    phase_disp_pair: List[Tuple[complex, complex]], 
    basis: List[qt.Qobj] | int | None = None
) -> qt.Qobj:
    """
    Return a cat state with given phase and displacement.

    Parameters
    ----------
    phase_disp_pair: List[Tuple[complex, complex]]
        for a two-legged cat: [(1, alpha), (1, -alpha)]

    basis: List[qt.Qobj] | int | None
        [ket0, ket1, ket2, ...]. 
        If None, use Fock basis with auto-generated dimension.
        If int, use Fock basis with given dimension.
    """
    if basis is None:
        disp_list = [disp for phase, disp in phase_disp_pair]
        max_disp = np.max(np.abs(disp_list))
        max_n = int(max_disp**2 + 5 * max_disp)
        basis = [qt.fock(max_n, n) for n in range(max_n)]
    elif isinstance(basis, int):
        basis = [qt.fock(basis, n) for n in range(basis)]
    else:
        pass
    
    dims = basis[0].dims

    if QUTIP_VERSION[0] >= 5:
        state = qt.zero_ket(dimensions=dims[0])
    else:
        N = np.prod(np.array(dims))
        state = qt.zero_ket(N, dims=dims)
    
    for phase, disp in phase_disp_pair:
        state += phase * coherent(basis, disp)

    return state.unit()