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
    N = np.prod(dims)

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

def cat(basis: List[qt.Qobj], phase_disp_pair: List[Tuple[float]]) -> qt.Qobj:
    """
    phase_disp_pair for a two-legged cat: [(1, alpha), (1, -alpha)]
    """
    dims = basis[0].dims
    N = np.prod(dims)
    state = qt.zero_ket(N, dims=dims)
    
    for phase, disp in phase_disp_pair:
        state += phase * coherent(basis, disp)

    return state.unit()

def dressed_basis(
    h_space: scq.HilbertSpace, 
    dim_list: List[float], 
    esys: Tuple[np.ndarray]
):
    if esys is None:
        esys = h_space.eigensys(evals_count=np.prod(dim_list))
    h_space.generate_lookup(dressed_esys=esys)

    # basis
    _, evecs = esys
    drs_basis: np.ndarray = np.ndarray(dim_list, dtype=qt.Qobj)
    for idx, bare_idx in enumerate(np.ndindex(dim_list)):

        drs_idx = h_space.dressed_index(bare_idx)
        if drs_idx is not None:
            evec = evecs[drs_idx]
            # make the "principle_val" have zero phase
            principle_val = evec[idx, 0]
            evec /= (principle_val) / np.abs(principle_val)
            drs_basis[bare_idx] = evec
        else:
            drs_basis[bare_idx] = None

    return drs_basis

def projector_w_basis(basis: List[qt.Qobj]) -> qt.Qobj:
    projector = 0
    for ket in basis:
        projector = projector + ket * ket.dag()
    return projector
