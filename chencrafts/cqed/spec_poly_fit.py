# fitting the resonator frequency to determine the chi and kerr

import numpy as np
import qutip as qt

from scqubits.core.hilbert_space import HilbertSpace

from typing import List, Tuple
import warnings

from chencrafts.cqed.mode_assignment import single_mode_dressed_esys

def to_normal_order(p: np.ndarray | List):
    """
    By fitting the resonator frequency as a polynomial, we are effectively using this 
    hamiltonian:
        H = p0 + p1 * a^\dag a + p2 * (a^\dag a)^2 + p3 * (a^\dag a)^3...
    It's effectively similar to fitting the frquency to a polynomial:
        f(n) = p0 + q1 * n + q2 * n^2 + q3 * n^3...
    
    It converts the fitting parameters (p0, p1, p2, ...) to ones for the normal order 
    hamiltonian:
        H = q0 + q1 * a^\dag a + q2 * (a^\dag)^2 a^2 + q3 * (a^\dag)^3 a^3...
    It's effectively similar to fitting the frquency to a polynomial:
        f(n) = q0 + q1 * n + q2 * n(n-1) + q3 * n(n-1)(n-2)...
             = q0 + q1 * n + q2 * n^2 - q2 * n + q3 * n^3 - 3 * q3 * n^2 + 2 * q3 * n...

    Parameters
    ----------
    [p0, p1, p2, ...]

    Returns
    -------
    [q0, q1, q2, ...]
    """
    degree = len(p)

    def n_prod(n: int, i: int):
        return np.prod(np.arange(n-i+1, n+1))
    def f(n: int, p = p):
        return np.sum(p * np.power(n, range(degree)))


    # It's obvious that f(1) = q1, f(2) = q1 + 2*q2, f(3) = q1 + 6*q2 + 6*q3, ...
    # So that we can solve q1, q2, q3, ... by solving a linear system:

    # cook the matrix
    A = np.zeros((degree, degree))
    for n in range(degree):
        for j in range(degree):
            A[n, j] = n_prod(n, j)
    
    # cook the vector
    b = np.zeros(degree)
    for n in range(degree):
        b[n] = f(n)

    # solve for q0, q1, q2, q3, ...
    q = np.linalg.solve(A, b)

    return q


def spec_poly_fit(
    hilbert_space: HilbertSpace,
    mode_idx: int,
    state_label: Tuple[int, ...] | List[int],
    fit_truncated_dim: int = 2,
    fit_polynomial_degree: int = 1,
    return_normal_order: bool = True,
    dressed_indices: np.ndarray | None = None,
    eigensys = None,
):
    """
    It fits the resonator frequency to a weakly non-linear polynomial to characterize the 
    resonator.

    Parameters
    ----------
    hilberspace:
        scq.HilberSpace object which include the desired mode
    mode_idx:
        The index of the resonator mode of interest in the hilberspace's subsystem_list
    state_label:
        The resonator frequencies are calculated with other modes staying at some bare 
        states. For example, we are looking for the spectrum of the resonator mode 0 in a
        three mode system with the last two mode fixed at bare state 0 and barea state 1, 
        we can set state_label to be (<any number>, 0, 1).
    fit_truncated_dim: 
        The number the resonator eigenvalues for fitting. Should be equal to or smaller 
        than the dimension of the bare resonator dimension and greater than 
        fit_polynimial_degree.
    fit_polynomial_degree: 
        The order of the polynomial used for fitting. Should be equal to or greater than 1.
    dressed_indeces: 
        An array mapping the FLATTENED bare state label to the eigenstate label. Usually
        given by `ParameterSweep["dressed_indices"]`.
    eigensys:
        The eigenenergies and eigenstates of the bare Hilbert space. Usually given by
        `ParameterSweep["evals"]` and `ParameterSweep["evecs"]`.

    Returns
    -------
    The fitting parameters of the resonator frequency.
    """
    
    evals, _ = single_mode_dressed_esys(
        hilbert_space, mode_idx, state_label, dressed_indices, eigensys, adjust_phase=False
    )

    # truncate the eigenvalues
    if fit_truncated_dim <= fit_polynomial_degree:
        raise ValueError("fit_truncated_dim must be greater than "
                         "fit_polynomial_order.")
    elif fit_truncated_dim > len(evals):
        fit_truncated_dim = len(evals)
        warnings.warn(f"fit_truncated_dim must be smaller than or equal to "
                      f"available resonator dimension {len(evals)}.\n")

    evals = evals[:fit_truncated_dim]

    result = np.polyfit(np.arange(fit_truncated_dim), evals, fit_polynomial_degree)

    if return_normal_order:
        return to_normal_order(result[-1::-1])
    else:
        return result[-1::-1]