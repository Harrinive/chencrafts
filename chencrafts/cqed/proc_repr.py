__all__ = [
    'pauli_basis',
    'pauli_col_vec_basis',
    'pauli_row_vec_basis',
    'ij_col_vec_basis',
    'pauli_stru_const',
    'bloch_vec_by_op', 'op_by_bloch_vec',
    'to_orth_chi', 'orth_chi_to_choi', 
]

import numpy as np
import qutip as qt
from typing import List

# Pauli basis, but normalized according to Hilbert-Schmidt inner product
pauli_basis = np.array([
    qt.qeye(2), 
    qt.sigmax(), 
    qt.sigmay(), 
    qt.sigmaz()
], dtype=qt.Qobj) / np.sqrt(2)

pauli_col_vec_basis = np.array([qt.operator_to_vector(pauli) for pauli in pauli_basis], dtype=qt.Qobj)
pauli_row_vec_basis = np.array([qt.operator_to_vector(pauli.trans()) for pauli in pauli_basis], dtype=qt.Qobj)

# |i><j| basis
ij_col_vec_basis = [qt.operator_to_vector(qt.basis(2, j) * qt.basis(2, i).dag()) for i in range(2) for j in range(2)]   # column stacking

# structure constant, determines the multiplication of Pauli operators
# \sigma_a \sigma_b = f_{abc} \sigma_c
# Given the Pauli operators are orthonormal, we can get f_{abc} by
# f_{abc} = \text{tr} (\sigma_a \sigma_b \sigma_c^\dagger)
pauli_stru_const = np.array([
    [
        [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
        [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
        [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
        [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j]
    ],
    [
        [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
        [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
        [0.+0.j, 0.+0.j, 0.+0.j, 0.+1.j],
        [0.+0.j, 0.+0.j, 0.-1.j, 0.+0.j]
    ],
    [
        [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
        [0.+0.j, 0.+0.j, 0.+0.j, 0.-1.j],
        [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
        [0.+0.j, 0.+1.j, 0.+0.j, 0.+0.j]
    ],
    [
        [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
        [0.+0.j, 0.+0.j, 0.+1.j, 0.+0.j],
        [0.+0.j, 0.-1.j, 0.+0.j, 0.+0.j],
        [1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]
    ]
]) / np.sqrt(2)

def bloch_vec_by_op(op: qt.Qobj) -> np.ndarray:
    """
    Given an 2*2 operator, return its Bloch vector representation
    """
    assert op.shape == (2, 2)
    return np.array([(pauli.dag() * op).tr() for pauli in pauli_basis], dtype=complex)

def op_by_bloch_vec(bloch_vec: np.ndarray) -> qt.Qobj:
    """
    Given a Bloch vector, return the corresponding 2*2 operator
    """
    assert bloch_vec.shape == (4,)
    return sum([bloch * pauli for bloch, pauli in zip(bloch_vec, pauli_basis)])

def to_orth_chi(
    superop: qt.Qobj, 
    basis: np.ndarray | List = pauli_col_vec_basis
) -> qt.Qobj:
    """
    Given a superoperator, return its orthogonal chi representation.
    
    Note that it is simply scaled from qt.to_chi(), it seems that qt.to_chi() 
    only uses the Pauli row vector basis, with a different scaling factor.
    """
    choi = qt.to_choi(superop)
    proc_orth_chi = np.zeros(choi.shape, dtype=complex)
    for i, pauli_i in enumerate(basis):
        for j, pauli_j in enumerate(basis):
            proc_orth_chi[i, j] = (pauli_i.dag() * choi * pauli_j)
    return qt.Qobj(proc_orth_chi, dims=choi.dims, superrep='orth_chi')

def orth_chi_to_choi(
    chi: qt.Qobj, 
    basis: np.ndarray | List = pauli_col_vec_basis
) -> qt.Qobj:
    """
    Given an orthogonal chi representation of a superoperator, return the 
    corresponding Choi matrix.
    """
    assert chi.superrep == 'orth_chi'
    choi = qt.Qobj(np.zeros(chi.shape, dtype=complex), dims=chi.dims, superrep='choi')
    for i, pauli_i in enumerate(basis):
        for j, pauli_j in enumerate(basis):
            choi += (pauli_i * pauli_j.dag() * chi[i, j])
    return choi