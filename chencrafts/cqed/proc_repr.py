__all__ = [
    'pauli_basis',
    'pauli_col_vec_basis',
    'pauli_row_vec_basis',
    'ij_col_vec_basis',
    'pauli_stru_const',
    'bloch_vec_by_op', 'op_by_bloch_vec',
    'to_orth_chi', 'orth_chi_to_choi', 
    'Stinespring_to_Kraus',
]

import numpy as np
import qutip as qt
from typing import List, Tuple

    
# Pauli basis, but normalized according to Hilbert-Schmidt inner product
pauli_basis: List[qt.Qobj] = [
    qt.qeye(2) / np.sqrt(2), 
    qt.sigmax() / np.sqrt(2), 
    qt.sigmay() / np.sqrt(2), 
    qt.sigmaz() / np.sqrt(2)
] 

pauli_col_vec_basis: List[qt.Qobj] = [
    qt.operator_to_vector(pauli) for pauli in pauli_basis
]
pauli_row_vec_basis: List[qt.Qobj] = [
    qt.operator_to_vector(pauli.trans()) for pauli in pauli_basis
]

# |i><j| basis
ij_col_vec_basis: List[qt.Qobj] = [
    qt.operator_to_vector(qt.basis(2, j) * qt.basis(2, i).dag()) 
    for i in range(2) for j in range(2)
]   # column stacking

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


def _construct_env_state(
    dims: List[int] | Tuple[int, ...],
    env_indices: List[int] | Tuple[int, ...],
    env_state_label: List[int] | Tuple[int, ...],
) -> qt.Qobj:
    """
    Construct the environment state vector, tensor product with the system
    identity operator.
    
    Parameters
    ----------
    dims: List[int]
        the dimensions of the composite Hilbert space
    env_indices: List[int]
        the indices of the environment in the composite Hilbert space
    env_state_label: List[int]
        the state of the environment
    """
    ingredients = []
    for idx, dim in enumerate(dims):
        if idx in env_indices:
            ingredients.append(qt.basis(dim, env_state_label[env_indices.index(idx)]))
        else:
            ingredients.append(qt.qeye(dim))
    return qt.tensor(*ingredients)
    
def Stinespring_to_Kraus(
    sys_env_prop: qt.Qobj,
    sys_indices: int | List[int],
    env_state_label: int | List[int] | Tuple[int, ...] | None = None,
):
    """
    Convert a system-environment unitary to a list of Kraus operators. It's like
    a partial trace of the propagator.
    
    sys_prop(rho) = Tr_env[sys_env_prop * (rho x env_state) * sys_env_prop.dag()]
    
    Parameters
    ----------
    sys_env_prop: qt.Qobj
        the propagator acting on the composite Hilbert space of system + environment.
    sys_indices: int | List[int]
        the indices of the system in the composite Hilbert space
    env_state_label: qt.Qobj | int | List[int] | Tuple[int, ...] | None = None
        the state of the environment. If None, the environment is set to be the 
        ground state.
        
    Returns
    -------
    List[qt.Qobj]
        a list of Kraus operators
    """
    dims: List[int] = sys_env_prop.dims[0]
    
    if isinstance(sys_indices, int):
        sys_indices = [sys_indices]
    all_indices = list(range(len(dims)))
    env_indices = [idx for idx in all_indices if idx not in sys_indices]
    
    env_dims = [dims[idx] for idx in env_indices]
    
    # construct the state of the environment when doing partial trace
    if env_state_label is None:
        env_state_label = [0] * len(env_indices)
    if isinstance(env_state_label, int):
        env_state_label = [env_state_label]
    env_state_vec = _construct_env_state(
        dims = dims,
        env_indices = env_indices,
        env_state_label = env_state_label,
    )
    
    # construct an orthonormal basis for the environment
    env_basis = []
    for state_label in np.ndindex(tuple(env_dims)):
        env_basis.append(_construct_env_state(
            dims = dims,
            env_indices = env_indices,
            env_state_label = state_label,
        ))
        
    # calculate the Kraus operators
    kraus_ops = [
        basis.dag() * sys_env_prop * env_state_vec 
        for basis in env_basis
    ]
    
    return kraus_ops    