import numpy as np
import qutip as qt

from typing import Literal, Callable, List, Tuple, overload

# ##############################################################################
def projector_w_basis(basis: List[qt.Qobj]) -> qt.Qobj:
    """
    Generate a projector onto the subspace spanned by the given basis.
    """

    projector: qt.Qobj = basis[0] * basis[0].dag()
    for ket in basis[1:]:
        projector = projector + ket * ket.dag()
    return projector

def basis_of_projector(projector: qt.Qobj) -> List[qt.Qobj]:
    """
    Return a basis of the subspace projected by the projector.
    """
    evals, evecs = projector.eigenstates()
    projected_basis = []
    for idx, val in enumerate(evals):
        if np.abs(val - 1) < 1e-6:
            projected_basis.append(evecs[idx])
        elif np.abs(val) < 1e-6:
            continue
        else:
            raise ValueError("The object is not a projector with an eigenvalue that is "
                             "neither 0 nor 1.")
    return projected_basis

def oprt_in_basis(
    oprt: np.ndarray | qt.Qobj, 
    states: List[np.ndarray] | List[qt.Qobj] | np.ndarray
):
    """
    Convert an operator to a matrix representation described by a given set of basis.
    If the number of basis is smaller than the dimension of the Hilbert space, the operator
    will be projected onto the subspace spanned by the basis.
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

def superop_in_basis(
    superop: np.ndarray | qt.Qobj,
    states: List[np.ndarray] | List[qt.Qobj] | np.ndarray,
):
    """
    Convert a superoperator to a matrix representation described by a given set of basis. 
    The basis should be a list of kets.

    If the number of basis is smaller than the dimension of the Hilbert space, the
    superoperator will be projected onto the subspace spanned by the basis.
    """
    length = len(states)

    # go through all states and oprt, to find a dimension
    if isinstance(superop, qt.Qobj):
        dim = superop.dims[0]
    elif isinstance(states[0], qt.Qobj):
        dim = [states[0].dims[0], states[0].dims[0]]
    else:
        # not tested...
        dim = [[states[0].shape[0]], [states[0].shape[0]]]

    # convert to qobj
    if isinstance(superop, np.ndarray):
        superop = qt.Qobj(superop, dims=[dim, dim])
    state_qobj = [qt.Qobj(state, dims=dim) for state in states] 

    # generata a basis of the operator space
    dm_qobj = [state_qobj[j] * state_qobj[k].dag() for j, k in np.ndindex(length, length)]

    # calculate matrix elements
    data = np.zeros((length**2, length**2), dtype=complex)
    for j in range(length**2):
        for k in range(length**2):
            data[j, k] = (dm_qobj[j].dag() * superop_evolve(superop, dm_qobj[k])).tr()

    return qt.Qobj(data, dims=[[[length]] * 2] * 2,)

# ##############################################################################
def superop_evolve(superop: qt.Qobj, state: qt.Qobj) -> qt.Qobj:
    """
    return a density matrix after evolving with a superoperator
    """
    if qt.isket(state):
        state = qt.ket2dm(state)

    return qt.vector_to_operator(superop * qt.operator_to_vector(state))

def projected_superop(
    superop: qt.Qobj,
    subspace_basis: List[qt.Qobj],
    in_new_basis: bool = False,
) -> qt.Qobj:
    """
    If provided a set of basis describing a subspace of a Hilbert space, return 
    the superoperator projected onto the subspace.

    If in_new_basis is True, the superoperator is represented in the new basis, i.e.,
    dimension becomes d^2 * d^2, where d = len(subspace_basis).
    """
    if not in_new_basis:
        # just do a simple projection
        projector = projector_w_basis(subspace_basis)
        superop_proj = qt.to_super(projector)
        return superop_proj * superop * superop_proj
    
    else:   
        # calculate the matrix elements of the superoperator in the new basis
        return superop_in_basis(superop, subspace_basis)

def process_fidelity(
    super_propagator_1: qt.Qobj, super_propagator_2: qt.Qobj, 
    subspace_basis: List[qt.Qobj] | None = None,
) -> float:
    """
    The process fidelity between two superoperators. The relationship between process and 
    qt.average_gate_fidelity is: 
        process_fidelity * d + 1 = (d + 1) * qt.average_gate_fidelity
    where d is the dimension of the Hilbert space.


    """

    if subspace_basis is not None:
        # write the superoperators in the new basis to reduce the dimension and speed up 
        # the calculation
        super_propagator_1 = projected_superop(super_propagator_1, subspace_basis, in_new_basis=True)
        super_propagator_2 = projected_superop(super_propagator_2, subspace_basis, in_new_basis=True)
        subspace_dim = len(subspace_basis)
    else:
        subspace_dim = np.sqrt(super_propagator_1.shape[0]).astype(int)

    return qt.fidelity(
        qt.to_choi(super_propagator_1) / subspace_dim,
        qt.to_choi(super_propagator_2) / subspace_dim
    )**2