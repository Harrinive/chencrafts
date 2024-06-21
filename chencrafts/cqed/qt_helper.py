import numpy as np
import qutip as qt
from scipy.sparse import csc_matrix
import functools

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

def trans_by_kets(
    kets: List[np.ndarray] | List[qt.Qobj] | np.ndarray,
):
    """
    Given a list of kets = [|k1>, |k2>, ...], 
    calculate a transformation matrix that can transform a state in the 
    basis of kets.
    """
    if isinstance(kets[0], qt.Qobj):
        kets = [ket.full().ravel() for ket in kets]
    
    # stack all column vectors 
    trans = np.stack(kets, axis=-1)
    return trans
    
def ket_in_basis(
    ket: np.ndarray | qt.Qobj | csc_matrix,
    states: List[np.ndarray] | List[qt.Qobj] | np.ndarray,
):
    """
    Convert a ket to a vector representation described by a given set of basis.
    If the number of basis is smaller than the dimension of the Hilbert space, the ket
    will be projected onto the subspace spanned by the basis.
    """
    length = len(states)

    # dimension check
    assert ket.shape[0] == states[0].shape[0], "Dimension mismatch."

    # go through all states and oprt, to find a dimension 
    if isinstance(ket, qt.Qobj):
        dim = ket.dims[0]
    elif isinstance(states[0], qt.Qobj):
        dim = states[0].dims[0]
    else:
        dim = [ket.shape[0]]

    # convert to qobj
    if isinstance(ket, np.ndarray | csc_matrix):
        ket = qt.Qobj(ket, dims=[dim, list(np.ones_like(dim).astype(int))])
    state_qobj = [qt.Qobj(state, dims=[dim, list(np.ones_like(dim).astype(int))]) for state in states]

    # calculate matrix elements
    data = np.zeros((length, 1), dtype=complex)
    for j in range(length):
        data[j, 0] = state_qobj[j].overlap(ket) 

    return qt.Qobj(data)

def _oprt_in_basis(
    oprt: np.ndarray | qt.Qobj | csc_matrix, 
    bra_basis: List[np.ndarray] | List[qt.Qobj] | np.ndarray,
    ket_basis: List[np.ndarray] | List[qt.Qobj] | np.ndarray | None = None,
) -> Tuple[qt.Qobj, List[qt.Qobj], List[qt.Qobj], List[int]]:
    """
    Internal function to realize oprt_in_basis, which returns more things 
    that could be potentially useful.

    Convert an operator to a matrix representation described by a given set of basis.
    If the number of basis is smaller than the dimension of the Hilbert space, the operator
    will be projected onto the subspace spanned by the basis.
    It can also calculate the off-diagonal elements if bra_states is provided.

    Parameters
    ----------
    oprt : np.ndarray | qt.Qobj | csc_matrix
        operator in matrix form
    bra_basis : List[np.ndarray] | List[qt.Qobj] | np.ndarray
        a list of states describing the basis, [<bra1|, <bra2|, ...]. Note that
        each <bra1| should still be represented as a column vector.
    ket_basis : List[np.ndarray] | List[qt.Qobj] | np.ndarray | None = None
        a list of kets, [|ket1>, |ket2>, ...].
        - If not provided, the returned operator will be O_{jk} = <bra_j|O|bra_k>.
        - If provided, the returned operator will be O_{jk} = <bra_j|O|ket_k>, and it's not a square matrix.
        Note that each |ket1> should be represented as a column vector.

    Returns
    -------
    Tuple[qt.Qobj, List[qt.Qobj], List[qt.Qobj], List[int]]
        the operator in the new basis, 
        a list of qobj of the bra basis (Qobj), 
        a list of qobj of the ket basis (Qobj),
        and the dimension of the Qobj
    """
    if ket_basis is None:
        ket_basis = bra_basis
    #     ident_bra_ket_list = True
    # else:
    #     # treat the basis as a list of bras
    #     ident_bra_ket_list = False

    # dimension check
    assert oprt.shape[0] == bra_basis[0].shape[0], "Dimension mismatch."
    assert oprt.shape[1] == ket_basis[0].shape[0], "Dimension mismatch."

    # # go through all states and oprt, to find a dimension 
    # if isinstance(oprt, qt.Qobj):
    #     dim = oprt.dims[0]
    # elif isinstance(bra_basis[0], qt.Qobj):
    #     dim = bra_basis[0].dims[0]
    # elif isinstance(ket_basis[0], qt.Qobj):
    #     dim = bra_basis[0].dims[0]
    # else:
    #     dim = [oprt.shape[0]]

    # # convert to qobj
    # if isinstance(oprt, np.ndarray | csc_matrix):
    #     oprt = qt.Qobj(oprt, dims=[dim, dim])
    # ket_qobj = [qt.Qobj(state, dims=[dim, list(np.ones_like(dim).astype(int))]) for state in ket_basis]
    # bra_qobj = [qt.Qobj(state, dims=[dim, list(np.ones_like(dim).astype(int))]) for state in bra_basis]

    # # calculate matrix elements
    # data = np.zeros((len(bra_basis), len(ket_basis)), dtype=complex)
    # for j, bra in enumerate(bra_qobj):
    #     for k, ket in enumerate(ket_qobj):
    #         if ident_bra_ket_list and oprt.isherm and j > k:
    #             data[j, k] = data[k, j].conjugate()
    #             continue

    #         data[j, k] = oprt.matrix_element(bra, ket)
            
    # return qt.Qobj(data), bra_qobj, ket_qobj, dim
    
    if isinstance(oprt, qt.Qobj):
        oprt = oprt.full()
    
    # convert bra_list and ket list to transformation matrix
    bra_trans = trans_by_kets(bra_basis)
    ket_trans = trans_by_kets(ket_basis)
    
    data = np.conj(bra_trans.T) @ oprt @ ket_trans
    return qt.Qobj(data)

def oprt_in_basis(
    oprt: np.ndarray | qt.Qobj | csc_matrix,
    bra_basis: List[np.ndarray] | List[qt.Qobj] | np.ndarray,
    ket_basis: List[np.ndarray] | List[qt.Qobj] | np.ndarray | None = None,
) -> qt.Qobj:
    """
    Convert an operator to a matrix representation described by a given set of basis.
    If the number of basis is smaller than the dimension of the Hilbert space, the operator
    will be projected onto the subspace spanned by the basis.
    It can also calculate the off-diagonal elements if bra_states is provided.

    Parameters
    ----------
    oprt : np.ndarray | qt.Qobj | csc_matrix
        operator in matrix form
    bra_basis : List[np.ndarray] | List[qt.Qobj] | np.ndarray
        a list of states describing the basis, [<bra1|, <bra2|, ...].
    ket_basis : List[np.ndarray] | List[qt.Qobj] | np.ndarray | None = None
        a list of kets, [|ket1>, |ket2>, ...].
        - If not provided, the returned operator will be O_{jk} = <bra_j|O|bra_k>.
        - If provided, the returned operator will be O_{jk} = <bra_j|O|ket_k>, and it's not a square matrix.

    Returns
    -------
    qt.Qobj
        the operator in the new basis
    """
    oprt = _oprt_in_basis(oprt, bra_basis, ket_basis)

    return oprt

def superop_in_basis(
    superop: np.ndarray | qt.Qobj | csc_matrix,
    states: List[np.ndarray] | List[qt.Qobj] | np.ndarray,
):
    """
    Convert a superoperator to a matrix representation described by a given set of basis. 
    The basis should be a list of kets.

    If the number of basis is smaller than the dimension of the Hilbert space, the
    superoperator will be projected onto the subspace spanned by the basis.
    """
    length = len(states)

    # dimension check
    assert superop.shape[1] == states[0].shape[0]**2, "Dimension mismatch."

    # go through all states and oprt, to find a dimension
    if isinstance(superop, qt.Qobj):
        dim = superop.dims[0]
    elif isinstance(states[0], qt.Qobj):
        dim = [states[0].dims[0], states[0].dims[0]]
    else:
        # not tested...
        dim = [[states[0].shape[0]], [states[0].shape[0]]]

    # convert to qobj
    if isinstance(superop, np.ndarray | csc_matrix):
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

def evecs_2_transformation(evecs: List[qt.Qobj]) -> qt.Qobj:
    """
    Convert n eigenvectors with length m, convert them to a qobj of size m x n.
    """
    length = len(evecs)
    dim = evecs[0].shape[0]

    data = np.zeros((dim, length), dtype=complex)
    for j in range(length):
        data[:, j] = evecs[j].full().squeeze()

    return qt.Qobj(data)


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

# ##############################################################################
def normalization_factor(ket_or_dm: qt.Qobj):
    """
    Return the normalization factor (N) of a ket or a density matrix (Qobj).
    Such factor makes Qobj / N normalized.
    """
    if qt.isket(ket_or_dm):
        return np.sqrt((ket_or_dm * ket_or_dm.dag()).tr()).real
    elif qt.isoper(ket_or_dm):
        return (ket_or_dm.tr()).real
    else:
        raise ValueError("The object is neither a ket nor a density matrix.")
    
# ##############################################################################
# direct sum
def _direct_sum_ket(ket1: qt.Qobj, ket2: qt.Qobj) -> qt.Qobj:
    return qt.Qobj(np.concatenate((ket1.full(), ket2.full())))

def _direct_sum_op(A: qt.Qobj, B: qt.Qobj) -> qt.Qobj:
    shape_A = np.array(A.shape)
    shape_B = np.array(B.shape)

    A = np.pad(A.full(), ((0, shape_B[0]), (0, shape_B[1])), mode="constant")
    B = np.pad(B.full(), ((shape_A[0], 0), (shape_A[1], 0)), mode="constant")

    return qt.Qobj(A + B)

def _direct_sum_superop(A: qt.Qobj, B: qt.Qobj) -> qt.Qobj:
    raise NotImplementedError(
        "It seems that there is no general way to direct sum two superoperators."
        "For two subsystem's evolution, their noises may be correlated, and a simple"
        "direct-sum-like operation may not know the information and thus"
        "is impossible to find a correct outcome."
    )

def direct_sum(*args: qt.Qobj) -> qt.Qobj:
    """
    Given a few operators (Qobj), return their direct sum.
    """
    if len(args) == 0:
        raise ValueError("No operator is given.")
    elif len(args) == 1:
        return args[0]
    
    if args[0].type == "ket":
        return functools.reduce(_direct_sum_ket, args)
    elif args[0].type == "oper":
        return functools.reduce(_direct_sum_op, args)
    elif args[0].type == "super":
        return functools.reduce(_direct_sum_superop, args)


# ##############################################################################
# fidelity conversion
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

def ave_fid_2_proc_fid(ave_fid, d):
    """
    Convert average gate fidelity to process fidelity using the formula:
        proc_fid = (ave_fid * (d + 1) - 1) / d
    """
    return (ave_fid * (d + 1) - 1) / d

def proc_fid_2_ave_fid(proc_fid, d):
    """
    Convert process fidelity to average gate fidelity using the formula:
        ave_fid = (proc_fid * d + 1) / (d + 1)
    """
    return (proc_fid * d + 1) / (d + 1)

def fid_in_dim(fid, d0, d1, type="ave"):
    """
    Convert a fidelity calculated with operators in (truncated) hilbert space dimension d0
    to a number in hilbert space dimension d1.

    Parameters
    ----------
    fid : float
        fidelity, either average gate fidelity or process fidelity, specified by type
    d0 : int
        dimension of the Hilbert space of the original fidelity
    d1 : int
        dimension of the Hilbert space of the new fidelity
    type : str, optional
        type of the fidelity, by default "ave"

    Returns
    -------
    float
        fidelity in the new Hilbert space dimension
    """
    if type == "ave":
        proc_fid = ave_fid_2_proc_fid(fid, d0)
    elif type == "proc":
        proc_fid = fid
    else:
        raise ValueError("type should be 'ave' or 'proc'")
    
    # this one is only valid for process fidelity
    proc_fid *= (d0 / d1)**2

    if type == "ave":
        fid = proc_fid_2_ave_fid(proc_fid, d1)
    elif type == "proc":
        fid = proc_fid

    return fid