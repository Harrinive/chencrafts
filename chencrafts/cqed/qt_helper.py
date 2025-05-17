__all__ = [
    'projector_w_basis',
    'ket_in_basis',
    'oprt_in_basis',
    'superop_in_basis',
    'basis_of_projector',
    'trans_by_kets',
    'superop_evolve',
    'projected_superop',
    'evecs_2_transformation',
    'qobj_submatrix',

    'normalization_factor',
    'direct_sum',

    'process_fidelity',
    'ave_fid_2_proc_fid',
    'proc_fid_2_ave_fid',
    'fid_in_dim',
    
    'Pauli_twirl',
    'Pauli_distance',
    'Pauli_twirled_dnorm',
    
    'old_leakage_amount',
    'state_leakage_amount',
    'leakage_rate',
    'seepage_rate',
    
    'qobj_sparsity',
    
    'gram_schmidt',
    'complete_basis_set',
    
    'sparsify_qobj',
]

import numpy as np
import qutip as qt
from scipy.sparse import csr_matrix
import functools
from .proc_repr import to_orth_chi, orth_chi_to_choi

import warnings
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
) -> np.ndarray | qt.Qobj:
    """
    Given a list of kets = [|k1>, |k2>, ...], 
    calculate a unitary operator that can perform basis transformation 
    from the current basis to the basis of kets.
    """
    if isinstance(kets[0], qt.Qobj):        
        dim = kets[0].dims[0]
        if kets[0].type == "ket":
            new_dim = [dim, [len(kets)]] 
        elif kets[0].type == "operator-ket":
            new_dim = [dim, [[len(kets)], [1]]]
            # Without the last [1], I will get the following error:
            # NotImplementedError: Operator with both space and 
            # superspace dimensions are not supported
        else:
            raise ValueError("Only ket and operator-ket are supported.")
        
        # turn them into ndarray for stacking purposes
        kets = [ket.full().ravel() for ket in kets]
    else:
        new_dim = None
    
    # stack all column vectors 
    trans = np.stack(kets, axis=-1)
    
    if new_dim is not None:
        trans = qt.Qobj(trans, dims=new_dim)
    
    return trans
    
def ket_in_basis(
    ket: np.ndarray | qt.Qobj | csr_matrix,
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
    if isinstance(ket, np.ndarray | csr_matrix):
        ket = qt.Qobj(ket, dims=[dim, list(np.ones_like(dim).astype(int))])
    state_qobj = [qt.Qobj(state, dims=[dim, list(np.ones_like(dim).astype(int))]) for state in states]

    # calculate matrix elements
    data = np.zeros((length, 1), dtype=complex)
    for j in range(length):
        data[j, 0] = state_qobj[j].overlap(ket) 

    return qt.Qobj(data)

def _oprt_in_basis(
    oprt: np.ndarray | qt.Qobj | csr_matrix, 
    bra_basis: List[np.ndarray] | List[qt.Qobj] | np.ndarray,
    ket_basis: List[np.ndarray] | List[qt.Qobj] | np.ndarray | None = None,
    to_sparse: bool = True,
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
    oprt : np.ndarray | qt.Qobj | csr_matrix
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
        
    # dimension check
    assert oprt.shape[0] == bra_basis[0].shape[0], "Dimension mismatch."
    assert oprt.shape[1] == ket_basis[0].shape[0], "Dimension mismatch."
    
    if not isinstance(oprt, qt.Qobj):
        oprt = qt.Qobj(oprt)
    
    # convert bra_list and ket list to transformation matrix
    bra_trans = qt.Qobj(trans_by_kets(bra_basis))
    ket_trans = qt.Qobj(trans_by_kets(ket_basis))
    
    data = bra_trans.dag() * oprt * ket_trans
    
    if to_sparse:
        data = sparsify_qobj(data)
    
    return data

def oprt_in_basis(
    oprt: np.ndarray | qt.Qobj | csr_matrix,
    bra_basis: List[np.ndarray] | List[qt.Qobj] | np.ndarray,
    ket_basis: List[np.ndarray] | List[qt.Qobj] | np.ndarray | None = None,
    to_sparse: bool = True,
) -> qt.Qobj:
    """
    Convert an operator to a matrix representation described by a given set of basis.
    If the number of basis is smaller than the dimension of the Hilbert space, the operator
    will be projected onto the subspace spanned by the basis.
    It can also calculate the off-diagonal elements if bra_states is provided.

    Parameters
    ----------
    oprt : np.ndarray | qt.Qobj | csr_matrix
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
    oprt = _oprt_in_basis(oprt, bra_basis, ket_basis, to_sparse)

    return oprt

def superop_in_basis(
    superop: np.ndarray | qt.Qobj | csr_matrix,
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
    if isinstance(superop, np.ndarray | csr_matrix):
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
    Convert n eigenvectors |e1>, |e2>, ... with dimension m, convert them to a 
    qobj of size m x n, which reads as:
        |e1><1| + |e2><2| + ... + |en><n|
    """
    length = len(evecs)
    dim = evecs[0].shape[0]

    transformation = np.zeros((dim, length), dtype=complex)
    for j in range(length):
        transformation[:, j] = evecs[j].full().squeeze()

    trans_dims = [evecs[0].dims[0], [length]]
    return qt.Qobj(transformation, dims=trans_dims)

def qobj_submatrix(self: qt.Qobj, states_inds, normalize=False):
    """
    Qobj with states in state_inds only. It's a copy of the deleted (maybe??)
    function in qutip (qutip.Qobj.extract_states). 
    It works similar to oprt_in_basis and ket_in_basis, while it takes in a list 
    of indices instead of a Qobj basis.

    Parameters
    ----------
    states_inds : list of integer
        The states that should be kept.

    normalize : True / False
        Weather or not the new Qobj instance should be normalized (default
        is False). For Qobjs that represents density matrices or state
        vectors normalized should probably be set to True, but for Qobjs
        that represents operators in for example an Hamiltonian, normalize
        should be False.

    Returns
    -------
    q : Qobj
        A new instance of :class:`qutip.Qobj` that contains only the states
        corresponding to the indices in `state_inds`.

    Notes
    -----
    Experimental.
    """
    if self.isoper:
        q = qt.Qobj(self.full()[states_inds, :][:, states_inds])
    elif self.isket:
        q = qt.Qobj(self.full()[states_inds, :])
    elif self.isbra:
        q = qt.Qobj(self.full()[:, states_inds])
    else:
        raise TypeError("Can only eliminate states from operators or " +
                        "state vectors")

    return q.unit() if normalize else q


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
    assert super_propagator_1.type == "super" and super_propagator_2.type == "super", "The input should be superoperators."
    assert super_propagator_1.superrep == "super" and super_propagator_2.superrep == "super", "The superoperators should be in super operator representation."
    
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

def Pauli_twirl(
    superop: qt.Qobj,
) -> qt.Qobj:
    """
    Given a superoperator, return its Pauli twirl.
    """
    original_rep = superop.superrep
    if original_rep == 'orth_chi':
        # convert to qutip choi representation
        superop = orth_chi_to_choi(superop)
        
    chi = qt.to_chi(superop)
    dims = chi.dims
    # Pauli twirl can be done by taking the diagonal of the chi matrix
    chi_diag = np.diag(np.diag(chi.full()))
    result = qt.Qobj(chi_diag, dims=dims, superrep='chi')
    
    if original_rep == 'orth_chi':
        # convert back to orth_chi representation
        return to_orth_chi(result)
    elif original_rep == "super":
        return qt.to_super(result)
    elif original_rep == "choi":
        return qt.to_choi(result)
    elif original_rep == "chi":
        return qt.to_chi(result)
    else:
        raise ValueError(
            "The original representation of the superoperator should be "
            "either 'super', 'choi', 'chi', or 'orth_chi'."
        )

def Pauli_distance(
    superop: qt.Qobj,
) -> float:
    """
    Calculate the "Pauli distance" of a superoperator, which is 
    its diamond norm distance to its Pauli twirl.
    """
    pauli_twirl = Pauli_twirl(superop)
    
    if superop.superrep == "orth_chi":
        superop = orth_chi_to_choi(superop)
        pauli_twirl = orth_chi_to_choi(pauli_twirl)

    return (superop - pauli_twirl).dnorm()

def Pauli_twirled_dnorm(superop: qt.Qobj) -> float:
    """
    Calculate the diamond norm of a Pauli twirled superoperator.
    """
    result = Pauli_twirl(superop)
    
    if superop.superrep == "orth_chi":
        result = orth_chi_to_choi(result)
    
    return result.dnorm()

# Sparsity ================================================================
def qobj_sparsity(oprt: qt.Qobj) -> float:
    try:
        return oprt.data.as_scipy().nnz / np.prod(oprt.shape)
    except:
        warnings.warn(
            "The operator is not in sparse format. Return 0.",
            stacklevel=2,
        )
        return 0

# #############################################################################
def old_leakage_amount(U: qt.Qobj) -> float:
    """
    Calculate the leakage of a quantum channel. It seems to have no reference about
    this definition...
    """
    dim = U.shape[0]
    return 1 - np.abs((U * U.dag()).tr()) / dim

def state_leakage_amount(
    state: qt.Qobj,
    subspace_basis: List[qt.Qobj],
) -> float:
    """
    Calculate the leakage of a state (occupation of the leakage space)
    , according to Eq. (1) in Wood 2018.
    
    Parameters
    ----------
    state : qt.Qobj
        the state to be leaked
    subspace_basis : List[qt.Qobj]
        the basis of the computational subspace. 
        
    Returns
    -------
    float
        the leakage of the state
    """
    if state.isket:
        state = qt.ket2dm(state)
    
    eye = qt.qeye_like(state)
    subspace_projector = projector_w_basis(subspace_basis)
    leakage_space_projector = eye - subspace_projector
    
    return (state * leakage_space_projector).tr().real

def leakage_rate(
    process: qt.Qobj,
    init_subspace_basis: List[qt.Qobj],
    final_subspace_basis: List[qt.Qobj] | None = None,
    normalize: bool = True,
) -> float:
    """
    Calculate the leakage rate of a process, according to Eq. (2) in Wood 2018.
    
    Parameters
    ----------
    process : qt.Qobj
        can be an operator or a superoperator in different representations
    init_subspace_basis : List[qt.Qobj]
        the basis of the computational subspace before the process
    final_subspace_basis : List[qt.Qobj] | None, optional
        the basis of the computational subspace after the process, by 
        default None, which means it's the same as the init_subspace_basis
    normalize : bool, optional
        whether to normalize the maximally mixed state. If True, the definition
        is the same as Wood (2018). However, the result is dependent on whether
        a irrelevant dimension is padded to the computation subspace or leakage
        space. This padding usually happens when we truncate the Hilbert space
        differently.
        If False, we won't normalize the maximally mixed state, making the
        result indepedent of such padding.
        
    Returns
    -------
    float
        the leakage rate of the process
    """
    if final_subspace_basis is None:
        final_subspace_basis = init_subspace_basis

    process = qt.to_super(process)
    init_subspace_projector = projector_w_basis(init_subspace_basis)
    max_mixed_state = init_subspace_projector
    if normalize:
        max_mixed_state = max_mixed_state.unit()
    
    return state_leakage_amount(
        superop_evolve(process, max_mixed_state),
        final_subspace_basis,
    )
    
def seepage_rate(
    process: qt.Qobj,
    init_subspace_basis: List[qt.Qobj],
    final_subspace_basis: List[qt.Qobj] | None = None,
    normalize: bool = True,
) -> float:
    """
    Calculate the seepage rate of a process, according to Eq. (3) in Wood 2018.
    
    Parameters
    ----------
    process : qt.Qobj
        can be an operator or a superoperator in different representations
    init_subspace_basis : List[qt.Qobj]
        the basis of the computational subspace before the process
    final_subspace_basis : List[qt.Qobj] | None, optional
        the basis of the computational subspace after the process, by 
        default None, which means it's the same as the init_subspace_basis
    normalize : bool, optional
        whether to normalize the maximally mixed state. If True, the definition
        is the same as Wood (2018). However, the result is dependent on whether
        a irrelevant dimension is padded to the computation subspace or leakage
        space. This padding usually happens when we truncate the Hilbert space
        differently.
        If False, we won't normalize the maximally mixed state, making the
        result indepedent of such padding.
        
    Returns
    -------
    float
        the seepage rate of the process
    """
    if final_subspace_basis is None:
        final_subspace_basis = init_subspace_basis

    process = qt.to_super(process)
    init_subspace_projector = projector_w_basis(init_subspace_basis)
    eye = qt.qeye_like(init_subspace_projector)
    max_mixed_state = (eye - init_subspace_projector)
    if normalize:
        max_mixed_state = max_mixed_state.unit()
    
    return max_mixed_state.norm() - state_leakage_amount(
        superop_evolve(process, max_mixed_state),
        final_subspace_basis,
    )
    

# ##############################################################################
def gram_schmidt(vectors: List[qt.Qobj]) -> List[qt.Qobj]:
    """
    Given a list of vectors, return their orthonormal basis using the 
    Gram-Schmidt process.
    """
    orthonormal_basis = []
    for vec in vectors:
        for basis in orthonormal_basis:
            vec = vec - basis.overlap(vec) * basis
        orthonormal_basis.append(vec.unit())
    return orthonormal_basis

def complete_basis_set(basis: List[qt.Qobj]) -> List[qt.Qobj]:
    """
    Given a basis that does not span the whole Hilbert space, return a complete 
    basis set that includes the original basis and spans the whole Hilbert space.
    """
    dims = basis[0].dims[0]
    
    # the Fock basis
    Fock_basis = [qt.basis(dims, list(idx)) for idx in np.ndindex(tuple(dims))]
    
    # pick out the basis that has the largest overlap with the original basis
    for original_base in basis:
        overlaps = [np.abs(original_base.overlap(base)) for base in Fock_basis]
        max_overlap_idx = np.argmax(overlaps)
        Fock_basis.pop(max_overlap_idx)
        
    return gram_schmidt(basis + Fock_basis)

# misc ########################################################################
def sparsify_qobj(qobj: qt.Qobj) -> qt.Qobj:
    """
    Given a Qobj, return its sparse representation.
    """
    return qt.Qobj(
        csr_matrix(qobj.full()),
        dims=qobj.dims,
        superrep=qobj.superrep,
    )
