__all__ = [
    'StateNode',
    'StateEnsemble',
]

import qutip as qt
import numpy as np
from copy import deepcopy
from warnings import warn

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False 

from chencrafts.cqed.qt_helper import (
    projector_w_basis,
    normalization_factor,
    evecs_2_transformation,
    proc_fid_2_ave_fid,
    complete_basis_set,
    evecs_2_transformation,
)
from chencrafts.cqed.proc_repr import (
    pauli_col_vec_basis, pauli_basis, to_orth_chi, orth_chi_to_choi
)

from typing import List, Tuple, Any, TYPE_CHECKING, Dict, Literal
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from .edge import Edge

MeasurementRecord = List[Tuple[int, ...]]


to_choi_vec = np.vectorize(qt.to_choi)
to_chi_vec = np.vectorize(qt.to_chi)
to_orth_chi_vec = np.vectorize(to_orth_chi)
to_super_vec = np.vectorize(qt.to_super)
orth_chi_to_choi_vec = np.vectorize(orth_chi_to_choi)


def effective_logical_process(
    process: qt.Qobj,
    init_encoders: np.ndarray[qt.Qobj] | List[qt.Qobj],
    final_encoders: np.ndarray[qt.Qobj] | List[qt.Qobj],
    repr: Literal["super", "choi", "chi", "orth_chi"] = "super",
) -> np.ndarray[qt.Qobj]:
    """
    The effective logical process in the computational basis.
    
    Note that the initial and final states may be in multiple logical subspaces,
    say the initial node have i subspaces and the final node have f subspaces.
    Then the effective logical process is a f*i matrix, with each element being
    a superoperator representation of the logical process.
    """
    len_init_subspace = len(init_encoders)
    len_final_subspace = len(final_encoders)
    
    init_encoders_superop = [
        qt.sprepost(enc, enc.dag()) for enc in init_encoders
    ]
    final_decoders_superop = [
        qt.sprepost(enc.dag(), enc) for enc in final_encoders
    ]

    effective_logical_process = np.ndarray(
        (len_final_subspace, len_init_subspace), 
        dtype=qt.Qobj
    )
    
    for idx_final, idx_init in np.ndindex(*effective_logical_process.shape):
        proc = (
            final_decoders_superop[idx_final] * process * init_encoders_superop[idx_init]
        )
        
        effective_logical_process[idx_final, idx_init] = proc
        
    if repr == "super":
        return effective_logical_process
    elif repr == "choi":
        return to_choi_vec(effective_logical_process)
    elif repr == "chi":
        return to_chi_vec(effective_logical_process)
    elif repr == "orth_chi":
        return to_orth_chi_vec(effective_logical_process)
    else:
        raise ValueError(
            "The type of effective logical process should be "
            "either 'super', 'choi', 'chi', or 'orth_chi'."
        )

def target_process_for_dnorm(
    process: qt.Qobj,
    I_prob: float | None = None,
) -> qt.Qobj:
    """
    To calculate the diamond norm of the a process, we need to
    specify the target process to compare to. It's typically a scaled 
    identity process.
    It supports super, choi, chi, and orth_chi representations.
    
    I_prob: the probability of having I * rho * I in the compared process. 
    If not specified, it will be calculated from the process.
    """
    assert process.shape == (4, 4), "Only support single qubit process for now."
    
    if I_prob is None:
        # choi matrix is a unitary transformation of the orth_chi matrix
        # if it's CPTP, the trace of choi matrix is 2 (dimension)
        if process.superrep == "orth_chi":
            choi = orth_chi_to_choi(process)
        else:
            choi = qt.to_choi(process)
        I_prob = choi.tr()
    else:
        # we need to scale the give I prob as I_op has a normalization factor 
        # of sqrt(2)
        I_prob = I_prob * 2 
        
    I_op = pauli_basis[0]   # I operator / sqrt(2)
    
    # find the superoperator representation of I*rho*I
    I_super = qt.sprepost(I_op, I_op) * I_prob
    
    if process.superrep == "orth_chi":
        I_super = to_orth_chi(I_super)
    elif process.superrep == "choi":
        I_super = qt.to_choi(I_super)
    elif process.superrep == "chi":
        I_super = qt.to_chi(I_super)
    elif process.superrep == "super":
        pass
    else:
        raise ValueError("The superoperator representation of the process should be "
                         "either 'super', 'choi', 'chi', or 'orth_chi'.")

    return I_super

def choi_conv_change(choi: qt.Qobj):
    """
    Change a Choi matrix between column-convension and row-convension.
    """
    
    dim_1, dim_2 = choi.dims[0]
    shape_1, shape_2 = np.prod(dim_1), np.prod(dim_2)
    
    # reshape to 4d array
    choi_array = choi.full().reshape(shape_1, shape_2, shape_1, shape_2)
    # change the convension by trasposition
    choi_array = np.transpose(choi_array, (1, 0, 3, 2))
    # reshape back to 2d array
    choi_array = choi_array.reshape(shape_1*shape_2, shape_1*shape_2)
    
    return qt.Qobj(
        choi_array, 
        dims = [[dim_2, dim_1], [dim_2, dim_1]], 
        superrep = 'choi'
    )

def choi_multiply(
    choi_1: qt.Qobj, 
    choi_2: qt.Qobj, 
    convension: str = 'row'
) -> qt.Qobj:
    """
    Multiply two processes in Choi representation.
    
    convension: 'row' or 'col'
    """
    if convension == 'row':
        choi_1 = choi_conv_change(choi_1)
        choi_2 = choi_conv_change(choi_2)
    
    superop_1 = qt.to_super(choi_1)
    superop_2 = qt.to_super(choi_2)
    return qt.to_choi(superop_1 * superop_2)

def complete_basis_by_encoder(
    encoder: qt.Qobj,
) -> List[qt.Qobj]:
    """
    Given an encoder, return the complete basis set that includes the original 
    basis and spans the whole Hilbert space.
    
    The encoder is a 2D Qobj, formed by a few columns of basis states.
    """
    dims = encoder.dims[0]
    ones = [1] * len(dims)
    basis_dims = [dims, ones]
    original_basis = [
        qt.Qobj(
            np.array(encoder[:, idx:idx+1]), 
            dims=basis_dims,
        ) for idx in range(encoder.shape[1])
    ]
    
    return complete_basis_set(original_basis)
    
def leakage_process(
    process: qt.Qobj,
    init_encoders: np.ndarray[qt.Qobj] | List[qt.Qobj],
    final_encoders: np.ndarray[qt.Qobj] | List[qt.Qobj],
    truncate_init_space: bool = True,
    reorder_final_space: bool = True,
) -> np.ndarray[qt.Qobj]:
    """
    The processes that map the logical states outside of the computational subspaces.
    
    For one initial subspace projector P_i, one final subspace projector 
    P_f and the final leakage subspace projector P_L, we first define the 
    projector superoperator 
    Ps_fL (rho) = P_L * rho * P_f + P_f * rho * P_L
    and
    Ps_ii (rho) = P_i * rho * P_i
    
    the leakage process can be represented as:
    S_ia = Ps_fL * process * Ps_ii
    
    Note that the initial and final states may be in multiple logical subspaces,
    say the initial node have i subspaces and the final node have f subspaces.
    Then the effective logical process is a f*i matrix, with each element being
    a superoperator representation of the logical process.
    
    Parameters:
    truncate_init_space: bool
        Whether to truncate the initial space to a smaller dimension. If 
        False, the initial space is not truncated although the rank is 
        smaller than the dimension.
    reorder_final_space: bool
        Whether to reorder the final space, so that the first few (two)
        index of the final states are final logical states. The rest of the 
        basis are obtained by Gram-Schmidt orthogonalization.
        
    """
    len_init_subspace = len(init_encoders)
    len_final_subspace = len(final_encoders)
    
    if truncate_init_space:
        init_superop = [
            qt.sprepost(enc, enc.dag()) for enc in init_encoders
        ]
    else:
        init_projectors = [
            enc * enc.dag() for enc in init_encoders
        ]
        init_superop = [
            qt.sprepost(proj, proj) for proj in init_projectors
        ]
        
    final_projectors = [
        enc * enc.dag() for enc in final_encoders
    ]
    leakage_projector = qt.qeye_like(final_projectors[0]) - sum(final_projectors)
    final_superop = [
        (
            qt.sprepost(proj, leakage_projector)
            + qt.sprepost(leakage_projector, proj)
        ) for proj in final_projectors
    ]
    
    if reorder_final_space:
        final_transformations = []
        for enc in final_encoders:
            basis = complete_basis_by_encoder(enc)
            trans = evecs_2_transformation(basis).dag()
            final_transformations.append(qt.to_super(trans))
            
        final_superop = [
            final_transformation * superop 
            for final_transformation, superop 
            in zip(final_transformations, final_superop)
        ]
    
    leakage_process = np.ndarray(
        (len_final_subspace, len_init_subspace), 
        dtype=qt.Qobj
    )
    
    for idx_final, idx_init in np.ndindex(*leakage_process.shape):
        proc = (
            final_superop[idx_final] * process * init_superop[idx_init]
        )
        
        leakage_process[idx_final, idx_init] = proc
        
    return leakage_process

def truncate_leakage_process(
    leakage_process: qt.Qobj,
    logical_dim: int,
    threshold: float,
) -> Tuple[List[int], qt.Qobj]:
    """
    Determine whether to keep the level l by the smallness of 
    matrix element of the superoperator: proc_lijk.
    If the matrix elements with all ijk < logical_dim is 
    smaller than a threshold, then remove it.
    
    Returns:
    kept_levels: List[int]
        The levels that are kept.
    new_process: qt.Qobj
        The truncated leakage process. Say 5 levels are kept, 
        the resulting process is a 5*5*logical_dim*logical_dim
        superoperator.
    """
    dim = int(np.sqrt(leakage_process.shape[0]))
    kept_levels = list(range(dim))
    
    for level in range(logical_dim, dim):
        l_slice = [
            level * dim + idx for idx in range(logical_dim)
        ]
        if np.max(np.abs(leakage_process[l_slice, :])) < threshold:
            kept_levels.remove(level)
       
    # slice the leakage process to keep the levels
    new_dim = len(kept_levels)
    
    kept_idx = [    
        idx_1 * dim + idx_2 
        for idx_1 in kept_levels 
        for idx_2 in kept_levels 
    ] 
    kept_idx = list(set(kept_idx))
    new_process = leakage_process.full()[kept_idx, :]
    new_dims = [[[new_dim], [new_dim]], leakage_process.dims[1]]
    
    return kept_levels, qt.Qobj(new_process, dims=new_dims, superrep='super')

def bound_fidelity(
    process_1: qt.Qobj, 
    logical_dim: int,
    leakage_space_dim: int,
    maximize: bool = True,
    constr_proc2_fid: float | None = None,
    constr_proc2_seepage: float | None = None,
) -> Dict[str, float]:
    """
    Bound the fidelity of process_2 * process_1, where process_1 is given by the 
    user, and process_2 is maximized or minimized over all possible CPTP processes.
    
    The process_1 maps the operators on H_1 to operators on H_2, which is specified 
    by a Choi matrix. The following process_2 must maps the operators on H_2 
    to operators on H_3. We require H_3 to include H_1, so that the full process 
    (maybe projected) maps operators on H_1 to operators on H_1, which makes the 
    fidelity calculation well defined.
    
    Parameters:
    process_1: qt.Qobj
        Superoperator representation of the first process. Note that the 
        computational basis of H_1 and H_2 must be ordered in the first two
        dimensions of the density matrix.
    logical_dim: int
        The dimension of the logical subspace, the fidelity is divided over
        logical_dim**2. Besides, the fidelity and seepage constraint is 
        applied assuming the logical dimension is logical_dim.
    leakage_space_dim: int
        The dimension of the leakage space. In a simple case, it's the 
        dim(H_2) - logical_dim, but it can be smaller when there are 
        multiple allowed computational subspaces.
    choi_conv: 'row' or 'col'
        The convension of the Choi matrix.
    maximize: bool
        Whether to maximize the fidelity.
    constr_proc2_fid: float | None
        The fidelity lower bound for the process_2.
    constr_proc2_seepage: float | None
        The seepage upper bound for the process_2.
        
    Returns:
    Dict[str, float]
        "fidelity": the fidelity of the process_2 * process_1
        "process_2": the process_2 that maximizes the fidelity
        "process_total": process_2 * process_1
    """
    if not CVXPY_AVAILABLE:
        raise ImportError("CVXPY is required for bound_fidelity function.")
    
    if logical_dim != 2:
        raise ValueError(
            "Only support logical_dim = 2 for now, not for the fidelity"
            "calculation, but for the fidelity and seepage constraint."
        )
    
    # convert process_1 to Choi matrix with row-convension
    choi_1 = choi_conv_change(qt.to_choi(process_1))
    
    # define the dimensions --------------------------------------------
    # dimension for individual subsystems in H_2 and H_1
    subsys_dim_2, subsys_dim_1 = choi_1.dims[0]   
    # dimension for H_2 and H_1
    dim_2, dim_1 = np.prod(subsys_dim_2), np.prod(subsys_dim_1)
    # for a choi matrix, it maps a state from H1 \otimes H2 to H1 \otimes H2.
    # so the dimension of the choi matrix is shape_1 * shape_2
    choi_1_dim = dim_1 * dim_2
    
    if constr_proc2_seepage is not None:
        # if we need to constraint on seepage rate, then
        # we need to provide the full choi_2.
        use_proj_choi = False
        
        # dim_3 should be greater than logical_dim. If not, the seepage
        # rate is always 1, as the logical subspace is the only destination
        # of the process_2.
        # choice 1: 
        # dim_3 = dim_2
        # choice 2: 
        dim_3 = logical_dim + 1
    else:
        # if we don't need to constraint on seepage rate, then
        # we only need to optimize the projected choi_2
        use_proj_choi = True
        dim_3 = logical_dim
        
    # for process_2, it maps a state from H2 to H3
    choi_2_dim = dim_3 * dim_2

    # Define the variables ---------------------------------------------
    # Flattened 2D form of the 4D choi matrix
    if use_proj_choi:
        # if we don't need to constraint on seepage rate, then
        # we only need to optimize the projected choi_2
        choi_2_proj = cp.Variable((choi_2_dim, choi_2_dim), hermitian=True)  
    else:
        choi_2 = cp.Variable((choi_2_dim, choi_2_dim), hermitian=True)  
        
        # project the final state after E to the logical subspace:
        # choi_2_mpnq with m, n < 2
        proj_slice = [
            m_idx * dim_2 + p_idx 
            for m_idx in range(logical_dim)
            for p_idx in range(dim_2)
        ]
        choi_2_proj = choi_2[proj_slice, :][:, proj_slice]

    # Objective --------------------------------------------------------
    # Compute the fidelity by trace(process_1 * process_2) = Sum_mnpq (choi_2_proj_mpnq * choi_1_pmqn)
    # it's done by elementwise multiplication after transposing choi_1 to (m, p, n, q) ordering
    choi_1_4d = np.reshape(choi_1.full(), (dim_2, dim_1, dim_2, dim_1))
    choi_1_4d_T = np.transpose(choi_1_4d, (1, 0, 3, 2)) 
    choi_1_2d_T = np.reshape(choi_1_4d_T, (choi_1_dim, choi_1_dim))

    fidelity = (
        cp.real(cp.sum(cp.multiply(choi_2_proj, choi_1_2d_T))) 
        / logical_dim**2 
    )
    
    # Constraints ------------------------------------------------------
    constaints = []
    if use_proj_choi:
        const_choi = choi_2_proj
    else:
        const_choi = choi_2
        
    # 1. trace-preserving process: Partial trace of choi_2 is an 
    # identity operator, i.e.
    # sum_m choi_2_mpmq = delta_pq   (row-convension)
    for p_idx in range(dim_2):
        for q_idx in range(dim_2):
            choi_2_idx1 = slice(p_idx, None, dim_2)
            choi_2_idx2 = slice(q_idx, None, dim_2)
            trace_elem = cp.sum(cp.diag(const_choi[choi_2_idx1, choi_2_idx2]))
            constaints.append(trace_elem == int(p_idx == q_idx))
            
    # 2. completely positive process: choi_2 is positive semidefinite
    constaints.append(const_choi >> 0)
    
    # 3. (Optional) process_2 realizes a high fidelity gate within the logical subspace
    # fidelity = trace(superop(process_2)) = sum_(m<2, n<2) choi_2_mmnn
    if constr_proc2_fid is not None:
        slice_mm = [m_idx * dim_2 + m_idx for m_idx in range(logical_dim)]
        slice_nn = [n_idx * dim_2 + n_idx for n_idx in range(logical_dim)]
        subspace_fidelity = cp.real(cp.sum(const_choi[slice_mm, :][:, slice_nn])) / logical_dim**2
        constaints.append(
            subspace_fidelity >= constr_proc2_fid
        )
        
    # 4. (Optional) the seepage rate of the process is less than a threshold
    # For seepage rate, see Wood (2018)
    # which is calculated by 1 - sum_(n>2, p>2) choi_2_npnp / leakage_space_dim
    if constr_proc2_seepage is not None:
        slice_np = [
            n_idx * dim_2 + p_idx 
            for n_idx in range(logical_dim, dim_3)
            for p_idx in range(logical_dim, dim_2)
        ]
        seepage_rate = cp.real(1 - cp.trace(const_choi[slice_np, :][:, slice_np]) / leakage_space_dim)
        constaints.append(seepage_rate <= constr_proc2_seepage)
            
    # solve the problem ------------------------------------------------
    if maximize:
        objective = cp.Maximize(fidelity)
    else:
        objective = cp.Minimize(fidelity)

    # Define and solve the problem
    problem = cp.Problem(objective, constaints)
    result = problem.solve(solver=cp.MOSEK)

    if problem.status != 'optimal':
        raise ValueError(f"The result is not optimal: {problem.status}")
    
    # post-processing -------------------------------------------------
    if use_proj_choi:
        process_2 = None
        process_total = None
    else:
        choi_2_dims = [[[dim_3], [dim_2]], [[dim_3], [dim_2]]]
        choi_2 = qt.Qobj(choi_2.value, dims=choi_2_dims, superrep='choi')
        choi_2 = choi_conv_change(choi_2)
        process_2 = qt.to_super(choi_2)
        process_total = process_2 * process_1
    
    # for choi_2_proj, it always has dimension logical_dim * dim_2,
    # regardless of use_proj_choi
    choi_2_proj_dims = [[[logical_dim], [dim_2]], [[logical_dim], [dim_2]]]
    choi_2_proj = qt.Qobj(choi_2_proj.value, dims=choi_2_proj_dims, superrep='choi')
    choi_2_proj = choi_conv_change(choi_2_proj)
    process_2_proj = qt.to_super(choi_2_proj)
    process_total_proj = process_2_proj * process_1
    
    return {
        "fidelity": float(result),
        "process_2": process_2,
        "process_total": process_total,
        "process_2_proj": process_2_proj,
        "process_total_proj": process_total_proj,
    }


class TerminationError(Exception):
    """
    Error raised when trying to perform operations on a terminated node.
    
    This typically happens when trying to:
    - Calculate effective logical process
    - Calculate process fidelity
    - Calculate process diamond norm
    - Or other operations that require valid process information
    """
    def __init__(self, message: str = "Operations not allowed on terminated node"):
        self.message = message
        super().__init__(self.message)


class NodeBase(ABC):
    # current state as a density matrix
    state: qt.Qobj
    
    # process that evolves the initial node to the current node
    process: qt.Qobj
    
    # initial encoders that encode the logical states to the physical states    
    # it is used for calculating the effective logical process
    init_encoders: np.ndarray[qt.Qobj]
    
    index: int

    def __init__(
        self, 
    ):
        """
        A node that represents a state in the QEC trajectory
        """
        self.out_edges: List["Edge"] = []

    @property
    @abstractmethod
    def fidelity(self) -> float:
        """
        Calculate the fidelity of the state
        """
        pass
    
    def assign_index(self, index: int):
        self.index = index

    def to_nx(self) -> Tuple[int, Dict[str, Any]]:
        """
        Convert to a networkx node
        """
        return (
            self.index,
            {
                "state": self,
            }
        )

    @abstractmethod
    def deepcopy(self):
        """
        1. Not storing the edge information and avoiding circular reference
        2. deepcopy the Qobj
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    def add_out_edges(self, edge):
        self.out_edges.append(edge)

    @abstractmethod
    def clear_evolution_data(self):
        pass

    def expect(self, op: qt.Qobj) -> float:
        """
        Calculate the expectation value of the operator
        """
        return qt.expect(op, self.state)
    
    @abstractmethod
    def accept(self, **kwargs):
        """
        Accept the evolution data from the edge and overwrite the current state
        (if exists). It's useful for a node in a tree structure.
        """
        pass

    @abstractmethod
    def join(self, **kwargs):
        """
        Accpet the evolution data from the edges and add them to the current
        state (if exists)
        """
        pass
    
    def effective_logical_process(self):
        """
        The effective logical process of the node in the computational basis.
        """
        pass
    
    def fidelity_by_process(self):
        """
        The fidelity of the process since the initial node to the current node
        by comparing the effective logical process with the ideal logical process
        (identity superoperator)
        """
        pass

    def error_rate_by_process(self):
        """
        The error rate of the process since the initial node to the current node
        by comparing the effective logical process with the ideal logical process
        (identity superoperator)
        """
        pass

    
class StateNode(NodeBase):
    """
    State node that keep track of the ideal states and the measurement record
    """
    # options:
    ORTHOGONALIZE_LOGICAL_STATES = True
    ORTHOGONALIZE_METHOD: Literal["GS", "symm"] = "GS"

    # measurement record
    meas_record: MeasurementRecord

    # probability amplitude of |0> and |1>
    _prob_amp_01: Tuple[float, float]

    # ideal states, organized in an ndarray, with dimension n*3
    # the first dimension counts the number of correctable errors
    # the second dimension enumerates: logical state 0 and logical state 1
    _raw_ideal_logical_states: np.ndarray[qt.Qobj]
    _effective_logical_process: np.ndarray[qt.Qobj] | None  # store for efficiency, always a choi matrix representation

    # mark that the node will not be further evolved and reduce the compu
    # time. It does not mean that the node is a final state in the diagram,
    # but a state that will stay in the ensemble forever.
    terminated: bool = False

    # fidelity warning issued when the fidelity is larger than 0 in a 
    # terminated branch
    term_fid_warning_issued = False
    
    # trajectory probability to get to this node's allowed computational subspaces. 
    # It's obtained from the product of the trace of choi matrix of each process 
    # leading to the node.
    # The sum of the array should be first-order close to the probability 
    # of the node.
    traj_prob: np.ndarray[float] | None = None
    
    # the accumulated logical process.
    # it's the product of the process matrix on each edge leading to the node's
    # computational subspaces. 
    _accum_logical_process: np.ndarray[qt.Qobj] | None = None

    def accept(
        self, 
        meas_record: MeasurementRecord,
        state: qt.Qobj,
        prob_amp_01: Tuple[float, float],
        raw_ideal_logical_states: np.ndarray[qt.Qobj],
        process: qt.Qobj,
        init_encoders: np.ndarray[qt.Qobj],
        **kwargs,
    ):
        """
        Accept the evolution data from the edge and overwrite the current state. 
        It's useful for a node in a tree structure.

        For StateNode, it takes the following arguments:
        - meas_record: the measurement record
        - state: the state after the evolution
        - prob_amp_01: the probability amplitude of |0> and |1>
        - raw_ideal_logical_states: N*2 array of the ideal logical states (before orthogonalization)
        - process: the process that evolves the initial node to the current node
        - init_encoders: the initial encoders that encode the logical states to the
        initial physical states
        """
        # basic type and validity checks:
        for ideal_state in raw_ideal_logical_states.ravel():
            assert ideal_state.type == "ket"
            assert np.allclose(normalization_factor(ideal_state), 1)
        assert np.allclose(np.sum(np.abs(prob_amp_01)**2), 1)
        assert process.type == "super" and process.superrep == "super"
        
        # shape consistency:
        subsys_dims = state.dims[0]
        assert process.dims == [[subsys_dims] * 2] * 2
        for ideal_state in raw_ideal_logical_states.ravel():
            assert ideal_state.dims[0] == subsys_dims

        self.meas_record = meas_record
        self.state = state
        self._prob_amp_01 = prob_amp_01
        self._raw_ideal_logical_states = raw_ideal_logical_states
        self.process = process
        self.init_encoders = init_encoders
        
        # reset and wait for calculation
        self._effective_logical_process = None      

    @property
    def prob_amp_01(self) -> Tuple[float, float]:
        return self._prob_amp_01
    
    @prob_amp_01.setter
    def prob_amp_01(self, prob_amp_01: Tuple[float, float]):
        """
        Reset the probability amplitude of |0> and |1> and automatically
        set a new state
        """
        self._prob_amp_01 = prob_amp_01 / np.sqrt(np.sum(np.abs(prob_amp_01)**2))

        if self.terminated:
            warn("The probability amplitude of |0> and |1> is reset manually. "
                 "Usually it's not allowed for a terminated node. \n")
            return
        elif self._raw_ideal_logical_states.shape[0] > 1:
            warn("The probability amplitude of |0> and |1> is reset manually. "
                 "While the state is not reset as the ideal logical states are "
                 "not unique. \n")
            return
        elif self._raw_ideal_logical_states.shape[0] == 1:
            warn("The probability amplitude of |0> and |1> and the state are "
                 "reset manually. \n")
            if self.ORTHOGONALIZE_LOGICAL_STATES:
                logical_states = self._orthogonalize(self._raw_ideal_logical_states)
            else:
                logical_states = self._raw_ideal_logical_states

            self.state = qt.ket2dm(
                self._prob_amp_01[0] * logical_states[0, 0] 
                + self._prob_amp_01[1] * logical_states[0, 1]
            ).unit()
        else:
            warn("The probability amplitude of |0> and |1> is reset manually, "
                 "but the situation is not expected. \n")
            return 

    def join(self, **kwargs):
        raise ValueError("StateNode does not support join method.")
    
    def add_out_edges(self, edge):
        if self.terminated:
            raise ValueError("The node is terminated and cannot have out edges.")
        
        super().add_out_edges(edge)

    @staticmethod
    def _GS_orthogonalize(state_0, state_1):
        """
        Gram-Schmidt orthogonalization. 
        """
        state_0 = state_0.unit()
        state_1 = state_1.unit()
        
        new_state_0 = state_0
        new_state_1 = (
            state_1 - state_1.overlap(new_state_0) * new_state_0
        ).unit()
        
        return new_state_0, new_state_1

    @staticmethod
    def _symmtrized_orthogonalize(state_0, state_1):
        """
        A little bit more generalized version of Gram-Schmidt orthogonalization?
        Don't know whether there is a reference.
        """
        state_0 = state_0.unit()
        state_1 = state_1.unit()
        
        overlap = (state_0.overlap(state_1))
        theta = - np.angle(overlap)   # to make the ovrlap real
        state_1_w_phase = state_1 * np.exp(1j * theta)

        x = 2 * (state_0.overlap(state_1_w_phase)).real
        sq2mx = np.sqrt(2 - x)
        sq2px = np.sqrt(2 + x)
        sq8mx2 = np.sqrt(8 - 2 * x**2)
        p = (sq2mx + sq2px) / sq8mx2
        q = (sq2mx - sq2px) / sq8mx2

        new_state_0 = p * state_0 + q * state_1_w_phase
        new_state_1 = p * state_1_w_phase + q * state_0

        return new_state_0, new_state_1 * np.exp(-1j * theta)
    
    @staticmethod
    def _orthogonalize(
        state_arr: np.ndarray[qt.Qobj],
    ) -> np.ndarray[qt.Qobj]:
        """
        Orthogonalize the states in the N*2 array, return a N*2 array
        """
        if StateNode.ORTHOGONALIZE_METHOD == "GS":
            func = StateNode._GS_orthogonalize
        elif StateNode.ORTHOGONALIZE_METHOD == "symm":
            func = StateNode._symmtrized_orthogonalize
        else:
            raise ValueError(
                "The orthogonalization method should be either "
                "'GS' or 'symm'."
            )

        new_state_arr = np.empty_like(state_arr)
        for i in range(len(state_arr)):
            (
                new_state_arr[i, 0], new_state_arr[i, 1]
            ) = func(
                *state_arr[i]
            )
        
        return new_state_arr
    
    def fidelity_drop_by_orth(self):
        """
        by orthorgonalize (redefine) the logical states, the fidelity will drop. 
        This method returns the the amount of such drop.

        Note: it only work if there is only one pair of ideal logical states
        """
        if self.terminated:
            return 0

        ideal_state_wo_orth = self._ideal_states(orthogonalize=False)
        ideal_state_w_orth = self._ideal_states(orthogonalize=True)

        if not len(ideal_state_wo_orth) == 1:
            raise ValueError("This method only works if there is only one pair "
                            "of ideal logical states.")
        
        fid = 1 - np.abs(ideal_state_w_orth[0].overlap(ideal_state_wo_orth[0]))**2
        fid *= self.probability     # normalize by the probability
        
        return fid
    
    @staticmethod
    def _qobj_unit(qobj: qt.Qobj) -> qt.Qobj:
        """
        used for vectorization of qobj.unit()
        """
        return qobj.unit()
    
    def _ideal_states(
        self,
        orthogonalize: bool,
    ) -> np.ndarray[qt.Qobj]:
        """
        Return the ideal state by logical states
        """
        if len(self._raw_ideal_logical_states) == 0:
            # the states' norm is too small and thrown away
            dim = self.state.dims[0]
            return np.array([qt.Qobj(
                np.zeros(self.state.shape), 
                dims=[dim, np.ones_like(dim).astype(int).tolist()]
            )], dtype=qt.Qobj)
        
        # need to be modified as the logical states are not necessarily
        # orthogonal
        if orthogonalize:
            othogonalized_states = self._orthogonalize(self._raw_ideal_logical_states)
            return (
                self._prob_amp_01[0] * othogonalized_states[:, 0]
                + self._prob_amp_01[1] * othogonalized_states[:, 1]
            )
        else:
            qobj_array_unit = np.vectorize(
                self._qobj_unit, otypes = [qt.Qobj]
            )   # apply qobj.unit() to each element in the array
            return qobj_array_unit(
                self._prob_amp_01[0] * self._raw_ideal_logical_states[:, 0] 
                + self._prob_amp_01[1] * self._raw_ideal_logical_states[:, 1]
            )

    @property
    def ideal_states(
        self,
    ) -> np.ndarray[qt.Qobj]:
        """
        Return the ideal state by logical states
        """
        return self._ideal_states(self.ORTHOGONALIZE_LOGICAL_STATES)
    
    @property
    def ideal_logical_states(self) -> np.ndarray[qt.Qobj]:
        if self.ORTHOGONALIZE_LOGICAL_STATES:
            return self._orthogonalize(self._raw_ideal_logical_states)
        else:
            return self._raw_ideal_logical_states
    
    @property
    def ideal_projector(self) -> qt.Qobj:
        return projector_w_basis(self.ideal_states)

    @property
    def fidelity(self) -> float:
        fid = ((self.state * self.ideal_projector).tr()).real

        if not self.term_fid_warning_issued:
            # term_fid_warning_issued is to avoid infinite fidelity calculation
            # as print out self requires fidelity calculation as well
            if self.terminated and fid > 1e-10:
                self.term_fid_warning_issued = True
                warn(f"Terminated branch [{self}] has a total fidelity larger than 1e-10.\n")

        return fid
    
    @property
    def probability(self) -> float:
        return (self.state.tr()).real

    def deepcopy(self) -> "StateNode":
        """
        1. Not storing the edge information and avoiding circular reference
        2. deepcopy the Qobj
        """

        copied_node = StateNode()
        copied_node.meas_record = deepcopy(self.meas_record)
        copied_node.state = deepcopy(self.state)
        copied_node._raw_ideal_logical_states = deepcopy(self._raw_ideal_logical_states)

        return copied_node
    
    @classmethod
    def initial_note(
        cls, 
        init_prob_amp_01: Tuple[float, float],
        logical_0: qt.Qobj,
        logical_1: qt.Qobj,
    ) -> "StateNode":
        # put the logical states in an array, as the other part of the code
        # only accepts ndarray
        logical_state_arr = np.empty((1, 2), dtype=object)
        logical_state_arr[:] = [[logical_0, logical_1]]

        # need to be modified as the logical states are not necessarily
        # orthogonal
        if cls.ORTHOGONALIZE_LOGICAL_STATES:
            logical_state_arr = cls._orthogonalize(logical_state_arr)
        state = (
            init_prob_amp_01[0] * logical_state_arr[0, 0] 
            + init_prob_amp_01[1] * logical_state_arr[0, 1]
        ).unit()
        state_dm = qt.ket2dm(state)
        
        # initial process is identity
        eye_op = qt.qeye_like(state_dm)
        eye_super = qt.sprepost(eye_op, eye_op)
        
        init_node = cls()
        init_node.accept(
            meas_record = [], 
            state = state_dm,
            prob_amp_01 = init_prob_amp_01,
            raw_ideal_logical_states = logical_state_arr,
            process = eye_super,
            init_encoders = np.array([]),  # add in the next line
        )
        init_node.init_encoders = init_node.ideal_encoders()

        return init_node

    def to_nx(self) -> Tuple[int, Dict[str, Any]]:
        """
        Convert to a networkx node
        """
        try:
            fidelity = self.fidelity
            probability = self.probability
        except AttributeError:
            fidelity = np.nan
            probability = np.nan

        return (
            self.index,
            {
                "state": self,
                "fidelity": fidelity,
                "probability": probability,
            }
        )

    def clear_evolution_data(self):
        try:
            del self.state
            del self._raw_ideal_logical_states
            del self.fidelity
            del self.meas_record
        except AttributeError:
            pass

    def __str__(self) -> str:
        try:
            idx = self.index
        except AttributeError:
            idx = "No Index"
            
        try:
            fail = ", Terminated" if self.terminated else ""
            return (
                f"StateNode ({idx}){fail}, record {self.meas_record}, "
                + f"prob {self.probability:.3f}, fid {self.fidelity:.3f}"
            )
        except AttributeError:
            return f"StateNode ({idx})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def bloch_vector(self) -> np.ndarray:
        """
        Calculate the bloch vector of the state
        """
        if self.terminated:
            return np.zeros(4)

        if self._raw_ideal_logical_states.shape[0] > 1:
            warn("The ideal logical states are not unique. Returned nan.\n")
            return np.nan * np.ones(4)
        
        trans = evecs_2_transformation(self.ideal_logical_states[0])

        X = trans * qt.sigmax() * trans.dag()
        Y = trans * qt.sigmay() * trans.dag()
        Z = trans * qt.sigmaz() * trans.dag()
        I = trans * qt.qeye(2) * trans.dag()
        op_list: List[qt.Qobj] = [X, Y, Z, I]

        dims = self.state.dims[0]
        for op in op_list:
            op.dims = [dims, dims]

        return np.array([
            self.expect(op) for op in op_list
        ])
        
    def ideal_encoders(self) -> np.ndarray[qt.Qobj]:
        """
        The ideal decoder for the state - map the state to a two-level system.
        
        Note that the state may have multiple logical subspaces, so there
        are multiple possible decoders.
        """
        ideal_logical_states = self.ideal_logical_states
        len_subspace = ideal_logical_states.shape[0]
        encoders = np.ndarray(len_subspace, dtype=qt.Qobj)
        
        # a list like [1, 1, ..., 1], to make the dimensions consistent
        subsys_ones = ideal_logical_states[0, 0].dims[1]
        logical_0 = qt.basis(2, 0)
        logical_0.dims = [[2], subsys_ones]
        logical_1 = qt.basis(2, 1)
        logical_1.dims = [[2], subsys_ones]
        
        for idx in range(len_subspace):
            encoders[idx] = (
                ideal_logical_states[idx, 0] * logical_0.dag()
                + ideal_logical_states[idx, 1] * logical_1.dag()
            )
            
        return encoders
    
    def effective_logical_process(
        self,
        repr: Literal["super", "choi", "chi", "orth_chi"] = "super",
    ) -> np.ndarray[qt.Qobj]:
        """
        Calculate the effective logical process since the initial state
        to the current state.
        
        If there is already a cached result, return it.
        
        It is always an N*1 array (col vector) of Qobj.
        """
        if self.terminated:
            raise TerminationError("The node is terminated. No effective logical process.")
        
        # if there is no cached result, calculate it
        # the cache is reset when `accept` is called
        if self._effective_logical_process is None:
            self._effective_logical_process = effective_logical_process(
                process = self.process, 
                init_encoders = self.init_encoders, 
                final_encoders = self.ideal_encoders(), 
                repr = "choi",
            )
            
        if repr == "super":
            return to_super_vec(self._effective_logical_process)
        elif repr == "choi":
            return self._effective_logical_process
        elif repr == "chi":
            return to_chi_vec(self._effective_logical_process)
        elif repr == "orth_chi":
            return to_orth_chi_vec(self._effective_logical_process)
        else:
            raise ValueError(f"The representation {repr} is not supported.")

    def fidelity_by_process(
        self, 
        type: Literal["avg", "etg"] = "etg",
        approx_proc: bool = False,
    ) -> np.ndarray[float]:
        """
        The fidelity of the process since the initial node to the current node.
        
        Type:
            "avg": average fidelity
            "etg": enranglement fidelity
            
        approx_proc: bool = False
            whether to use the effective process from the full numerics
            or the multiplication of individual truncated processes
        """
        if self.terminated:
            raise TerminationError("The node is terminated. No effective logical process.")
        
        if not approx_proc:
            realized_process = self.effective_logical_process(repr="super")
        else:
            realized_process = self.accum_logical_process(repr="super")
        
        fidelity = np.zeros(realized_process.shape)
        for idx, proc in np.ndenumerate(realized_process):
            process_fidelity = qt.process_fidelity(proc, qt.qeye_like(proc))
            
            if type == "avg":
                raise NotImplementedError(
                    "Average fidelity is not implemented. As the conversion "
                    "from process fidelity isn't clear when the process isn't "
                    "CPTP."
                )
                # wrong when the process is not TP
                # fidelity[idx] = proc_fid_2_ave_fid(process_fidelity, 2)
            elif type == "etg":
                fidelity[idx] = process_fidelity
            else:
                raise ValueError("The type of fidelity should be either 'avg' or 'etg'.")

        return fidelity
        
    def process_dnorm(
        self,
        approx_proc: bool = False,
    ) -> np.ndarray[float]:
        """
        The diamond norm of the processes since the initial node to the current node.
        
        For each process, we compare it with a scaled identity process
        (compare_process = I_prob * internal_ratio * identity),
        where the scaling factor can be determined by the trace of 
        the overall choi matrix. 
        """        
        if self.terminated:
            raise TerminationError("The node is terminated. No effective logical process.")
        
        if not approx_proc:
            processes = self.effective_logical_process(repr = "choi")
        else:
            processes = self.accum_logical_process(repr = "choi")
            
        dnorms = np.zeros(processes.shape)
        for idx, process in np.ndenumerate(processes):
            compare_process = target_process_for_dnorm(process)
            dnorms[idx] = process.dnorm(compare_process)
            
        return dnorms
    
    def process_choi_trace(
        self,
        approx_proc: bool = False,
    ) -> float:
        """
        The trace of the choi matrix of the effective logical processes
        """
        if self.terminated:
            raise TerminationError("The node is terminated. No effective logical process.")

        if not approx_proc:
            processes = self.effective_logical_process(repr = "choi")
        else:
            processes = self.accum_logical_process(repr = "choi")
            
        traces = np.zeros(processes.shape)
        for idx, process in np.ndenumerate(processes):
            traces[idx] = process.tr()
            
        return traces
    
    def outgoing_dnorm(self) -> np.ndarray[float]:
        """
        The dnorm of the outgoing edges, for each allowed computational subspace,
        conditioned on the node & subspace is reached.
        """
        if len(self.out_edges) == 0:
            raise ValueError("The node has no outgoing edges.")
        
        dnorms = np.zeros(self._raw_ideal_logical_states.shape[0])
        for edge in self.out_edges:
            edge_dnorm = edge.process_dnorm()
            dnorms += np.sum(edge_dnorm, axis=0)        # sum over destination subspaces
            
        return dnorms   
    
    def outgoing_infid(
        self,
        type: Literal["avg", "etg"] = "etg",
    ) -> np.ndarray[float]:
        """
        The infidelity of the outgoing edges, for each allowed computational subspace,
        conditioned on the node & subspace is reached.
        """
        if len(self.out_edges) == 0:
            raise ValueError("The node has no outgoing edges.")
        
        fid = np.zeros(self._raw_ideal_logical_states.shape[0])
        for edge in self.out_edges:
            edge_infid = edge.fidelity_by_process(type)
            fid += np.sum(edge_infid, axis=0)   # sum over destination subspaces
            
        # for each of the initial subspaces, the fidelity should sum up to 1
        # if not, there is infidelity
        return 1 - fid
    
    def outgoing_leakage_prob(self) -> np.ndarray[float]:
        """
        The probability of the leakage process of the outgoing edges, for each
        allowed computational subspace, conditioned on the node & subspace is reached.
        """
        if len(self.out_edges) == 0:
            raise ValueError("The node has no outgoing edges.")
        
        total_trace = np.zeros(self._raw_ideal_logical_states.shape[0])
        for edge in self.out_edges:
            total_trace += np.sum(edge.process_choi_trace(), axis=0)
            
        # the total trace should be 2, if not, there is leakage
        return 1 - total_trace / 2
    
    def accum_logical_process(
        self,
        repr: Literal["super", "choi", "chi", "orth_chi"] = "super",
    ) -> np.ndarray[qt.Qobj]:
        """
        The accumulated logical process since the initial node to the current node.
        """
        if repr == "super":
            return self._accum_logical_process
        elif repr == "choi":
            return to_choi_vec(self._accum_logical_process)
        elif repr == "chi":
            return to_chi_vec(self._accum_logical_process)
        elif repr == "orth_chi":
            return to_orth_chi_vec(self._accum_logical_process)
        else:
            raise ValueError(f"The representation {repr} is not supported.")
        
    def leakage_process(
        self,
        truncate_init_space: bool = True,
        reorder_final_space: bool = True,
    ) -> np.ndarray[qt.Qobj]:
        """
        The leakage process since the initial node to the current node.
        
        The leakage process is defined as:
        S_ia = Ps_fL * process * Ps_ii
        where Ps_fL is the projector onto the final coherent leakage subspace, 
        and Ps_ii is the projector onto the initial subspaces.
        
        If truncate_init_space is False, the initial space is not truncated,
        even if the rank of the initial space is smaller than the dimension.
        
        If reorder_final_space is True, the final state is in the basis of the
        final logical states + orthonormalized leakage states.
        """
        if self.terminated:
            raise TerminationError("The node is terminated. No leakage process.")
        
        # if there is no cached result, calculate it
        # the cache is reset when `accept` is called
        return leakage_process(
            process = self.process, 
            init_encoders = self.init_encoders, 
            final_encoders = self.ideal_encoders(), 
            truncate_init_space = truncate_init_space,
            reorder_final_space = reorder_final_space,
        )
            
    def leakage_fidelity_bound(
        self,
        return_full_result: bool = False,
        final_space_trunc_threshold: float | None = None,
        constr_proc2_fid: float | None = None,
        constr_proc2_seepage: float | None = None,
    ) -> float | Dict[str, float]:
        """
        Estimate the leakage processes' effect by bounding the potential 
        fidelity increase or decrease.
        
        Parameters
        ----------
        return_full_result: bool = False
            Whether to return the full result, including the optimal fidelity, 
            the process that maximizes/minimizes the fidelity, and the total process.   
        final_space_trunc_threshold: float | None = None
            We will truncate the final space if the maximum element of the leakage
            process is smaller than this threshold.
        constr_proc2_fid: float | None = None
            The bound of the fidelity of the leakage process.
        constr_proc2_seepage: float | None = None
            The bound of the seepage rate of the leakage process.
        """
        # get the leakage process. We must reorder the final space
        # as the truncation, constraints etc. rely on the computational
        # subspace being the first part of the final space.
        processes = self.leakage_process(
            truncate_init_space = True,
            reorder_final_space = True, 
        )
        
        # dimension of the leakage space
        logical_dim = 2
        full_dim = int(np.sqrt(processes.shape[0]))
        comp_dim = self.ideal_logical_states.size
        leakage_dim = full_dim - comp_dim
        
        # data holder
        bound = np.zeros_like(processes)
        if return_full_result:
            process_2 = np.zeros_like(processes, dtype=qt.Qobj)
            process_total = np.zeros_like(processes, dtype=qt.Qobj)
        
        for idx, proc in np.ndenumerate(processes):
            if final_space_trunc_threshold is not None:
                kept_levels, proc = truncate_leakage_process(
                    proc,
                    logical_dim = logical_dim,
                    threshold = final_space_trunc_threshold,
                )
                
                # update the leakage dimension
                # note: during the truncation, the other computational subspaces
                # are truncated out due to no matrix elements.
                # so we only need to subtract one logical dimension
                leakage_dim = len(kept_levels) - logical_dim
            
            result = bound_fidelity(
                process_1 = proc,
                logical_dim = logical_dim,
                leakage_space_dim = leakage_dim,
                maximize = True,
                constr_proc2_fid = constr_proc2_fid,
                constr_proc2_seepage = constr_proc2_seepage,
            )
            bound[idx] = result["fidelity"]
            if return_full_result:
                process_2[idx] = result["process_2"]
                process_total[idx] = result["process_total"]
            
        if return_full_result:
            return {
                "fidelity": bound,
                "process_2": process_2,
                "process_total": process_total,
            }
        else:
            return np.sum(bound)
    

Node = StateNode    # for now, the only node type is StateNode


class StateEnsemble:

    def __init__(
        self, 
        nodes: List[StateNode] | None = None,
        # note: Do not use [] as the default value, it will be shared by 
        # all the instances, as it's a mutable object
    ):
        if nodes is None:
            nodes = []
        self.nodes: List[StateNode] = nodes

    @property
    def no_further_evolution(self) -> bool:
        """
        Determine if the ensemble is a final state in the diagram, namely
        no node has out edges.
        """
        no_further_evolution = True
        for node in self.active_nodes():
            if node.out_edges != []:
                no_further_evolution = False
                break

        return no_further_evolution

    def append(self, node: StateNode):
        if node in self:
            raise ValueError("The node is already in the ensemble.")
        self.nodes.append(node)

    def is_trace_1(self) -> bool:
        """
        Check if the total trace is 1
        """
        return np.abs(self.probability - 1) < 1e-6
    
    def fidelity_drop_by_orth(self) -> float:
        """
        Calculate the fidelity drop by orthogonalization
        """
        return sum([node.fidelity_drop_by_orth() for node in self.nodes])
    
    @property
    def probability(self) -> float:
        """
        Calculate the total probability
        """
        for node in self.nodes:
            try: 
                node.state
            except AttributeError:
                raise RuntimeError("A node has not been evolved.")
            
        return sum([node.probability for node in self.nodes])
    
    @property
    def state(self) -> qt.Qobj:
        """
        Calculate the total state
        """
        for node in self.nodes:
            try:
                node.state
            except AttributeError:
                raise AttributeError(f"A node {node} has not been evolved.")

        if not self.is_trace_1():
            warn("The total trace is not 1. The averaged state is not "
                 "physical. \n")
        return sum([node.state for node in self.nodes])

    @property
    def fidelity(self) -> float:
        """
        Calculate the total fidelity
        """
        return sum([node.fidelity for node in self.nodes])

    def deepcopy(self):
        """
        1. Not storing the edge information
        2. deepcopy the Qobj
        """ 
        return [
            node.deepcopy() for node in self.nodes
        ]
    
    def __iter__(self):
        return iter(self.nodes)
    
    def __getitem__(self, index) -> StateNode:
        return self.nodes[index]
    
    def __len__(self):
        return len(self.nodes)
    
    def order_by_fidelity(self) -> List[StateNode]:
        """
        Return the nodes ordered by fidelity
        """
        return sorted(self.nodes, key=lambda node: node.fidelity, reverse=True)
    
    def order_by_probability(self) -> List[StateNode]:
        """
        Return the nodes ordered by probability
        """
        return sorted(self.nodes, key=lambda node: node.probability, reverse=True)
    
    def expect(self, op: qt.Qobj) -> float:
        """
        Calculate the expectation value of the operator
        """
        return sum([node.expect(op) for node in self.nodes])
    
    def next_step_name(self) -> str:
        """
        Usually, the edges that the state nodes are connected to are named
        similarly as operations are applied to the whole ensemble.
        """
        if self.nodes == []:
            return "[NO NEXT STEP DUE TO EMPTY ENSEMBLE]"
        
        if self.nodes[0].out_edges == []:
            return "[NO NEXT STEP DUE TO NO OUT EDGES]"

        return self.nodes[0].out_edges[0].name
    
    def __str__(self) -> str:
        try:
            return (
                f"StateEnsemble before {self.next_step_name()}, "
                + f"prob {self.state.tr().real:.3f}, fid {self.fidelity:.3f}"
            )
        except AttributeError:
            return f"StateEnsemble before {self.next_step_name()}"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def active_nodes(self) -> "StateEnsemble":
        """
        Return the nodes that are not terminated
        """
        return StateEnsemble([
            node for node in self.nodes if not node.terminated
        ])
    
    def terminated_nodes(self) -> "StateEnsemble":
        """
        Return the nodes that are terminated
        """
        return StateEnsemble([
            node for node in self.nodes if node.terminated
        ])
    
    def bloch_vectors(self) -> np.ndarray:
        """
        Calculate the bloch vectors of the states
        """
        return np.sum([
            node.bloch_vector() for node in self.nodes
        ], axis=0)
        
    def process(self,) -> qt.Qobj:
        """
        Calculate the ensemble averaged process starting from the initial state
        to the ensemble.
        """
        return sum([node.process for node in self.nodes])

    def effective_logical_process(
        self,
        repr: Literal["super", "choi", "chi", "orth_chi"] = "super",
        force_sum: bool = False,
    ) -> qt.Qobj:
        """
        Calculate the effective logical process since the initial state
        to the current state.
        
        If force_sum is True and the effective process is not unique, we will
        sum over all the decoded processes, as if they are all decoded right 
        after the step.
        """
        processes = []
        for node in self.active_nodes():
            proc = node.effective_logical_process(repr)

            if proc.shape != (1, 1) and not force_sum:
                raise ValueError(
                    "To get the effective logical process for an ensemble, "
                    "the effective logical process for each node should be "
                    "unique / well defined, or in other words, the ideal "
                    "final decoders should be unique."
                )
            else:
                processes.append(np.sum(proc))

        return sum(processes)
    
    def accum_logical_process(
        self,
        repr: Literal["super", "choi", "chi", "orth_chi"] = "super",
        force_sum: bool = False,
    ) -> qt.Qobj:
        """
        Calculate the accumulated logical process since the initial state
        to the current state.
        
        If force_sum is True and the accumulated process is not unique, we will
        sum over all the accumulated processes, as if they are all accumulated 
        right after the step.
        """
        processes = []
        for node in self.active_nodes():
            proc = node.accum_logical_process(repr)

            if proc.shape != (1, 1) and not force_sum:
                raise ValueError(
                    "To get the accumulated logical process for an ensemble, "
                    "the accumulated logical process for each node should be "
                    "unique / well defined, or in other words, the ideal "
                    "final decoders should be unique."
                )
            else:
                processes.append(np.sum(proc))

        return sum(processes)

    def fidelity_by_process(
        self,
        type: Literal["avg", "etg"] = "etg",
        approx_proc: bool = False,
        force_sum: bool = False,
    ) -> float:
        """
        The fidelity of the process since the initial node to the current node.
        
        Type:
            "avg": average fidelity
            "etg": enranglement fidelity
        """
        fids = []
        for node in self.active_nodes():
            fid = node.fidelity_by_process(
                type = type,
                approx_proc = approx_proc,
            )

            if fid.shape != (1, 1) and not force_sum:
                raise ValueError(
                    "To get the fidelity by process for an ensemble, the fidelity "
                    "by process for each node should be unique / well defined."
                )
            else:
                fids.append(np.sum(fid))
            
        ensemble_fid = sum(fids)
        
        # double check, the fidelity calculation is linear w.r.t. the actual process
        if not approx_proc:
            eff_proc = self.effective_logical_process("super", force_sum)
            fid_compare = qt.process_fidelity(eff_proc, qt.qeye_like(eff_proc))
            assert np.abs(fid_compare - ensemble_fid) < 1e-6, "The fidelity by process should be equal to the process fidelity."
        
        return ensemble_fid
    
    def process_dnorm(
        self,
        force_sum: bool = False,
        approx_proc: bool = False,
    ) -> float:
        """
        The diamond norm of the processes since the initial node to the 
        current ensemble.
        
        dnorm(sum_i E_i - I), where E_i is the process of the i-th node.
        """
        if not approx_proc:
            proc = self.effective_logical_process(
                repr="super", 
                force_sum=force_sum,
            )
        else:
            proc = self.accum_logical_process(
                repr="super", 
                force_sum=force_sum,
            )
        return (proc - qt.qeye_like(proc)).dnorm()
    
    def process_choi_trace(
        self,
        force_sum: bool = False,
        approx_proc: bool = False,
    ) -> float:
        """
        The trace of the choi matrix of the effective logical processes
        from the initial node to the current ensemble.
        """
        traces = []
        for node in self.active_nodes():
            tr = node.process_choi_trace(
                approx_proc=approx_proc,
            )

            if tr.shape != (1, 1) and not force_sum:
                raise ValueError(
                    "To get the process choi trace for an ensemble, the process "
                    "choi trace for each node should be unique / well defined."
                )
            else:
                traces.append(np.sum(tr))
                
        return sum(traces)
    
    def process_dnorm_by_sum(
        self,
        force_sum: bool = False,
        approx_proc: bool = False,
    ) -> float:
        """
        The diamond norm of the processes since the initial node to the 
        current ensemble.
        
        Say the process is sum_i E_i, where E_i is the process of the i-th node.
        Then the diamond norm is calculated as   
        dnorm(sum_i E_i - I) 
        = dnorm(sum_i (E_i - p_i*I) + p_leak*I)
        >= sum_i dnorm(E_i - p_i*I) + p_leak*dnorm(I)   (triangle inequality)
        
        where p_i is the probability (trace of the choi matrix) of the i-th node,
        and p_leak is the probability of the leakage process.
        """
        dnorms = []
        for node in self.active_nodes():
            dn = node.process_dnorm(
                approx_proc=approx_proc,
            )

            if dn.shape != (1, 1) and not force_sum:
                raise ValueError(
                    "To get the process dnorm for an ensemble, the process "
                    "dnorm for each node should be unique / well defined."
                )
            else:
                dnorms.append(np.sum(dn))
        
        # tackel the leakage
        p_leak = 1 - self.process_choi_trace(force_sum) / 2
        
        return sum(dnorms) + p_leak

    def outgoing_dnorm(self) -> np.ndarray[float]:
        """
        For each node, we have calculated the dnorm of the outgoing edges, for 
        each allowed computational subspace, conditioned on the node & subspace 
        is reached.
        
        Here we sum over all the active nodes and subspaces with the corresponding
        weights.
        
        We also sum over the leakage probability of each node, as it is missing 
        in the calculation of the dnorm of the process.
        """
        return np.sum([
            node.outgoing_dnorm() * node.traj_prob + node.outgoing_leakage_prob()
            for node in self.active_nodes()
        ])

    def outgoing_infid(self) -> np.ndarray[float]:
        """
        The trace of the choi matrix of the outgoing edges, for each allowed 
        computational subspace, conditioned on the node & subspace is reached.
        """
        return np.sum([
            node.outgoing_infid() * node.traj_prob
            for node in self.active_nodes()
        ])
    