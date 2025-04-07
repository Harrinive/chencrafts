import qutip as qt
import numpy as np

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False 

from chencrafts.cqed.qt_helper import (
    evecs_2_transformation,
    complete_basis_set,
    evecs_2_transformation,
)
from chencrafts.cqed.proc_repr import (
    pauli_basis, to_orth_chi, orth_chi_to_choi
)

from typing import List, Tuple, Dict, Literal
from . import settings

to_choi_vec = np.vectorize(qt.to_choi)
to_chi_vec = np.vectorize(qt.to_chi)
to_orth_chi_vec = np.vectorize(to_orth_chi)
to_super_vec = np.vectorize(qt.to_super)
orth_chi_to_choi_vec = np.vectorize(orth_chi_to_choi)

_subspace_types = Literal["L", "p"] | int

def _abstract_error_op(
    logical_dim: int, num_errors: int, which_error: int
) -> qt.Qobj:
    """
    Map a logical state to an error state.
    Error subspace is num_errors copies of the logical space, by specifying 
    which_error, we generate such "identity" error operator.
    """
    logical_basis = [
        qt.basis(logical_dim, idx) for idx in range(logical_dim)
    ]
    error_basis = [
        qt.basis(num_errors * logical_dim, which_error * logical_dim + idx)
        for idx in range(logical_dim)
    ]
    
    logi_error_map = [
        error_state * logical_state.dag()
        for error_state, logical_state in zip(error_basis, logical_basis)
    ]
    
    return sum(logi_error_map)

def _constr_enc_by_subspace(
    encoders: np.ndarray[qt.Qobj] | List[qt.Qobj],
    subspace_type: _subspace_types,
    return_proj: bool = True,
    enc_from_error_space: bool = False,
) -> qt.Qobj:
    """
    A helper in process_block method. Pick the right encoder / projector
    by name or index.
    
    Parameters
    ----------
    encoders: np.ndarray[qt.Qobj] | List[qt.Qobj]
        The encoders of the subspaces.
    subspace_type: Literal["L", "p"] | int
        - int: the index of the correctable subspaces (subspace related to encoders[a])
        - "L": the total correctable subspace (sum of all correctable subspaces)
        - "p": the leakage subspace
    return_proj: bool
        Whether to return the projector instead of the encoder. 
        IMPORTANT: 
        - For "p" subspace, we don't have an encoder. Setting it to False 
        will still return the projector (just to be careful) without raising an error.
    enc_from_error_space: bool
        Only valid when subspace_type is an int. If True, the encoder takes 
        a state in error space (multiple copies of the logical space) to 
        the corresponding physical space.
        
    Returns
    -------
    enc: qt.Qobj
        The encoder or projector. 
        If return_proj is True, its dimensionality is the same as the physical space. 
        If return_proj is False, its left-dimension is the same as the physical space, 
        and its right-dimension (rd) is related to the subspace_type.
        - If subspace_type is int, rd = dim of the logical space
        - If subspace_type is "L", rd = dim of the error space
        - If subspace_type is "p", it does not support return_proj = False
    """
    if isinstance(subspace_type, int):
        enc = encoders[subspace_type]
        
        if enc_from_error_space:
            len_encs = len(encoders)
            dim_logical = encoders[0].dims[1][0]
            error_to_logi_op = _abstract_error_op(
                dim_logical, len_encs, subspace_type
            ).dag()
            enc = enc * error_to_logi_op
            
        if return_proj:
            return enc * enc.dag()
        else:
            return enc
        
    elif subspace_type == "L":
        if not enc_from_error_space:
            raise ValueError("enc_from_error_space should be True for 'L' subspace.")
        
        if return_proj:
            # physical space projector
            return sum([enc * enc.dag() for enc in encoders])
        else:
            # # method 1: stack the encoders, i.e., the encoded logical states
            # # are different in different subspaces
            # enc_dim = encoders[0].dims[0]       # a list of integers
            # logi_dim = encoders[0].dims[1][0]   # an integer
            # len_enc = len(encoders)

            # stacked_enc = np.concatenate([
            #     enc.full() for enc in encoders
            # ], axis=1)
            # stacked_enc = qt.Qobj(stacked_enc, dims=[enc_dim, [logi_dim * len_enc]])
            # return stacked_enc
            
            # method 2: map the logical states back to the corresponding error space
            len_errors = len(encoders)
            dim_logical = encoders[0].dims[1][0]
            new_encoders = [
                enc * _abstract_error_op(
                    dim_logical, len_errors, idx
                ).dag()
                for idx, enc in enumerate(encoders)
            ]
            return sum(new_encoders)
        
    elif subspace_type == "p":
        # leakage subspace does not care about the enc_from_error_space 
        # (not well-defined to say it's true or false)
        # it will always return the projector in the physical space
        if return_proj == False:
            raise ValueError("The leakage subspace does not support return_proj = False.")
        
        proj_L = sum([enc * enc.dag() for enc in encoders])
        return qt.qeye_like(proj_L) - proj_L
    
    else:
        raise ValueError("The subspace_type should be either 'L', 'p', or an int.")

def _constr_enc_superop_by_subspace(
    encoders: np.ndarray[qt.Qobj] | List[qt.Qobj],
    subspace_type: Tuple[_subspace_types, _subspace_types] | Literal["LC", "LD"], 
    return_type: Literal["enc", "proj", "dec"] = "enc",
    enc_from_error_space_if_possible: bool = False,
) -> qt.Qobj:
    """
    A helper in process_block method. Pick the right superoperator encoder 
    / projector / decoder by name or index.
    
    The subspace_type can be a tuple of two elements a and b,
    denoting the subspaces of the hilbert space.
    That defines a projector superoperator by
    P_ab (.) = P_a (.) P_b, if a = b
    P_ab (.) = P_a (.) P_b + h.c., if a != b
    where P_a and P_b are projector operators to the corresponding subspaces.
    The allowed subspace indices (a) are:
    - int: the index of the correctable subspaces (node.ideal_logical_states[a])
    - "L": the total correctable subspace (sum of all correctable subspaces)
    - "p": the leakage subspace
    
    The subspace_type can also be a single string
    - "LD", which stands for the sum_a P_a (.) P_a
    - "LC", which stands for the sum_{a!=b} P_a (.) P_b + h.c.
    here a and b run over all indices of the logical subspaces.
    
    If return_type is "enc", it maps the logical space (or error space, 
    the sum of all correctable subspaces, if enc_from_error_space_if_possible is True) 
    to the physical space.
    If return_type is "proj", its dimensionality is the same as the physical space.
    If return_type is "dec", it maps the physical space to the logical space 
    (or error space, if enc_from_error_space_if_possible is True).
    """            
    if isinstance(subspace_type, tuple):
        if subspace_type[0] != subspace_type[1]:
            # in this case, it is possible that qt.sprepost(enc_0, enc_1.dag()) 
            # and qt.sprepost(enc_1, enc_0.dag()) are the different in dimension
            
            if (
                isinstance(subspace_type[0], int) 
                or subspace_type[0] == "L"
            ) and (
                isinstance(subspace_type[1], int)
                or subspace_type[1] == "L"
            ):
                # both of them are from the correctable subspaces
                # they can't be further truncated to the individual logical subspaces
                if return_type != "proj" and not enc_from_error_space_if_possible:
                    raise ValueError("enc_from_error_space_if_possible must be True "
                                     "for different logical subspaces.")

            if subspace_type[0] == "p" or subspace_type[1] == "p":
                # at least one of them is from the leakage subspace
                if return_type != "proj":
                    raise ValueError("return_type must be 'proj' for leakage subspaces.")
                # the returned operator will be in the physical space, 
                # and we don't need to check the enc_from_error_space_if_possible
        
        enc_0 = _constr_enc_by_subspace(
            encoders, subspace_type[0], 
            return_proj = (return_type == "proj"),
            enc_from_error_space = (
                enc_from_error_space_if_possible 
                or not isinstance(subspace_type[0], int)
            )
        )
            
        enc_1 = _constr_enc_by_subspace(
            encoders, subspace_type[1], 
            return_proj = (return_type == "proj"),
            enc_from_error_space = (
                enc_from_error_space_if_possible 
                or not isinstance(subspace_type[1], int)
            )
        )
        
        # remove double counting if the two subspaces are the same
        if subspace_type[0] == subspace_type[1]:
            prefactor = 1/2     
        else:
            prefactor = 1   
        
        if return_type == "enc" or return_type == "proj":
            return prefactor * (
                qt.sprepost(enc_0, enc_1.dag())
                + qt.sprepost(enc_1, enc_0.dag())
            )
        else:
            # return_type == "dec":
            return prefactor * (
                qt.sprepost(enc_0.dag(), enc_1)
                + qt.sprepost(enc_1.dag(), enc_0)
            )
            
    # subspace_type == "LD" or "LC":
    # first, get projector version of the superoperator
    if return_type != "proj" and not enc_from_error_space_if_possible:
        raise ValueError("enc_from_error_space_if_possible must be True "
                         "for 'LD' or 'LC' subspace.")
    
    len_encs = len(encoders)
    superops = []
    for idx_0, idx_1 in np.ndindex(len_encs, len_encs):
        if idx_0 <= idx_1 and subspace_type == "LC":
            continue    
        if idx_0 != idx_1 and subspace_type == "LD":
            continue
        
        op = _constr_enc_superop_by_subspace(
            encoders, (idx_0, idx_1), 
            return_type,
            enc_from_error_space_if_possible
        )
        
        superops.append(op)
        
    return sum(superops)
    
def process_block(
    process: qt.Qobj,
    init_subspace: Tuple[_subspace_types, _subspace_types] | Literal["C"], 
    final_subspace: Tuple[_subspace_types, _subspace_types] | Literal["C"],
    init_encoders: np.ndarray[qt.Qobj] | List[qt.Qobj],
    final_encoders: np.ndarray[qt.Qobj] | List[qt.Qobj],
    truncate_leakage_space: bool = True,
    truncate_irrelevant_logical_space_if_possible: bool = True,
) -> qt.Qobj:
    """
    Return P_final * process * P_init, where P_init and P_final
    are projector maps determined by the initial and final subspaces.
    
    The init_subspace (final_subspace) can be a tuple of two elements a and b,
    denoting the subspaces of the hilbert space.
    That defines a projector superoperator by
    P_ab (.) = P_a (.) P_b, if a = b
    P_ab (.) = P_a (.) P_b + h.c., if a != b
    where P_a and P_b are projector operators to the corresponding subspaces.
    The allowed subspace indices (a) are:
    - int: the index of the correctable subspaces (node.ideal_logical_states[a])
    - "L": the total correctable subspace (sum of all correctable subspaces)
    - "p": the leakage subspace
    
    The init_subspace (final_subspace) can also be a single string
    - "LD", which stands for the sum_a P_a (.) P_a
    - "LC", which stands for the sum_{a!=b} P_a (.) P_b + h.c.
    here a and b run over all indices of the logical subspaces.
    
    Parameters
    ----------
    truncate_leakage_space: bool
        Whether to truncate out the leakage space. 
    truncate_irrelevant_logical_space: bool
        Whether to truncate out the irrelevant logical space, i.e., when init_subspace
        = [0, 0], the rest of the logical space (1, 2, ...) is truncated.
    """
    # if truncate_leakage_space, we actually decode the logical states 
    # from the physical space to the logical space (or error space)
    if truncate_leakage_space and (
        init_subspace[0] != "p"
        and init_subspace[1] != "p"
        and final_subspace[0] != "p"
        and final_subspace[1] != "p"
    ):
        init_return_type = "enc"
        final_return_type = "dec"
    else:
        init_return_type = "proj"
        final_return_type = "proj"
    
    init_superop = _constr_enc_superop_by_subspace(
        init_encoders, init_subspace, return_type=init_return_type,
        enc_from_error_space_if_possible=(
            not truncate_irrelevant_logical_space_if_possible
            or init_subspace[0] != init_subspace[1]
            or init_subspace[0] == "L"
            or init_subspace[1] == "L"
        )
    )
    final_superop = _constr_enc_superop_by_subspace(
        final_encoders, final_subspace, return_type=final_return_type,
        enc_from_error_space_if_possible=(
            not truncate_irrelevant_logical_space_if_possible
            or final_subspace[0] != final_subspace[1]
            or final_subspace[0] == "L"
            or final_subspace[1] == "L"
        )
    )
    
    return final_superop * process * init_superop

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
    
    # init_encoders_superop = [
    #     qt.sprepost(enc, enc.dag()) for enc in init_encoders
    # ]
    # final_decoders_superop = [
    #     qt.sprepost(enc.dag(), enc) for enc in final_encoders
    # ]

    ELP = np.ndarray(
        (len_final_subspace, len_init_subspace), 
        dtype=qt.Qobj
    )
    
    for idx_final, idx_init in np.ndindex(*ELP.shape):
        # proc = (
        #     final_decoders_superop[idx_final] * process * init_encoders_superop[idx_init]
        # )
        
        proc = process_block(
            process,
            (idx_init, idx_init),
            (idx_final, idx_final),
            init_encoders,
            final_encoders,
            truncate_leakage_space = True,
            truncate_irrelevant_logical_space_if_possible = True,
        )
        
        ELP[idx_final, idx_init] = proc
        
    if repr == "super":
        return ELP
    elif repr == "choi":
        return to_choi_vec(ELP)
    elif repr == "chi":
        return to_chi_vec(ELP)
    elif repr == "orth_chi":
        return to_orth_chi_vec(ELP)
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

def process_block_dnorm(
    process: qt.Qobj,
    init_subspace: Tuple[_subspace_types, _subspace_types] | Literal["C"], 
    final_subspace: Tuple[_subspace_types, _subspace_types] | Literal["C"],
    init_encoders: np.ndarray[qt.Qobj] | List[qt.Qobj],
    final_encoders: np.ndarray[qt.Qobj] | List[qt.Qobj],
):
    """
    Return the diamond norm of P_final * process * P_init, 
    where P_init and P_final are projector maps determined by the initial 
    and final subspaces.
    
    The init_subspace (final_subspace) can be a tuple of two elements a and b,
    denoting the subspaces of the hilbert space.
    That defines a projector superoperator by
    P_ab (.) = P_a (.) P_b, if a = b
    P_ab (.) = P_a (.) P_b + h.c., if a != b
    where P_a and P_b are projector operators to the corresponding subspaces.
    The allowed subspace indices (a) are:
    - int: the index of the correctable subspaces (node.ideal_logical_states[a])
    - "L": the total correctable subspace (sum of all correctable subspaces)
    - "p": the leakage subspace
    
    The init_subspace (final_subspace) can also be a single string
    - "LD", which stands for the sum_a P_a (.) P_a
    - "LC", which stands for the sum_{a!=b} P_a (.) P_b + h.c.
    here a and b run over all indices of the logical subspaces.
    """
    block = process_block(
        process, init_subspace, final_subspace, init_encoders, final_encoders,
        truncate_leakage_space = True,
        truncate_irrelevant_logical_space_if_possible = True,
    )

    return block.dnorm()
    

# Manipulation of Choi matrix ==============================================
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

# Bound the fidelity contribution from leakage process =========================
# Now we got a better way to do this, using diamond norm of the process_block ==
# so it is kind-of deprecated. =================================================
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
        
    Note:
    This method is a little redundant compared to process_block, but it 
    provides "recorder_final_space" option, which is not available in 
    process_block method. Now it's only used for bound the effect of the leakage 
    process.
        
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
        The (unnormalized) seepage upper bound for the process_2.
        
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
    # The normalized seepage rate is calculated by 1 - sum_(n>2, p>2) choi_2_npnp / leakage_space_dim (Wood (2018))
    # Since we do truncations to the H_2, so we use the unnormalized seepage rate
    # seepage_rate = leakage_space_dim - sum_(n>2, p>2) choi_2_npnp
    if constr_proc2_seepage is not None:
        slice_np = [
            n_idx * dim_2 + p_idx 
            for n_idx in range(logical_dim, dim_3)
            for p_idx in range(logical_dim, dim_2)
        ]
        seepage_rate = cp.real(leakage_space_dim - cp.trace(const_choi[slice_np, :][:, slice_np]))
        constaints.append(seepage_rate <= constr_proc2_seepage)
            
    # solve the problem ------------------------------------------------
    if maximize:
        objective = cp.Maximize(fidelity)
    else:
        objective = cp.Minimize(fidelity)

    # Define and solve the problem
    problem = cp.Problem(objective, constaints)
    result = problem.solve(solver=settings.CVX_SOLVER)

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