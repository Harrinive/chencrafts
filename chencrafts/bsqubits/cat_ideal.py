import numpy as np
import math
import qutip as qt
from typing import Literal, Callable, List, Tuple, overload, TYPE_CHECKING

from chencrafts.cqed import oprt_in_basis, superop_evolve
if TYPE_CHECKING:
    from chencrafts.bsqubits.QEC_graph.node import MeasurementRecord

# ##############################################################################
def _res_qubit_tensor(
    res_op, qubit_op, 
    res_mode_idx: Literal[0, 1],
) -> qt.Qobj:
    if res_mode_idx == 0:
        return qt.tensor(res_op, qubit_op)
    elif res_mode_idx == 1:
        return qt.tensor(qubit_op, res_op)

def _eye(
    res_dim: int, qubit_dim: int,
    res_mode_idx: Literal[0, 1] = 0,
) -> qt.Qobj:
    return _res_qubit_tensor(qt.qeye(res_dim), qt.qeye(qubit_dim), res_mode_idx)

def _res_destroy(
    res_dim: int, qubit_dim: int,
    res_mode_idx: Literal[0, 1] = 0, 
) -> qt.Qobj:
    return _res_qubit_tensor(qt.destroy(res_dim), qt.qeye(qubit_dim), res_mode_idx)
    
def _qubit_pauli(
    res_dim: int, qubit_dim: int,
    res_mode_idx: Literal[0, 1] = 0,
    axis: Literal['x', 'y', 'z'] = 'x',
) -> qt.Qobj:
    qubit_oprt = np.eye(qubit_dim, dtype=complex)
    if axis == 'x':
        qubit_oprt[:2, :2] = qt.sigmax().full()
    elif axis == 'y':
        qubit_oprt[:2, :2] = qt.sigmay().full()
    elif axis == 'z':
        qubit_oprt[:2, :2] = qt.sigmaz().full()
    else:
        raise ValueError(f'Invalid axis {axis}')
    qubit_oprt = qt.Qobj(qubit_oprt)

    return _res_qubit_tensor(qt.qeye(res_dim), qubit_oprt, res_mode_idx)

def _res_number(
    res_dim: int, qubit_dim: int,
    res_mode_idx: Literal[0, 1] = 0,
    qubit_state: int | None = None,
) -> qt.Qobj:
    """
    If qubit_state is None, return the resonator number operator tensor qubit identity.

    If qubit_state is an integer, return the resonator number operator tensor a qubit 
    projection operator.
    """
    res_oprt = qt.num(res_dim)
    if qubit_state is None:
        qubit_oprt = qt.qeye(qubit_dim)
    else:
        qubit_oprt = qt.projection(qubit_dim, qubit_state, qubit_state)

    return _res_qubit_tensor(res_oprt, qubit_oprt, res_mode_idx)
    
def _qubit_proj(
    res_dim: int, qubit_dim: int,
    res_mode_idx: Literal[0, 1] = 0,
    qubit_state: int = 0,
    qubit_state_2: int | None = None,
) -> qt.Qobj:
    """For qubit measurement"""
    res_oprt = qt.qeye(res_dim)
    
    if qubit_state_2 is not None:
        qubit_oprt = qt.projection(qubit_dim, qubit_state, qubit_state_2)
    else:
        qubit_oprt = qt.projection(qubit_dim, qubit_state, qubit_state)

    return _res_qubit_tensor(res_oprt, qubit_oprt, res_mode_idx)

def _res_rotation(
    res_dim: int, qubit_dim: int,
    res_mode_idx: Literal[0, 1] = 0,
    angle: float = np.pi / 2,
) -> qt.Qobj:
    res_oprt = (-1j * angle * qt.num(res_dim)).expm()
    qubit_oprt = qt.qeye(qubit_dim)

    return _res_qubit_tensor(res_oprt, qubit_oprt, res_mode_idx)

def res_qubit_basis(
    res_dim: int, qubit_dim: int,
    bare_label: Tuple[int, int] = (0, 0),
    res_mode_idx: Literal[0, 1] = 0,
):
    """
    Generate a basis state for the resonator-qubit Hilbert space.
    """
    res_basis = qt.basis(res_dim, bare_label[res_mode_idx])
    qubit_basis = qt.basis(qubit_dim, bare_label[1 - res_mode_idx])

    return _res_qubit_tensor(res_basis, qubit_basis, res_mode_idx)

# #############################################################################
def idling_maps(
    res_dim: int, qubit_dim: int,
    res_mode_idx: Literal[0, 1], 
    static_hamiltonian: qt.Qobj,
    time: float,
    decay_rate: float = 0.0,
    self_Kerr: float = 0.0,
) -> List[qt.Qobj]:
    """
    Return the correctable maps for the idling process.
    """
    # free evolution
    shrinkage_oprt = (
        -decay_rate * time / 2 * _res_number(res_dim, qubit_dim, res_mode_idx)
    ).expm()
    free_evolution_oprt = shrinkage_oprt * (-1j * static_hamiltonian * time).expm()

    # single-photon loss related operators
    spl_rotation_oprt = _res_rotation(
        res_dim, qubit_dim, res_mode_idx, angle = self_Kerr * time
    )   # average rotation due to self-Kerr
    a_oprt = _res_destroy(res_dim, qubit_dim, res_mode_idx)

    return [
        free_evolution_oprt,    # zero-photon loss
        free_evolution_oprt * spl_rotation_oprt * a_oprt    # single-photon loss
    ]

@overload
def idling_w_decay_propagator(
    res_dim: int, qubit_dim: int,
    res_mode_idx: Literal[0, 1] = 0,
    decay_prob: float = 0.0,
    max_photon_loss: int = 0,
    superop: Literal[False] = False,
) -> List[qt.Qobj]:
    ...

@overload
def idling_w_decay_propagator(
    res_dim: int, qubit_dim: int,
    res_mode_idx: Literal[0, 1] = 0,
    decay_prob: float = 0.0,
    max_photon_loss: int = 0,
    superop: Literal[True] = True,
) -> qt.Qobj:
    ...

def idling_w_decay_propagator(
    res_dim: int, qubit_dim: int,
    res_mode_idx: Literal[0, 1] = 0,
    decay_prob: float = 0.0,
    max_photon_loss: int = 0,
    superop: bool = False,
) -> qt.Qobj | List[qt.Qobj]:
    """
    The evolution in the presence of resonator decay.
    The operator is given by exp(-i * H * dt), where H = - decay_prob * a^\dagger a.

    The resulting state is not normalized when max_photon_loss is small.

    A superoperator is returned.

    Parameters
    ----------
    decay_prob: float
        The decay probability, defined to be decay_rate * time.
    max_photon_loss: int
        When representing the decay channel by Kraus operators, the number of Kraus operators
        is max_photon_loss + 1.
    superop: bool
        If False, return a list of Kraus operators. If True, return the superoperator.
    """
    shrinkage_oprt = (-decay_prob / 2 * _res_number(res_dim, qubit_dim, res_mode_idx)).expm()
    a_oprt = _res_destroy(res_dim, qubit_dim, res_mode_idx)

    # Kraus representation of the decay channel
    kraus_op = lambda k: (
        1 / np.sqrt(math.factorial(k))
        * (1 - np.exp(-decay_prob)) ** (k / 2)
        * a_oprt ** k
    )    
    kraus_op_list = [kraus_op(k) for k in range(max_photon_loss + 1)]

    if superop:
        super_kraus = [qt.to_super(kraus) for kraus in kraus_op_list]
        return qt.to_super(shrinkage_oprt) * sum(super_kraus)
    else:
        shrink_kraus = [shrinkage_oprt * kraus for kraus in kraus_op_list]
        return shrink_kraus

def qubit_rot_propagator(
    res_dim: int, qubit_dim: int,
    res_mode_idx: Literal[0, 1] = 0,
    angle: float = np.pi / 2,
    axis: Literal['x', 'y', 'z', '-x', '-y', '-z'] = 'x',
    superop: bool = False,
) -> qt.Qobj:
    """
    The ideal qubit rotation propagator in the rotating frame.
    """
    if axis.startswith('-'):
        angle = -angle
        axis = axis[1:]

    generator = _qubit_pauli(res_dim, qubit_dim, res_mode_idx, axis) / 2
    unitary = (-1j * angle * generator).expm()

    if superop:
        return qt.to_super(unitary)
    else:
        return unitary
    
def parity_mapping_propagator(
    res_dim: int, qubit_dim: int,
    angle: float = np.pi,
    res_mode_idx: Literal[0, 1] = 0,
    superop: bool = False,
) -> qt.Qobj:
    """
    The ideal parity mapping propagator.
    """
    generator = _res_number(res_dim, qubit_dim, res_mode_idx, qubit_state=1)
    unitary = (-1j * angle * generator).expm()

    if superop:
        return qt.to_super(unitary)
    else:
        return unitary
    
def qubit_projectors(
    res_dim: int, qubit_dim: int,
    res_mode_idx: Literal[0, 1] = 0,
    superop: bool = False,
) -> List[qt.Qobj]:
    """
    The ideal qubit measurement projectors, projecting to qubit states 
    0, 1, ..., qubit_dim - 1.
    """
    ops = [
        _qubit_proj(res_dim, qubit_dim, res_mode_idx, qubit_state=i) 
        for i in range(qubit_dim)
    ]
    if superop:
        return [qt.to_super(op) for op in ops]
    else:
        return ops
    
def qubit_measurement_func(
    res_dim: int, qubit_dim: int,
    res_mode_idx: Literal[0, 1] = 0,
    order_by_prob: bool = True,
) -> Callable: 
    """
    The ideal qubit measurement operation. Returns a function that mimic the measurement, 
    which takes in a state and returns the measurement result. The measurement
    result is a list of pairs of (probability, post-measurement state). 

    Parameters
    ----------
    order_by_prob: bool
        If True, the measurement results are ordered by the probability of the measurement.

    Returns
    -------
    A function that takes in a state and returns the measurement result.

    """
    proj_list = qubit_projectors(
        res_dim, qubit_dim, res_mode_idx, superop=False
    )

    def measurement(state: qt.Qobj):
        prob_list = np.array([qt.expect(proj, state) for proj in proj_list], dtype=float)

        if order_by_prob:
            sorted_idx = np.argsort(prob_list)[::-1]
            sorted_prob_list = prob_list[sorted_idx]
            sorted_proj_list = [proj_list[i] for i in sorted_idx]
        else:
            sorted_prob_list = prob_list
            sorted_proj_list = proj_list

        if qt.isket(state):
            post_state_list = [
                proj * state / np.sqrt(prob) for proj, prob in zip(sorted_proj_list, sorted_prob_list)
            ]
        else:
            post_state_list = [
                proj * state * proj / prob for proj, prob in zip(sorted_proj_list, sorted_prob_list)
            ]
        
        return list(zip(sorted_prob_list, post_state_list))

    return measurement

def qubit_reset_propagator(
    res_dim: int, qubit_dim: int,
    res_mode_idx: Literal[0, 1] = 0,
    superop: bool = False,
    axis: Literal['x', 'y', 'z'] = 'x',
):
    """
    The ideal qubit reset operation (an X gate).
    """
    return qubit_rot_propagator(
        res_dim, qubit_dim, res_mode_idx, angle=np.pi, axis=axis, superop=superop
    )

def identity(
    res_dim: int, qubit_dim: int,
    res_mode_idx: Literal[0, 1] = 0,
    superop: bool = False,
):
    """
    The identity propagator.
    """
    if superop:
        return qt.to_super(_eye(res_dim, qubit_dim, res_mode_idx))
    else:
        return _eye(res_dim, qubit_dim, res_mode_idx)