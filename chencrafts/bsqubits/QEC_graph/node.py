import qutip as qt
import numpy as np
import networkx as nx
from copy import deepcopy
from warnings import warn

from chencrafts.cqed.qt_helper import (
    projector_w_basis,
    normalization_factor,
)

from typing import List, Tuple, Any, TYPE_CHECKING, Dict, Callable

if TYPE_CHECKING:
    from chencrafts.bsqubits.QEC_graph.edge import Edge

MeasurementRecord = List[Tuple[int, ...]]

class StateNode:

    # options:
    OTHOGONALIZE_LOGICAL_STATES = True

    meas_record: MeasurementRecord

    # current state as a density matrix
    state: qt.Qobj

    # probability amplitude of |0> and |1>
    prob_amp_01: Tuple[float, float]

    # ideal states, organized in an ndarray, with dimension n*3
    # the first dimension counts the number of correctable errors
    # the second dimension enumerates: logical state 0 and logical state 1
    ideal_logical_states: np.ndarray[qt.Qobj]
    
    index: int

    def __init__(
        self, 
    ):
        """
        A node that represents a state in the QEC trajectory
        """
        self.out_edges: List["Edge"] = []

    def add_out_edges(self, edge):
        self.out_edges.append(edge)

    def accept(
        self, 
        meas_record: MeasurementRecord,
        state: qt.Qobj,
        prob_amp_01: Tuple[float, float],
        ideal_logical_states: np.ndarray[qt.Qobj],
    ):
        # basic type checks:
        for ideal_state in ideal_logical_states.ravel():
            assert ideal_state.type == "ket"
            assert np.allclose(normalization_factor(ideal_state), 1)
        assert np.allclose(np.sum(np.abs(prob_amp_01)**2), 1)

        self.meas_record = meas_record
        self.state = state
        self.prob_amp_01 = prob_amp_01
        self.ideal_logical_states = ideal_logical_states

    @staticmethod
    def _symmtrized_orthogonalize(state_0, state_1):
        """
        A little bit more generalized version of Gram-Schmidt orthogonalization?
        Don't know whether there is a reference.
        """
        overlap = (state_0.overlap(state_1))
        theta = - np.angle(overlap)   # to make the ovrlap real
        state_1_w_phase= state_1 * np.exp(1j * theta)

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
    def _orthogonalize(state_arr: np.ndarray[qt.Qobj]) -> np.ndarray[qt.Qobj]:
        """
        Orthogonalize the states in the array
        """
        new_state_arr = np.empty_like(state_arr)
        for i in range(len(state_arr)):
            (
                new_state_arr[i, 0], new_state_arr[i, 1]
            ) = StateNode._symmtrized_orthogonalize(
                *state_arr[i]
            )
        
        return new_state_arr

    @property
    def ideal_states(self) -> np.ndarray[qt.Qobj]:
        """
        Return the ideal state by logical states
        """
        if len(self.ideal_logical_states) == 0:
            # the states' norm is too small and thrown away
            dim = self.state.dims[0]
            return np.array([qt.Qobj(
                np.zeros(self.state.shape), 
                dims=[dim, np.ones_like(dim).astype(int).tolist()]
            )], dtype=qt.Qobj)
        
        # need to be modified as the logical states are not necessarily
        # orthogonal
        if self.OTHOGONALIZE_LOGICAL_STATES:
            othogonalized_states = self._orthogonalize(self.ideal_logical_states)
            return (
                self.prob_amp_01[0] * othogonalized_states[:, 0]
                + self.prob_amp_01[1] * othogonalized_states[:, 1]
            )
        else:
            return (
                self.prob_amp_01[0] * self.ideal_logical_states[:, 0] 
                + self.prob_amp_01[1] * self.ideal_logical_states[:, 1]
            )
    
    @property
    def ideal_projector(self) -> qt.Qobj:
        return projector_w_basis(self.ideal_states)

    @property
    def fidelity(self) -> float:
        return ((self.state * self.ideal_projector).tr()).real

    def deepcopy(self):
        """
        1. Not storing the edge information and avoiding circular reference
        2. deepcopy the Qobj
        """

        copied_node = StateNode()
        copied_node.meas_record = deepcopy(self.meas_record)
        copied_node.state = deepcopy(self.state)
        copied_node.ideal_logical_states = deepcopy(self.ideal_logical_states)

        return copied_node
    
    @classmethod
    def initial_note(
        cls, 
        init_prob_amp_01: Tuple[float, float],
        logical_0: qt.Qobj,
        logical_1: qt.Qobj,
    ) -> "StateNode":
        state = init_prob_amp_01[0] * logical_0 + init_prob_amp_01[1] * logical_1
        logical_state_arr = np.empty((1, 2), dtype=object)
        logical_state_arr[:] = [[logical_0, logical_1]]

        init_node = cls()
        init_node.accept(
            meas_record = [], 
            state = qt.ket2dm(state),
            prob_amp_01 = init_prob_amp_01,
            ideal_logical_states = logical_state_arr,
        )

        return init_node
    
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

    def clear_evolution_data(self):
        try:
            del self.state
            del self.ideal_logical_states
            del self.fidelity
            del self.meas_record
        except AttributeError:
            pass

    def __str__(self) -> str:
        try:
            idx = self.index
        except AttributeError:
            idx = "No Index"
        return f"StateNode ({idx}), record {self.meas_record}, fidelity {self.fidelity:.3f}"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def expect(self, op: qt.Qobj) -> float:
        """
        Calculate the expectation value of the operator
        """
        return self.state.expect(op)


# class CatNode(StateNode):
#     state_vector: List[complex]
#     logical_states: List[qt.Qobj]

#     def accept(
#         self, 
#         meas_record: MeasurementRecord, 
#         state: qt.Qobj, 
#         state_vector: List[complex],
#         logical_states: List[qt.Qobj],
#     ):
#         self.state_vector = state_vector
#         self.logical_states = logical_states

#         ideal_state
#         ideal_projector = qt.ket2dm(state)
        
#         super().accept(meas_record, state, ideal_projector)


class StateEnsemble:

    def __init__(
        self, 
        nodes: [List[StateNode] | None] = None,
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
        for node in self:
            if node.out_edges != []:
                no_further_evolution = False
                break

        return no_further_evolution

    def append(self, node: StateNode):
        self.nodes.append(node)

    def is_trace_1(self) -> bool:
        """
        Check if the total trace is 1
        """
        for node in self.nodes:
            try: 
                node.state
            except AttributeError:
                return False

        trace = sum([node.state.tr() for node in self.nodes])
        return np.abs(trace - 1) < 1e-8
    
    @property
    def state(self) -> qt.Qobj:
        """
        Calculate the total state
        """
        if not self.is_trace_1():
            warn("The total trace is not 1. The averaged state is not"
                 " physical. ")
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
    
    def expect(self, op: qt.Qobj) -> float:
        """
        Calculate the expectation value of the operator
        """
        return sum([node.expect(op) for node in self.nodes])