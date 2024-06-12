import qutip as qt
import numpy as np
from copy import deepcopy
from warnings import warn

from chencrafts.cqed.qt_helper import (
    projector_w_basis,
    normalization_factor,
    evecs_2_transformation,
)

from typing import List, Tuple, Any, TYPE_CHECKING, Dict, Callable, Literal
from abc import ABC, abstractmethod, abstractproperty

if TYPE_CHECKING:
    from chencrafts.bsqubits.QEC_graph.edge import Edge

MeasurementRecord = List[Tuple[int, ...]]

class NodeBase(ABC):
    # current state as a density matrix
    state: qt.Qobj
    index: int

    def __init__(
        self, 
    ):
        """
        A node that represents a state in the QEC trajectory
        """
        self.out_edges: List["Edge"] = []

    @abstractproperty
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
    ideal_logical_states: np.ndarray[qt.Qobj]

    # mark that the node will not be further evolved and reduce the compu
    # time. It does not mean that the node is a final state in the diagram,
    # but a state that will stay in the ensemble forever.
    terminated: bool = False

    # fidelity warning issued when the fidelity is larger than 0 in a 
    # terminated branch
    term_fid_warning_issued = False

    def accept(
        self, 
        meas_record: MeasurementRecord,
        state: qt.Qobj,
        prob_amp_01: Tuple[float, float],
        ideal_logical_states: np.ndarray[qt.Qobj],
        **kwargs,
    ):
        """
        Accept the evolution data from the edge and overwrite the current state. 
        It's useful for a node in a tree structure.

        For StateNode, it takes the following arguments:
        - meas_record: the measurement record
        - state: the state after the evolution
        - prob_amp_01: the probability amplitude of |0> and |1>
        - ideal_logical_states: the ideal logical states
        """
        # basic type checks:
        for ideal_state in ideal_logical_states.ravel():
            assert ideal_state.type == "ket"
            assert np.allclose(normalization_factor(ideal_state), 1)
        assert np.allclose(np.sum(np.abs(prob_amp_01)**2), 1)

        self.meas_record = meas_record
        self.state = state
        self._prob_amp_01 = prob_amp_01
        self.ideal_logical_states = ideal_logical_states

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
        elif self.ideal_logical_states.shape[0] > 1:
            warn("The probability amplitude of |0> and |1> is reset manually. "
                 "While the state is not reset as the ideal logical states are "
                 "not unique. \n")
            return
        elif self.ideal_logical_states.shape[0] == 1:
            warn("The probability amplitude of |0> and |1> and the state are "
                 "reset manually. \n")
            if self.ORTHOGONALIZE_LOGICAL_STATES:
                logical_states = self._orthogonalize(self.ideal_logical_states)
            else:
                logical_states = self.ideal_logical_states

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
        Gram-Schmidt orthogonalization
        """
        new_state_0 = state_0.unit()
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
        if len(self.ideal_logical_states) == 0:
            # the states' norm is too small and thrown away
            dim = self.state.dims[0]
            return np.array([qt.Qobj(
                np.zeros(self.state.shape), 
                dims=[dim, np.ones_like(dim).astype(int).tolist()]
            )], dtype=qt.Qobj)
        
        # need to be modified as the logical states are not necessarily
        # orthogonal
        if orthogonalize:
            othogonalized_states = self._orthogonalize(self.ideal_logical_states)
            return (
                self._prob_amp_01[0] * othogonalized_states[:, 0]
                + self._prob_amp_01[1] * othogonalized_states[:, 1]
            )
        else:
            qobj_array_unit = np.vectorize(
                self._qobj_unit, otypes = [qt.Qobj]
            )   # apply qobj.unit() to each element in the array
            return qobj_array_unit(
                self._prob_amp_01[0] * self.ideal_logical_states[:, 0] 
                + self._prob_amp_01[1] * self.ideal_logical_states[:, 1]
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
        copied_node.ideal_logical_states = deepcopy(self.ideal_logical_states)

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
            othogonalized_states = cls._orthogonalize(logical_state_arr)
            state = (
                init_prob_amp_01[0] * othogonalized_states[0, 0]
                + init_prob_amp_01[1] * othogonalized_states[0, 1]
            )
        else:
            state = (
                init_prob_amp_01[0] * logical_state_arr[0, 0] 
                + init_prob_amp_01[1] * logical_state_arr[0, 1]
            ).unit()
        
        init_node = cls()
        init_node.accept(
            meas_record = [], 
            state = qt.ket2dm(state),
            prob_amp_01 = init_prob_amp_01,
            ideal_logical_states = logical_state_arr,
        )

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

        if self.ideal_logical_states.shape[0] > 1:
            warn("The ideal logical states are not unique. Returned nan.\n")
            return np.nan * np.ones(4)
        
        if self.ORTHOGONALIZE_LOGICAL_STATES:
            logical_states = self._orthogonalize(self.ideal_logical_states)
        else:
            logical_states = self.ideal_logical_states
        
        trans = evecs_2_transformation(logical_states[0])

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