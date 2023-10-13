import qutip as qt
import numpy as np
import networkx as nx
from copy import deepcopy

from typing import List, Tuple, Any, TYPE_CHECKING, Dict, Callable

if TYPE_CHECKING:
    from chencrafts.bsqubits.QEC_graph.edge import Edge

MeasurementRecord = List[Tuple[int, ...]]

class StateNode:

    meas_record: MeasurementRecord

    state: qt.Qobj
    ideal_projector: qt.Qobj
    fidelity: float

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
        ideal_projector: qt.Qobj,
    ):
        self.meas_record = meas_record
        
        self.state = state
        self.ideal_projector = ideal_projector
        self.fidelity = ((state * ideal_projector).tr()).real

    def deepcopy(self):
        """
        1. Not storing the edge information and avoiding circular reference
        2. deepcopy the Qobj
        """

        copied_node = StateNode()
        copied_node.meas_record = deepcopy(self.meas_record)
        copied_node.state = deepcopy(self.state)
        copied_node.ideal_projector = deepcopy(self.ideal_projector)
        copied_node.fidelity = deepcopy(self.fidelity)

        return copied_node
    
    @classmethod
    def initial_note(cls, init_state: qt.Qobj) -> "StateNode":
        init_node = cls()
        init_node.accept(
            [], 
            qt.ket2dm(init_state),
            init_state
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


class StateEnsemble:

    def __init__(
        self, 
        nodes: [List[StateNode] | None] = None,
        # note: Do not use [] as the default value, it will be shared by 
        # all the instances, as it's a mutable object
    ):
        if nodes is None:
            nodes = []
        self.nodes = nodes

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
    