import qutip as qt
import numpy as np
from copy import deepcopy

from typing import List, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from chencrafts.bsqubits.QEC_trajectory.edge import Edge

class StateNode:

    meas_record: List[Tuple[int, ...]]

    state: qt.Qobj | None
    ideal_state: qt.Qobj | None
    fidelity: float | None

    index: int

    def __init__(
        self, 
    ):
        """
        A node that represents a state in the QEC trajectory

        Parameters
        ----------
        name : str
            Name of the node
        """
        self.out_edges: List[Edge] = []

    def add_out_edges(self, edge):
        self.out_edges.append(edge)

    def accept(
        self, 
        meas_record: List[Tuple[int, ...]],
        state: qt.Qobj,
        ideal_state: qt.Qobj,
    ):
        self.meas_record = meas_record
        
        self.state = state
        self.ideal_state = ideal_state
        self.fidelity = state.overlap(ideal_state)

    def deepcopy(self):
        """
        1. Not storing the edge information and avoiding circular reference
        2. deepcopy the Qobj
        """

        copied_node = StateNode()
        copied_node.meas_record = deepcopy(self.meas_record)
        copied_node.state = deepcopy(self.state)
        copied_node.ideal_state = deepcopy(self.ideal_state)
        copied_node.fidelity = deepcopy(self.fidelity)

        return copied_node
    
    @classmethod
    def initial_note(init_state: qt.Qobj) -> "StateNode":
        init_node = StateNode()
        init_node.accept(
            [], 
            qt.ket2dm(init_state),
            init_state
        )
        return init_node
    
    def assign_index(self, index: int):
        self.index = index


class StateEnsemble:

    def __init__(
        self, 
        node_list: List[StateNode],
    ):
        self.node_list = node_list

    def add(self, node: StateNode):
        self.node_list.append(node)

    def is_valid(self) -> bool:
        """
        Check if the total trace is 1
        """
        for node in self.node_list:
            try: 
                node.state
            except AttributeError:
                return False

        trace = sum([node.state.tr() for node in self.node_list])
        return np.abs(trace - 1) < 1e-8
    
    @property
    def fidelity(self) -> float:
        """
        Calculate the total fidelity
        """
        return sum([node.fidelity for node in self.node_list])

    def deepcopy(self):
        """
        1. Not storing the edge information
        2. deepcopy the Qobj
        """ 
        return [
            node.deepcopy() for node in self.node_list
        ]
    
    def __iter__(self):
        return iter(self.node_list)