import qutip as qt

from chencrafts.cqed import superop_evolve

from typing import List, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from chencrafts.bsqubits.QEC_trajectory.node import StateNode

class EdgeBase:
    init_state: StateNode
    final_state: StateNode

    index: int

    def __init__(
        self, 
        map: qt.Qobj,
        ideal_map: qt.Qobj,
    ):
        """
        Edge that connects two StateNodes.

        Parameters
        ----------
        init_state : StateNode
            The initial state node
        final_state : StateNode
            The final state node
        map : qt.Qobj
            The actual map that evolves the initial state to the final state.
            Should be a superoperator.
        ideal_map : qt.Qobj
            The ideal map that evolves the initial state to the final state.
            Should be a regular operator.
        """
        self.map = map
        self.ideal_map = ideal_map

    def connect(self, init_state: StateNode, final_state: StateNode):
        """
        Connect the edge to the initial state and the final state
        """
        self.init_state = init_state
        self.final_state = final_state

    def evolve(self):
        """
        Evolve the initial state to the final state using the map
        """
        try:
            self.init_state
            self.final_state
        except AttributeError:
            raise AttributeError("The initial state and the final state not connected.")

        try:
            final_state = superop_evolve(self.map, self.init_state.state)
            ideal_final_state = self.ideal_map * self.init_state.ideal_state
        except AttributeError:
            raise AttributeError("The initial state not evolved.")
        
        self.final_state.accept(
            self.init_state.meas_record, 
            final_state, 
            ideal_final_state,
        )

    def assign_index(self, index: int):
        self.index = index
       

class PropagatorEdge(EdgeBase):
    pass


class MeasurementEdge(EdgeBase):
    def __init__(
        self, 
        outcome: float,
        map: qt.Qobj,
        ideal_map: qt.Qobj,
    ):
        """
        One of the measurement outcomes and projections
        """
        super().__init__(map, ideal_map)

        self.outcome = outcome

    def evolve(self):
        """
        Evolve the initial state to the final state using the map 
        and then append the measurement outcome to the measurement record
        """
        super().evolve()
        self.final_state.meas_record.append(self.outcome)


Edge = PropagatorEdge | MeasurementEdge