import copy
import qutip as qt
from typing import List, Tuple, Any, TYPE_CHECKING, Callable, Dict

from chencrafts.cqed import superop_evolve

if TYPE_CHECKING:
    from chencrafts.bsqubits.QEC_graph.node import (
        StateNode, 
        MeasurementRecord,
    )

class EdgeBase:
    init_state: "StateNode"
    final_state: "StateNode"

    index: int

    def __init__(
        self, 
        name: str,
        real_map: qt.Qobj | Callable[["MeasurementRecord"], qt.Qobj],
        ideal_map: qt.Qobj | Callable[[qt.Qobj, "MeasurementRecord"], qt.Qobj],
    ):
        """
        Edge that connects two StateNodes.

        Parameters
        ----------
        name : str
            Name of the edge
        map : qt.Qobj | Callable[[MeasurementRecord], qt.Qobj]
            The actual map that evolves the initial state to the final state.
            Should be a superoperator or a function that takes the measurement
            record as the input and returns a superoperator.
        ideal_map : qt.Qobj | Callable[[qt.Qobj, MeasurementRecord], qt.Qobj]
            The ideal map that evolves the initial ideal projector to 
            the final ideal projector. It could be a superoperator or a function. When it's a function, the measurement record is 
            needed as one of the inputs.
        """
        self.name = name
        self.real_map = real_map
        self.ideal_map = ideal_map

    def connect(self, init_state: "StateNode", final_state: "StateNode"):
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
            raise AttributeError("The initial state and the final state are not connected.")
        
        try:
            # real map
            if isinstance(self.real_map, qt.Qobj):
                map_superop = self.real_map
            else:
                map_superop = self.real_map(self.init_state.meas_record) 
            final_state = superop_evolve(
                map_superop, self.init_state.state
            )
            # ideal map
            if isinstance(self.ideal_map, qt.Qobj):
                ideal_final_projector = superop_evolve(
                    self.ideal_map, self.init_state.ideal_projector
                )
            else:
                ideal_final_projector = self.ideal_map(
                    self.init_state.ideal_projector,
                    self.init_state.meas_record
                )
        except AttributeError:
            raise AttributeError("The initial state are not evolved.")
        
        self.final_state.accept(
            self.init_state.meas_record, 
            final_state, 
            ideal_final_projector,
        )

    def assign_index(self, index: int):
        self.index = index

    def to_nx(self) -> Tuple[int, int, Dict[str, Any]]:
        """
        Convert to a networkx edge
        """
        return (
            self.init_state.index,
            self.final_state.index,
            {
                "name": self.name,
                "type": "propagator" if isinstance(self, PropagatorEdge) else "measurement",
                "process": self,
            }
        )       

class PropagatorEdge(EdgeBase):
    pass


class MeasurementEdge(EdgeBase):
    def __init__(
        self, 
        name: str,
        outcome: float,
        real_map: qt.Qobj,
        ideal_map: qt.Qobj,
    ):
        """
        One of the measurement outcomes and projections
        """
        super().__init__(name, real_map, ideal_map)

        self.outcome = outcome

    def evolve(self):
        """
        Evolve the initial state to the final state using the map 
        and then append the measurement outcome to the measurement record
        """
        super().evolve()
        init_record = copy.copy(self.init_state.meas_record)
        self.final_state.meas_record = init_record + [self.outcome]


Edge = PropagatorEdge | MeasurementEdge