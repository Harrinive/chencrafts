import copy
import qutip as qt
import numpy as np
from typing import List, Tuple, Any, TYPE_CHECKING, Callable, Dict

from chencrafts.cqed.qt_helper import (
    superop_evolve,
    normalization_factor,
)

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
        ideal_maps: List[qt.Qobj] | Callable[["MeasurementRecord"], List[qt.Qobj]],
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
        ideal_maps : List[qt.Qobj] | List[Callable[[MeasurementRecord], qt.Qobj]]
            The ideal map that evolves the initial ideal state (pure) to 
            the final ideal state (pure, but may not be properly normalized). 
            It could be a operator or a function. When it's a function, 
            the measurement record is needed as the input.
        """
        self.name = name
        self.real_map = real_map
        self.ideal_maps = ideal_maps

    def connect(self, init_state: "StateNode", final_state: "StateNode"):
        """
        Connect the edge to the initial state and the final state
        """
        self.init_state = init_state
        self.final_state = final_state

    def evolve(self):
        """
        Evolve the initial state to the final state using the map. 
        
        All of the evolved ideal states are normalized to norm 1.
        """
        try:
            self.init_state
            self.final_state
        except AttributeError:
            raise AttributeError("The initial state and the final state are not connected.")
        
        try:
            self.init_state.state
            self.init_state.prob_amp_01
            self.init_state.ideal_logical_states
        except AttributeError:
            raise AttributeError("The initial state are not evolved.")

        # evolve the state using the real map
        if isinstance(self.real_map, qt.Qobj):
            map_superop = self.real_map
        else:
            map_superop = self.real_map(self.init_state.meas_record) 
        final_state = superop_evolve(
            map_superop, self.init_state.state
        )

        # evolve the ideal states using the ideal maps
        if isinstance(self.ideal_maps, list):
            map_op_list = self.ideal_maps
        else:
            map_op_list = self.ideal_maps(self.init_state.meas_record)

        new_ideal_logical_states = []
        for map_op in map_op_list:
            for logical_0, logical_1 in self.init_state.ideal_logical_states:
                new_logical_0 = map_op * logical_0
                new_logical_1 = map_op * logical_1

                # when a syndrome measurement is done, it's likely that the 
                # number of ideal states will be reduced and the state is not 
                # normalized anymore. Only add the state to the list if it's 
                # not zero norm.
                norm_0 = normalization_factor(new_logical_0)
                norm_1 = normalization_factor(new_logical_1)
                if norm_0 > 1e-13 or norm_1 > 1e-13:
                    new_ideal_logical_states.append(
                        [new_logical_0 / norm_0, new_logical_1 / norm_1]
                    )
        
        # no any ideal state component, usually means the state is in it's 
        # steady state - no single photon loss anymore
        if len(new_ideal_logical_states) == 0:
            raise ValueError("No ideal state component left. "
                             "The state is likely in it's steady state.")

        # convert to ndarray
        new_ideal_logical_state_array = np.empty(
            (len(new_ideal_logical_states), 2), dtype=object
        )
        new_ideal_logical_state_array[:] = new_ideal_logical_states

        # feed the result to the final state
        self.final_state.accept(
            copy.copy(self.init_state.meas_record), 
            final_state, 
            copy.copy(self.init_state.prob_amp_01),
            new_ideal_logical_state_array,
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
    
    def __str__(self) -> str:
        return f"{self.name}"
    
    def __repr__(self) -> str:
        return self.__str__()

class PropagatorEdge(EdgeBase):
    pass


class MeasurementEdge(EdgeBase):
    def __init__(
        self, 
        name: str,
        outcome: float,
        real_map: qt.Qobj | Callable[["MeasurementRecord"], qt.Qobj],
        ideal_map: List[qt.Qobj] | Callable[["MeasurementRecord"], List[qt.Qobj]],
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

    def __str__(self) -> str:
        return super().__str__() + f" ({self.outcome})"

    


Edge = PropagatorEdge | MeasurementEdge