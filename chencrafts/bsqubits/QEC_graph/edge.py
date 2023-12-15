import copy
import qutip as qt
import numpy as np
from warnings import warn
from abc import ABC, abstractmethod
from typing import List, Tuple, Any, TYPE_CHECKING, Callable, Dict, overload

from chencrafts.cqed.qt_helper import (
    superop_evolve,
    normalization_factor,
)
import chencrafts.bsqubits.QEC_graph.settings as settings

if TYPE_CHECKING:
    from chencrafts.bsqubits.QEC_graph.node import (
        Node,
        MeasurementRecord,
    )


class EdgeBase(ABC):
    name: str

    init_state: "Node"
    final_state: "Node"

    index: int

    # deprecated now !!!!
    # if the evolved states are added to an ensemble (for example, trash bin)
    # represents an ensemble (for example, trash bin)
    # it should be True
    to_ensemble: bool

    def connect(self, init_state: "Node", final_state: "Node"):
        """
        Connect the edge to the initial state and the final state
        """
        self.init_state = init_state
        self.final_state = final_state

    def assign_index(self, index: int):
        self.index = index

    @abstractmethod
    def evolve(self):
        """
        Evolve the initial state to the final state
        """
        pass

    def to_nx(self) -> Tuple[int, int, Dict[str, Any]]:
        """
        Convert to a networkx edge
        """
        try:
            prob = self.branching_probability
        except AttributeError:
            prob = np.nan

        return (
            self.init_state.index,
            self.final_state.index,
            {
                "name": self.name,
                "type": type(self).__name__,
                "process": self,
                "branching_probability": prob,
            }
        )  

    @property
    def branching_probability(self) -> float:
        """
        The probability of going onto this edge from the initial state
        """
        return self.final_state.probability / self.init_state.probability     


class EvolutionEdge(EdgeBase):

    def __init__(
        self, 
        name: str,
        real_map: qt.Qobj | Callable[["MeasurementRecord"], qt.Qobj],
        ideal_maps: List[qt.Qobj] | Callable[["MeasurementRecord"], List[qt.Qobj]],
        to_ensemble: bool = False,
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
        
        self.to_ensemble = to_ensemble

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

                norm_0 = normalization_factor(new_logical_0)
                norm_1 = normalization_factor(new_logical_1)

                threshold_0 = np.sqrt(settings.IDEAL_STATE_THRESHOLD_0)
                if norm_0 < threshold_0 or norm_1 < threshold_0:
                    # when a syndrome measurement is done, it's likely that the 
                    # number of ideal states will be reduced and the state is not 
                    # normalized anymore. Only add the state to the list if it's 
                    # not zero norm.
                    continue

                threshold_1 = np.sqrt(settings.IDEAL_STATE_THRESHOLD_1)
                if norm_0 < threshold_1 or norm_1 < threshold_1:
                    # it's possible that the ideal evolution will give a state
                    # that has some small components. (For exmaple, the parity
                    # mapping with chi_prime can never be perfect, but we 
                    # still want to keep track of the chi_prime during evolution)
                    warn("Non-negligible small components are found in the ideal "
                         "states, for simplicity, they are ignored.\n")
                    continue

                new_ideal_logical_states.append(
                    [new_logical_0 / norm_0, new_logical_1 / norm_1]
                )
        
        # no any ideal state component, usually because: 
        # 1. the state is in it's steady state - no single photon loss anymore
        # 2. the state is in a branch where talking about ideal state is not
        #    meaningful anymore (failures like leakage)
        if len(new_ideal_logical_states) == 0:
            warn("Can't find ideal logical states. Use the previous ideal logical states.\n")
            new_ideal_logical_states = copy.deepcopy(self.init_state.ideal_logical_states)
            self.final_state.terminated = True

        # convert to ndarray
        new_ideal_logical_state_array = np.empty(
            (len(new_ideal_logical_states), 2), dtype=object
        )
        new_ideal_logical_state_array[:] = new_ideal_logical_states

        # feed the result to the final state
        if not self.to_ensemble:
            self.final_state.accept(
                meas_record = copy.copy(self.init_state.meas_record), 
                state = final_state, 
                prob_amp_01 = copy.copy(self.init_state._prob_amp_01),
                ideal_logical_states = new_ideal_logical_state_array,
            )
        else:
            self.final_state.join(
                meas_record = copy.copy(self.init_state.meas_record), 
                state = final_state, 
                prob_amp_01 = copy.copy(self.init_state._prob_amp_01),
                ideal_logical_states = new_ideal_logical_state_array,
            )

    def __str__(self) -> str:
        return f"{self.name}"
    
    def __repr__(self) -> str:
        return self.__str__()


class PropagatorEdge(EvolutionEdge):
    pass


class MeasurementEdge(EvolutionEdge):
    def __init__(
        self, 
        name: str,
        outcome: float,
        real_map: qt.Qobj | Callable[["MeasurementRecord"], qt.Qobj],
        ideal_map: List[qt.Qobj] | Callable[["MeasurementRecord"], List[qt.Qobj]],
        to_ensemble: bool = False,
    ):
        """
        One of the measurement outcomes and projections
        """
        super().__init__(name, real_map, ideal_map, to_ensemble)

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


class CheckPointEdge(EdgeBase):
    def __init__(
        self, 
        name: str,
        success: bool,
    ):
        self.name = name
        self.success = success

    def evolve(self):
        """
        Project the initial state onto the logical subspace and store the
        result in the final state. It gives a first order approximation of the 
        failure probability.
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

        # project the state onto the logical subspace
        state = self.init_state.state
        projector = self.init_state.ideal_projector
        success_state = projector * state * projector.dag()
        failure_state = state - success_state

        # feed the result to the final state
        self.final_state.accept(
            meas_record = copy.copy(self.init_state.meas_record), 
            state = success_state if self.success else failure_state, 
            prob_amp_01 = copy.copy(self.init_state._prob_amp_01),
            ideal_logical_states = copy.deepcopy(self.init_state.ideal_logical_states),
        )

    def __str__(self) -> str:
        return f"{self.name}"
    
    def __repr__(self) -> str:
        return self.__str__()