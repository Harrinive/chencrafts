__all__ = [
    'EvolutionEdge',
    'PropagatorEdge',
    'MeasurementEdge',
    'CheckPointEdge',
]

import copy
import qutip as qt
import numpy as np
from warnings import warn
from abc import ABC, abstractmethod
from typing import List, Tuple, Any, TYPE_CHECKING, Callable, Dict, Literal

from chencrafts.cqed.qt_helper import (
    superop_evolve,
    normalization_factor,
)
from . import settings
from .utils import (
    effective_logical_process, 
    target_process_for_dnorm, 
    _subspace_types,
    process_block_dnorm,
)
from .node import TerminationError

if TYPE_CHECKING:
    from .node import (
        Node,
        MeasurementRecord,
    )


class EdgeBase(ABC):
    name: str

    init_state: "Node"
    final_state: "Node"

    index: int
    
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

    @property
    def terminated(self) -> bool:
        """
        Whether the edge is linked to a terminated state
        """
        return self.final_state.terminated

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
        if self.to_ensemble:
            raise NotImplementedError("to_ensemble = True is deprecated.")

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
            self.init_state._raw_ideal_logical_states
            self.init_state.process
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
        
        # evolve the process using the real map
        final_process = map_superop * self.init_state.process

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
                    if settings.ISSUE_WARNING:
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
            if settings.ISSUE_WARNING:
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
                meas_record = copy.deepcopy(self.init_state.meas_record), 
                state = final_state, 
                prob_amp_01 = copy.deepcopy(self.init_state._prob_amp_01),
                raw_ideal_logical_states = new_ideal_logical_state_array,
                process = final_process,
                init_encoders = copy.deepcopy(self.init_state.init_encoders),
            )
        else:
            # deprecated now !!!!
            
            # self.final_state.join(
            #     meas_record = copy.copy(self.init_state.meas_record), 
            #     state = final_state, 
            #     prob_amp_01 = copy.copy(self.init_state._prob_amp_01),
            #     ideal_logical_states = new_ideal_logical_state_array,
            # )
            pass

    def __str__(self) -> str:
        return f"{self.name}"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def effective_logical_process(
        self,
        repr: Literal["super", "choi", "chi", "orth_chi"] = "super",
    ):
        """
        The effective logical process of the edge in the computational basis.
        
        Note that the initial and final states may be in multiple logical subspaces,
        say the initial node have i subspaces and the final node have f subspaces.
        Then the effective logical process is a f*i matrix, with each element being
        a superoperator representation of the logical process.
        """
        init_encoders = self.init_state.ideal_encoders()
        final_encoders = self.final_state.ideal_encoders()
        
        if isinstance(self.real_map, qt.Qobj):
            map_superop = self.real_map
        else:
            map_superop = self.real_map(self.init_state.meas_record) 
            
        return effective_logical_process(
            process = map_superop,
            init_encoders = init_encoders,
            final_encoders = final_encoders,
            repr = repr,
        )
        
    def fidelity_by_process(self, type: Literal["avg", "etg"] = "avg"):
        """
        The fidelity of the processes on the edge.
        
        Type:
            "avg": average fidelity
            "etg": enranglement fidelity
        """
        realized_process = self.effective_logical_process("super")
        
        fidelity = np.zeros(realized_process.shape)
        for idx, proc in np.ndenumerate(realized_process):
            process_fidelity = qt.process_fidelity(proc, qt.qeye_like(proc))
            
            if type == "avg":
                raise NotImplementedError(
                    "Average fidelity is not implemented. As the conversion "
                    "from process fidelity isn't clear when the process isn't "
                    "CPTP."
                )
                # wrong when the process is not TP
                # fidelity[idx] = proc_fid_2_ave_fid(process_fidelity, 2)
            elif type == "etg":
                fidelity[idx] = process_fidelity
            else:
                raise ValueError("The type of fidelity should be either 'avg' or 'etg'.")

        return fidelity
        
    def process_dnorm(self):
        """
        The diamond norm of the processes on the edge.
        """
        processes = self.effective_logical_process("super")
        dnorms = np.zeros(processes.shape)
        
        for idx, process in np.ndenumerate(processes):
            dnorms[idx] = process.dnorm(target_process_for_dnorm(process))
        
        return dnorms
    
    def process_fidelity(self):
        """
        The process fidelity of the processes on the edge.
        """
        processes = self.effective_logical_process(repr="chi") / 4
        fidelities = np.zeros(processes.shape)
        
        for idx, process in np.ndenumerate(processes):
            fidelities[idx] = process[0, 0].real
        
        return fidelities
    
    def process_choi_trace(self) -> float:
        """
        The trace of the choi matrix of the effective logical processes
        """
        processes = self.effective_logical_process(repr = "choi")
        traces = np.zeros(processes.shape, dtype=float)
        for idx, process in np.ndenumerate(processes):
            traces[idx] = process.tr().real
            
        return traces
    
    def process_block_dnorm(
        self,
        init_subspace: Tuple[_subspace_types, _subspace_types] | Literal["LC", "LD"], 
        final_subspace: Tuple[_subspace_types, _subspace_types] | Literal["LC", "LD"],
    ):
        """
        Return the diamond norm of P_final * self.real_map * P_init, 
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
        init_encoders = self.init_state.ideal_encoders()
        final_encoders = self.final_state.ideal_encoders()
        
        if isinstance(self.real_map, qt.Qobj):
            map_superop = self.real_map
        else:
            map_superop = self.real_map(self.init_state.meas_record) 
            
        return process_block_dnorm(
            process = map_superop,
            init_subspace = init_subspace,
            final_subspace = final_subspace,
            init_encoders = init_encoders,
            final_encoders = final_encoders,
        )
        

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
            self.init_state._raw_ideal_logical_states
        except AttributeError:
            raise AttributeError("The initial state are not evolved.")

        # process: a projection on to the logical subspace or its complement
        projector = self.init_state.ideal_projector
        if self.success:
            map_op = projector
        else:
            eye_op = qt.qeye_like(projector)
            map_op = eye_op - projector
        map_superop = qt.sprepost(map_op, map_op.dag())
        
        # evolve the state and the process
        process = map_superop * self.init_state.process
        final_state = superop_evolve(map_superop, self.init_state.state)

        # feed the result to the final state
        self.final_state.accept(
            meas_record = copy.copy(self.init_state.meas_record), 
            state = final_state, 
            prob_amp_01 = copy.copy(self.init_state._prob_amp_01),
            raw_ideal_logical_states = copy.deepcopy(self.init_state._raw_ideal_logical_states),
            process = process,
            init_encoders = copy.deepcopy(self.init_state.init_encoders),
        )

    def __str__(self) -> str:
        return f"{self.name}"
    
    def __repr__(self) -> str:
        return self.__str__()