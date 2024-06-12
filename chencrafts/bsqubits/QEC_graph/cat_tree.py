from typing import List, Tuple, Any, TYPE_CHECKING, Dict, Callable, overload, Literal
from warnings import warn
from copy import deepcopy
from tqdm.notebook import tqdm
from abc import ABC, abstractmethod

import numpy as np
import qutip as qt
import scqubits as scq

from chencrafts.cqed import FlexibleSweep, superop_evolve

from chencrafts.bsqubits.QEC_graph.node import StateNode, StateEnsemble, MeasurementRecord
from chencrafts.bsqubits.QEC_graph.edge import (
    PropagatorEdge, MeasurementEdge, Edge, CheckPointEdge)
from chencrafts.bsqubits.QEC_graph.graph import EvolutionGraph, EvolutionTree

import chencrafts.bsqubits.cat_ideal as cat_ideal
import chencrafts.bsqubits.cat_real as cat_real
import chencrafts.settings as settings


class CatTreeBuilder(ABC):

    graph = EvolutionGraph()

    # ideal_process_switches
    idling_is_ideal: bool = False
    gate_1_is_ideal: bool = True
    parity_mapping_is_ideal: bool = False
    gate_2_is_ideal: bool = True
    qubit_measurement_is_ideal: bool = False
    qubit_reset_is_ideal: bool = True

    # utils ############################################################
    @staticmethod
    def _current_parity(meas_record: MeasurementRecord):
        # with adaptive qubit pulse, detecting "1" meaning a parity flip
        return sum(meas_record) % 2

    @staticmethod
    @overload
    def _kraus_to_super(
        qobj_list: List[qt.Qobj]
    ) -> qt.Qobj:
        ...

    @staticmethod
    @overload
    def _kraus_to_super(
        qobj_list: Callable[[MeasurementRecord], List[qt.Qobj]]
    ) -> Callable[[MeasurementRecord], qt.Qobj]:
        ...
        
    @staticmethod
    def _kraus_to_super(
        qobj_list: List[qt.Qobj] | Callable[[MeasurementRecord], List[qt.Qobj]]
    ) -> qt.Qobj | Callable[[MeasurementRecord], qt.Qobj]:
        """
        Convert a list of Kraus operators to a superoperator. It also works
        for a function that returns a list of Kraus operators.
        """
        if isinstance(qobj_list, qt.Qobj):
            # Qobj is callable and we should detect it before checking callable
            raise TypeError("The input should be a list of Qobj or a function.")
        
        if isinstance(qobj_list, list):
            return sum([qt.to_super(qobj) for qobj in qobj_list])
        elif callable(qobj_list):
            return lambda meas_record: sum([qt.to_super(qobj) for qobj in qobj_list(meas_record)])
        else:
            raise TypeError("The input should be a list of Qobj or a function.")
        
    @staticmethod
    def _add_leaves(
        graph: EvolutionTree,
        nodes: StateEnsemble,
        edge: Edge,
    ) -> StateEnsemble:
        final_nodes = StateEnsemble()

        for node in nodes.active_nodes():
            edge = deepcopy(edge)
            f_node = StateNode()
            final_nodes.append(f_node)
            graph.add_node(f_node)
            graph.add_edge_connect(
                edge,
                node,
                f_node,
            )

        return final_nodes
        
    # idling ###########################################################
    _idling_real: qt.Qobj
    _idling_ideal: List[qt.Qobj]

    def idle(
        self,
        step: int | str,
        graph: EvolutionTree,
        init_nodes: StateEnsemble,
        correctable_single_photon_loss: bool = True,
    ) -> StateEnsemble:
        """
        Idling process.

        Add one idling edge to every node in the initial ensemble.

        Note that it doesn't support idling_is_ideal.

        Parameters
        ----------
        correctable_single_photon_loss: bool
            Denote single-photon loss as a correctable error, which enables 
            the fidelity calculation when QEC is on. When QEC is off, we should 
            set it to False.
        """
        if correctable_single_photon_loss:
            accepted_states_num = 2
        else:
            accepted_states_num = 1

        edge_idling = PropagatorEdge(
            f"ID ({step})",
            self._idling_real,  # when ideal, time=0, it's already the identity, no need to change
            self._idling_ideal[:accepted_states_num],
        )

        return self._add_leaves(graph, init_nodes.active_nodes(), edge_idling)
    
    # qubit gate #######################################################
    _qubit_gate_p_ideal: List[qt.Qobj]
    _qubit_gate_m_ideal: List[qt.Qobj]
    _qubit_gate_p_real: qt.Qobj
    _qubit_gate_m_real: qt.Qobj

    @abstractmethod
    def _qubit_gate_2_map_real(
        self,
        meas_record: MeasurementRecord,
    ) -> qt.Qobj:
        pass
        
    @abstractmethod
    def _qubit_gate_2_map_ideal(
        self,
        meas_record: MeasurementRecord,
    ) -> List[qt.Qobj]:
        pass

    def qubit_gate_1(
        self,
        step: int | str,
        graph: EvolutionTree,
        init_nodes: StateEnsemble,
    ) -> StateEnsemble:
        edge_qubit_gate = PropagatorEdge(
            f"G1 ({step})",
            (
                self._qubit_gate_p_real 
                if not self.gate_1_is_ideal 
                else self._kraus_to_super(self._qubit_gate_p_ideal)
            ),
            self._qubit_gate_p_ideal,
        )
        return self._add_leaves(graph, init_nodes.active_nodes(), edge_qubit_gate)
    
    def qubit_gate_2(
        self,
        step: int | str,
        graph: EvolutionTree,
        init_nodes: StateEnsemble,
    ) -> StateEnsemble:
        edge_qubit_gate = PropagatorEdge(
            f"G2 ({step})",
            (
                self._qubit_gate_2_map_real 
                if not self.gate_2_is_ideal 
                else self._kraus_to_super(self._qubit_gate_2_map_ideal)
            ),
            self._qubit_gate_2_map_ideal,
        )

        return self._add_leaves(graph, init_nodes.active_nodes(), edge_qubit_gate)
    
    # parity mapping ###################################################
    _parity_mapping_ideal: List[qt.Qobj]
    _parity_mapping_real: qt.Qobj

    def parity_mapping(
        self,
        step: int | str,
        graph: EvolutionTree,
        init_nodes: StateEnsemble,
    ) -> StateEnsemble:
        edge_parity_mapping = PropagatorEdge(
            name = f"PM ({step})",
            real_map = (
                self._parity_mapping_real
                if not self.parity_mapping_is_ideal 
                else self._kraus_to_super(self._parity_mapping_ideal)
            ),
            ideal_maps = self._parity_mapping_ideal
        )

        return self._add_leaves(graph, init_nodes.active_nodes(), edge_parity_mapping)
    
    # qubit measurement ################################################
    # the ideal process and the real process share the same outcomes,
    # as the actual action is based on the real outcomes.
    _accepted_measurement_outcome_pool: List[int]
    _measurement_outcome_pool: List[int]

    _qubit_projs_ideal: List[qt.Qobj]
    _qubit_projs_real: List[qt.Qobj]

    def qubit_measurement(
        self,
        step: int | str,
        graph: EvolutionTree,
        init_nodes: StateEnsemble,
    ) -> StateEnsemble:
        all_final_nodes = StateEnsemble()

        for idx in range(len(self._qubit_projs_ideal)):
            # final_node is trash if not accepted
            trashed = idx >= len(self._accepted_measurement_outcome_pool)
            edge = MeasurementEdge(
                name = f"MS ({step})",
                outcome = self._measurement_outcome_pool[idx],
                real_map = (
                    self._qubit_projs_real[idx] 
                    if not self.qubit_measurement_is_ideal 
                    else self._qubit_projs_ideal[idx]
                ),
                ideal_map = [self._qubit_projs_ideal[idx]],
            )              

            final_nodes = self._add_leaves(
                graph, init_nodes.active_nodes(), edge
            )

            for node in final_nodes.active_nodes():
                node.terminated = trashed

            all_final_nodes.nodes += final_nodes.nodes

        return all_final_nodes
    
    # qubit reset ######################################################
    @abstractmethod
    def _qubit_reset_map_real(
        self,
        meas_record: MeasurementRecord,
    ) -> qt.Qobj:
        pass
            
    @abstractmethod
    def _qubit_reset_map_ideal(
        self,
        meas_record: MeasurementRecord,
    ) -> List[qt.Qobj]:
        pass

    def qubit_reset(
        self,
        step: int | str,
        graph: EvolutionTree,
        init_nodes: StateEnsemble,
    ) -> StateEnsemble:
        final_nodes = StateEnsemble()

        for node in init_nodes.active_nodes():
            edge_qubit_reset = PropagatorEdge(
                f"RS ({step})",
                (
                    self._qubit_reset_map_real 
                    if not self.qubit_reset_is_ideal 
                    else self._kraus_to_super(self._qubit_reset_map_ideal)
                ),
                self._qubit_reset_map_ideal,
            )
            final_nodes.append(StateNode())

            graph.add_node(final_nodes[-1])

            graph.add_edge_connect(
                edge_qubit_reset,
                node,
                final_nodes[-1],
            )

        return final_nodes
    
    # check point ######################################################
    def check_point(
        self,
        step: int | str,
        graph: EvolutionTree,
        init_nodes: StateEnsemble,
    ) -> StateEnsemble:
        all_final_nodes = StateEnsemble()

        for success in [True, False]:
            edge_check_point = CheckPointEdge(
                name = f"CP ({step})",
                success = success,
            )
            final_nodes = self._add_leaves(
                graph, init_nodes.active_nodes(), edge_check_point
            )

            for node in final_nodes.active_nodes():
                node.terminated = not success

            all_final_nodes.nodes += final_nodes.nodes

        
        
        return all_final_nodes


class FullCatTreeBuilder(CatTreeBuilder):
    _res_mode_idx: int
    _qubit_mode_idx: int
    res_dim: int
    qubit_dim: int

    static_hamiltonian: qt.Qobj
    c_ops: List[qt.Qobj]
    esys: Tuple[np.ndarray, np.ndarray]
    frame_hamiltonian: qt.Qobj

    _accepted_measurement_outcome_pool = [0, 1]

    def __init__(
        self,
        fsweep: FlexibleSweep,
        sim_para: Dict[str, Any],
    ):
        self.fsweep = fsweep
        self.sim_para = sim_para

        self._find_sim_param()

        self._generate_cat_ingredients()

    def _generate_cat_ingredients(
        self,
    ):
        """
        Construct the static Hamiltonian and the collapse operators
        in the diagonalized & rotating frame.
        """
        (
            self.static_hamiltonian, self.c_ops, self.esys,
            self.frame_hamiltonian
        ) = cat_real.cavity_ancilla_me_ingredients(
            hilbertspace=self.fsweep.hilbertspace,
            res_mode_idx=self._res_mode_idx, qubit_mode_idx=self._qubit_mode_idx,
            res_truncated_dim=self.res_dim, qubit_truncated_dim=self.qubit_dim,
            collapse_parameters={
                "res_decay": self.fsweep["kappa_s"],
                "res_excite": self.fsweep["kappa_s"] * self.fsweep["n_th_s"],
                "res_dephase": 0,
                "qubit_decay": [
                    [0, self.fsweep["Gamma_up"]], 
                    [self.fsweep["Gamma_down"], 0]],
                "qubit_dephase": [0, self.fsweep["Gamma_phi"]]
            },
            in_rot_frame=True,
        )

        # change unit from GHz to rad / ns
        self.static_hamiltonian = self.static_hamiltonian * np.pi * 2
        self.frame_hamiltonian = self.frame_hamiltonian * np.pi * 2

    def _find_sim_param(
        self,
    ):
        """
        Find the resonator mode index and the truncated dimension
        """
        assert len(self.fsweep.hilbertspace.subsystem_list) == 2

        # determine the resonator mode index
        if type(self.fsweep.hilbertspace.subsystem_list[0]) == scq.Oscillator:
            self._res_mode_idx = 0
            self._qubit_mode_idx = 1
        elif type(self.fsweep.hilbertspace.subsystem_list[1]) == scq.Oscillator:
            self._res_mode_idx = 1
            self._qubit_mode_idx = 0
        else:
            raise ValueError("The Hilbert space does not contain a resonator.")
        
        # determine the truncated dimension
        self.res_dim = self.sim_para["res_dim"]
        self.qubit_dim = self.sim_para["qubit_dim"]

        # which qubit gate is used
        # qubit = self.fsweep.hilbertspace.subsystem_list[self._qubit_mode_idx]
        # if type(qubit) == scq.Transmon:
        #     self._gate_axis = "x"
        # elif type(qubit) == scq.Fluxonium:
        #     self._gate_axis = "x"
        # else:
        #     raise ValueError("Unknown qubit type.")
        self._gate_axis = "x"
        
    # overall properties ################################################
    @property
    def _total_simulation_time(self) -> float:
        return (
            self._idling_time
            + self._parity_mapping_time
            + self._qubit_gate_1_time
            + self._qubit_gate_2_time
            + self._qubit_measurement_time
            + self._qubit_reset_time
        )

    # idling ###########################################################
    def _idling_real_by_time(self, time: float) -> qt.Qobj:
        return cat_real.idling_propagator(
            self.static_hamiltonian,
            self.c_ops,
            float(time),    
            # float() is here because it will throw an error when time 
            # is a NamedSlotNdArray / ndarray object, because 
            # Qobj * array(1.0) will return a scipy sparse matrix
            # while Qobj * 1.0 returns a Qobj
        )
    
    def _idling_ideal_by_time(self, time: float) -> List[qt.Qobj]:
        return cat_ideal.idling_maps(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            res_mode_idx=self._res_mode_idx,
            static_hamiltonian=self.static_hamiltonian,
            time=time,
            decay_rate=self.fsweep["kappa_s"],
            self_Kerr=self.fsweep["K_s"],
        )
    
    def _build_idling_process(
        self,
    ):
        """
        Build ingredients for the idling process including 
        - the real propagator
        - the ideal (correctable) Kraus operators
        - the identity superoperator, used for many other processes
        """
        self._idling_real = self._idling_real_by_time(self._idling_time)
        self._idling_ideal = self._idling_ideal_by_time(self._idling_time)

        self._identity = cat_ideal.identity(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            res_mode_idx=self._res_mode_idx, superop=False,
        )   # may be used later

    @property
    def _idling_time(self) -> float:
        return float(self.fsweep["T_W"] * (1 - self.idling_is_ideal))

    # qubit gate #######################################################
    def frame_transform(
        self,
        propagator: qt.Qobj,
        total_time: float,
        type: Literal[
            "rot2current", 
            "current2rot", 
            "lab2current",
            "current2lab",
            "lab2rot",
            "rot2lab",
        ],
        init_time: float = 0,
    ) -> qt.Qobj:
        """
        Transform the propagator to the rotating frame.

        There are three frames that we need to consider:
        1. [lab] The lab frame.
        2. [rot] The rotating frame, basically all of the natural frequencies are set to 0.
        3. [current] The current frame, linear part of the natural frequencies are removed.
        """
        init, final = type.split("2")

        if init == "rot":
            init_ham = 0
        elif init == "current":
            init_ham = self.static_hamiltonian
        elif init == "lab":
            init_ham = self.frame_hamiltonian + self.static_hamiltonian

        if final == "rot":
            final_ham = 0
        elif final == "current":
            final_ham = self.static_hamiltonian
        elif final == "lab":
            final_ham = self.frame_hamiltonian + self.static_hamiltonian

        trans_ham = init_ham - final_ham

        prop_free = lambda t: (-1j * trans_ham * t).expm()
        return (
            prop_free(total_time + init_time).dag() 
            * propagator 
            * prop_free(init_time)
        )
    
    def _build_qubit_gate_process(
        self,
        num_cpus: int = 8,
    ):
        self._gate_p_rot_ideal = cat_ideal.qubit_rot_propagator(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            res_mode_idx=self._res_mode_idx,
            angle=np.pi/2, axis=self._gate_axis, superop=False,
        )   # p stands for angle's sign plus
        self._qubit_gate_p_ideal = [self.frame_transform(
            self._gate_p_rot_ideal, 
            total_time = self.fsweep["tau_p_eff"], 
            type = "rot2current",
            # init_time = -self.fsweep["tau_p_eff"] / 2,
        )]  
        self._gate_m_rot_ideal = cat_ideal.qubit_rot_propagator(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            res_mode_idx=self._res_mode_idx,
            angle=-np.pi/2, axis=self._gate_axis, superop=False,
        )
        self._qubit_gate_m_ideal = [self.frame_transform(
            self._gate_m_rot_ideal, 
            total_time = self.fsweep["tau_p_eff"], 
            type = "rot2current",
            # init_time = -self.fsweep["tau_p_eff"] / 2,
        )]

        self._gate_p_lab_real = cat_real.qubit_gate(
            self.fsweep.hilbertspace,
            self._res_mode_idx, self._qubit_mode_idx,
            self.res_dim, self.qubit_dim,
            # eigensys = self.esys, # self.esys is not full esys
            rotation_angle = np.pi / 2,
            gate_params = self.fsweep,
            num_cpus = num_cpus,
        )
        self._qubit_gate_p_real = self._kraus_to_super(
            [self.frame_transform(
                self._gate_p_lab_real, 
                total_time = self.fsweep["tau_p_eff"], 
                type = "lab2current",
                # init_time = -self.fsweep["tau_p_eff"] / 2,
            )]
        )
        self._gate_m_lab_real = cat_real.qubit_gate(
            self.fsweep.hilbertspace,
            self._res_mode_idx, self._qubit_mode_idx,
            self.res_dim, self.qubit_dim,
            # eigensys = self.esys,  # self.esys is not full esys
            rotation_angle = - np.pi / 2,
            gate_params = self.fsweep,
            num_cpus = num_cpus,
        )
        self._qubit_gate_m_real = self._kraus_to_super(
            [self.frame_transform(
                self._gate_m_lab_real, 
                total_time = self.fsweep["tau_p_eff"], 
                type = "lab2current",
                # init_time = -self.fsweep["tau_p_eff"] / 2,
            )]
        )

    # the qubit gate after parity mapping
    def _qubit_gate_2_map_real(
        self,
        meas_record: MeasurementRecord,
    ) -> qt.Qobj:
        """
        Different from the gate 1, the gate 2 depend on previous measurement results
        to minimize the possibility of being at the excited state.
        Try to keep the qubit at |0> while the parity isn't changed
        """
        if self._current_parity(meas_record) == 0:
            # even parity, use the opposite gate
            return self._qubit_gate_m_real
        else:
            # odd parity, use the same gate
            return self._qubit_gate_p_real
        
    def _qubit_gate_2_map_ideal(
        self,
        meas_record: MeasurementRecord,
    ) -> List[qt.Qobj]:
        """
        Different from the gate 1, the gate 2 depend on previous measurement results
        to minimize the possibility of being at the excited state.
        Try to keep the qubit at |0> while the parity isn't changed
        """
        if self._current_parity(meas_record) == 0:
            # even parity, use the opposite gate
            return self._qubit_gate_m_ideal
        else:
            # odd parity, use the same gate
            return self._qubit_gate_p_ideal

    @property
    def _qubit_gate_1_time(self) -> float:
        """Currently the gate time is the same for both ideal and real cases."""
        return float(self.fsweep["tau_p_eff"])
        # return float(self.fsweep["tau_p_eff"] * (1 - self.gate_1_is_ideal))
    
    @property
    def _qubit_gate_2_time(self) -> float:
        """Currently the gate time is the same for both ideal and real cases."""
        return float(self.fsweep["tau_p_eff"])
        # return float(self.fsweep["tau_p_eff"] * (1 - self.gate_2_is_ideal))

    # parity mapping ###################################################
    def _build_parity_mapping_process(
        self,
    ):
        self._parity_mapping_ideal = self._idling_ideal_by_time(self._parity_mapping_time)[:1]

        self._parity_mapping_real = self._idling_real_by_time(self._parity_mapping_time)

        # when self.parity_mapping_is_ideal == True, we should switch to the constructed
        # parity mapping, as the self._parity_mapping_ideal is not TP
        self._parity_mapping_constructed = [cat_ideal.parity_mapping_propagator(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            angle=self._parity_mapping_angle,
            res_mode_idx=self._res_mode_idx, superop=False,
        )]

    @property
    def _average_pm_interaction(self) -> float:
        n_bar = np.abs(self.fsweep["disp"])**2  # TODO: is actually time-dependent
        return float(
            self.fsweep["chi_sa"] 
            + (2 * n_bar - 1) * self.fsweep["chi_prime"]
        )

    @property
    def _parity_mapping_time(self) -> float:
        """
        Either real or ideal, the parity mapping takes the same amount of time.

        TODO: why is time = time - qubit_gate_time / 2?
        """
        t = (
            float(np.abs(np.pi / self._average_pm_interaction))
            # During the first (ideal) qubit gates we get a phase when 
            # transfroming to the current frame (linear)
            - self._qubit_gate_1_time
        )

        if t > self.fsweep["T_W"]:
            # usually because a unreasonable parameter is set, making the chi_sa too small
            warn("The parity mapping time is longer than the waiting time. Set the time to be the waiting time.\n")
            t = self.fsweep["T_W"]

        return t
    
    @property
    def _parity_mapping_angle(self) -> float:
        """
        Since a bit of phase has been accumulated during the qubit gates,
        we don't need to apply a full pi rotation.

        This property returns the actual rotation needed.
        """
        qubit_gate_time_tot = self._qubit_gate_1_time + self._qubit_gate_2_time
        return np.pi - self._average_pm_interaction * qubit_gate_time_tot / 2
    
    def parity_mapping(
        self,
        step: int | str,
        graph: EvolutionTree,
        init_nodes: StateEnsemble,
    ) -> StateEnsemble:
        """
        Overwrite the base class parity mapping - as when the parity_mapping_is_ideal 
        is true, the self._parity_mapping_ideal is not TP and can't serve as the 
        ideal map for actual process.
        """
        edge_parity_mapping = PropagatorEdge(
            name = f"PM ({step})",
            real_map = (
                self._parity_mapping_real
                if not self.parity_mapping_is_ideal 
                else self._kraus_to_super(self._parity_mapping_constructed)
            ),
            ideal_maps = (
                self._parity_mapping_ideal
                if not self.parity_mapping_is_ideal
                else self._parity_mapping_constructed
            )
        )

        return self._add_leaves(graph, init_nodes.active_nodes(), edge_parity_mapping)

    # qubit measurement ################################################
    def _build_qubit_measurement_process(
        self,
    ):
        # ideal process
        self._qubit_projs_constructed = cat_ideal.qubit_projectors(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            res_mode_idx=self._res_mode_idx, superop=False,
        )
        # multiply by the idling propagator to mimic the time it takes to measure
        idling_prop = self._idling_ideal_by_time(self._qubit_measurement_time)[0]
        self._qubit_projs_ideal = [
            proj * idling_prop for proj in self._qubit_projs_constructed
        ]

        # real process
        confusion_matrix = np.eye(self.qubit_dim)
        confusion_matrix[0, 0] = 1 - self.fsweep["M_ge"]
        confusion_matrix[0, 1] = self.fsweep["M_ge"]
        confusion_matrix[1, 0] = self.fsweep["M_eg"]
        confusion_matrix[1, 1] = 1 - self.fsweep["M_eg"]

        real_projs = cat_real.qubit_projectors(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            res_mode_idx=self._res_mode_idx,
            confusion_matrix=confusion_matrix,
            ensamble_average=True,  # currently, only support True because the length of the list should be the same as the ideal one
            superop=True,
        )
        idling_prop = self._idling_real_by_time(self._qubit_measurement_time)
        self._qubit_projs_real = [
            proj * idling_prop for proj in real_projs
        ]

        # all of the lists should have the same length
        if len(self._qubit_projs_ideal) != len(self._accepted_measurement_outcome_pool):
            # update the measurement outcome pool (if there are outcomes that are not accepted)
            self._measurement_outcome_pool = (
                self._accepted_measurement_outcome_pool
                + list(range(len(self._accepted_measurement_outcome_pool), len(self._qubit_projs_ideal)))
            )
            # check if there are duplicate values
            if len(self._measurement_outcome_pool) != len(set(self._measurement_outcome_pool)):
                raise ValueError("The measurement outcome pool should not contain duplicate values.")
            else:
                warn("The number of accepted measurement outcomes is not equal to"
                    f"defined projectors, use {self._measurement_outcome_pool} instead.\n")
        else:
            self._measurement_outcome_pool = self._accepted_measurement_outcome_pool

        assert len(self._qubit_projs_ideal) == len(self._qubit_projs_real)

    @property
    def _qubit_measurement_time(self) -> float:
        return float(self.fsweep["tau_m"] * (1 - self.qubit_measurement_is_ideal))

    def qubit_measurement(
        self,
        step: int | str,
        graph: EvolutionTree,
        init_nodes: StateEnsemble,
    ) -> StateEnsemble:
        all_final_nodes = StateEnsemble()

        for idx in range(len(self._qubit_projs_ideal)):
            # final_node is trash if not accepted
            trashed = idx >= len(self._accepted_measurement_outcome_pool)

            edge_qubit_measurement = MeasurementEdge(
                name = f"MS ({step})",
                outcome = self._measurement_outcome_pool[idx],
                real_map = (
                    self._qubit_projs_real[idx] 
                    if not self.qubit_measurement_is_ideal 
                    else qt.to_super(self._qubit_projs_constructed[idx])
                ),
                ideal_map = (
                    [self._qubit_projs_ideal[idx]]
                    if not self.qubit_measurement_is_ideal 
                    else [self._qubit_projs_constructed[idx]]
                ),
            )                    

            final_nodes = self._add_leaves(
                graph, init_nodes.active_nodes(), edge_qubit_measurement
            )

            for node in final_nodes.active_nodes():
                node.terminated = trashed

            all_final_nodes.nodes += final_nodes.nodes
            

        return all_final_nodes

    # qubit reset ######################################################
    def _build_qubit_reset_process(
        self,
    ):
        reset_rot_ideal = cat_ideal.qubit_rot_propagator(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            res_mode_idx=self._res_mode_idx,
            angle=np.pi, axis=self._gate_axis, superop=False,
        )
        self._qubit_reset_ideal = [self.frame_transform(
            reset_rot_ideal, 
            total_time = self.fsweep["tau_p_eff"] * 2, 
            type = "rot2current",
            # init_time = -self.fsweep["tau_p_eff"],
        )]

        reset_lab_real = cat_real.qubit_gate(
            self.fsweep.hilbertspace,
            self._res_mode_idx, self._qubit_mode_idx,
            self.res_dim, self.qubit_dim,
            eigensys = self.esys,
            rotation_angle = np.pi,
            gate_params = self.fsweep,
        )
        self._qubit_reset_real = self._kraus_to_super(
            [self.frame_transform(
                reset_lab_real, 
                total_time = self.fsweep["tau_p_eff"] * 2, 
                type = "lab2current",
                # init_time = -self.fsweep["tau_p_eff"],
            )]
        )

    def _qubit_reset_map_real(
        self,
        meas_record: MeasurementRecord,
    ) -> qt.Qobj:
        if meas_record[-1] == 1:
            return self._qubit_reset_real
        elif meas_record[-1] == 0:
            return qt.to_super(self._identity)
        else:
            # already failed
            return qt.to_super(self._identity)
            
    def _qubit_reset_map_ideal(
        self,
        meas_record: MeasurementRecord,
    ) -> List[qt.Qobj]:
        if meas_record[-1] == 1:
            return self._qubit_reset_ideal
        elif meas_record[-1] == 0:
            return [self._identity]
        else:
            # already failed
            return [self._identity]
        
    @property
    def _qubit_reset_time(self) -> float:
        return float(self.fsweep["tau_p_eff"] * 2 * (1 - self.qubit_reset_is_ideal))
        
    # overall generation ###############################################
    def build_all_processes(
        self,
    ):
        
        builds = [
            self._build_idling_process,
            self._build_qubit_gate_process,
            self._build_parity_mapping_process,
            self._build_qubit_measurement_process,
            self._build_qubit_reset_process,
        ]

        for build in tqdm(builds, disable=settings.PROGRESSBAR_DISABLED):
            build()

    def generate_tree(
        self,
        init_prob_amp_01: Tuple[float, float],
        logical_0: qt.Qobj,
        logical_1: qt.Qobj,
        QEC_rounds: int = 1,
        with_check_point: bool = False,
    ) -> EvolutionTree:

        # construct an operation list:
        edge_method_names = [
            "idle",
            "qubit_gate_1",
            "parity_mapping",
            "qubit_gate_2",
            "qubit_measurement",
            "qubit_reset",
        ]
        if with_check_point:
            # after each step, add a check point
            edge_w_check_point = []
            for method_name in edge_method_names:
                edge_w_check_point.append(method_name)
                edge_w_check_point.append("check_point")
            edge_method_names = edge_w_check_point

        # generate the tree
        graph = EvolutionTree()

        # initial node
        init_state_node = StateNode.initial_note(
            init_prob_amp_01, logical_0, logical_1,
        )
        graph.add_node(init_state_node)

        # update on the current ensemble by adding edges on it
        current_ensemble = StateEnsemble([init_state_node])
        step_counter = 0
        for round in range(QEC_rounds):
            for method_name in edge_method_names:
                current_ensemble = getattr(self, method_name)(
                    f"{round}.{step_counter}", graph, current_ensemble,
                )
                step_counter += 1
            
        return graph

    def generate_wo_QEC(
        self,
        init_prob_amp_01: Tuple[float, float],
        logical_states: np.ndarray[qt.Qobj],
        QEC_rounds: int = 1,
        with_check_point: bool = False,
    ) -> EvolutionTree:
        """
        Currently, it only contains the idling process.

        logical_states: np.ndarray[qt.Qobj]
            a 2D array, each row is a pair of logical states
        """
        graph = EvolutionTree()

        # initial node
        logical_0, logical_1 = logical_states[0]
        init_state_node = StateNode.initial_note(
            init_prob_amp_01, logical_0, logical_1,
        )
        init_state_node.ideal_logical_states = logical_states
        graph.add_node(init_state_node)

        # current ensemble
        current_ensemble = StateEnsemble([init_state_node])

        # add the idling edge
        step_counter = 0
        for round in range(QEC_rounds):
            current_ensemble = self.idle(
                f"{round}.{step_counter}", graph, current_ensemble,
                correctable_single_photon_loss=False,
            )
            step_counter += 1

            if with_check_point:
                current_ensemble = self.check_point(
                    f"{round}.{step_counter}", graph, current_ensemble,
                )
                step_counter += 1

        return graph
    

class KerrTreeBuilder(CatTreeBuilder):
    """
    Cat code in a Kerr oscillator

    Parameters
    ----------
    para: Dict[str, Any]
        System and protocol parameters. Should include keys:
        "K", "lbd", "M_ge", "M_eg",
        "T_W",
    sim_para: Dict[str, Any]
        Simulation parameters. Should include keys:
        "res_dim"
    """

    def __init__(
        self,
        para: Dict[str, Any],
        sim_para: Dict[str, Any],
    ):
        self.para = para
        self.sim_para = sim_para

        self._generate_cat_ingredients()
        self.build_all_processes()

    def _generate_cat_ingredients(
        self,
    ):
        self._a_op = qt.destroy(self.sim_para["res_dim"])
        self._n_op = qt.num(self.sim_para["res_dim"])
        self._c_ops = [np.sqrt(self.para["lbd"]) * self._a_op]

    def build_all_processes(
        self,
    ):
        builds = [
            self._build_idling_process,
            self._build_parity_measurement_process,
        ]

        for build in tqdm(builds, disable=settings.PROGRESSBAR_DISABLED):
            build()

    # idling ###########################################################
    def _H_K(self) -> qt.Qobj:
        return self.para["K"] * self._a_op.dag()**2 * self._a_op**2

    def _U_K(self, time: float) -> qt.Qobj:
        return (-1j * time * self._H_K()).expm()
    
    def _U_R(self, theta: float) -> qt.Qobj:
        return (-1j * theta * self._n_op).expm()
    
    def _T_K(self, time: float) -> qt.Qobj:
        return self._U_K(time) * (
            -1/2 * self.para["lbd"] * time * self._n_op
        ).expm()
    
    def _idling_liouv(self, time: float) -> qt.Qobj:
        liouv = qt.liouvillian(self._H_K(), self._c_ops)
        return (liouv * time).expm()

    def _build_idling_process(self):
        """
        Idling process for the cat code in a Kerr oscillator.
        """
        self._idling_real = self._idling_liouv(self.para["T_W"])

        # the following content is the same as cat_ideal.idling_maps(...)
        # no jump 
        free_evolution_oprt = self._T_K(self.para["T_W"])

        # single-photon loss related operators
        spl_rotation_oprt = self._U_R(
            self.para["K"] * self.para["T_W"]
        )   # average rotation due to self-Kerr

        self._idling_ideal = [
            free_evolution_oprt,
            free_evolution_oprt * spl_rotation_oprt * self._a_op    
        ]

    # parity measurement ###############################################
    _measurement_outcome_pool = [0, 1]
    _accepted_measurement_outcome_pool = [0, 1]

    def _build_parity_measurement_process(
        self,
    ):
        """
        Measure the parity of the cavity with a probability of making 
        assignment error.
        """

        # ideal process
        L = self.sim_para["res_dim"]
        alternating_list = np.tile([1, 0], L // 2 + 1)
        self._parity_projs_ideal = [
            qt.qdiags(alternating_list[:L]),
            qt.qdiags(alternating_list[1:L+1]),
        ]

        # real process
        confusion_matrix = np.eye(2)
        confusion_matrix[0, 0] = 1 - self.para["M_ge"]
        confusion_matrix[0, 1] = self.para["M_ge"]
        confusion_matrix[1, 0] = self.para["M_eg"]
        confusion_matrix[1, 1] = 1 - self.para["M_eg"]

        measurement_ops = np.empty_like(confusion_matrix, dtype=object)
        for idx, prob in np.ndenumerate(confusion_matrix):
            measurement_ops[idx] = prob * qt.to_super(
                self._parity_projs_ideal[idx[1]])
        self._parity_projs_real = np.sum(measurement_ops, axis=1).tolist()

    def parity_measurement(
        self,
        step: int | str,
        graph: EvolutionTree,
        init_nodes: StateEnsemble,
    ) -> StateEnsemble:
        all_final_nodes = StateEnsemble()

        for idx in range(len(self._parity_projs_ideal)):
            # final_node is trash if not accepted
            trashed = idx >= len(self._accepted_measurement_outcome_pool)

            edge_parity_measurement = MeasurementEdge(
                name = f"MS ({step})",
                outcome = self._measurement_outcome_pool[idx],
                real_map = self._parity_projs_real[idx],
                ideal_map = [self._parity_projs_ideal[idx]],
            )                    

            final_nodes = self._add_leaves(
                graph, init_nodes.active_nodes(), edge_parity_measurement
            )

            for node in final_nodes.active_nodes():
                node.terminated = trashed

            all_final_nodes.nodes += final_nodes.nodes

        return all_final_nodes
    
    def generate_tree(
        self,
        init_prob_amp_01: Tuple[float, float],
        logical_0: qt.Qobj,
        logical_1: qt.Qobj,
        QEC_rounds: int = 1,
        with_check_point: bool = False,
    ) -> EvolutionTree:
        graph = EvolutionTree()

        # initial node
        init_state_node = StateNode.initial_note(
            init_prob_amp_01, logical_0, logical_1,
        )
        graph.add_node(init_state_node)

        # current ensemble
        current_ensemble = StateEnsemble([init_state_node])

        # add edges step by step
        edge_method_names = [
            "idle",
            "parity_measurement",
        ]
        if with_check_point:
            # after each step, add a check point
            edge_w_check_point = []
            for method_name in edge_method_names:
                edge_w_check_point.append(method_name)
                edge_w_check_point.append("check_point")
            edge_method_names = edge_w_check_point

        step_counter = 0
        for round in range(QEC_rounds):
            for method_name in edge_method_names:
                current_ensemble = getattr(self, method_name)(
                    f"{round}.{step_counter}", graph, current_ensemble,
                )
                step_counter += 1
            
        return graph
    
    # other unused abstract methods ####################################
    def _qubit_gate_2_map_ideal(self): pass
    def _qubit_gate_2_map_real(self): pass
    def _qubit_gate_2_time(self): pass
    def _qubit_measurement_time(self): pass
    def _qubit_reset_map_ideal(self): pass
    def _qubit_reset_map_real(self): pass