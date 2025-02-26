__all__ = [
    'FullCatTreeBuilder',
    'KerrTreeBuilder',
]

from typing import List, Tuple, Any, TYPE_CHECKING, Dict, Callable, overload, Literal
from warnings import warn
from copy import deepcopy
from tqdm.notebook import tqdm
from abc import ABC, abstractmethod

import numpy as np
import qutip as qt
import scqubits as scq

from chencrafts.cqed import FlexibleSweep, superop_evolve

import chencrafts.bsqubits.cat_ideal as cat_ideal
import chencrafts.bsqubits.cat_recipe as cat_recipe
import chencrafts.bsqubits.cat_real as cat_real
import chencrafts.settings as global_settings
import chencrafts.bsqubits.QEC_graph.settings as graph_settings


from .node import StateNode, StateEnsemble, MeasurementRecord
from .edge import (
    PropagatorEdge, MeasurementEdge, Edge, CheckPointEdge)
from .graph import EvolutionGraph, EvolutionTree

class CatTreeBuilder(ABC):

    graph = EvolutionGraph()

    # ideal_process_switches
    idling_is_ideal: bool = False
    gate_1_is_ideal: bool = True
    parity_mapping_is_ideal: bool = False
    gate_2_is_ideal: bool = True
    qubit_measurement_is_ideal: bool = False
    qubit_reset_is_ideal: bool = True
    
    # ideal_process_type
    use_improved_gate: bool = True
    use_improved_parity_mapping: bool = True

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
    res_me_dim: int
    qubit_me_dim: int
    
    # the integer part of the mean photon number
    # partially determines the frame transformation
    n_bar_ref: int  
    
    c_ops: List[qt.Qobj]
    partial_esys: Tuple[np.ndarray, np.ndarray]
    frame_hamiltonian: qt.Qobj

    _accepted_measurement_outcome_pool = [0, 1]

    def __init__(
        self,
        fsweep: FlexibleSweep,
        sim_para: Dict[str, Any],
        new_recipe: bool = False,
    ):
        self.fsweep = fsweep
        self.sim_para = sim_para
        self.new_recipe = new_recipe

        self._find_sim_param()
        self._generate_cat_ingredients()
        
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

        # if exist, store the truncated dim for dynamical simulation
        try:
            self.res_me_dim = self.sim_para["res_me_dim"]
            self.qubit_me_dim = self.sim_para["qubit_me_dim"]
        except KeyError:
            pass

        # which qubit gate is used
        # qubit = self.fsweep.hilbertspace.subsystem_list[self._qubit_mode_idx]
        # if type(qubit) == scq.Transmon:
        #     self._gate_axis = "x"
        # elif type(qubit) == scq.Fluxonium:
        #     self._gate_axis = "x"
        # else:
        #     raise ValueError("Unknown qubit type.")
        self._gate_axis = "x"
        
    def _generate_cat_ingredients(
        self,
    ):
        """
        Construct the static Hamiltonian and the collapse operators
        in the diagonalized & rotating frame.
        """
        if not self.new_recipe:
            (
                self.static_hamiltonian, self.c_ops, self.partial_esys,
                self.frame_hamiltonian
            ) = cat_real.cavity_ancilla_me_ingredients(
                hilbertspace=self.fsweep.hilbertspace,
                res_mode_idx=self._res_mode_idx, 
                qubit_mode_idx=self._qubit_mode_idx,
                res_truncated_dim=self.res_dim, 
                qubit_truncated_dim=self.qubit_dim,
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
        else:
            self.n_bar = np.abs(self.fsweep["disp"])**2
            self.n_bar_ref = int(np.round(self.n_bar))  
            # reference photon number, determining the rotating frame 
            
            (
                self.static_hamiltonian, self.c_ops, self.partial_esys,
                self.frame_hamiltonian
            ) = cat_recipe.cavity_ancilla_me_ingredients(
                fsweep = self.fsweep, 
                res_mode_idx=self._res_mode_idx,
                qubit_mode_idx=self._qubit_mode_idx, 
                res_dim=self.res_dim,
                qubit_dim=self.qubit_dim, 
                res_me_dim=self.res_me_dim,
                qubit_me_dim=self.qubit_me_dim,
                in_rot_frame=True,
                res_n_ref=self.n_bar_ref,
            )
        
        # esys
        self.full_esys = (self.fsweep["evals"], self.fsweep["evecs"])
        self.full_dressed_indices = self.fsweep["dressed_indices"]

        # change unit from GHz to rad / ns
        self.static_hamiltonian = self.static_hamiltonian * np.pi * 2
        self.frame_hamiltonian = self.frame_hamiltonian * np.pi * 2
        
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
            decay_rate=self.fsweep["jump_a"] if self.new_recipe else self.fsweep["kappa_s"],
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
        self._gate_p_lab_real = cat_real.qubit_gate(
            self.fsweep.hilbertspace,
            self._res_mode_idx, self._qubit_mode_idx,
            self.res_dim, self.qubit_dim,
            eigensys = self.full_esys,
            dressed_indices = self.full_dressed_indices,
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
            eigensys = self.full_esys,
            dressed_indices = self.full_dressed_indices,
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
        
        # ideal process: a pi rotation in the current frame
        gate_p_ideal = cat_ideal.qubit_rot_propagator(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            res_mode_idx=self._res_mode_idx,
            angle=np.pi/2, axis=self._gate_axis, superop=False,
        )   # p stands for angle's sign plus
        gate_m_ideal = cat_ideal.qubit_rot_propagator(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            res_mode_idx=self._res_mode_idx,
            angle=-np.pi/2, axis=self._gate_axis, superop=False,
        )
        
        # resonator phase (global phase in the submat gate)
        assert self._qubit_gate_1_rabi_freq == self._qubit_gate_2_rabi_freq
        res_angle = self._average_pm_interaction * (
            self._qubit_gate_1_time - 1 / self._qubit_gate_1_rabi_freq
        ) / 2
        gate_res_rot_op = cat_ideal.res_rotation(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            res_mode_idx=self._res_mode_idx,
            angle=res_angle,
        ) * np.exp(1j * res_angle * self.n_bar_ref)

        # partial parity mapping (ideal)
        # note that the angle at submat = 0 is zero, while in the current frame 
        # submat = n_bar should have zero angle, so it should be corrected by a 
        # qubit phase gate
        gate_pm_op = self._parity_mapping_ideal_n_ref_frame(
            angle=self._average_pm_interaction / 2 / self._qubit_gate_1_rabi_freq,
            pm_qubit_state=1,
        )
        
        # Kerr rotation during the qubit gate
        gate_res_Kerr_op_1 = cat_ideal.res_Kerr_rotation(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            res_mode_idx=self._res_mode_idx,
            Kerr=self.fsweep["K_s"], time=self._qubit_gate_1_time,
        )
        gate_res_Kerr_op_2 = self._Kerr_rotation_n_ref_frame(
            angle=self.fsweep["chi_prime"]/2 * self._qubit_gate_1_time,
        )

        if self.use_improved_gate:
            self._qubit_gate_p_ideal = [
                gate_res_Kerr_op_1 * gate_res_Kerr_op_2
                * gate_res_rot_op * gate_pm_op * gate_p_ideal * gate_pm_op
            ]
            self._qubit_gate_m_ideal = [
                gate_res_Kerr_op_1 * gate_res_Kerr_op_2
                * gate_res_rot_op * gate_pm_op * gate_m_ideal * gate_pm_op
            ]
        else:
            self._qubit_gate_p_ideal = [gate_p_ideal]
            self._qubit_gate_m_ideal = [gate_m_ideal]

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
        
    @property
    def _qubit_gate_1_rabi_freq(self) -> float:
        return np.pi / 4 / self._qubit_gate_1_time
    
    @property
    def _qubit_gate_2_rabi_freq(self) -> float:
        return np.pi / 4 / self._qubit_gate_2_time

    # parity mapping ###################################################
    def _build_parity_mapping_process(
        self,
    ):
        self._parity_mapping_real = self._idling_real_by_time(self._parity_mapping_time)

        # constructed parity mapping
        # note that the angle at submat = 0 is zero, while in the current frame 
        # submat = n_bar should have zero angle, so it should be corrected by a 
        # qubit phase gate
        parity_mapping_ideal = self._parity_mapping_ideal_n_ref_frame(
            angle=self._parity_mapping_angle,
            pm_qubit_state=1,
        )

        # add resonator non-linearity, induced from Kerr and chi_prime
        pm_res_Kerr_op_1 = cat_ideal.res_Kerr_rotation(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            res_mode_idx=self._res_mode_idx,
            Kerr=self.fsweep["K_s"], time=self._parity_mapping_time,
        )
        pm_res_Kerr_op_2 = self._Kerr_rotation_n_ref_frame(
            angle=self.fsweep["chi_prime"]/2 * self._parity_mapping_time,
        )
        
        if self.use_improved_parity_mapping:
            self._parity_mapping_ideal = [
                pm_res_Kerr_op_1 * pm_res_Kerr_op_2 * parity_mapping_ideal
            ]
        else:
            self._parity_mapping_ideal = [
                self._parity_mapping_ideal_n_ref_frame(
                    angle=np.pi,
                    pm_qubit_state=1,
                )
            ]

    @property
    def _average_pm_interaction(self) -> float:
        """
        Current implementation:
        we compute the local dispersive shift per photon at n = n_bar.
        """
        # dispersive shift = chi * n + chi_prime * n * (n - 1)
        # we take the average by the local derivative at n_bar
        return float(
            self.fsweep["chi_sa"] 
            + (2 * self.n_bar - 1) * self.fsweep["chi_prime"]
        )
        
        # # another option:
        # # take the dispersive shift difference between n = n_bar and n = n_bar - 0
        # return float(
        #     self.fsweep["chi_sa"] * n_bar
        #     + n_bar * (n_bar - 1) * self.fsweep["chi_prime"]
        # ) / n_bar   # divide by n_bar to get the frequency shift per photon

    @property
    def _parity_mapping_angle(self) -> float:
        """
        The angle has the same sign as the _average_pm_interaction,
        the partial angle accumulated in the qubit gate is subtracted.
        """
        if self._average_pm_interaction >= 0:
            full_angle = np.pi
        else:
            full_angle = -np.pi
            
        # calculate the angle of the partial parity mapping from the qubit gate
        partial_angle = (
            self._average_pm_interaction / 2 / self._qubit_gate_1_rabi_freq
            + self._average_pm_interaction / 2 / self._qubit_gate_2_rabi_freq
        )
        return full_angle - partial_angle

    @property
    def _parity_mapping_time(self) -> float:
        """
        Either real or ideal, the parity mapping takes the same amount of time.
        """
        t = float(self._parity_mapping_angle / self._average_pm_interaction)

        if t > self.fsweep["T_W"]:
            # usually because a unreasonable parameter is set, making the chi_sa too small
            if graph_settings.ISSUE_WARNING:
                warn("The parity mapping time is longer than the waiting time. Set the time to be the waiting time.\n")
            t = self.fsweep["T_W"]

        return t
    
    def _parity_mapping_ideal_n_ref_frame(
        self,
        angle: float,
        pm_qubit_state: int = 1,
    ) -> qt.Qobj:
        """
        Construct the parity mapping operator with arbitrary angle:
        exp (
            -1j * angle 
            * (adag * a - n_bar_ref) 
            * |pm_qubit_state><pm_qubit_state|
        )
        
        In the current frame, the pm angle at subsystem with resonator
        photon = n_bar_int is zero, while in the constructed propagator, the 
        angle at photon = 0 is zero.
        So we need to add a qubit rotation to correct the phase.
        
        The phase accumulated is put on states with qubit state = pm_qubit_state.
        """
        assert pm_qubit_state in [0, 1]
        
        parity_mapping_ideal = cat_ideal.parity_mapping_propagator(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            angle=angle,
            res_mode_idx=self._res_mode_idx, superop=False,
            qubit_state=pm_qubit_state,
        )

        # transform to the current frame 
        # equivalently, set the propagator submat at n_bar_int to be identity
        extra_phase = - (angle * self.n_bar_ref) % (2 * np.pi)
        qubit_rot_ideal = cat_ideal.qubit_rot_propagator(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            res_mode_idx=self._res_mode_idx,
            axis="z" if pm_qubit_state == 0 else "-z",  
            # make 0 or 1 have the lower eigenvalue (-1/2)
            angle= extra_phase,
            superop=False,
        ) * np.exp(
            -1 *        # cancel out the phase for the ground state
            -1/2 *      # ground state eigenvalue
            -1j *       # exp(-1j * phase) natural evolution
            extra_phase
        )

        return parity_mapping_ideal * qubit_rot_ideal
    
    def _Kerr_rotation_n_ref_frame(
        self,
        angle: float,
    ) -> qt.Qobj:
        """
        Construct the operator:
        exp (
            -1j * angle 
            * (adag * a - n_bar_ref)**2 
        )
        """
        res_num_op = cat_ideal.res_number(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            res_mode_idx=self._res_mode_idx,
        ) - self.n_bar_ref
        return (
            -1j * angle 
            * (res_num_op)**2
        ).expm()
        
    # def parity_mapping(
    #     self,
    #     step: int | str,
    #     graph: EvolutionTree,
    #     init_nodes: StateEnsemble,
    # ) -> StateEnsemble:
    #     """
    #     Overwrite the base class parity mapping - as when the parity_mapping_is_ideal 
    #     is true, the self._parity_mapping_ideal is not TP and can't serve as the 
    #     ideal map for actual process.
    #     """
    #     edge_parity_mapping = PropagatorEdge(
    #         name = f"PM ({step})",
    #         real_map = (
    #             self._parity_mapping_real
    #             if not self.parity_mapping_is_ideal 
    #             else self._kraus_to_super(self._parity_mapping_constructed)
    #         ),
    #         ideal_maps = (
    #             self._parity_mapping_ideal
    #             if not self.parity_mapping_is_ideal
    #             else self._parity_mapping_constructed
    #         )
    #     )

    #     return self._add_leaves(graph, init_nodes.active_nodes(), edge_parity_mapping)

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
                if graph_settings.ISSUE_WARNING:
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
        num_cpus: int = 8,
    ):
        reset_lab_real = cat_real.qubit_gate(
            self.fsweep.hilbertspace,
            self._res_mode_idx, self._qubit_mode_idx,
            self.res_dim, self.qubit_dim,
            eigensys = self.full_esys,
            dressed_indices = self.full_dressed_indices,
            rotation_angle = np.pi,
            gate_params = self.fsweep,
            num_cpus = num_cpus,
        )
        self._qubit_reset_real = self._kraus_to_super(
            [self.frame_transform(
                reset_lab_real, 
                total_time = self._qubit_reset_time, 
                type = "lab2current",
            )]
        )
        
        reset_ideal = cat_ideal.qubit_rot_propagator(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            res_mode_idx=self._res_mode_idx,
            angle=np.pi, axis=self._gate_axis, superop=False,
        )
        # resonator phase (global phase from the submat gate)
        res_rot_op = cat_ideal.res_rotation(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            res_mode_idx=self._res_mode_idx,
            angle=self._average_pm_interaction * self._qubit_reset_time / 2,
        )
        # resonator non-linearity, induced from Kerr and chi_prime
        gate_res_Kerr_op_1 = cat_ideal.res_Kerr_rotation(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            res_mode_idx=self._res_mode_idx,
            Kerr=self.fsweep["K_s"], time=self._qubit_reset_time,
        )
        gate_res_Kerr_op_2 = self._Kerr_rotation_n_ref_frame(
            angle=self.fsweep["chi_prime"]/2 * self._qubit_reset_time,
        )
        
        if self.use_improved_gate:
            # no need to transform to the current frame
            self._qubit_reset_ideal = [
                gate_res_Kerr_op_1 * gate_res_Kerr_op_2 * res_rot_op 
                * reset_ideal
            ]
        else:
            self._qubit_reset_ideal = [reset_ideal]

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
        num_cpus: int = 8,
    ):
        
        builds = [
            self._build_idling_process,
            self._build_qubit_gate_process,
            self._build_parity_mapping_process,
            self._build_qubit_measurement_process,
            self._build_qubit_reset_process,
        ]

        for build in tqdm(builds, disable=global_settings.PROGRESSBAR_DISABLED):
            try:
                build(num_cpus=num_cpus)
            except TypeError as e:
                if "num_cpus" in str(e):
                    # unexpected keyword argument num_cpus
                    build()
                else:
                    raise e

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
        init_state_node._raw_ideal_logical_states = logical_states
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

        for build in tqdm(builds, disable=global_settings.PROGRESSBAR_DISABLED):
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