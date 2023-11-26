import numpy as np
import qutip as qt
import scqubits as scq


from chencrafts.cqed import FlexibleSweep, superop_evolve

from chencrafts.bsqubits.QEC_graph.node import StateNode, TrashNode, StateEnsemble, MeasurementRecord
from chencrafts.bsqubits.QEC_graph.edge import PropagatorEdge, MeasurementEdge, Edge, TrashEdge
from chencrafts.bsqubits.QEC_graph.graph import EvolutionGraph, EvolutionTree

import chencrafts.bsqubits.cat_ideal as cat_ideal
import chencrafts.bsqubits.cat_real as cat_real

from tqdm.notebook import tqdm
from typing import List, Tuple, Any, TYPE_CHECKING, Dict, Callable, overload
from warnings import warn


class GraphBuilder:

    graph = EvolutionGraph()

    def __init__(
        self,
        fsweep: FlexibleSweep,
        sim_para: Dict[str, Any],
    ):
        self.fsweep = fsweep
        self.sim_para = sim_para


class CatGraphBuilder(GraphBuilder):
    _res_mode_idx: int
    _qubit_mode_idx: int
    res_dim: int
    qubit_dim: int

    static_hamiltonian: qt.Qobj
    c_ops: List[qt.Qobj]
    esys: Tuple[np.ndarray, np.ndarray]
    frame_hamiltonian: qt.Qobj

    # the ideal process and the real process share the same outcomes,
    # as the actual action is based on the real outcomes.
    _accepted_measurement_outcome_pool = [0, 1]
    _measurement_outcome_pool: List

    # processes:
    _identity: qt.Qobj
    _idling_real: qt.Qobj
    _idling_ideal: List[qt.Qobj]
    _qubit_gate_m_ideal: List[qt.Qobj]
    _qubit_gate_p_ideal: List[qt.Qobj]
    _qubit_gate_m_real: qt.Qobj
    _qubit_gate_p_real: qt.Qobj
    _parity_mapping_ideal: List[qt.Qobj]
    _parity_mapping_real: qt.Qobj
    _qubit_projs_ideal: List[qt.Qobj]
    _qubit_projs_real: List[qt.Qobj]
    _qubit_reset_real: qt.Qobj
    _qubit_reset_ideal: [qt.Qobj]

    # ideal_process_switches
    idling_is_ideal: bool = False
    gate_1_is_ideal: bool = True
    parity_mapping_is_ideal: bool = False
    gate_2_is_ideal: bool = True
    qubit_measurement_is_ideal: bool = False
    qubit_reset_is_ideal: bool = True

    def __init__(
        self,
        fsweep: FlexibleSweep,
        sim_para: Dict[str, Any],
    ):
        super().__init__(fsweep, sim_para)

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
        if isinstance(qobj_list, list):
            return sum([qt.to_super(qobj) for qobj in qobj_list])
        elif callable(qobj_list):
            return lambda meas_record: sum([qt.to_super(qobj) for qobj in qobj_list(meas_record)])
        else:
            raise TypeError("The input should be a list of Qobj or a function.")
        
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

    def idle(
        self,
        step: int | str,
        graph: EvolutionTree,
        init_nodes: StateEnsemble,
    ) -> StateEnsemble:
        """
        Idling process.

        Add one idling edge to every node in the initial ensemble.
        """

        final_nodes = StateEnsemble()

        for node in init_nodes:
            edge_idling = PropagatorEdge(
                f"idling ({step})",
                self._idling_real,  # when ideal, time=0, it's already the identity, no need to change
                self._idling_ideal,
            )
            final_nodes.append(StateNode())

            graph.add_node(final_nodes[-1])

            graph.add_edge_connect(
                edge_idling,
                node,
                final_nodes[-1],
            )

        return final_nodes
    
    # qubit gate #######################################################
    def frame_transform(
        self,
        propagator: qt.Qobj,
        time: float,
    ) -> qt.Qobj:
        """
        Transform the propagator to the rotating frame.
        """
        prop_free = lambda t: (-1j * self.frame_hamiltonian * t).expm()
        return prop_free(time) * propagator * prop_free(0)
    
    def _build_qubit_gate_process(
        self,
    ):
        self._qubit_gate_p_ideal = [cat_ideal.qubit_rot_propagator(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            res_mode_idx=self._res_mode_idx,
            angle=np.pi/2, axis="x", superop=False,
        )]   # p stands for angle's sign plus
        self._qubit_gate_m_ideal = [cat_ideal.qubit_rot_propagator(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            res_mode_idx=self._res_mode_idx,
            angle=-np.pi/2, axis="x", superop=False,
        )]

        _gate_p_lab_frame = cat_real.qubit_gate(
            self.fsweep.hilbertspace,
            self._res_mode_idx, self._qubit_mode_idx,
            self.res_dim, self.qubit_dim,
            # eigensys = self.esys, # self.esys is not full esys
            rotation_angle = np.pi / 2,
            gate_params = self.fsweep,
        )
        self._qubit_gate_p_real = self._kraus_to_super(
            self.frame_transform(_gate_p_lab_frame, self.fsweep["tau_p_eff"])
        )
        _qubit_gate_m_real = cat_real.qubit_gate(
            self.fsweep.hilbertspace,
            self._res_mode_idx, self._qubit_mode_idx,
            self.res_dim, self.qubit_dim,
            # eigensys = self.esys,  # self.esys is not full esys
            rotation_angle = - np.pi / 2,
            gate_params = self.fsweep,
        )
        self._qubit_gate_m_real = self._kraus_to_super(
            self.frame_transform(_qubit_gate_m_real, self.fsweep["tau_p_eff"])
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
        return float(self.fsweep["tau_p_eff"] * (1 - self.gate_1_is_ideal))
    
    @property
    def _qubit_gate_2_time(self) -> float:
        return float(self.fsweep["tau_p_eff"] * (1 - self.gate_2_is_ideal))

    def qubit_gate_1(
        self,
        step: int | str,
        graph: EvolutionTree,
        init_nodes: StateEnsemble,
    ) -> StateEnsemble:
        final_nodes = StateEnsemble()

        for node in init_nodes:
            edge_qubit_gate = PropagatorEdge(
                f"qubit_gate_1 ({step})",
                (self._qubit_gate_p_real if not self.gate_1_is_ideal 
                    else self._kraus_to_super(self._qubit_gate_p_ideal)),
                self._qubit_gate_p_ideal,
            )
            final_nodes.append(StateNode())

            graph.add_node(final_nodes[-1])

            graph.add_edge_connect(
                edge_qubit_gate,
                node,
                final_nodes[-1],
            )

        return final_nodes
    
    def qubit_gate_2(
        self,
        step: int | str,
        graph: EvolutionTree,
        init_nodes: StateEnsemble,
    ) -> StateEnsemble:
        final_nodes = StateEnsemble()

        for node in init_nodes:
            edge_qubit_gate = PropagatorEdge(
                f"qubit_gate_2 ({step})",
                (self._qubit_gate_2_map_real if not self.gate_2_is_ideal 
                    else self._kraus_to_super(self._qubit_gate_2_map_ideal)),
                self._qubit_gate_2_map_ideal,
            )
            final_nodes.append(StateNode())

            graph.add_node(final_nodes[-1])

            graph.add_edge_connect(
                edge_qubit_gate,
                node,
                final_nodes[-1],
            )

        return final_nodes
    
    # parity mapping ###################################################
    def _build_parity_mapping_process(
        self,
    ):
        # previous implementation: a constructed parity mapping
        # self._parity_mapping_ideal = [cat_ideal.parity_mapping_propagator(
        #     res_dim=self.res_dim, qubit_dim=self.qubit_dim,
        #     res_mode_idx=self._res_mode_idx, superop=False,
        # )]

        # new implementation: actual idling evolution
        self._parity_mapping_ideal = self._idling_ideal_by_time(self._parity_mapping_time)[:1]

        self._parity_mapping_real = self._idling_real_by_time(self._parity_mapping_time)

    @property
    def _parity_mapping_time(self) -> float:
        """
        Either real or ideal, the parity mapping takes the same amount of time.
        """
        t = (
            float(np.abs(np.pi / self.fsweep["chi_sa"]))
            # half of the qubit gate time is used for parity mapping
            - self._qubit_gate_1_time / 2
            - self._qubit_gate_2_time / 2
        )

        if t > self.fsweep["T_W"]:
            # usually because a unreasonable parameter is set, making the chi_sa too small
            warn("The parity mapping time is longer than the waiting time. Set the time to be the waiting time.")
            t = self.fsweep["T_W"]

        return t

    def parity_mapping(
        self,
        step: int | str,
        graph: EvolutionTree,
        init_nodes: StateEnsemble,
    ) -> StateEnsemble:
        final_nodes = StateEnsemble()

        for node in init_nodes:
            edge_parity_mapping = PropagatorEdge(
                name = f"parity_mapping ({step})",
                real_map = (self._parity_mapping_real
                    if not self.parity_mapping_is_ideal 
                    else self._kraus_to_super(self._parity_mapping_ideal)),
                ideal_maps = self._parity_mapping_ideal,
            )
            final_nodes.append(StateNode())

            graph.add_node(final_nodes[-1])

            graph.add_edge_connect(
                edge_parity_mapping,
                node,
                final_nodes[-1],
            )

        return final_nodes

    # qubit measurement ################################################
    def _build_qubit_measurement_process(
        self,
    ):
        # ideal process
        ideal_projs = cat_ideal.qubit_projectors(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            res_mode_idx=self._res_mode_idx, superop=False,
        )
        # multiply by the idling propagator to mimic the time it takes to measure
        idling_prop = self._idling_ideal_by_time(self._qubit_measurement_time)[0]
        self._qubit_projs_ideal = [
            proj * idling_prop for proj in ideal_projs
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
                    f"defined projectors, use {self._measurement_outcome_pool} instead.")
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
        trash_node: TrashNode,
    ) -> StateEnsemble:
        """
        Special case: when the measurement is not accepted, the init node is 
        then connected to a trash node.
        """
        final_nodes = StateEnsemble()

        for node in init_nodes:
            for idx in range(len(self._qubit_projs_ideal)):
                # final_node is trash if not accepted
                trashed = idx >= len(self._accepted_measurement_outcome_pool)

                edge_qubit_measurement = MeasurementEdge(
                    name = f"qubit_meas ({step})",
                    outcome = self._measurement_outcome_pool[idx],
                    real_map = (
                        self._qubit_projs_real[idx] if not self.qubit_measurement_is_ideal 
                        else qt.to_super(self._qubit_projs_ideal[idx])),
                    ideal_map = [self._qubit_projs_ideal[idx]],
                    to_ensemble = trashed,     
                )                    

                if trashed:
                    final_node = trash_node
                    # we don't add the trash node to the final_nodes,
                    # because it will not be evolved anymore (it has its
                    # own out edge, already connected to itself)
                else:
                    final_node = StateNode()
                    final_nodes.append(final_node)

                graph.add_node(final_node)

                graph.add_edge_connect(
                    edge_qubit_measurement,
                    node,
                    final_node,
                )

        return final_nodes

    # qubit reset ######################################################
    def _build_qubit_reset_process(
        self,
    ):
        self._qubit_reset_ideal = [cat_ideal.qubit_reset_propagator(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            res_mode_idx=self._res_mode_idx, superop=False,
        )]
        # currently the same as the ideal one
        _qubit_reset_real = cat_real.qubit_gate(
            self.fsweep.hilbertspace,
            self._res_mode_idx, self._qubit_mode_idx,
            self.res_dim, self.qubit_dim,
            eigensys = self.esys,
            rotation_angle = np.pi,
            gate_params = self.fsweep,
        )
        self._qubit_reset_real = self._kraus_to_super(
            self.frame_transform(_qubit_reset_real, self.fsweep["tau_p_eff"] * 2)
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
        
    def qubit_reset(
        self,
        step: int | str,
        graph: EvolutionTree,
        init_nodes: StateEnsemble,
    ) -> StateEnsemble:
        final_nodes = StateEnsemble()

        for node in init_nodes:
            edge_qubit_reset = PropagatorEdge(
                f"qubit_reset ({step})",
                (self._qubit_reset_map_real if not self.qubit_reset_is_ideal 
                    else self._kraus_to_super(self._qubit_reset_map_ideal)),
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

        for build in tqdm(builds):
            build()

    def generate_tree(
        self,
        init_prob_amp_01: Tuple[float, float],
        logical_0: qt.Qobj,
        logical_1: qt.Qobj,
        QEC_rounds: int = 1,
    ) -> EvolutionTree:
        """
        Currently, it only contains the idling process.
        """
        graph = EvolutionTree()

        # add some important nodes
        init_state_node = StateNode.initial_note(
            init_prob_amp_01, logical_0, logical_1,
        )
        graph.add_node(init_state_node)
        trash_node = TrashNode()
        trash_edge = TrashEdge("Trash - Trash", to_ensemble=False)
        graph.add_node(trash_node)
        graph.add_edge_connect(
            trash_edge,
            trash_node,
            trash_node,
        )

        # current ensemble
        current_ensemble = StateEnsemble([init_state_node])

        # add the idling edge
        step_counter = 0
        for round in range(QEC_rounds):
            current_ensemble = self.idle(
                f"{round}.{step_counter}", graph, current_ensemble,
            )
            step_counter += 1
            current_ensemble = self.qubit_gate_1(
                f"{round}.{step_counter}", graph, current_ensemble,
            )
            step_counter += 1
            current_ensemble = self.parity_mapping(
                f"{round}.{step_counter}", graph, current_ensemble,
            )
            step_counter += 1
            current_ensemble = self.qubit_gate_2(
                f"{round}.{step_counter}", graph, current_ensemble,
            )
            step_counter += 1
            current_ensemble = self.qubit_measurement(
                f"{round}.{step_counter}", graph, current_ensemble,
                trash_node,
            )
            step_counter += 1
            current_ensemble = self.qubit_reset(
                f"{round}.{step_counter}", graph, current_ensemble,
            )
            step_counter += 1

        return graph

    # def generate_tree_wo_QEC(
    #     self,
    #     init_prob_amp_01: Tuple[float, float],
    #     logical_0: qt.Qobj,
    #     logical_1: qt.Qobj,
    #     QEC_rounds: int = 1,
    # ) -> EvolutionTree:
    #     """
    #     Currently, it only contains the idling process.
    #     """
    #     graph = EvolutionTree()

    #     # add some important nodes
    #     init_state_node = StateNode.initial_note(
    #         init_prob_amp_01, logical_0, logical_1,
    #     )
    #     graph.add_node(init_state_node)
    #     trash_node = TrashNode()
    #     trash_edge = TrashEdge("Trash - Trash", to_ensemble=False)
    #     graph.add_node(trash_node)
    #     graph.add_edge_connect(
    #         trash_edge,
    #         trash_node,
    #         trash_node,
    #     )

    #     # current ensemble
    #     current_ensemble = StateEnsemble([init_state_node])

    #     # add the idling edge
    #     step_counter = 0
    #     for round in range(QEC_rounds):
    #         current_ensemble = self.idle(
    #             f"{round}.{step_counter}", graph, current_ensemble,
    #         )
    #         step_counter += 1

    #     return graph