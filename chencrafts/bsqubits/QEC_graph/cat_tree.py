import numpy as np
import qutip as qt
import scqubits as scq

from typing import List, Tuple, Any, TYPE_CHECKING, Dict, Callable

from chencrafts.cqed import FlexibleSweep, superop_evolve

from chencrafts.bsqubits.QEC_graph.node import StateNode, StateEnsemble, MeasurementRecord
from chencrafts.bsqubits.QEC_graph.edge import PropagatorEdge, MeasurementEdge, Edge
from chencrafts.bsqubits.QEC_graph.graph import EvolutionGraph, EvolutionTree

import chencrafts.bsqubits.cat_ideal as cat_ideal
import chencrafts.bsqubits.cat_real as cat_real


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

    # the ideal process and the real process share the same outcomes,
    # as the actual action is based on the real outcomes.
    _measurement_outcome_pool = [0, 1]

    # processes:
    _idling_real: qt.Qobj
    _idling_ideal: List[qt.Qobj]
    _qubit_gate_m_ideal: List[qt.Qobj]
    _qubit_gate_p_ideal: List[qt.Qobj]
    _qubit_gate_m_real: qt.Qobj
    _qubit_gate_p_real: qt.Qobj
    _parity_mapping_ideal: List[qt.Qobj]
    _parity_mapping_real: qt.Qobj
    _measurement_outcome_pool: List[int]
    _qubit_projs_ideal: List[qt.Qobj]
    _qubit_projs_real: List[qt.Qobj]
    _qubit_reset_real: qt.Qobj
    _qubit_reset_ideal: [qt.Qobj]
    _identity: qt.Qobj

    def __init__(
        self,
        fsweep: FlexibleSweep,
        sim_para: Dict[str, Any],
    ):
        super().__init__(fsweep, sim_para)

        self._find_sim_param()

        self._generate_cat_ingredients()

        self._build_all_processes()

    def _generate_cat_ingredients(
        self,
    ):
        """
        Construct the static Hamiltonian and the collapse operators
        in the diagonalized & rotating frame.
        """
        self.static_hamiltonian, self.c_ops, self.esys = cat_real.cavity_ancilla_me_ingredients(
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
        qubit_dim: int = 2,
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
        self.res_dim = self.fsweep.hilbertspace.subsystem_list[self._res_mode_idx].truncated_dim
        self.qubit_dim = qubit_dim

    # utils ############################################################
    @staticmethod
    def _current_parity(meas_record: MeasurementRecord):
        # with adaptive qubit pulse, detecting "1" meaning a parity flip
        return sum(meas_record) % 2

    # idling ###########################################################
    def _build_idling_process(
        self,
    ):
        """
        Add idling process edges to all of the initial nodes:
        - add the idling propagator edge
        - add the final node

        Parameters
        ----------
        graph : EvolutionTree
            The graph to be built
        init_nodes : StateEnsemble
            A collection of initial nodes that 
        """
        self._idling_real = cat_real.idling_propagator(
            self.static_hamiltonian,
            self.c_ops,
            # Qobj * array(1.0) will return a scipy sparse matrix
            # while Qobj * 1.0 returns a Qobj
            float(self.fsweep["T_W"]),
        )
        self._idling_ideal = cat_ideal.idling_proj_maps(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            res_mode_idx=self._res_mode_idx,
            static_hamiltonian=self.static_hamiltonian,
            time=float(self.fsweep["T_W"]),
            decay_rate=self.fsweep["kappa_s"],
            self_Kerr=self.fsweep["K_s"],
            # basis=self.esys[1],
        )

    def idle(
        self,
        step: int | str,
        graph: EvolutionTree,
        init_nodes: StateEnsemble,
    ) -> StateEnsemble:
        final_nodes = StateEnsemble()

        for node in init_nodes:
            edge_idling = PropagatorEdge(
                f"idling ({step})",
                self._idling_real,
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

        # currently the same as the ideal one   
        self._qubit_gate_p_real = qt.to_super(self._qubit_gate_p_ideal[0])
        self._qubit_gate_m_real = qt.to_super(self._qubit_gate_m_ideal[0])

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
                self._qubit_gate_p_real,
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
                self._qubit_gate_2_map_real,
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
        self._parity_mapping_ideal = [cat_ideal.parity_mapping_propagator(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            res_mode_idx=self._res_mode_idx, superop=False,
        )]

        # currently the same as the ideal one
        self._parity_mapping_real = qt.to_super(self._parity_mapping_ideal[0],)

    def parity_mapping(
        self,
        step: int | str,
        graph: EvolutionTree,
        init_nodes: StateEnsemble,
    ) -> StateEnsemble:
        final_nodes = StateEnsemble()

        for node in init_nodes:
            edge_parity_mapping = PropagatorEdge(
                f"parity_mapping ({step})",
                self._parity_mapping_real,
                self._parity_mapping_ideal,
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
        self._qubit_projs_ideal = cat_ideal.qubit_projectors(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            res_mode_idx=self._res_mode_idx, superop=False,
        )

        # currently the same as the ideal one
        self._qubit_projs_real = [
            qt.to_super(proj) for proj in self._qubit_projs_ideal
        ]

        # all of the lists should have the same length
        assert len(self._qubit_projs_ideal) == len(self._measurement_outcome_pool)
        assert len(self._qubit_projs_ideal) == len(self._qubit_projs_real)

    def qubit_measurement(
        self,
        step: int | str,
        graph: EvolutionTree,
        init_nodes: StateEnsemble,
    ) -> StateEnsemble:
        final_nodes = StateEnsemble()

        for node in init_nodes:
            for idx in range(len(self._qubit_projs_ideal)):
                edge_qubit_measurement = MeasurementEdge(
                    f"qubit_meas ({step})",
                    self._measurement_outcome_pool[idx],
                    self._qubit_projs_real[idx],
                    [self._qubit_projs_ideal[idx]],
                )
                final_nodes.append(StateNode())

                graph.add_node(final_nodes[-1])

                graph.add_edge_connect(
                    edge_qubit_measurement,
                    node,
                    final_nodes[-1],
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
        self._qubit_reset_real = qt.to_super(self._qubit_reset_ideal[0])

        # if there is no need to reset
        self._identity = cat_ideal.identity(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            res_mode_idx=self._res_mode_idx, superop=False,
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
            raise ValueError("The measurement outcome should be 0 or 1.")
        
    def _qubit_reset_map_ideal(
        self,
        meas_record: MeasurementRecord,
    ) -> List[qt.Qobj]:
        if meas_record[-1] == 1:
            return self._qubit_reset_ideal
        elif meas_record[-1] == 0:
            return [self._identity]
        else:
            raise ValueError("The measurement outcome should be 0 or 1.")
        
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
                self._qubit_reset_map_real,
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
    def _build_all_processes(
        self,
    ):
        self._build_idling_process()
        self._build_qubit_gate_process()
        self._build_parity_mapping_process()
        self._build_qubit_measurement_process()
        self._build_qubit_reset_process()

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

        # add the initial state
        init_state_node = StateNode.initial_note(
            init_prob_amp_01, logical_0, logical_1,
        )
        graph.add_node(init_state_node)
        current_ensemble = StateEnsemble([init_state_node])

        # add the idling edge
        step_counter = 0
        for round in range(QEC_rounds):
            current_ensemble = self.idle(
                f"{round}.{step_counter}", graph, current_ensemble)
            step_counter += 1
            current_ensemble = self.qubit_gate_1(
                f"{round}.{step_counter}", graph, current_ensemble)
            step_counter += 1
            current_ensemble = self.parity_mapping(
                f"{round}.{step_counter}", graph, current_ensemble)
            step_counter += 1
            current_ensemble = self.qubit_gate_2(
                f"{round}.{step_counter}", graph, current_ensemble)
            step_counter += 1
            current_ensemble = self.qubit_measurement(
                f"{round}.{step_counter}", graph, current_ensemble)
            step_counter += 1
            current_ensemble = self.qubit_reset(
                f"{round}.{step_counter}", graph, current_ensemble)
            step_counter += 1

        return graph

