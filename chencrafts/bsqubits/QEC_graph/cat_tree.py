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

    # processes:
    idling_real: qt.Qobj
    idling_ideal: Callable
    qubit_gate_m_ideal: qt.Qobj
    qubit_gate_p_ideal: qt.Qobj
    qubit_gate_m_real: qt.Qobj
    qubit_gate_p_real: qt.Qobj
    parity_mapping_ideal: qt.Qobj
    parity_mapping_real: qt.Qobj

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
        self.idling_real = cat_real.idling_propagator(
            self.static_hamiltonian,
            self.c_ops,
            # Qobj * array(1.0) will return a scipy sparse matrix
            # while Qobj * 1.0 returns a Qobj
            float(self.fsweep["T_W"]),
        )
        self.idling_ideal = cat_ideal.idling_proj_map(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            res_mode_idx=self._res_mode_idx,
            static_hamiltonian=self.static_hamiltonian,
            time=float(self.fsweep["T_W"]),
            decay_rate=self.fsweep["kappa_s"],
            self_Kerr=self.fsweep["K_s"],
            # basis=self.esys[1],
        )

    def _idle(
        self,
        graph: EvolutionTree,
        init_nodes: StateEnsemble,
    ) -> StateEnsemble:
        final_nodes = StateEnsemble()

        for node in init_nodes:
            edge_idling = PropagatorEdge(
                "idling",
                self.idling_real,
                self.idling_ideal,
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
        self.qubit_gate_p_ideal = cat_ideal.qubit_rot_propagator(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            res_mode_idx=self._res_mode_idx,
            angle=np.pi/2, axis="x", superop=True,
        )   # p stands for angle's sign plus
        self.qubit_gate_m_ideal = cat_ideal.qubit_rot_propagator(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            res_mode_idx=self._res_mode_idx,
            angle=-np.pi/2, axis="x", superop=True,
        )

        # currently the same as the ideal one   
        self.qubit_gate_p_real = self.qubit_gate_p_ideal
        self.qubit_gate_m_real = self.qubit_gate_m_ideal

    # the qubit gate after parity mapping
    def qubit_gate_2_real(
        self,
        meas_record: MeasurementRecord,
    ) -> qt.Qobj:
        """
        Try to keep the qubit at |0> while the parity isn't changed
        """
        if self._current_parity(meas_record) == 0:
            # even parity, use the opposite gate
            return self.qubit_gate_m_real
        else:
            # odd parity, use the same gate
            return self.qubit_gate_p_real
        
    def qubit_gate_2_ideal(
        self,
        proj: qt.Qobj,
        meas_record: MeasurementRecord,
    ) -> qt.Qobj:
        """
        Try to keep the qubit at |0> while the parity isn't changed
        """
        if self._current_parity(meas_record) == 0:
            # even parity, use the opposite gate
            return superop_evolve(self.qubit_gate_m_ideal, proj)
        else:
            # odd parity, use the same gate
            return superop_evolve(self.qubit_gate_p_ideal, proj)

    def _qubit_gate_1(
        self,
        graph: EvolutionTree,
        init_nodes: StateEnsemble,
    ) -> StateEnsemble:
        final_nodes = StateEnsemble()

        for node in init_nodes:
            edge_qubit_gate = PropagatorEdge(
                "qubit_gate_1",
                self.qubit_gate_p_real,
                lambda proj, _: superop_evolve(self.qubit_gate_p_ideal, proj),
            )
            final_nodes.append(StateNode())

            graph.add_node(final_nodes[-1])

            graph.add_edge_connect(
                edge_qubit_gate,
                node,
                final_nodes[-1],
            )

        return final_nodes
    
    def _qubit_gate_2(
        self,
        graph: EvolutionTree,
        init_nodes: StateEnsemble,
    ) -> StateEnsemble:
        final_nodes = StateEnsemble()

        for node in init_nodes:
            edge_qubit_gate = PropagatorEdge(
                "qubit_gate_2",
                self.qubit_gate_2_real,
                self.qubit_gate_2_ideal,
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
        self.parity_mapping_ideal = cat_ideal.parity_mapping_propagator(
            res_dim=self.res_dim, qubit_dim=self.qubit_dim,
            res_mode_idx=self._res_mode_idx, superop=True,
        )

        # currently the same as the ideal one
        self.parity_mapping_real = self.parity_mapping_ideal

    def _parity_mapping(
        self,
        graph: EvolutionTree,
        init_nodes: StateEnsemble,
    ) -> StateEnsemble:
        final_nodes = StateEnsemble()

        for node in init_nodes:
            edge_parity_mapping = PropagatorEdge(
                "parity_mapping",
                self.parity_mapping_real,
                self.parity_mapping_ideal,
            )
            final_nodes.append(StateNode())

            graph.add_node(final_nodes[-1])

            graph.add_edge_connect(
                edge_parity_mapping,
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

    def generate_tree(
        self,
        init_state: qt.Qobj,
    ) -> EvolutionTree:
        """
        Currently, it only contains the idling process.
        """
        graph = EvolutionTree()

        # add the initial state
        init_state_node = StateNode.initial_note(init_state)
        graph.add_node(init_state_node)
        init_ensemble = StateEnsemble([init_state_node])

        # add the idling edge
        last_ensemble = self._idle(graph, init_ensemble)
        last_ensemble = self._qubit_gate_1(graph, last_ensemble)
        last_ensemble = self._parity_mapping(graph, last_ensemble)
        last_ensemble = self._qubit_gate_2(graph, last_ensemble)

        return graph

